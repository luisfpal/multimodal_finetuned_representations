"""SentEval-style linguistic probing: extract per-layer activations and fit linear probes.

Inputs: --model_name_or_path and other CLI args; SentEval task data.
Outputs: one JSON file of per-layer accuracies per pooling strategy.
"""

import argparse
import gc
import json
import os
from collections import defaultdict
from functools import partial

import numpy as np
import torch

# Compat shim for OneVision remote code that imports a symbol
# transformers>=4.56 moved into _lazy_imports. Mirrors the shim in
# embeddings_vlms.py — required even in text-only mode because the OneVision
# remote module load happens at AutoModel.from_pretrained time.
import transformers.modeling_flash_attention_utils as _fa_utils
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from utils.probing.activation_extraction import extract_activations
from utils.probing.layer_detection import get_embdims, get_target_layers
from utils.probing.probes import fit_sklearn_no_cv
from utils.probing.senteval_data import DataCollatorForCausalLM, load_senteval_data

if not hasattr(_fa_utils, "flash_attn_varlen_func"):
    try:
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

        _fa_utils.flash_attn_varlen_func = _flash_attn_varlen_func
    except ImportError:
        _fa_utils.flash_attn_varlen_func = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/probing",
        help="Path to SentEval-format probing task data.",
    )
    parser.add_argument("--model_name_or_path", type=str, default="./tokenizer")
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=6,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--use_last_token",
        action="store_true",
        help="If passed, use the last token representation instead of averaging.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=5000,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=5000,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--n_rep",
        type=int,
        default=3,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="all",
        help="Name of the SentEval task to run. Accepts an individual task name "
        "(SentLen, WC, TreeDepth, ...), a group ('surface' / 'syntactic' / "
        "'semantic'), or 'all' for every task.",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default="norm1",
        help="The name of the layer to analyze.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-device batch size for the activation forward passes.",
    )
    args = parser.parse_args()
    return args


TASK_FILE_NAMES = {
    "SentLen": "sentence_length.txt",
    "WC": "word_content.txt",
    "TreeDepth": "tree_depth.txt",
    "TopConst": "top_constituents.txt",
    "BShift": "bigram_shift.txt",
    "Tense": "past_present.txt",
    "SubjNum": "subj_number.txt",
    "ObjNum": "obj_number.txt",
    "SOMO": "odd_man_out.txt",
    "CoordInv": "coordination_inversion.txt",
}

# SentEval task groupings from Conneau et al. 2018 ("What you can cram into a
# single $&!#* vector"). Used by --task_name to run a subset of related probes.
TASK_GROUPS = {
    "surface": ["SentLen", "WC"],
    "syntactic": ["BShift", "TreeDepth", "TopConst"],
    "semantic": ["Tense", "SubjNum", "ObjNum", "SOMO", "CoordInv"],
}


# Family tags, mirrored from embeddings_vlms.py's dispatch (model_name_or_path
# substring match) so the probing script accepts the same six VLMs plus the
# original Vicuna baseline.
FAMILY_VICUNA = "vicuna"
FAMILY_LLAVA_HF = "llava_hf"
FAMILY_ONEVISION = "onevision"
FAMILY_INTERNVL = "internvl"
FAMILY_CAMBRIAN = "cambrian"


def detect_family(model_name_or_path: str) -> str:
    s = model_name_or_path.lower()
    if "internvl" in s:
        return FAMILY_INTERNVL
    if "cambrian" in s:
        return FAMILY_CAMBRIAN
    if "onevision" in s:
        return FAMILY_ONEVISION
    if "llava" in s:
        return FAMILY_LLAVA_HF
    return FAMILY_VICUNA


SIMPLIFY_NAME = {
    "lmsys/vicuna-7b-v1.5": "vicuna",
    "llava-hf/llava-1.5-7b-hf": "llava_7b_hf",
    "llava-hf/llava-1.5-13b-hf": "llava_13b_hf",
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct": "onevision_4b_hf",
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": "onevision_8b_hf",
    "OpenGVLab/InternVL2-8B": "internvl_8b_hf",
    "nyu-visionx/cambrian-8b": "cambrian_8b",
    "lbasile/llava_13b": "llava_13b",
    "lbasile/llava_7b": "llava_7b",
}


def vicuna_instruct_template(text):
    return f"USER: {text}"


def make_instruction_template(family, processor=None, cambrian_conv_mode=None):
    """Returns a callable text -> rendered prompt string for the given family.

    Each rendering produces a single user turn with the SentEval sentence as
    its content (no assistant scaffold), matching the original Vicuna
    behavior of ``USER: {text}``.
    """
    if family == FAMILY_VICUNA:
        return vicuna_instruct_template

    if family in (FAMILY_LLAVA_HF, FAMILY_ONEVISION):
        def fn(text):
            return processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": text}]}],
                add_generation_prompt=False,
                tokenize=False,
            )
        return fn

    if family == FAMILY_INTERNVL:
        from utils.internvl_conversations import get_conv_template

        def fn(text):
            tmpl = get_conv_template("internlm2-chat")
            tmpl.append_message(tmpl.roles[0], text)
            return tmpl.get_prompt()
        return fn

    if family == FAMILY_CAMBRIAN:
        assert cambrian_conv_mode is not None
        from cambrian.conversation import conv_templates

        def fn(text):
            conv = conv_templates[cambrian_conv_mode].copy()
            conv.append_message(conv.roles[0], text)
            return conv.get_prompt()
        return fn

    raise ValueError(f"Unknown family: {family!r}")


def load_model_and_tokenizer(args, family):
    """Family-aware load. Returns (model, tokenizer, cambrian_conv_mode_or_None).

    `tokenizer` is the underlying token-string tokenizer (the SentEval
    pipeline only ever needs text-side tokenization). For LLaVA-hf and
    OneVision we also pull the processor out for chat-template rendering;
    we expose it here as `processor_for_chat` via attribute on tokenizer
    only when the chat template lives on the processor — but because the
    processor's tokenizer is shared, returning the processor itself for
    those families and the bare tokenizer otherwise is simpler.
    """
    if family == FAMILY_VICUNA:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=False
        )
        print("loading model")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, dtype=torch.bfloat16,
        ).to("cuda")
        return model, tokenizer, None, tokenizer

    if family == FAMILY_CAMBRIAN:
        # Cambrian has no remote .py code in the snapshot — load via the
        # vendored bridge, then install the modern-transformers compat
        # patch so model.model(**batch) (used by our text-only forward_fn)
        # accepts position_embeddings the way recent Llama decoders expect.
        from utils.embeddings_extraction.cambrian_helpers import load_cambrian
        from utils.vlm_layer_skipping.patch_cambrian import (
            install_compat as install_cambrian_compat,
        )

        print("loading cambrian")
        model, tokenizer, _image_processor, conv_mode = load_cambrian(
            args.model_name_or_path, attn_implementation="eager",
        )
        install_cambrian_compat(model)
        return model, tokenizer, conv_mode, tokenizer

    if family == FAMILY_INTERNVL:
        from utils.internvl2_compat import (
            patch_internvl_model_post_instantiation,
        )
        from utils.internvl_load_helpers import (
            force_internvl_attn_implementation,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        print("loading internvl")
        with force_internvl_attn_implementation("flash_attention_2"):
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to("cuda")
        patch_internvl_model_post_instantiation(model)
        return model, tokenizer, None, tokenizer

    # llava-1.5-hf or onevision: AutoProcessor + AutoModel; the chat
    # template lives on the processor, so we return it as processor_for_chat.
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    from_pretrained_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if family == FAMILY_ONEVISION:
        # OneVision's vision tower would OOM under eager attn even in
        # text-only mode (the LM forward still allocates KV-cache layout
        # consistent with flash). Same setting embeddings_vlms.py uses.
        from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"
    print(f"loading {family}")
    model = AutoModel.from_pretrained(
        args.model_name_or_path, **from_pretrained_kwargs,
    ).to("cuda")
    return model, processor.tokenizer, None, processor


def get_n_layer(model, family):
    if family == FAMILY_VICUNA:
        return model.config.num_hidden_layers
    if family == FAMILY_CAMBRIAN:
        return model.config.num_hidden_layers
    # All HF wrappers (LLaVA-hf, OneVision, InternVL2) expose the LM via
    # model.language_model whose .config carries num_hidden_layers.
    return model.language_model.config.num_hidden_layers


def make_forward_fn(model, family, args):
    """Routing for the forward pass during activation extraction.

    LLaVA-1.5-hf and OneVision wrappers handle pixel_values=None natively,
    so the default ``model(**batch)`` works. InternVL2's wrapper calls
    ``image_flags.squeeze(-1)`` unconditionally, and Cambrian's outer cmm
    forward inspects the ``images`` argument — both require routing
    through the LM body to skip the multimodal path entirely.
    """
    if family == FAMILY_INTERNVL:
        return lambda batch: model.language_model(**batch, use_cache=False)
    if family == FAMILY_CAMBRIAN:
        # install_cambrian_compat (applied in load_model_and_tokenizer)
        # replaces model.model.forward with the modern-transformers
        # llama_forward, so this call gets the position_embeddings fix.
        return lambda batch: model.model(**batch, use_cache=False)
    return None


def get_pad_token_id(tokenizer):
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        pad = 0
    return pad


def process_data(inputs, tokenizer, apply_chat_template, instruction_template):

    if apply_chat_template:
        inputs = [instruction_template(text) for text in inputs["text"]]
    else:
        inputs = [text for text in inputs["text"]]

    inputs = tokenizer(inputs, add_special_tokens=False)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }


def get_activations(
    dataset,
    process_data_func,
    model,
    target_layer_names,
    pad_token_id,
    forward_fn=None,
    special_token_ids=None,
    extract_pools=None,
):
    """Run extraction over `dataset` and return per-layer pooled activations.

    When `extract_pools` is non-empty the result is a dict keyed by pool name
    (`text_avg`, `text_special`, `last_text`) mapping to the standard
    {layer_id: (nsamples, embdim) tensor}. Otherwise the legacy single-dict
    return is preserved (driven by `args.use_last_token` for last-vs-mean).
    """
    dataset = dataset.map(
        process_data_func,
        batched=True,
        batch_size=args.preprocessing_num_workers,
        num_proc=1,
    )
    # Left-padding is required by the OneVision wrapper under flash_attention_2
    # (its _update_causal_mask raises on right-padded batches), and is safe for
    # the other families in forward-only inference. The extractor's pooling
    # detects left-padding via mask[:, -1].sum() == batch_size and reads the
    # last real token at index -1.
    collate_fn = DataCollatorForCausalLM(
        pad_token_id=pad_token_id, padding_side="left"
    )
    loader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.preprocessing_num_workers,
        sampler=None,
        pin_memory=True,
    )
    embdims, dtypes = get_embdims(
        model, loader, target_layer_names, forward_fn=forward_fn
    )
    extractor = extract_activations(
        model=model,
        dataloader=loader,
        target_layers=target_layer_names,
        embdims=embdims,
        dtypes=dtypes,
        use_last_token=args.use_last_token,
        print_every=args.print_every,
        forward_fn=forward_fn,
        special_token_ids=special_token_ids,
        extract_pools=extract_pools,
    )

    extractor.extract(loader)
    if extract_pools:
        result = {}
        if "text_avg" in extract_pools:
            result["text_avg"] = extractor.hidden_states_text_avg
        if "text_special" in extract_pools:
            result["text_special"] = extractor.hidden_states_text_special
        if "last_text" in extract_pools:
            result["last_text"] = extractor.hidden_states_last_text
        return result
    return extractor.hidden_states


def get_train_subsets(dataset, subset_size=5000, n_rep=5):
    assert subset_size < 8000
    labels = dataset["label"]

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    num_labels = len(label_to_indices)

    if num_labels*100 > 8000:
        subset_size = 8000
    elif num_labels*100 > subset_size:
        subset_size = num_labels*5000

    rng = np.random.default_rng(seed=42)
    for label in label_to_indices:
        rng.shuffle(label_to_indices[label])

    n = len(dataset) // subset_size
    assert n_rep <= n, f"n_rep ({n_rep}) must be <= n ({n})"

    label_chunks = {}
    for label, indices in label_to_indices.items():
        if len(indices) < n:
            raise ValueError(f"Label {label} has fewer samples than subsets")
        label_chunks[label] = np.array_split(indices, n)
    subsets = []

    for i in range(n_rep):
        subset_indices = []
        for label in label_chunks:
            subset_indices.extend(label_chunks[label][i])

        subsets.append(dataset.select(subset_indices))
    subsets = [subset.shuffle(seed=42) for subset in subsets]

    return subsets


def main(args):

    family = detect_family(args.model_name_or_path)
    print(f"family: {family}")

    model, tokenizer, cambrian_conv_mode, processor_for_chat = (
        load_model_and_tokenizer(args, family)
    )

    model = model.eval()
    n_layer = get_n_layer(model, family)

    instruction_template = make_instruction_template(
        family,
        processor=processor_for_chat,
        cambrian_conv_mode=cambrian_conv_mode,
    )

    target_layers = get_target_layers(
        model=model,
        n_layer=n_layer,
        option=args.target_layer,
        every=1,
        is_internvl=(family == FAMILY_INTERNVL),
    )
    target_layer_names = list(target_layers.values())
    print("target_layers", target_layers)

    forward_fn = make_forward_fn(model, family, args)
    pad_token_id = get_pad_token_id(tokenizer)

    # Tokenizer special-token ids (BOS / EOS / PAD plus chatml tokens like
    # <|im_start|> / <|im_end|> when present). The extractor uses these to
    # split each rendered prompt into "text" (real, non-special) and
    # "special" (chat scaffold) positions. For families whose chat template
    # renders entirely as plain tokens (Vicuna / LLaVA-1.5-hf default), this
    # list is BOS/EOS/etc. — none of which appear in the rendered output —
    # so the special pool is empty and the text pool degenerates to the
    # full mean. Same convention as embeddings_vlms.py text-only mode.
    special_token_ids = (
        list(tokenizer.all_special_ids)
        if hasattr(tokenizer, "all_special_ids")
        else []
    )

    # SentEval setup: probe the user-supplied sentence only — chat templates
    # are rendered with no system prompt and no assistant header, and we
    # always extract two pools per run:
    #   - text_avg: mean over real, non-special positions (user text content).
    #   - last_text: rightmost real, non-special position (last user-text
    #     token; for chatml-style templates this lands BEFORE the closing
    #     <|im_end|> rather than on it).
    # The text_special pool from embeddings_vlms.py is intentionally dropped
    # here because we only care about the user content, not the scaffold.
    # --use_last_token is left as a no-op (pool layout is fixed for SentEval).
    extract_pools = ["text_avg", "last_text"]

    if args.model_name_or_path in SIMPLIFY_NAME:
        model_name = SIMPLIFY_NAME[args.model_name_or_path]
    else:
        # Fall back to a slug derived from the path so the script still
        # runs on user-provided checkpoints not in the table.
        model_name = args.model_name_or_path.rstrip("/").split("/")[-1]
    # Layout: {out_dir}/{model_name}/{target_layer}/{pool_name}/{task}_accuracy_logistic.json
    # The pool_name leaf replaces the previous {last|mean} subdir; a "mean"-
    # mode run now produces TWO sibling subdirs (text_avg/ and text_special/).
    dirpath_base = os.path.join(args.out_dir, model_name, args.target_layer)
    os.makedirs(dirpath_base, exist_ok=True)

    fn = ""
    if args.filename:
        fn += args.filename
    # *******************************************************************************

    if args.task_name == "all":
        task_names = list(TASK_FILE_NAMES.keys())
    elif args.task_name in TASK_GROUPS:
        task_names = TASK_GROUPS[args.task_name]
    elif args.task_name in TASK_FILE_NAMES:
        task_names = [args.task_name]
    else:
        valid = (
            list(TASK_FILE_NAMES.keys()) + list(TASK_GROUPS.keys()) + ["all"]
        )
        raise ValueError(f"{args.task_name} not valid. Choose from: {valid}")

    for task_name in task_names:
        print(task_name, "\n")
        train_dataset, test_dataset, val_dataset = load_senteval_data(
            task_name, args.data_path
        )
        train_subsets = get_train_subsets(
            train_dataset, subset_size=args.subset_size, n_rep=args.n_rep
        )

        process_data_func = partial(
            process_data,
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            apply_chat_template=True,
        )

        test_pools = get_activations(
            test_dataset,
            process_data_func,
            model,
            target_layer_names,
            pad_token_id=pad_token_id,
            forward_fn=forward_fn,
            special_token_ids=special_token_ids,
            extract_pools=extract_pools,
        )
        # test_pools: {pool_name: {layer_id: tensor}}. Convert each pool's
        # per-layer tensor to a numpy float64 array up front — these survive
        # across all train reps and feed every fit_layer call.
        test_pools_np = {
            pool_name: {
                layer: tensor.to(torch.float64).numpy()
                for layer, tensor in pool_data.items()
            }
            for pool_name, pool_data in test_pools.items()
        }
        y_test = np.array(test_dataset["label"])

        accuracy_per_pool = {
            pool_name: {layer: [] for layer in test_pools_np[pool_name].keys()}
            for pool_name in extract_pools
        }

        for i, train_subset in enumerate(train_subsets):
            print(f"rep {i}/{args.n_rep}")
            train_pools = get_activations(
                train_subset,
                process_data_func,
                model,
                target_layer_names,
                pad_token_id=pad_token_id,
                forward_fn=forward_fn,
                special_token_ids=special_token_ids,
                extract_pools=extract_pools,
            )
            y_train_sub = np.array(train_subset["label"])

            for pool_name in extract_pools:
                X_train_sub = train_pools[pool_name]
                test_layers = test_pools_np[pool_name]

                def fit_layer(
                    layer, X_test, X_train_sub=X_train_sub, y_train_sub=y_train_sub
                ):
                    X_train = X_train_sub[layer].to(torch.float64).numpy()
                    return layer, fit_sklearn_no_cv(
                        X_train, y_train_sub, X_test, y_test
                    )

                results = Parallel(n_jobs=8)(
                    delayed(fit_layer)(layer, X_test)
                    for layer, X_test in test_layers.items()
                )
                for layer, acc in results:
                    accuracy_per_pool[pool_name][layer].append(acc)

            del train_pools
            gc.collect()

            for pool_name in extract_pools:
                running = [
                    np.mean(acc)
                    for acc in accuracy_per_pool[pool_name].values()
                    if acc
                ]
                if running:
                    print(
                        f"  {pool_name}: mean-over-layers running acc = "
                        f"{np.mean(running):.4f}"
                    )

        for pool_name in extract_pools:
            pool_dir = os.path.join(dirpath_base, pool_name)
            os.makedirs(pool_dir, exist_ok=True)
            out_path = os.path.join(
                pool_dir, f"{task_name}_accuracy_logistic.json"
            )
            with open(out_path, "w") as f:
                json.dump(accuracy_per_pool[pool_name], f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
