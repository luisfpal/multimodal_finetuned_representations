import argparse
import os
from typing import List

import torch
import torch.nn.functional as F

# Compat shim: the 8B OneVision remote code does `from
# transformers.modeling_flash_attention_utils import flash_attn_varlen_func`
# at module load (gated on is_flash_attn_available(), which is True on GPU
# nodes with flash_attn installed). transformers>=4.56 moved that symbol
# into _lazy_imports, so the import fails. We run eager attention and never
# call the function — rebind the symbol so the import just succeeds.
import transformers.modeling_flash_attention_utils as _fa_utils
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from utils.embeddings_extraction.cambrian_helpers import (
    CAMBRIAN_LAYER_PREFIX,
    CAMBRIAN_NORM_PATH,
    CambrianImageDataset,
    cambrian_collator,
    load_cambrian,
    make_cambrian_pool_info_provider,
    wrap_cambrian_for_pooling,
)
from utils.embeddings_extraction.cambrian_helpers import (
    IMAGE_NEWLINE_SENTINEL as CAMBRIAN_NEWLINE_SENTINEL,
)
from utils.embeddings_extraction.cambrian_helpers import (
    IMAGE_TOKEN_SENTINEL as CAMBRIAN_IMAGE_SENTINEL,
)
from utils.embeddings_extraction.caption_starters import starters
from utils.embeddings_extraction.dataset_utils import (
    DataCollatorForVisionLM,
    DetailImageDataset,
    InternVLImageDataset,
    TextOnlyDataset,
    build_coco_karpathy_records,
    build_imagenet_records,
    build_winoground_records,
    default_winoground_cache_dir,
    load_and_validate_detail23k,
)
from utils.embeddings_extraction.extraction_engine import save_representations
from utils.internvl2_compat import (
    patch_internvl_model_post_instantiation,
)
from utils.internvl_load_helpers import force_internvl_attn_implementation
from utils.vlm_layer_skipping.patch_cambrian import (
    install_compat as install_cambrian_compat,
)

if not hasattr(_fa_utils, "flash_attn_varlen_func"):
    try:
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

        _fa_utils.flash_attn_varlen_func = _flash_attn_varlen_func
    except ImportError:
        _fa_utils.flash_attn_varlen_func = None


def _detect_lm_layer_layout(model, n_layer):
    """Auto-detect the named_modules path prefix where the language model's
    decoder layers live, by checking common patterns in priority order.

    Different VLM families (and different transformers versions) place the
    LM decoder under different attribute paths. Hardcoding per-model
    branches drifts as HF restructures classes — this scan is robust to
    that. Returns ``(layer_prefix, norm_path_or_None)`` such that
    ``f"{layer_prefix}.{i}"`` is a valid named_modules entry for every
    ``i`` in ``[0, n_layer)``.

    Patterns tried (priority order, most specific first):
      * ``model.language_model.layers``  — OneVision (Qwen2.5-VL family);
        older LLaVA-hf where AutoModel returned the outer wrapper.
      * ``language_model.layers``        — modern LLaVA-1.5-hf (transformers
        >= 4.45 dropped the outer .model wrapper).
      * ``language_model.model.layers``  — InternVL2 (no outer .model
        wrapper, but its language_model is a CausalLM with a nested .model).
      * ``model.layers``                 — Cambrian, plain LlamaModel.
    """
    names = {name for name, _ in model.named_modules()}
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.layers",
    ]
    for cand in candidates:
        # Endpoints suffice — decoder layers are contiguous 0..n_layer-1.
        if f"{cand}.0" in names and f"{cand}.{n_layer - 1}" in names:
            norm_candidate = cand[: -len(".layers")] + ".norm"
            return cand, (norm_candidate if norm_candidate in names else None)
    raise RuntimeError(
        f"Could not auto-detect LM layer prefix for n_layer={n_layer}. "
        f"Tried: {candidates}. Top-level module names: "
        f"{sorted({n.split('.', 1)[0] for n in names if n})}"
    )


def get_target_layers(
    model,
    n_layer,
    option="",
    every=1,
    start=0,
    is_internvl=False,
    layer_prefix=None,
    norm_path=None,
):
    if is_internvl:
        # InternVL2 uses an InternLM2 backbone whose decoder norm modules are
        # named .attention_norm / .ffn_norm (not the LLaMA-style names).
        map_names = dict(
            norm1=".attention_norm",
            norm2=".ffn_norm",
            res2="",
        )
    else:
        map_names = dict(
            norm1=".input_layernorm",
            norm2=".post_attention_layernorm",
            res2="",
        )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    # Resolve layer path. Different VLM families place the LM decoder under
    # different attribute paths, and HF restructures classes between
    # transformers versions (e.g. modern LlavaForConditionalGeneration drops
    # the outer .model wrapper, so layers live at "language_model.layers.{i}"
    # rather than "model.language_model.layers.{i}"). Hardcoding per-model
    # branches drifts over time — auto-detect from named_modules() instead.
    # Explicit layer_prefix wins (used by the Cambrian branch in main()),
    # is_internvl is preserved as a hard override for backward compat.
    if layer_prefix is None:
        if is_internvl:
            layer_prefix = "language_model.model.layers"
            norm_path = "language_model.model.norm"
        else:
            layer_prefix, norm_path = _detect_lm_layer_layout(model, n_layer)
    elif norm_path is None:
        # Derive norm sibling from prefix: replace ".layers" suffix with ".norm".
        assert layer_prefix.endswith(
            ".layers"
        ), f"Cannot derive norm_path from {layer_prefix!r}; pass it explicitly."
        norm_path = layer_prefix[: -len(".layers")] + ".norm"

    target_layers = {
        i: f"{layer_prefix}.{i}{suffix}" for i in range(start, n_layer, every)
    }
    # Post-stack norm has label `n_layer`. Only attach it on a "default" sweep
    # (start=0); offset sweeps (e.g., --odd_layers fills in start=1) skip it
    # to avoid redundantly re-saving a tensor produced by the start=0 run.
    if (option == "norm1" or option == "norm2") and start == 0:
        if norm_path is None:
            raise RuntimeError(
                f"target_layer={option!r} requires a norm sibling next to "
                f"{layer_prefix!r}, but it's absent from named_modules()."
            )
        target_layers[n_layer] = norm_path

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./tokenizer")
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=6,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--avg_token",
        action="store_true",
        help="If passed, ID will be measured on the last token represenation.",
    )
    parser.add_argument(
        "--extract_all",
        action="store_true",
        help="Extract per-sample additional pooled activations: (1) average "
        "over image-token positions only (the IMG_CONTEXT positions for "
        "InternVL), and (2) the </img> end-of-image token representation. "
        "Currently supported only for InternVL2; saved alongside the main "
        "layer files with _imgavg / _eoi suffixes.",
    )
    parser.add_argument(
        "--save_distances",
        action="store_true",
        help="If passed, the distance matrices will be saved",
    )
    parser.add_argument(
        "--remove_duplicates",
        action="store_true",
        help="If passed, duplicate datapoints will be removed",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default="norm1",
        help="The name of the layer to analyze.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Stride over decoder layers when selecting which layers to extract: "
        "every=1 saves all layers, every=2 saves layers 0,2,4,...; etc.",
    )
    parser.add_argument(
        "--odd_layers",
        action="store_true",
        help="Shift the layer sweep to start at index 1 instead of 0 — combined with "
        "--every 2 this extracts only odd-indexed layers (1,3,5,...), useful for "
        "filling in the layers a previous --every 2 run skipped. The post-stack "
        "norm tensor (label n_layer) is omitted in this mode to avoid overwriting "
        "the one produced by the default sweep.",
    )
    parser.add_argument(
        "--similar_captions_n_prompts",
        type=int,
        default=50,
        help="First N dataset captions used as queries for per-layer neighbor JSON.",
    )
    parser.add_argument(
        "--similar_captions_k",
        type=int,
        default=10,
        help="Top-K most similar captions (cosine) per query at each layer.",
    )
    parser.add_argument(
        "--similar_captions_json_suffix",
        type=str,
        default="_similar_captions.json",
        help="Suffix for per-layer JSON files (after l{layer_id}{filename_stem}).",
    )
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=None,
        help="Suffix for per-layer JSON files (after l{layer_id}{filename_stem}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10],
        help="k values for k-NN overlap vs Qwen embeddings (e.g. --k 5 10 20).",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="qwen0.6b",
        choices=["qwen0.6b", "qwen8b"],
        help="Which Qwen embedder was used to produce the reference embeddings.",
    )
    parser.add_argument(
        "--gaussian_cka_sigma",
        type=float,
        nargs="+",
        default=[0.1],
        help="RBF bandwidth scale passed to gram_rbf(..., threshold=sigma) for Gaussian-kernel CKA.",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="Basename for the combined metrics JSON under out_dir/model/coco_image_question. "
        "Default: {target_layer}_qwen06b_llava13b_all_metrics.json",
    )
    parser.add_argument(
        "--detail_json",
        type=str,
        help="Path to detail_23k.json. Used when --dataset=coco.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="COCO: directory with COCO_train2014_* images. "
        "ImageNet: root folder with one subdir per class (n01440764/, ...).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["coco", "imagenet", "coco_karpathy", "winoground"],
        default="coco",
    )
    parser.add_argument(
        "--karpathy_split",
        type=str,
        default="test",
        help="Split to load from yerevann/coco-karpathy when --dataset=coco_karpathy.",
    )
    parser.add_argument(
        "--winoground_split",
        type=str,
        default="test",
        help="Split to load from facebook/winoground when --dataset=winoground "
        "(only 'test' is published, kept for symmetry).",
    )
    parser.add_argument(
        "--winoground_cache_dir",
        type=str,
        default=None,
        help="Where facebook/winoground PNGs are cached. Defaults to "
        "$HF_HOME/winoground_images (or ~/.cache/huggingface/winoground_images). "
        "When --dataset=winoground, --image_dir is ignored.",
    )
    parser.add_argument(
        "--class_subset",
        type=str,
        default=None,
        help="ImageNet: name of a list in animal_lists.py with class indices "
        "(e.g. animals_10). Overrides --num_classes / --seed.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="ImageNet: number of classes to sample (paired with --seed).",
    )
    parser.add_argument(
        "--num_images_per_class",
        type=int,
        default=None,
        help="ImageNet: images per class (deterministic, sorted filename order).",
    )
    parser.add_argument(
        "--no_generation_prompt",
        action="store_true",
        help="If passed, do not append the assistant generation prompt (e.g. 'ASSISTANT:') "
        "to the chat-template output.",
    )
    parser.add_argument(
        "--image_only",
        action="store_true",
        help="If passed, build prompts containing only the image (no question text). "
        "Any end-of-image tokens in the chat template are preserved.",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="If passed, feed only the detail_23k assistant 'detailed description' "
        "as the user message (no image, no question). Mutually exclusive with "
        "--image_only. Output goes under {out_dir}/{model}/coco_text_only/.",
    )
    args = parser.parse_args()
    if args.image_only and args.text_only:
        parser.error("--image_only and --text_only are mutually exclusive")
    return args


def first_prompts_topk_neighbors_json(
    embeddings: torch.Tensor,
    texts: List[str],
    n_prompts: int = 50,
    k: int = 10,
) -> list:
    """Cosine similarity in representation space; excludes self from neighbors."""
    n_unique = embeddings.shape[0]
    n_q = min(n_prompts, n_unique)
    k_eff = min(k, max(0, n_unique - 1))
    if k_eff == 0:
        return [{"index": i, "prompt": texts[i], "neighbors": []} for i in range(n_q)]

    emb = F.normalize(embeddings.float(), dim=1)
    sim = emb[:n_q] @ emb.T

    out = []
    for i in range(n_q):
        row = sim[i].clone()
        row[i] = -float("inf")
        vals, idx = torch.topk(row, k=k_eff)
        neighbors = [
            {
                "index": int(j),
                "prompt": texts[int(j)],
                "cosine_similarity": float(v),
            }
            for j, v in zip(idx.tolist(), vals.tolist())
        ]
        out.append({"index": i, "prompt": texts[i], "neighbors": neighbors})
    return out


def main():
    args = parse_args()

    # Winoground caches its own PIL images under --winoground_cache_dir, so
    # --image_dir is irrelevant for that dataset; every other path needs it.
    if args.dataset != "winoground" and not args.image_dir:
        raise ValueError("--image_dir is required")

    is_internvl = "internvl" in args.model_name_or_path.lower()
    is_cambrian = "cambrian" in args.model_name_or_path.lower()

    # Cambrian needs its full triple (model, tokenizer, image_processor) plus
    # conv_mode loaded together via the bridge — we capture them now so the
    # rest of main() can branch off `processor` (= tokenizer) without
    # re-loading. The actual model handle is stashed for later assignment.
    cambrian_state = None
    if is_cambrian:
        # eager attn here so the cross-modal forward path doesn't go through
        # flash-attn (consistent with lmm_eval.py's cambrian branch).
        cambrian_model, cambrian_tokenizer, cambrian_image_proc, cambrian_conv = (
            load_cambrian(args.model_name_or_path, attn_implementation="eager")
        )
        cambrian_state = {
            "model": cambrian_model,
            "tokenizer": cambrian_tokenizer,
            "image_processor": cambrian_image_proc,
            "conv_mode": cambrian_conv,
        }
        processor = cambrian_tokenizer
    elif is_internvl:
        # InternVL2 has no unified Processor — its tokenizer chat_template can't
        # handle list-content messages and it does its own image tiling. Load
        # the tokenizer directly and run the InternVL-specific dataset path.
        processor = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
    else:
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )

    folder = None
    if args.dataset == "coco":
        if not args.detail_json:
            raise ValueError("--detail_json is required when --dataset=coco")
        if args.text_only:
            # Text-only path doesn't open the COCO images, so skip the
            # existence check; just pull the gpt detailed descriptions.
            records = load_and_validate_detail23k(
                args.detail_json,
                args.image_dir,
                with_descriptions=True,
                validate_images=False,
            )
            print(f"Loaded {len(records)} detail_23k descriptions.")
            dataset_tag = "coco_text_only"
        else:
            print(f"Validating COCO images in {args.image_dir} ...")
            records = load_and_validate_detail23k(args.detail_json, args.image_dir)
            print(f"All {len(records)} images found.")
            dataset_tag = (
                "coco_image_only" if args.image_only else "coco_image_question"
            )
    elif args.dataset == "coco_karpathy":
        # yerevann/coco-karpathy: per-image record carries the original
        # captions list and a `description` = ". "-joined captions (via
        # return_caption). text_only feeds the description through
        # TextOnlyDataset; image_only skips text entirely; the combined
        # mode (neither flag) feeds image + description as the question.
        split = args.karpathy_split
        if args.text_only:
            records = build_coco_karpathy_records(
                args.image_dir, split=split, validate_images=False,
            )
            print(f"Loaded {len(records)} coco_karpathy {split} captions.")
            dataset_tag = f"coco_karpathy_{split}_text_only"
        else:
            print(f"Validating COCO Karpathy {split} images in {args.image_dir} ...")
            records = build_coco_karpathy_records(args.image_dir, split=split)
            print(f"All {len(records)} images found.")
            if not args.image_only:
                # Use concatenated captions as the question for the combined
                # image+text mode; image_only ignores the question entirely.
                for r in records:
                    r["question"] = r["description"]
            dataset_tag = (
                f"coco_karpathy_{split}_image_only" if args.image_only
                else f"coco_karpathy_{split}_image_caption"
            )
    elif args.dataset == "winoground":
        # facebook/winoground: 400 entries × (image_0/1, caption_0/1).
        # Flattened side-major: 800 records [(0,0),(0,1),(1,0),(1,1),...],
        # so the saved [2N, D] tensors reshape to [N, 2, D]. PIL images are
        # cached as PNGs under --winoground_cache_dir (default:
        # $HF_HOME/winoground_images) so the existing image_path-based
        # dataset classes work unchanged. text_only skips the PNG dump.
        split = args.winoground_split
        cache_dir = args.winoground_cache_dir or default_winoground_cache_dir()
        records = build_winoground_records(
            cache_dir, split=split, cache_images=not args.text_only,
        )
        print(f"Loaded {len(records)} winoground {split} records "
              f"({len(records) // 2} entries × 2 sides). "
              f"cache_dir={cache_dir} (cached={'no' if args.text_only else 'yes'})")
        if not (args.image_only or args.text_only):
            for r in records:
                r["question"] = r["caption"]
        if args.text_only:
            dataset_tag = f"winoground_{split}_text_only"
        elif args.image_only:
            dataset_tag = f"winoground_{split}_image_only"
        else:
            dataset_tag = f"winoground_{split}_image_caption"
    else:
        class_indices = None
        if args.class_subset is not None:
            import animal_lists

            if not hasattr(animal_lists, args.class_subset):
                available = sorted(
                    n
                    for n in dir(animal_lists)
                    if not n.startswith("_")
                    and isinstance(getattr(animal_lists, n), list)
                )
                raise ValueError(
                    f"Unknown class subset '{args.class_subset}'. Available: {available}"
                )
            class_indices = getattr(animal_lists, args.class_subset)
        elif args.num_classes is None:
            raise ValueError(
                "--class_subset or --num_classes is required when --dataset=imagenet"
            )
        print(f"Loading ImageNet from {args.image_dir} ...")
        records, folder = build_imagenet_records(
            args.image_dir,
            class_indices=class_indices,
            num_classes=args.num_classes,
            num_images_per_class=args.num_images_per_class,
            seed=args.seed if args.seed is not None else 0,
        )
        print(f"Loaded {len(records)} images across {len(folder.classes)} classes.")
        dataset_tag = (
            f"imagenet_{args.class_subset}"
            if args.class_subset is not None
            else "imagenet_captions"
        )

    if args.text_only:
        # Family resolution mirrors the model_name-based dispatch used to
        # pick processor / model loaders above. OneVision and LLaVA-1.5
        # share the HF processor path; only the chat template differs and
        # apply_chat_template handles that.
        is_onevision = "onevision" in args.model_name_or_path.lower()
        if is_cambrian:
            family = TextOnlyDataset.FAMILY_CAMBRIAN
        elif is_internvl:
            family = TextOnlyDataset.FAMILY_INTERNVL
        elif is_onevision:
            family = TextOnlyDataset.FAMILY_ONEVISION
        else:
            family = TextOnlyDataset.FAMILY_LLAVA_HF
        dataset = TextOnlyDataset(
            records,
            processor,
            family=family,
            cambrian_conv_mode=cambrian_state["conv_mode"] if is_cambrian else None,
            add_generation_prompt=not args.no_generation_prompt,
        )
        if is_cambrian or is_internvl:
            pad_token_id = processor.pad_token_id or processor.eos_token_id
        else:
            pad_token_id = processor.tokenizer.pad_token_id
    elif is_cambrian:
        dataset = CambrianImageDataset(
            records,
            tokenizer=cambrian_state["tokenizer"],
            image_processor=cambrian_state["image_processor"],
            model_config=cambrian_state["model"].config,
            conv_mode=cambrian_state["conv_mode"],
            image_only=args.image_only,
            image_dtype=torch.float16,
        )
        pad_token_id = processor.pad_token_id or processor.eos_token_id
    elif is_internvl:
        # num_image_token is fixed by the architecture; matches what
        # InternVLChatModel.__init__ computes from config:
        #   (image_size // patch_size) ** 2 * downsample_ratio ** 2
        # For InternVL2-8B (448 / 14, downsample 0.5) this is 256.
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        num_image_token = int(
            (config.force_image_size // config.vision_config.patch_size) ** 2
            * (config.downsample_ratio**2)
        )
        dataset = InternVLImageDataset(
            records,
            processor,
            num_image_token=num_image_token,
            image_only=args.image_only,
            add_generation_prompt=not args.no_generation_prompt,
        )
        pad_token_id = processor.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.eos_token_id
    else:
        dataset = DetailImageDataset(
            records,
            processor,
            image_only=args.image_only,
            add_generation_prompt=not args.no_generation_prompt,
        )
        pad_token_id = processor.tokenizer.pad_token_id
    n_expected = len(dataset)

    print(dataset[0]["text_prompts"])

    # Text-only mode produces only text tensors + userlast_pos; the generic
    # vision-LM collator handles the input_ids/attention_mask padding and
    # concatenates userlast_pos along dim 0. Use it for text-only Cambrian
    # too, since cambrian_collator assumes the per-sample image-tower list.
    if is_cambrian and not args.text_only:
        collate_fn = cambrian_collator
    else:
        collate_fn = DataCollatorForVisionLM(pad_token_id=pad_token_id)

    # batch_size MUST be 1 for InternVL2 (per-image variable tile count) and
    # for Cambrian (per-sample variable expansion offset of the -200 sentinel
    # would need a custom collator + per-sample expansion bookkeeping in the
    # pool_info_provider). The other paths also use 1 here, matching the
    # existing constraint.
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.preprocessing_num_workers,
        sampler=None,
        pin_memory=True,
    )

    from_pretrained_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # OneVision's vision tower has ~16K patch tokens per image; eager attention
    # materializes a [heads, N, N] float32 softmax (~15 GiB) and OOMs the A100.
    # flash_attn_varlen_func is available (shim above + flash_attn installed),
    # and unlike lmm_eval.py we don't apply patch_onevision, which is the only
    # thing that required eager.
    if "onevision" in args.model_name_or_path.lower():
        from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"

    if is_cambrian:
        # Cambrian was loaded above via load_cambrian (uses cambrian.model.builder
        # — AutoModel doesn't work, the snapshot has no remote .py code).
        model = cambrian_state["model"]
        # install_cambrian_compat is required on EVERY code path (see its
        # docstring): the vendored CambrianLlamaModel.forward is
        # incompatible with transformers >= 4.45 — it calls decoder layers
        # without the position_embeddings (cos, sin) tuple, raising
        # "cannot unpack non-iterable NoneType". The compat replacement
        # also patches prepare_inputs_labels_for_multimodal + generate;
        # both are harmless in text-only mode (cmm short-circuits when
        # images=None, and we never call generate here).
        install_cambrian_compat(model)
        if not args.text_only:
            # wrap_cambrian_for_pooling sets an instance-level attribute that
            # shadows the class-level cmm method, so it must run AFTER
            # install_cambrian_compat — it then wraps the cmm version as the
            # "original" that gets stashed. Skipped in text-only mode: no
            # -200 sentinel to expand, no _cambrian_pool_aux to record.
            wrap_cambrian_for_pooling(model)
    elif is_internvl:
        # - force_internvl_attn_implementation injects attn_implementation +
        #   use_flash_attn into the from_pretrained call (InternVLChatModel
        #   reads use_flash_attn at __init__ to pick the attention class).
        #   We use flash_attention_2 here — eager was only required by
        #   patch_internvl_lorenzo (cross-modal ablation, not applied here).
        # - patch_internvl_model_post_instantiation injects GenerationMixin
        #   into InternLM2ForCausalLM (transformers >= 4.50 dropped it from
        #   PreTrainedModel) and disables DynamicCache. We don't generate
        #   here, but the patch also fixes meta-tensor init in InternViT.
        with force_internvl_attn_implementation("flash_attention_2"):
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                **from_pretrained_kwargs,
            ).to("cuda")
        patch_internvl_model_post_instantiation(model)
        if not args.text_only:
            # Required: the forward replaces input_embeds at positions where
            # input_ids == img_context_token_id with vit features. chat() sets
            # this per-call; we set it once here. Skipped in text-only mode
            # — we bypass the multimodal forward entirely via forward_fn.
            model.img_context_token_id = processor.convert_tokens_to_ids(
                InternVLImageDataset.IMG_CONTEXT_TOKEN
            )
    else:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            **from_pretrained_kwargs,
        ).to("cuda")
    model = model.eval()

    # print(model)
    if is_cambrian:
        # CambrianLlamaForCausalLM is a Llama-3 backbone with vision attached;
        # config is a LlamaConfig (no nested language_model.config).
        n_layer = model.config.num_hidden_layers
    else:
        n_layer = model.language_model.config.num_hidden_layers

    # ***********************************************************************

    target_layers = get_target_layers(
        model=model,
        n_layer=n_layer,
        option=args.target_layer,
        is_internvl=is_internvl,
        layer_prefix=CAMBRIAN_LAYER_PREFIX if is_cambrian else None,
        norm_path=CAMBRIAN_NORM_PATH if is_cambrian else None,
        every=args.every,
        start=1 if args.odd_layers else 0,
    )
    nsamples = len(dataloader.dataset)
    target_layer_labels = list(target_layers.keys())

    print(f"target layer: {args.target_layer}")
    print("num_total_samples", nsamples)
    print("target_layers", target_layers)

    simplify_name = {
        "llava-hf/llava-1.5-7b-hf": "llava_7b_hf",
        "llava-hf/llava-1.5-13b-hf": "llava_13b_hf",
        "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct": "onevision_4b_hf",
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": "onevision_8b_hf",
        "OpenGVLab/InternVL2-8B": "internvl_8b_hf",
        "nyu-visionx/cambrian-8b": "cambrian_8b",
        "lbasile/llava_13b": "llava_13b",
        "lbasile/llava_7b": "llava_7b",
    }

    model_name = simplify_name[args.model_name_or_path]
    dirpath = args.out_dir + f"/{model_name}/{dataset_tag}"
    os.makedirs(dirpath, exist_ok=True)

    layers_dir = f"{dirpath}/layers"
    os.makedirs(layers_dir, exist_ok=True)
    token_tag = "avg" if args.avg_token else "last"
    filename = args.filename
    if args.filename != "":
        filename = f"_{args.filename}"

    layer_paths = {
        layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_{token_tag}{filename}.pt"
        for layer_id in target_layer_labels
    }
    predictions_path = f"{dirpath}/predicted_tokens.pt"

    # --extract_all: resolve token ids that select image-token positions and
    # the end-of-image marker. Per-layer extras are saved with _imgavg / _eoi
    # tags alongside the main file.
    #   InternVL2: <IMG_CONTEXT> for image positions, </img> for EOI.
    #   OneVision (Qwen2.5-VL family): <|image_pad|> and <|vision_end|>.
    #   Cambrian: synthetic sentinels (-200 / -201) injected by
    #     wrap_cambrian_for_pooling — Cambrian has no real EOI token, so the
    #     post-image-span position (first text activation that has attended
    #     over the full image) is used as the analog.
    image_token_id = None
    image_newline_token_id = None
    eoi_token_id = None
    special_token_ids = None
    layer_paths_imgavg = {}
    layer_paths_imgavg_with_newline = {}
    layer_paths_eoi = {}
    layer_paths_imgeoi = {}
    layer_paths_last = {}
    layer_paths_textavg = {}
    layer_paths_textspec = {}
    layer_paths_userlast = {}
    # Cambrian (image mode) needs the pool_info_provider unconditionally — the
    # model expands the -200 sentinel inside its forward, so activations come
    # out at length T_exp = T_orig - 1 + image_token_len. Pooling against the
    # input attention_mask (length T_orig) would silently mis-index. The
    # provider reads the post-expansion mask that wrap_cambrian_for_pooling
    # stashes on every forward. In text-only mode there's no -200 to expand,
    # so the input attention_mask already matches activation length.
    pool_info_provider = (
        make_cambrian_pool_info_provider(model)
        if (is_cambrian and not args.text_only)
        else None
    )

    # Forward routing for text-only mode: the InternVL2 wrapper requires
    # pixel_values as a positional arg, and Cambrian's outer cmm forward
    # would still try to look up ``images`` even though install_compat
    # short-circuits when it's None. Route through the LM body directly —
    # the activation hooks are bound to module objects
    # (model.language_model.model.layers.{i} / model.model.layers.{i}),
    # so they fire either way. install_cambrian_compat (applied above
    # unconditionally) replaces model.model.forward with the modern-
    # transformers-compatible llama_forward, so model.model(**batch) here
    # gets the position_embeddings fix. LLaVA-1.5-hf and OneVision
    # wrappers handle pixel_values=None natively, so no shim.
    forward_fn = None
    if args.text_only:
        if is_internvl:
            forward_fn = lambda batch: model.language_model(**batch, use_cache=False)
        elif is_cambrian:
            forward_fn = lambda batch: model.model(**batch, use_cache=False)
    if args.extract_all:
        is_onevision = "onevision" in args.model_name_or_path.lower()
        is_llava = "llava" in args.model_name_or_path.lower() and not is_onevision
        if args.text_only:
            # No image/EOI in text-only mode: leave both ids None so no
            # _imgavg / _eoi / _imgeoi files are emitted. text_avg pools
            # over all non-special positions (= description tokens) since
            # there's no boundary token to gate it. Specials come from the
            # model's tokenizer (HF processor wraps it at .tokenizer).
            tok = processor if (is_internvl or is_cambrian) else processor.tokenizer
            special_token_ids = list(tok.all_special_ids)
        elif is_internvl:
            image_token_id = processor.convert_tokens_to_ids(
                InternVLImageDataset.IMG_CONTEXT_TOKEN
            )
            eoi_token_id = processor.convert_tokens_to_ids(
                InternVLImageDataset.IMG_END_TOKEN
            )
            # All tokenizer special-token ids for InternVL minus image and
            # EOI; also drop <img> (the start-of-image marker) so that one
            # doesn't bleed into text_special.
            tok = processor
            img_start_id = tok.convert_tokens_to_ids(
                InternVLImageDataset.IMG_START_TOKEN
            )
            extras_drop = {image_token_id, eoi_token_id, img_start_id}
            special_token_ids = [
                tid for tid in tok.all_special_ids if tid not in extras_drop
            ]
        elif is_onevision:
            # The OneVision processor expands a single <image> placeholder into
            # <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>; the
            # repeat count is driven by image_grid_thw at processing time.
            tok = processor.tokenizer
            image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")
            eoi_token_id = tok.convert_tokens_to_ids("<|vision_end|>")
            # Drop image, EOI, and the matching <|vision_start|> / <|video_pad|>
            # so text_special only collects conversation-structure tokens.
            extras_drop = {
                image_token_id,
                eoi_token_id,
                tok.convert_tokens_to_ids("<|vision_start|>"),
                tok.convert_tokens_to_ids("<|video_pad|>"),
            }
            extras_drop.discard(None)
            special_token_ids = [
                tid for tid in tok.all_special_ids if tid not in extras_drop
            ]
        elif is_cambrian:
            # Cambrian has no image–text separator. We use -200 (the
            # synthetic image sentinel injected into synthetic_input_ids
            # by wrap_cambrian_for_pooling) to identify image positions
            # for _imgavg, and leave eoi_token_id unset — the post-image
            # boundary for _textavg / _textspec falls back to the last
            # image position. _eoi and _imgeoi files are intentionally
            # not written for Cambrian.
            image_token_id = CAMBRIAN_IMAGE_SENTINEL
            # -201 marks the per-row image_newline positions inside the
            # image span; this gives a separate _imgavg_with_newline pool
            # that includes them, while the default _imgavg excludes them.
            image_newline_token_id = CAMBRIAN_NEWLINE_SENTINEL
            eoi_token_id = None
            # The -200 / -201 sentinels aren't in the tokenizer's
            # all_special_ids, so no exclusion is needed here. Real
            # tokenizer ids at non-image positions in synthetic_input_ids
            # feed _textspec's all_special_ids matching directly.
            special_token_ids = list(processor.all_special_ids)
        elif is_llava:
            # LLaVA-1.5 has no end-of-image token. The processor expands
            # one <image> placeholder into image_seq_length copies of the
            # same image_token_id (32000 for llava-1.5-7b-hf), so the post-
            # image boundary that text_avg / text_special use is derived
            # downstream from the last image-token position (eoi_token_id
            # left as None — extract_activations handles this fallback).
            tok = processor.tokenizer
            image_token_id = tok.convert_tokens_to_ids("<image>")
            eoi_token_id = None
            extras_drop = {image_token_id}
            special_token_ids = [
                tid for tid in tok.all_special_ids if tid not in extras_drop
            ]
        else:
            raise NotImplementedError(
                "--extract_all is currently wired up for InternVL2, "
                "OneVision, Cambrian, and LLaVA-1.5; other model families "
                "need their image-token / end-of-image token ids resolved here."
            )
        layer_paths_imgavg = {
            layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_imgavg{filename}.pt"
            for layer_id in target_layer_labels
        }
        if image_newline_token_id is not None:
            layer_paths_imgavg_with_newline = {
                layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_imgavg_with_newline{filename}.pt"
                for layer_id in target_layer_labels
            }
        layer_paths_eoi = {
            layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_eoi{filename}.pt"
            for layer_id in target_layer_labels
        }
        layer_paths_imgeoi = {
            layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_imgeoi{filename}.pt"
            for layer_id in target_layer_labels
        }
        layer_paths_last = {
            layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_last{filename}.pt"
            for layer_id in target_layer_labels
        }
        layer_paths_textavg = {
            layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_textavg{filename}.pt"
            for layer_id in target_layer_labels
        }
        layer_paths_textspec = {
            layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_textspec{filename}.pt"
            for layer_id in target_layer_labels
        }
        if args.text_only:
            layer_paths_userlast = {
                layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}_userlast{filename}.pt"
                for layer_id in target_layer_labels
            }

    extras = None
    if args.dataset == "imagenet":
        result = save_representations(
            model=model,
            dataloader=dataloader,
            target_layers=target_layers,
            nsamples=nsamples,
            dirpath=dirpath,
            filename=args.filename,
            use_last_token=not args.avg_token,
            print_every=1000,
            collect_predictions=True,
            image_token_id=image_token_id,
            eoi_token_id=eoi_token_id,
            pool_info_provider=pool_info_provider,
            special_token_ids=special_token_ids,
            forward_fn=forward_fn,
            track_user_last=args.text_only and args.extract_all,
            image_newline_token_id=image_newline_token_id,
        )
        if args.extract_all:
            act_dict, predicted_token_ids, extras = result
        else:
            act_dict, predicted_token_ids = result
        # `processor` is the tokenizer itself for InternVL2 (loaded via
        # AutoTokenizer) and for Cambrian (cambrian bridge); other paths
        # use a Processor whose tokenizer is at `.tokenizer`.
        decoder = processor if (is_internvl or is_cambrian) else processor.tokenizer
        predicted_token_strs = decoder.batch_decode(
            [[tid] for tid in predicted_token_ids], skip_special_tokens=False
        )
        torch.save(
            {"token_ids": predicted_token_ids, "token_strs": predicted_token_strs},
            predictions_path,
        )
    else:
        result = save_representations(
            model=model,
            dataloader=dataloader,
            target_layers=target_layers,
            nsamples=nsamples,
            dirpath=dirpath,
            filename=args.filename,
            use_last_token=not args.avg_token,
            print_every=1000,
            image_token_id=image_token_id,
            eoi_token_id=eoi_token_id,
            pool_info_provider=pool_info_provider,
            special_token_ids=special_token_ids,
            forward_fn=forward_fn,
            track_user_last=args.text_only and args.extract_all,
            image_newline_token_id=image_newline_token_id,
        )
        if args.extract_all:
            act_dict, extras = result
        else:
            act_dict = result
        predicted_token_ids = None
        predicted_token_strs = None
    for layer_id, layer_name in target_layers.items():
        torch.save(act_dict[layer_name], layer_paths[layer_id])
    if extras is not None:
        for layer_id, layer_name in target_layers.items():
            if extras["image_avg"] is not None:
                torch.save(
                    extras["image_avg"][layer_name],
                    layer_paths_imgavg[layer_id],
                )
            if extras.get("image_avg_with_newline") is not None:
                torch.save(
                    extras["image_avg_with_newline"][layer_name],
                    layer_paths_imgavg_with_newline[layer_id],
                )
            if extras["eoi"] is not None:
                torch.save(
                    extras["eoi"][layer_name],
                    layer_paths_eoi[layer_id],
                )
            if extras["image_eoi_avg"] is not None:
                torch.save(
                    extras["image_eoi_avg"][layer_name],
                    layer_paths_imgeoi[layer_id],
                )
            if extras["last"] is not None:
                torch.save(
                    extras["last"][layer_name],
                    layer_paths_last[layer_id],
                )
            if extras["text_avg"] is not None:
                torch.save(
                    extras["text_avg"][layer_name],
                    layer_paths_textavg[layer_id],
                )
            if extras["text_special"] is not None:
                torch.save(
                    extras["text_special"][layer_name],
                    layer_paths_textspec[layer_id],
                )
            if extras.get("user_last") is not None:
                torch.save(
                    extras["user_last"][layer_name],
                    layer_paths_userlast[layer_id],
                )

    if args.dataset == "imagenet":
        metadata_path = f"{dirpath}/metadata.pt"
        torch.save(
            {
                "image_paths": [r["image_path"] for r in records],
                "class_labels": [r["class_label"] for r in records],
                "imagenet_codes": [r["imagenet_code"] for r in records],
                "starters": [r["starter"] for r in records],
                "starter_indices": [r["starter_idx"] for r in records],
                "questions": [r["question"] for r in records],
                "text_prompts": dataset.text_prompts,
                "classes": folder.classes,
                "class_to_idx": folder.class_to_idx,
                "caption_starters": list(starters),
                "predicted_token_ids": predicted_token_ids,
                "predicted_token_strs": predicted_token_strs,
            },
            metadata_path,
        )
        print(f"Saved metadata to {metadata_path}")
    elif args.dataset == "coco_karpathy":
        metadata_path = f"{dirpath}/metadata.pt"
        torch.save(
            {
                "filenames": [r["filename"] for r in records],
                "image_paths": [r["image_path"] for r in records],
                "captions": [r["captions"] for r in records],
                "descriptions": [r["description"] for r in records],
                "text_prompts": dataset.text_prompts,
                "split": args.karpathy_split,
            },
            metadata_path,
        )
        print(f"Saved metadata to {metadata_path}")
    elif args.dataset == "winoground":
        metadata_path = f"{dirpath}/metadata.pt"
        torch.save(
            {
                "entry_ids": [r["entry_id"] for r in records],
                "sides": [r["side"] for r in records],
                "tags": [r["tag"] for r in records],
                "secondary_tags": [r["secondary_tag"] for r in records],
                "captions": [r["caption"] for r in records],
                "filenames": [r["filename"] for r in records],
                "image_paths": [r["image_path"] for r in records],
                "text_prompts": dataset.text_prompts,
                "split": args.winoground_split,
                "n_entries": len(records) // 2,
                "layout": "side-major: rows ordered (0,0),(0,1),(1,0),(1,1),... "
                          "— reshape [2N,D] -> [N,2,D] for per-entry pairs.",
            },
            metadata_path,
        )
        print(f"Saved metadata to {metadata_path}")

    caption_texts = dataset.text_prompts
    assert len(caption_texts) == nsamples, (
        len(caption_texts),
        nsamples,
    )


if __name__ == "__main__":

    main()
