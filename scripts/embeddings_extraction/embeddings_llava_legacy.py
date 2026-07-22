"""LEGACY: superseded by embeddings_vlms.py's LLaVA path. Kept for provenance/reference only; not actively maintained."""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from utils.embeddings_extraction.extraction_engine import save_representations
from utils.embeddings_extraction.imbalance_adapter import compute_imbalance_torch
from utils.embeddings_extraction.legacy_cka_overlap_helpers import (
    cka,
    compute_overlap_torch,
    feature_space_linear_cka,
    gram_rbf,
)


def get_target_layers(model, n_layer, option="", every=1):
    map_names = dict(
        norm1=".input_layernorm",
        norm2=".post_attention_layernorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    target_layers = {
        i: f"language_model.layers.{i}{suffix}" for i in range(0, n_layer, every)
    }
    if option == "norm1" or option == "norm2":
        target_layers[n_layer] = "language_model.norm"

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers


def return_caption(captions):
    return ". ".join(s.strip().rstrip(".") for s in captions) + "."


LLAVA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] + ' '}}{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)

# Plain-Python equivalent of LLAVA_CHAT_TEMPLATE.
# Each message is {"role": str, "content": list[{"type": "text"|"image", "text": str}]}.
# System messages are rendered without the role prefix.
# For every message: images (<image>\n) come first, then text tokens (each followed by a space).
# The {% generation %}...{% endgeneration %} markers in the Jinja template only affect
# tokenization boundaries for training; in plain text they produce the same output as
# non-assistant text, so the branch below is identical for both roles.
#
# def apply_llava_chat_template(
#     messages: list[dict],
#     add_generation_prompt: bool = True,
# ) -> str:
#     result = ""
#     for message in messages:
#         if message["role"] != "system":
#             result += message["role"].upper() + ": "
#         for content in message["content"]:          # images first
#             if content["type"] == "image":
#                 result += "<image>\n"
#         for content in message["content"]:          # text next
#             if content["type"] == "text":
#                 result += content["text"] + " "
#     if add_generation_prompt:
#         result += "ASSISTANT:"
#     return result


def process_data(inputs, tokenizer):
    descriptions = [return_caption(captions) for captions in inputs["captions"]]
    questions = [
        qas[0]["question"].strip().rstrip("?") + "?"
        for qas in inputs["questions_and_answers"]
    ]

    conversations = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": description + "\n" + question}],
            }
        ]
        for description, question in zip(descriptions, questions)
    ]

    formatted_inputs = [
        tokenizer.apply_chat_template(
            conversation,
            chat_template=LLAVA_CHAT_TEMPLATE,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conversation in conversations
    ]
    # Equivalent plain-string version of the template above (single user turn, text only):
    # formatted_inputs = [
    #     f"USER: {description}\n{question} ASSISTANT:"
    #     for description, question in zip(descriptions, questions)
    # ]

    tokenized = tokenizer(formatted_inputs, add_special_tokens=False)

    return {
        "text_prompts": formatted_inputs,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def load_and_validate_detail23k(json_path: str, image_dir: str) -> List[Dict]:
    """Load detail_23k.json, verify every referenced image exists, return list of {image_path, question}."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [
        os.path.join(image_dir, entry["image"])
        for entry in data
        if not os.path.isfile(os.path.join(image_dir, entry["image"]))
    ]
    if missing:
        sample = "\n  ".join(missing[:10])
        tail = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} image(s) missing from {image_dir}:\n  {sample}{tail}"
        )

    records = []
    for entry in data:
        human_turn = next(c for c in entry["conversations"] if c["from"] == "human")
        # strip the <image> placeholder that LLaVA annotations embed in the text
        question = human_turn["value"].replace("<image>", "").strip()
        records.append({
            "image_path": os.path.join(image_dir, entry["image"]),
            "question": question,
        })
    return records


def process_detail_data(batch: Dict, processor) -> Dict:
    """Build image-first prompts for a generic HF instruct model.

    The chat template receives one user turn whose content list starts with
    {"type": "image"} and is immediately followed by {"type": "text", ...},
    so the processor inserts the model-specific image token(s) before the
    question regardless of the underlying architecture.
    """
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        for question in batch["question"]
    ]

    text_prompts = [
        processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conv in conversations
    ]

    images = [Image.open(p).convert("RGB") for p in batch["image_path"]]

    encoded = processor(
        text=text_prompts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    return {
        "text_prompts": text_prompts,
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
        "pixel_values": encoded["pixel_values"].tolist(),
    }


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict:

        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        attention_mask = [
            torch.tensor(instance["attention_mask"]) for instance in instances
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0, padding_side="left"
        )

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


@dataclass
class DataCollatorForVisionLM:
    """Collator for multimodal (image + text) batches from process_detail_data."""
    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        input_ids = [torch.tensor(inst["input_ids"]) for inst in instances]
        attention_mask = [torch.tensor(inst["attention_mask"]) for inst in instances]
        pixel_values = torch.tensor([inst["pixel_values"] for inst in instances])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left"
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0, padding_side="left"
        )

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./tokenizer")
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Where to store the final model."
    )
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
        "--force_recompute",
        action="store_true",
        help="If passed, recompute representations even if saved files already exist.",
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
        default="",
        help="Path to detail_23k.json. When set, uses this dataset instead of COCO-QA.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="Directory containing the images referenced in --detail_json.",
    )
    args = parser.parse_args()
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

    # **************************************************************************************
    if args.detail_json:
        # --- detail_23k path: generic HF instruct model with images ---
        if not args.image_dir:
            raise ValueError("--image_dir is required when --detail_json is set")

        processor = AutoProcessor.from_pretrained(args.model_name_or_path)

        print(f"Validating images in {args.image_dir} ...")
        records = load_and_validate_detail23k(args.detail_json, args.image_dir)
        print(f"All {len(records)} images found.")

        if args.max_num_samples is not None:
            records = records[: args.max_num_samples]

        dataset = Dataset.from_list(records)
        n_expected = len(dataset)

        dataset = dataset.map(
            process_detail_data,
            fn_kwargs={"processor": processor},
            batched=True,
            batch_size=args.preprocessing_num_workers,
            num_proc=1,  # PIL + processor are not fork-safe; keep single process
        )

        print(dataset[0]["text_prompts"])

        collate_fn = DataCollatorForVisionLM(pad_token_id=processor.tokenizer.pad_token_id)
    else:
        # --- original COCO-QA path: text-only LLaVA model ---
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False,
        )
        dataset = load_dataset("l11p/COCOQA-restval-ICQA")["train"]
        assert args.max_num_samples is not None

        dataset = dataset.select(range(args.max_num_samples))
        n_expected = len(dataset)

        dataset = dataset.map(
            process_data,
            fn_kwargs={"tokenizer": tokenizer},
            batched=True,
            batch_size=args.preprocessing_num_workers,
            num_proc=args.preprocessing_num_workers,
        )

        print(dataset[0]["text_prompts"])

        collate_fn = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)

    embedder_filenames = {
        "qwen0.6b": f"qwen3_0.6b_{args.max_num_samples}.pt",
        "qwen8b":   f"qwen3_8b_{args.max_num_samples}.pt",
    }
    qwen_path = os.path.join(args.out_dir, embedder_filenames[args.embedder])
    qwen_embeddings_dict = torch.load(qwen_path, map_location="cpu")
    qwen_embeddings = qwen_embeddings_dict["embeddings"]

    # qwen_embedding.py deduplicates captions; "indices" maps each sample back to
    # its unique embedding row so that qwen_embeddings[i] matches act[i].
    if "indices" in qwen_embeddings_dict:
        qwen_embeddings = qwen_embeddings[qwen_embeddings_dict["indices"].long()]
    if qwen_embeddings.shape[0] != n_expected:
        raise ValueError(
            f"Qwen embedding rows {qwen_embeddings.shape[0]} != dataset rows {n_expected}"
        )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.preprocessing_num_workers,
        sampler=None,
        pin_memory=True,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model = model.eval()

    # print(model)
    n_layer = model.language_model.config.num_hidden_layers

    # ***********************************************************************

    target_layers = get_target_layers(
        model=model,
        n_layer=n_layer,
        option=args.target_layer,
    )
    nsamples = len(dataloader.dataset)
    target_layer_labels = list(target_layers.keys())

    print(f"target layer: {args.target_layer}")
    print("num_total_samples", nsamples)
    print("target_layers", target_layers)

    simplify_name = {
        "llava-hf/llava-1.5-7b-hf": "llava_7b_hf",
        "llava-hf/llava-1.5-13b-hf": "llava_13b_hf",
    }

    model_name = simplify_name[args.model_name_or_path]
    dirpath = args.out_dir + f"/{model_name}/coco_image_question"
    os.makedirs(dirpath, exist_ok=True)

    layers_dir = f"{dirpath}/layers"
    os.makedirs(layers_dir, exist_ok=True)
    layer_paths = {
        layer_id: f"{layers_dir}/l{layer_id}{args.target_layer}.pt"
        for layer_id in target_layer_labels
    }
    all_exist = all(os.path.isfile(p) for p in layer_paths.values())

    if all_exist and not args.force_recompute:
        print("Representations already exist on disk — loading from files (use --force_recompute to override).")
        act_dict = {
            layer_name: torch.load(layer_paths[layer_id], map_location="cpu")
            for layer_id, layer_name in target_layers.items()
        }
    else:
        act_dict = save_representations(
            model=model,
            dataloader=dataloader,
            target_layers=target_layers,
            nsamples=nsamples,
            dirpath=dirpath,
            filename=args.filename,
            use_last_token=not args.avg_token,
            print_every=1000,
        )
        for layer_id, layer_name in target_layers.items():
            torch.save(act_dict[layer_name], layer_paths[layer_id])

    caption_texts = dataset["text_prompts"]
    assert len(caption_texts) == nsamples, (
        len(caption_texts),
        nsamples,
    )

    # for i, (layer_name, act) in enumerate(act_dict.items()):
    #     layer_id = target_layer_labels[i]
    #     report = first_prompts_topk_neighbors_json(
    #         act, caption_texts, n_prompts=n_prompts, k=k
    #     )
    #     json_path = f"{dirpath}/layers/l{layer_id}{args.target_layer}_{json_suffix}"
    #     with open(json_path, "w", encoding="utf-8") as f:
    #         json.dump(
    #             {
    #                 "layer_key": layer_id,
    #                 "layer_module": layer_name,
    #                 "n_prompts": n_prompts,
    #                 "top_k": k,
    #                 "queries": report,
    #             },
    #             f,
    #             ensure_ascii=False,
    #             indent=2,
    #         )
    #     print(f"Saved layer-{layer_id} similar-captions report to {json_path}")

    print("computing metrics")
    overlaps_by_layer: dict[int, list[float]] = {}
    overlaps_by_layer_norm: dict[int, list[float]] = {}
    imbalances: dict[int, tuple[float, float]] = {}
    imbalances_norm: dict[int, tuple[float, float]] = {}
    cka_by_layer: dict[int, dict[float, float]] = {}
    cka_norm_by_layer: dict[int, dict[float, float]] = {}
    linear_cka_by_layer: dict[int, float] = {}
    linear_cka_norm_by_layer: dict[int, float] = {}

    qwen_norm = F.normalize(qwen_embeddings.float(), dim=1)
    gram_qwen_by_sigma = {
        sigma: gram_rbf(qwen_embeddings, sigma) for sigma in args.gaussian_cka_sigma
    }
    gram_qwen_norm_by_sigma = {
        sigma: gram_rbf(qwen_norm, sigma) for sigma in args.gaussian_cka_sigma
    }

    for layer_id, layer_name in target_layers.items():
        print(layer_id)
        act = act_dict[layer_name]

        overlaps_by_layer[layer_id] = compute_overlap_torch(qwen_embeddings, act, ks=tuple(args.k))
        i12, i21 = compute_imbalance_torch(qwen_embeddings, act, exact_rank=False)
        imbalances[layer_id] = (i12, i21)

        cka_by_layer[layer_id] = {
            sigma: float(cka(gram_qwen_by_sigma[sigma], gram_rbf(act, sigma)))
            for sigma in args.gaussian_cka_sigma
        }
        linear_cka_by_layer[layer_id] = float(feature_space_linear_cka(qwen_embeddings, act))

        act_n = F.normalize(act, dim=1)
        overlaps_by_layer_norm[layer_id] = compute_overlap_torch(qwen_embeddings, act_n, ks=tuple(args.k))
        i12n, i21n = compute_imbalance_torch(qwen_embeddings, act_n, exact_rank=False)
        imbalances_norm[layer_id] = (i12n, i21n)

        cka_norm_by_layer[layer_id] = {
            sigma: float(cka(gram_qwen_norm_by_sigma[sigma], gram_rbf(act_n, sigma)))
            for sigma in args.gaussian_cka_sigma
        }
        linear_cka_norm_by_layer[layer_id] = float(feature_space_linear_cka(qwen_norm, act_n))

    filename = f"{args.target_layer}_{args.embedder}_{model_name}"
    metrics_basename = args.metrics_json.strip() or f"{filename}_all_metrics.json"
    metrics_path = os.path.join(dirpath, metrics_basename)

    hyperparameters = {
        "overlap_k": list(args.k),
        "gaussian_cka_sigma": list(args.gaussian_cka_sigma),
        "max_num_samples": args.max_num_samples,
    }

    metrics: dict = {}
    for ki, k_val in enumerate(args.k):
        metrics[f"overlap__k_{k_val}"] = {
            str(lid): overlaps_by_layer[lid][ki] for lid in target_layer_labels
        }
        metrics[f"overlap_norm__k_{k_val}"] = {
            str(lid): overlaps_by_layer_norm[lid][ki] for lid in target_layer_labels
        }

    metrics["imbalance"] = {
        str(lid): {"qwen_to_llava": imbalances[lid][0], "llava_to_qwen": imbalances[lid][1]}
        for lid in target_layer_labels
    }
    metrics["imbalance_norm"] = {
        str(lid): {"qwen_to_llava": imbalances_norm[lid][0], "llava_to_qwen": imbalances_norm[lid][1]}
        for lid in target_layer_labels
    }

    for sigma in args.gaussian_cka_sigma:
        sig_key = str(sigma)
        metrics[f"gaussian_cka__sigma_{sig_key}"] = {
            str(lid): cka_by_layer[lid][sigma] for lid in target_layer_labels
        }
        metrics[f"gaussian_cka_norm__sigma_{sig_key}"] = {
            str(lid): cka_norm_by_layer[lid][sigma] for lid in target_layer_labels
        }

    metrics["linear_cka"] = {str(lid): linear_cka_by_layer[lid] for lid in target_layer_labels}
    metrics["linear_cka_norm"] = {str(lid): linear_cka_norm_by_layer[lid] for lid in target_layer_labels}

    payload = {"hyperparameters": hyperparameters, "metrics": metrics}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved combined metrics to {metrics_path}")


if __name__ == "__main__":

    main()
