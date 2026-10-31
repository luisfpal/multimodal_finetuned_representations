"""
Compute the frequency of appearance of each unique representation vector
per layer, for each model / dataset / token-type / data-type combination.

Output: one txt file per (model, dataset, token_type) with rows like:
    layer0 img: 0 -> 35, 1 -> 2, 2 -> 1
    layer0 txt: 0 -> 128, 1 -> 4
    layer1 img: ...
    ...

Usage:
    python compute_repr_frequency.py \
        --model_name llava_7b \
        --dataset_name vqav2 \
        --token_type last \
        --ft_mode ft
"""

import argparse
import os
from collections import Counter

import torch
from safetensors import safe_open

# ─── paths / model-dir mapping (mirrors compute_imbalance.py) ─────────────────
repr_path = os.environ.get(
    "REPRESENTATIONS_DIR", os.path.expanduser("~/scratch/representations")
)

models = {
    "llava_7b": {"pre": "llava-v1.5-7b-pretrained-lbasile", "ft": "llava_7b"},
    "llava_13b": {"pre": "llava-v1.5-13b-pretrained-lbasile", "ft": "llava_13b"},
    "internvl2_8b": {"pre": "InternVL2-8B-Pretrain", "ft": "InternVL2-8B"},
}

datasets = {
    "vqav2": "vqav2_coco_restval_icq",
    "cocoqa": "cocoqa_restval_icqa",
    "gqa": "gqa_val_balanced_noyesno_captioned",
    "mme": "mme_test_captioned",
    "mmbench": "mmbench_en_dev_captioned",
}

# folder suffix for txt representations differs per dataset
txt_suffix = {
    "vqav2": "human_txt",
    "cocoqa": "human_txt",
    "gqa": "synthetic_txt",
    "mme": "synthetic_txt",
    "mmbench": "synthetic_txt",
}

# ─── argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Compute frequency of unique representation vectors per layer"
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    choices=list(models.keys()),
)
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=list(datasets.keys()),
)
parser.add_argument(
    "--ft_mode",
    type=str,
    required=True,
    choices=["pre", "ft"],
)
parser.add_argument(
    "--token_type",
    type=str,
    default="last",
    choices=["last", "avg"],
    help="Which token pooling to use: 'last' or 'avg'",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="results/repr_frequency",
    help="Directory where output txt files are written",
)
args = parser.parse_args()

# token-type sub-folder inside the representation safetensors path
token_subdir = "last" if args.token_type == "last" else "mean_context"


def compute_freq_from_tensor(tensor: torch.Tensor) -> list[str]:
    """
    For a 2-D bfloat16 tensor of shape (N, D), compute a frequency-of-frequencies:
    how many unique vectors appear exactly n times.

    E.g. "4 -> 10" means 10 distinct vectors each appear 4 times.

    Output is sorted by descending repetition count (n).
    """
    _, _, counts = torch.unique(
        tensor, dim=0, sorted=False, return_inverse=True, return_counts=True
    )
    # counts: how many times each unique vector appears
    freq_of_freq = Counter(counts.cpu().tolist())
    # sort by descending repetition count
    parts = [
        f"{how_many} -> {n}"
        for n, how_many in sorted(freq_of_freq.items(), key=lambda x: -x[0])
    ]
    return parts


def load_and_process(
    model_dir: str, dataset_key: str, data_type: str
) -> dict[int, list[str]]:
    """
    Load representations.safetensors for a given model / dataset / data_type
    and return a dict  {layer_index: [freq_string, ...]}.
    """
    suffix = txt_suffix[dataset_key] if data_type == "txt" else data_type
    safe_path = (
        f"{repr_path}/{model_dir}/{datasets[dataset_key]}_{suffix}"
        f"/output_layer/{token_subdir}/representations.safetensors"
    )

    if not os.path.exists(safe_path):
        print(f"  [WARN] not found: {safe_path}")
        return {}

    print(f"  loading {safe_path}")
    freq_by_layer: dict[int, list[str]] = {}
    with safe_open(safe_path, framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            tensor = f.get_tensor(key)  # bfloat16, shape (num_layers, N, D)
            num_layers = tensor.shape[0]
            for layer_idx in range(num_layers):
                layer_mat = tensor[layer_idx]  # (N, D) bfloat16
                freq_by_layer[layer_idx] = compute_freq_from_tensor(layer_mat)
                print(
                    f"    layer {layer_idx:3d}: {len(freq_by_layer[layer_idx])} unique vectors"
                )
    return freq_by_layer


# ─── main ─────────────────────────────────────────────────────────────────────
model_dir = models[args.model_name][args.ft_mode]
dataset_key = args.dataset_name

os.makedirs(args.output_dir, exist_ok=True)
out_file = os.path.join(
    args.output_dir,
    f"{args.model_name}_{dataset_key}_{args.token_type}.txt",
)

results: dict[str, dict[int, list[str]]] = {}
for data_type in ["img", "txt"]:
    print(f"\nProcessing {data_type} ...")
    results[data_type] = load_and_process(model_dir, dataset_key, data_type)

# ─── write output ─────────────────────────────────────────────────────────────
# Determine total number of layers from whatever data was loaded
all_layers = sorted(set(results["img"].keys()) | set(results["txt"].keys()))

with open(out_file, "w") as fout:
    fout.write(
        f"# model={args.model_name}  dataset={dataset_key}  "
        f"ft_mode={args.ft_mode}  token_type={args.token_type}\n"
    )
    for layer_idx in all_layers:
        for data_type in ["img", "txt"]:
            freq_parts = results[data_type].get(layer_idx)
            if freq_parts is None:
                continue
            line = f"layer{layer_idx} {data_type}: " + ", ".join(freq_parts)
            fout.write(line + "\n")

print(f"\nSaved frequency results to {out_file}")
