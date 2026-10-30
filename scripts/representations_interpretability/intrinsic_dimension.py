"""Compute per-layer intrinsic dimension of cached residual-stream representations.

Inputs: --results-dir, --dataset-path-or-name, --model-name-or-path, --id-* estimator options.
Outputs: safetensors result files under the results directory.
"""

import argparse
import gc
import os

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file

from utils import compute_layers_intrinsic_dimension
from utils.dataset import MULTIPLE_CHOICE_BENCHMARKS

load_dotenv()

RESIDUAL_STREAM_TYPE = "output_layer"
POOLING_METHOD = "last"
SAMPLE_SET_TAG = "unique_images"


def load_representation(path: str) -> torch.Tensor:
    """Load representation from cache. Raises if not found or invalid."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cached representation not found: {path}\n"
            "Run multimodality_similarity.py first to extract representations."
        )
    print(f"  💾 Loading: {path}")
    loaded = load_file(path)
    if "residual_stream" not in loaded:
        raise KeyError(f"Key 'residual_stream' not found in: {path}")
    tensor = loaded["residual_stream"]
    if tensor.ndim < 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"Invalid tensor shape {tensor.shape} in: {path}")
    print(f"  ✅ Shape: {tensor.shape}")
    return tensor


def build_repr_path(representations_dir: str, model_name: str, dataset_name: str) -> str:
    return os.path.join(
        os.path.expanduser(representations_dir),
        "representations",
        model_name,
        dataset_name,
        RESIDUAL_STREAM_TYPE,
        POOLING_METHOD,
        SAMPLE_SET_TAG,
        "representations.safetensors",
    )


def build_id_filename(modality: str, algorithm: str) -> str:
    return f"{modality}_{algorithm}.safetensors"


def build_algorithm_kwargs(args: argparse.Namespace) -> dict:
    if args.id_algorithm == "scaling_gride":
        return {"k": args.id_nn_rank, "range_max": args.id_nn_range_max}
    if args.id_algorithm == "MLE":
        return {"k": args.id_nn_rank}
    if args.id_algorithm == "twoNN":
        return {"fraction": args.id_fraction}
    raise ValueError(f"Unsupported algorithm: {args.id_algorithm}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute intrinsic dimension of cached representations (last pooling). "
            "Representations must be pre-extracted via multimodality_similarity.py."
        )
    )
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--representations-dir",
        type=str,
        default=None,
        help="Base directory containing the representations cache.",
    )
    parser.add_argument("--dataset-path-or-name", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--caption-source",
        type=str,
        default="synthetic",
        choices=["human", "synthetic"],
    )
    parser.add_argument(
        "--open-ended",
        action="store_true",
        help="Use open-ended formatting tag (applies only to MMBench/SEED).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        help="Split tag for OpenVQA datasets (e.g. split01).",
    )
    parser.add_argument(
        "--id-algorithm",
        type=str,
        default="scaling_gride",
        choices=["scaling_gride", "MLE", "twoNN"],
        help="Intrinsic-dimension algorithm.",
    )
    parser.add_argument("--id-nn-rank", type=int, default=16)
    parser.add_argument("--id-nn-range-max", type=int, default=100)
    parser.add_argument("--id-fraction", type=float, default=0.9)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ID outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths without loading or computing.",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.results_dir = args.results_dir or "/tmp/debug_run"
        args.representations_dir = args.representations_dir or "/tmp/debug_run"
    elif not args.results_dir or not args.representations_dir:
        parser.error(
            "--results-dir and --representations-dir are required unless --dry-run"
        )

    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    is_mc = any(benchmark in dataset_slug for benchmark in MULTIPLE_CHOICE_BENCHMARKS)

    if args.dataset_split:
        if is_mc:
            parser.error("--dataset-split is only valid for OpenVQA datasets, not MC.")
        dataset_slug = f"{dataset_slug}_{args.dataset_split}"

    open_ended_tag = "open_ended" if (is_mc and args.open_ended) else ""
    dataset_prefix = (
        f"{dataset_slug}_{open_ended_tag}_{args.caption_source}"
        if open_ended_tag
        else f"{dataset_slug}_{args.caption_source}"
    )

    model_name = args.model_name_or_path.split("/")[-1]
    results_dir = os.path.expanduser(args.results_dir)

    algorithm_kwargs = build_algorithm_kwargs(args)

    print(f"\n{'─' * 80}")
    print(f"📐 Intrinsic Dimension | {dataset_prefix} | {model_name}")
    print(f"   Pooling: {POOLING_METHOD} | Algorithm: {args.id_algorithm} (kwargs={algorithm_kwargs})")
    print(f"{'─' * 80}")

    for modality in ("txt", "img"):
        dataset_name = f"{dataset_prefix}_{modality}"
        repr_path = build_repr_path(args.representations_dir, model_name, dataset_name)

        analysis_dir = os.path.join(
            results_dir,
            "intrinsic_dimension",
            dataset_prefix,
            POOLING_METHOD,
            SAMPLE_SET_TAG,
            model_name,
            RESIDUAL_STREAM_TYPE,
        )
        save_path = os.path.join(
            analysis_dir,
            build_id_filename(modality, args.id_algorithm),
        )

        print(f"\n  [{modality.upper()}]")

        if not args.force and os.path.exists(save_path):
            print(f"  ⏩ Already computed — skipping: {save_path}")
            continue

        if args.dry_run:
            print(f"  🔍 [DRY-RUN] repr path : {repr_path}")
            print(f"  🔍 [DRY-RUN] output path: {save_path}")
            continue

        representations = load_representation(repr_path)

        print(f"  👉 Computing ID with {args.id_algorithm}...")
        id_result = compute_layers_intrinsic_dimension(
            residual_stream=representations,
            algorithm=args.id_algorithm,
            **algorithm_kwargs,
        )
        os.makedirs(analysis_dir, exist_ok=True)
        save_file(
            {"intrinsic_dimension": id_result},
            save_path,
            metadata={
                "model": model_name,
                "dataset": dataset_prefix,
                "modality": modality,
                "pooling_method": POOLING_METHOD,
                "sample_set": SAMPLE_SET_TAG,
                "type": RESIDUAL_STREAM_TYPE,
                "id_algorithm": args.id_algorithm,
                "id_nn_rank": str(args.id_nn_rank),
                "id_nn_range_max": str(args.id_nn_range_max),
                "id_fraction": str(args.id_fraction),
            },
        )
        print(f"  💾 Saved: {save_path}")

        del representations
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ Done: {dataset_prefix} | {model_name}")


if __name__ == "__main__":
    main()
