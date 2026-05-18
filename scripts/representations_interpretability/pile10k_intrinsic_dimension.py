"""Compute intrinsic dimension of pile-10k residual-stream representations.

Reads pre-extracted safetensors written by pile10k_representations.py and
computes per-layer intrinsic dimension for every (model, pooling_method) pair.

Representation path (input):
  {repr_dir}/representations/{model_slug}/pile10k_train_txt/output_layer/{pm}/
    all_samples/representations.safetensors

ID result path (output):
  {results_dir}/intrinsic_dimension/pile10k_train_txt/{pm}/all_samples/
    {model_slug}/output_layer/txt_{algorithm}.safetensors

This script is intentionally standalone — intrinsic_dimension.py is the
workhorse for multimodal experiments and is not touched by this script.
"""

import argparse
import gc
import os

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file

from utils import compute_layers_intrinsic_dimension

load_dotenv()

DATASET_TAG = "pile10k_train_txt"
SAMPLE_SET_TAG = "all_samples"

POOLING_METHODS_DEFAULT = ["last", "last_user_text"]


def build_token_tag(min_tokens: int, max_tokens: int) -> str:
    return f"tokens_{min_tokens}_{max_tokens}"


def build_pooling_variant(
    pooling_method: str,
    prompt_format: str = "wrapped",
    remove_assistant_tokens: bool = False,
) -> str:
    parts = [pooling_method]
    if prompt_format == "plain":
        parts.append("plain")
    if remove_assistant_tokens:
        parts.append("no_asst")
    return "_".join(parts)


def build_repr_path(
    representations_dir: str,
    model_slug: str,
    pooling_variant: str,
    max_tokens: int,
    min_tokens: int,
    residual_stream_type: str = "output_layer",
) -> str:
    token_tag = build_token_tag(min_tokens, max_tokens)
    return os.path.join(
        os.path.expanduser(representations_dir),
        "representations",
        model_slug,
        DATASET_TAG,
        token_tag,
        residual_stream_type,
        pooling_variant,
        SAMPLE_SET_TAG,
        "representations.safetensors",
    )


def build_id_path(
    results_dir: str,
    model_slug: str,
    pooling_variant: str,
    algorithm: str,
    max_tokens: int,
    min_tokens: int,
    residual_stream_type: str = "output_layer",
) -> str:
    token_tag = build_token_tag(min_tokens, max_tokens)
    return os.path.join(
        os.path.expanduser(results_dir),
        "intrinsic_dimension",
        DATASET_TAG,
        token_tag,
        pooling_variant,
        SAMPLE_SET_TAG,
        model_slug,
        residual_stream_type,
        f"txt_{algorithm}.safetensors",
    )


def load_representation(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cached representation not found: {path}\n"
            "Run pile10k_representations.py first to extract representations."
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
            "Compute intrinsic dimension of pile-10k residual-stream representations. "
            "Representations must be pre-extracted via pile10k_representations.py."
        )
    )
    parser.add_argument("--representations-dir", type=str, required=True,
                        help="Base directory containing the representations cache.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for ID results. Required unless --dry-run.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--pooling-methods",
        nargs="+",
        default=POOLING_METHODS_DEFAULT,
        help=f"Pooling methods to compute ID for (default: {POOLING_METHODS_DEFAULT})",
    )
    parser.add_argument(
        "--id-algorithm",
        type=str,
        default="scaling_gride",
        choices=["scaling_gride", "MLE", "twoNN"],
    )
    parser.add_argument("--id-nn-rank", type=int, default=16)
    parser.add_argument("--id-nn-range-max", type=int, default=100)
    parser.add_argument("--id-fraction", type=float, default=0.9)
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Max token budget used during representation extraction (default: 128)",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=0,
        help="Min token filter used during representation extraction (default: 0)",
    )
    parser.add_argument(
        "--prompt-format", choices=["wrapped", "plain"], default="wrapped",
        help="Prompt format used during representation extraction (default: wrapped)",
    )
    parser.add_argument(
        "--remove-assistant-tokens", action="store_true",
        help="Whether assistant tokens were removed during representation extraction",
    )
    parser.add_argument(
        "--residual-stream-type", type=str, default="output_layer",
        help="Residual stream type used during representation extraction (default: output_layer)",
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing ID outputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print paths without loading or computing.")
    args = parser.parse_args()

    if args.dry_run:
        args.results_dir = args.results_dir or "/tmp/debug_run"
    elif not args.results_dir:
        parser.error("--results-dir is required unless --dry-run")

    model_slug = args.model_name_or_path.rstrip("/").split("/")[-1]
    algorithm_kwargs = build_algorithm_kwargs(args)
    results_dir = os.path.expanduser(args.results_dir)
    token_tag = build_token_tag(args.min_tokens, args.max_tokens)

    print(f"\n{'─' * 80}")
    print(f"📐 Intrinsic Dimension | {DATASET_TAG} | {model_slug}")
    print(
        f"   Token range: {token_tag} | Pooling: {args.pooling_methods} | "
        f"Algorithm: {args.id_algorithm} (kwargs={algorithm_kwargs})"
    )
    print(
        f"   Residual stream: {args.residual_stream_type} | "
        f"Prompt format: {args.prompt_format} | "
        f"Remove asst: {args.remove_assistant_tokens}"
    )
    print(f"{'─' * 80}")

    for pm in args.pooling_methods:
        pv = build_pooling_variant(pm, args.prompt_format, args.remove_assistant_tokens)
        repr_path = build_repr_path(
            args.representations_dir, model_slug, pv,
            max_tokens=args.max_tokens, min_tokens=args.min_tokens,
            residual_stream_type=args.residual_stream_type,
        )
        id_path = build_id_path(
            results_dir, model_slug, pv, args.id_algorithm,
            max_tokens=args.max_tokens, min_tokens=args.min_tokens,
            residual_stream_type=args.residual_stream_type,
        )

        print(f"\n  [TXT / {pv}]")

        if not args.force and os.path.exists(id_path):
            print(f"  ⏩ Already computed — skipping: {id_path}")
            continue

        if args.dry_run:
            print(f"  🔍 [DRY-RUN] repr path : {repr_path}")
            print(f"  🔍 [DRY-RUN] output path: {id_path}")
            continue

        representations = load_representation(repr_path)

        print(f"  👉 Computing ID with {args.id_algorithm}...")
        id_result = compute_layers_intrinsic_dimension(
            residual_stream=representations,
            algorithm=args.id_algorithm,
            **algorithm_kwargs,
        )
        os.makedirs(os.path.dirname(id_path), exist_ok=True)
        save_file(
            {"intrinsic_dimension": id_result},
            id_path,
            metadata={
                "model": model_slug,
                "dataset": DATASET_TAG,
                "modality": "txt",
                "pooling_variant": pv,
                "pooling_method": pm,
                "sample_set": SAMPLE_SET_TAG,
                "residual_stream_type": args.residual_stream_type,
                "prompt_format": args.prompt_format,
                "remove_assistant_tokens": str(args.remove_assistant_tokens),
                "max_tokens": str(args.max_tokens),
                "min_tokens": str(args.min_tokens),
                "id_algorithm": args.id_algorithm,
                "id_nn_rank": str(args.id_nn_rank),
                "id_nn_range_max": str(args.id_nn_range_max),
                "id_fraction": str(args.id_fraction),
            },
        )
        print(f"  💾 Saved: {id_path}")

        del representations
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ Done: {DATASET_TAG} | {model_slug}")


if __name__ == "__main__":
    main()
