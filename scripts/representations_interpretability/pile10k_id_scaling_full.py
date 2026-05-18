"""Compute per-layer ID-scaling profiles for pile-10k representations.

Uses GRIDE across ALL neighbour-pair scales (not just a single selected k).
The result is a 2-D array of shape (num_layers, n_scales) saved as safetensors.

Input  path: same structure as pile10k_intrinsic_dimension.py
Output file: txt_scaling_gride_full.safetensors  (different filename — no collision)
  key "intrinsic_dimension_all_scales"      shape (num_layers, n_scales)
  key "intrinsic_dimension_all_scales_err"  shape (num_layers, n_scales)

n_scales = floor(log2(min(N_samples, range_max))).
With N ≥ 550 and range_max=550 this gives 9 scales:
  scale i  →  neighbour pair (2^i, 2^(i+1))
  label    →  k={2**i}-{2**(i+1)}
"""

from __future__ import annotations

import argparse
import gc
import math
import os

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from utils.metrics_utils import id_scaling_gride_full

load_dotenv()

DATASET_TAG = "pile10k_train_txt"
SAMPLE_SET_TAG = "all_samples"
RANGE_MAX_DEFAULT = 550
OUTPUT_FILENAME = "txt_scaling_gride_full.safetensors"


# ── Path helpers (mirror pile10k_intrinsic_dimension.py) ─────────────────────


def build_token_tag(min_tokens: int, max_tokens: int) -> str:
    return f"tokens_{min_tokens}_{max_tokens}"


def build_pooling_variant(
    pooling_method: str,
    prompt_format: str = "wrapped",
    remove_assistant_tokens: bool = False,
) -> str:
    parts = [pooling_method]
    if prompt_format in ("plain", "raw"):
        parts.append(prompt_format)
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


def build_id_full_path(
    results_dir: str,
    model_slug: str,
    pooling_variant: str,
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
        OUTPUT_FILENAME,
    )


# ── Core computation ──────────────────────────────────────────────────────────


def compute_layers_id_scaling_full(
    residual_stream: torch.Tensor,
    range_max: int = RANGE_MAX_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GRIDE ID profile for every layer.

    Args:
        residual_stream: shape (num_layers, num_samples, hidden_size)
        range_max: passed to id_scaling_gride_full; controls number of scales.

    Returns:
        ids_all:     shape (num_layers, n_scales) — ID per layer per scale.
        ids_err_all: shape (num_layers, n_scales) — GRIDE error estimates.
    """
    num_layers = residual_stream.shape[0]
    ids_layers: list[torch.Tensor] = []
    ids_err_layers: list[torch.Tensor] = []
    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Intrinsic dimension", unit="layer") as pbar:
        for layer in range(num_layers):
            ids, ids_err = id_scaling_gride_full(residual_stream[layer], range_max=range_max)
            ids_layers.append(ids)
            ids_err_layers.append(ids_err)
            updates_pending += 1
            if (layer + 1) % update_every == 0 or (layer + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return torch.stack(ids_layers, dim=0), torch.stack(ids_err_layers, dim=0)


def load_representation(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Representation not found: {path}\n"
            "Run pile10k_representations.py first."
        )
    print(f"  💾 Loading: {path}")
    loaded = load_file(path)
    if "residual_stream" not in loaded:
        raise KeyError(f"Key 'residual_stream' not found in: {path}")
    tensor = loaded["residual_stream"]
    print(f"  ✅ Shape: {tensor.shape}")
    return tensor


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-layer ID-scaling profiles (all GRIDE scales) for pile-10k. "
            "Output: txt_scaling_gride_full.safetensors — does NOT overwrite "
            "txt_scaling_gride.safetensors from pile10k_intrinsic_dimension.py."
        )
    )
    parser.add_argument("--representations-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for output. Required unless --dry-run.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--pooling-methods", nargs="+", default=["last"],
        help="Pooling methods (default: last)",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-format", choices=["wrapped", "plain", "raw"], default="plain",
    )
    parser.add_argument("--remove-assistant-tokens", action="store_true")
    parser.add_argument(
        "--residual-stream-type", type=str, default="output_layer",
    )
    parser.add_argument(
        "--range-max", type=int, default=RANGE_MAX_DEFAULT,
        help=f"Maximum neighbour rank for GRIDE (default: {RANGE_MAX_DEFAULT})",
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print paths without computing.")
    args = parser.parse_args()

    if args.dry_run:
        args.results_dir = args.results_dir or "/tmp/debug_run"
    elif not args.results_dir:
        parser.error("--results-dir is required unless --dry-run")

    model_slug = args.model_name_or_path.rstrip("/").split("/")[-1]
    results_dir = os.path.expanduser(args.results_dir)
    token_tag = build_token_tag(args.min_tokens, args.max_tokens)

    # Compute expected n_scales for display (N unknown until file loaded)
    approx_n_scales = int(math.log2(args.range_max))
    scale_labels = [f"k={2**i}-{2**(i+1)}" for i in range(approx_n_scales)]

    print(f"\n{'─' * 80}")
    print(f"📐 ID Scaling Full | {DATASET_TAG} | {model_slug}")
    print(f"   Token range: {token_tag} | Pooling: {args.pooling_methods}")
    print(f"   range_max={args.range_max} → ~{approx_n_scales} scales: {scale_labels}")
    print(f"   Residual stream: {args.residual_stream_type} | "
          f"Prompt format: {args.prompt_format} | "
          f"Remove asst: {args.remove_assistant_tokens}")
    print(f"{'─' * 80}")

    for pm in args.pooling_methods:
        pv = build_pooling_variant(pm, args.prompt_format, args.remove_assistant_tokens)
        repr_path = build_repr_path(
            args.representations_dir, model_slug, pv,
            max_tokens=args.max_tokens, min_tokens=args.min_tokens,
            residual_stream_type=args.residual_stream_type,
        )
        id_path = build_id_full_path(
            results_dir, model_slug, pv,
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
        n_samples = representations.shape[1]
        actual_n_scales = int(math.log2(min(n_samples, args.range_max)))
        print(f"  ℹ  N={n_samples} samples → {actual_n_scales} scales "
              f"(range_max={args.range_max})")

        print("  👉 Computing ID scaling profile...")
        ids_all, ids_err_all = compute_layers_id_scaling_full(
            representations, range_max=args.range_max
        )
        print(f"  ✅ IDs shape: {tuple(ids_all.shape)}")  # (num_layers, n_scales)

        os.makedirs(os.path.dirname(id_path), exist_ok=True)
        save_file(
            {
                "intrinsic_dimension_all_scales": ids_all,
                "intrinsic_dimension_all_scales_err": ids_err_all,
            },
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
                "range_max": str(args.range_max),
                "n_scales": str(ids_all.shape[1]),
                "id_algorithm": "scaling_gride_full",
                "shape": f"{ids_all.shape[0]}x{ids_all.shape[1]}",
            },
        )
        print(f"  💾 Saved: {id_path}")

        del representations, ids_all, ids_err_all
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ Done: {DATASET_TAG} | {model_slug}")


if __name__ == "__main__":
    main()
