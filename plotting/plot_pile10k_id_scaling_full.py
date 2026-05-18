"""Plot per-layer ID-scaling profiles (all GRIDE scales) for pile-10k.

Reads txt_scaling_gride_full.safetensors produced by pile10k_id_scaling_full.py.
Each file has shape (num_layers, n_scales); each scale corresponds to GRIDE
neighbour pairs (2^i, 2^(i+1)), labelled k=1-2, k=2-4, ..., k=256-512.

Layout — 4 rows × 3 columns:
  Row 0: LLaVA 7B pretrained,   LLaVA-OV 4B pretrained,  InternVL2 8B pretrained
  Row 1: LLaVA 7B finetuned,    LLaVA-OV 4B finetuned,   InternVL2 8B finetuned
  Row 2: LLaVA 13B pretrained,  LLaVA-OV 8B pretrained,  Cambrian 8B pretrained
  Row 3: LLaVA 13B finetuned,   LLaVA-OV 8B finetuned,   Cambrian 8B finetuned

Legend (scale labels) is shared across all subplots and placed below the figure.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open

# HPC-safe matplotlib config directory.
_HOME = Path.home()
if "MPLCONFIGDIR" not in os.environ:
    for _mpl_dir in (
        _HOME / "scratch" / "matplotlib",
        Path("/dev/shm") / f"matplotlib-{os.environ.get('USER', 'user')}",
    ):
        try:
            _mpl_dir.mkdir(parents=True, exist_ok=True)
            _probe = _mpl_dir / ".write_test"
            _probe.write_text("", encoding="utf-8")
            _probe.unlink()
        except OSError:
            continue
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
        break

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402


@dataclass(frozen=True)
class ModelPair:
    finetuned_id: str
    pretrained_id: str
    label: str
    slug: str


PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
DATASET_TAG = "pile10k_train_txt"
SAMPLE_SET_TAG = "all_samples"
OUTPUT_FILENAME = "txt_scaling_gride_full.safetensors"
RANGE_MAX_DEFAULT = 550
DPI = 200

# ── Model pairs (same order as plot_pile10k_intrinsic_dimension.py) ───────────
MODEL_PAIRS: tuple[ModelPair, ...] = (
    ModelPair(
        finetuned_id="llava_7b",
        pretrained_id="llava-v1.5-7b-pretrained-lbasile",
        label="LLaVA 7B",
        slug="llava_7b",
    ),
    ModelPair(
        finetuned_id="LLaVA-OneVision-1.5-4B-Instruct",
        pretrained_id="LLaVA-OneVision-1.5-4B-Base",
        label="LLaVA-OV 4B",
        slug="onevision_4b",
    ),
    ModelPair(
        finetuned_id="InternVL2-8B",
        pretrained_id="InternVL2-8B-Pretrain",
        label="InternVL2 8B",
        slug="internvl2_8b",
    ),
    ModelPair(
        finetuned_id="llava_13b",
        pretrained_id="llava-v1.5-13b-pretrained-lbasile",
        label="LLaVA 13B",
        slug="llava_13b",
    ),
    ModelPair(
        finetuned_id="LLaVA-OneVision-1.5-8B-Instruct",
        pretrained_id="LLaVA-OneVision-1.5-8B-Base",
        label="LLaVA-OV 8B",
        slug="onevision_8b",
    ),
    ModelPair(
        finetuned_id="cambrian-8b",
        pretrained_id="cambrian-8b-pretrained",
        label="Cambrian 8B",
        slug="cambrian_8b",
    ),
)

FONT_SUPTITLE = 11
FONT_TITLE = 9
FONT_AXIS_LABEL = 8
FONT_TICK = 7
FONT_LEGEND = 7
LINE_WIDTH = 1.2
MARKER_SIZE = 2.5
GRID_ALPHA = 0.18


# ── Path builder ──────────────────────────────────────────────────────────────


def build_curve_path(
    results_root: Path,
    pooling_variant: str,
    model_id: str,
    token_tag: str,
    residual_stream_type: str = "output_layer",
) -> Path:
    return (
        results_root
        / token_tag
        / pooling_variant
        / SAMPLE_SET_TAG
        / model_id
        / residual_stream_type
        / OUTPUT_FILENAME
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def load_profile(
    results_root: Path,
    pooling_variant: str,
    model_id: str,
    token_tag: str,
    residual_stream_type: str = "output_layer",
) -> np.ndarray | None:
    """Load intrinsic_dimension_all_scales array; shape (num_layers, n_scales)."""
    path = build_curve_path(
        results_root, pooling_variant, model_id, token_tag, residual_stream_type
    )
    if not path.exists():
        return None
    try:
        with safe_open(str(path), framework="numpy") as handle:
            key = "intrinsic_dimension_all_scales"
            if key not in handle.keys():
                return None
            return handle.get_tensor(key).copy().astype(float)
    except Exception as exc:
        print(f"  ⚠ failed to load {path}: {exc}")
        return None


# ── Style helpers ─────────────────────────────────────────────────────────────


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_AXIS_LABEL,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "--",
            "lines.linewidth": LINE_WIDTH,
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, *, title: str) -> None:
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=4)
    ax.set_xlabel("Layer", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Intrinsic dimension", fontsize=FONT_AXIS_LABEL)
    ax.tick_params(direction="out", length=2.5, width=0.7, color="#444", labelsize=FONT_TICK)
    ax.set_axisbelow(True)


def mark_missing(ax: plt.Axes, label: str = "N/A") -> None:
    ax.set_facecolor("#f5f5f5")
    ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#aaa", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Scale labels ──────────────────────────────────────────────────────────────


def make_scale_labels(n_scales: int) -> list[str]:
    """Return GRIDE scale labels: 'k=1-2', 'k=2-4', ..., 'k=2^(n-1)-2^n'."""
    return [f"k={2**i}-{2**(i+1)}" for i in range(n_scales)]


def determine_n_scales(
    results_root: Path,
    pooling_variant: str,
    token_tag: str,
    residual_stream_type: str,
    range_max: int,
) -> int:
    """Detect n_scales from the first available file; fall back to range_max."""
    for pair in MODEL_PAIRS:
        for model_id in (pair.pretrained_id, pair.finetuned_id):
            profile = load_profile(
                results_root, pooling_variant, model_id, token_tag, residual_stream_type
            )
            if profile is not None:
                return profile.shape[1]
    # Fallback: compute from range_max (N unknown, assume large)
    return int(math.log2(range_max))


# ── Availability check ────────────────────────────────────────────────────────


def summarize_availability(
    results_root: Path,
    pooling_variant: str,
    token_tag: str,
    residual_stream_type: str,
) -> None:
    expected = len(MODEL_PAIRS) * 2
    found = 0
    missing: list[Path] = []
    for pair in MODEL_PAIRS:
        for model_id in (pair.pretrained_id, pair.finetuned_id):
            path = build_curve_path(
                results_root, pooling_variant, model_id, token_tag, residual_stream_type
            )
            if path.exists():
                found += 1
            else:
                missing.append(path)
    print(f"  {found}/{expected} profiles available")
    for p in missing:
        print(f"    - missing: {p}")


# ── Main plot ─────────────────────────────────────────────────────────────────


def plot_id_scaling_full(
    results_root: Path,
    plots_root: Path,
    pooling_variant: str,
    token_tag: str,
    residual_stream_type: str,
    n_scales: int,
    scale_indices: list[int] | None = None,
) -> None:
    """4×3 grid: rows 0-1 for first 3 model pairs, rows 2-3 for last 3.

    scale_indices: which scale indices to plot (0-based). None = all.
    """
    all_labels = make_scale_labels(n_scales)
    if scale_indices is None:
        scale_indices = list(range(n_scales))
    # Validate
    scale_indices = [i for i in scale_indices if 0 <= i < n_scales]
    n_plot = len(scale_indices)
    colors = [cm.tab10(j / max(n_plot - 1, 1)) for j in range(n_plot)]

    fig, axes = plt.subplots(4, 3, figsize=(13, 11), squeeze=False)
    fig.suptitle(
        f"Pile-10k ID scaling (GRIDE all scales) — {pooling_variant} pooling",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )

    # Layout: first 3 pairs in rows 0-1, last 3 in rows 2-3
    group_pairs = [MODEL_PAIRS[:3], MODEL_PAIRS[3:]]
    group_row_offsets = [0, 2]

    for group_idx, (pairs, row_offset) in enumerate(zip(group_pairs, group_row_offsets)):
        for col, pair in enumerate(pairs):
            for role_offset, (model_id, role_label) in enumerate([
                (pair.pretrained_id, "Pretrained"),
                (pair.finetuned_id, "Finetuned"),
            ]):
                row = row_offset + role_offset
                ax = axes[row, col]
                title = f"{pair.label} — {role_label}"
                style_axis(ax, title=title)

                profile = load_profile(
                    results_root, pooling_variant, model_id, token_tag, residual_stream_type
                )
                if profile is None:
                    mark_missing(ax)
                    continue

                n_layers, n_s = profile.shape
                x = np.arange(n_layers)
                for plot_idx, scale_idx in enumerate(scale_indices):
                    if scale_idx >= n_s:
                        continue
                    ax.plot(
                        x,
                        profile[:, scale_idx],
                        color=colors[plot_idx],
                        linewidth=LINE_WIDTH,
                        marker="o",
                        markersize=MARKER_SIZE,
                        markeredgewidth=0.0,
                    )

    # Shared legend: one entry per plotted scale, placed below the entire figure
    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=colors[j],
            linewidth=LINE_WIDTH + 0.3,
            label=all_labels[scale_indices[j]],
        )
        for j in range(n_plot)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(n_plot, 9),
        fontsize=FONT_LEGEND + 0.5,
        framealpha=0.95,
        edgecolor="#ccc",
        handlelength=2.0,
        columnspacing=0.8,
        title="Neighbour-pair scale",
        title_fontsize=FONT_LEGEND,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.09, hspace=0.45, wspace=0.32)

    scales_tag = "all" if len(scale_indices) == n_scales else "-".join(str(i) for i in scale_indices)
    out_path = plots_root / f"id_scaling_full_{token_tag}_{pooling_variant}_{residual_stream_type}_scales{scales_tag}_4x3.png"
    save_fig(fig, out_path)


def plot_id_scaling_full_sliding(
    results_root: Path,
    plots_root: Path,
    pooling_variant: str,
    token_tag: str,
    residual_stream_type: str,
    n_scales: int,
    scale_indices: list[int] | None = None,
) -> None:
    """Same 4×3 grid but each point is the average of two consecutive layers.

    Point at x = i + 0.5  =  mean(profile[i], profile[i+1]).
    For n_layers layers the result has n_layers-1 points.
    """
    all_labels = make_scale_labels(n_scales)
    if scale_indices is None:
        scale_indices = list(range(n_scales))
    scale_indices = [i for i in scale_indices if 0 <= i < n_scales]
    n_plot = len(scale_indices)
    colors = [cm.tab10(j / max(n_plot - 1, 1)) for j in range(n_plot)]

    fig, axes = plt.subplots(4, 3, figsize=(13, 11), squeeze=False)
    fig.suptitle(
        f"Pile-10k ID scaling (GRIDE, sliding window avg) — {pooling_variant} pooling",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )

    group_pairs = [MODEL_PAIRS[:3], MODEL_PAIRS[3:]]
    group_row_offsets = [0, 2]

    for pairs, row_offset in zip(group_pairs, group_row_offsets):
        for col, pair in enumerate(pairs):
            for role_offset, (model_id, role_label) in enumerate([
                (pair.pretrained_id, "Pretrained"),
                (pair.finetuned_id, "Finetuned"),
            ]):
                row = row_offset + role_offset
                ax = axes[row, col]
                title = f"{pair.label} — {role_label}"
                style_axis(ax, title=title)
                ax.set_xlabel("Layer (midpoint)", fontsize=FONT_AXIS_LABEL)

                profile = load_profile(
                    results_root, pooling_variant, model_id, token_tag, residual_stream_type
                )
                if profile is None:
                    mark_missing(ax)
                    continue

                n_layers, n_s = profile.shape
                if n_layers < 2:
                    mark_missing(ax, "< 2 layers")
                    continue

                # sliding mean: shape (n_layers-1, n_s)
                smoothed = (profile[:-1] + profile[1:]) / 2.0
                x = np.arange(n_layers - 1) + 0.5

                for plot_idx, scale_idx in enumerate(scale_indices):
                    if scale_idx >= n_s:
                        continue
                    ax.plot(
                        x,
                        smoothed[:, scale_idx],
                        color=colors[plot_idx],
                        linewidth=LINE_WIDTH,
                        marker="o",
                        markersize=MARKER_SIZE,
                        markeredgewidth=0.0,
                    )

    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=colors[j],
            linewidth=LINE_WIDTH + 0.3,
            label=all_labels[scale_indices[j]],
        )
        for j in range(n_plot)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(n_plot, 9),
        fontsize=FONT_LEGEND + 0.5,
        framealpha=0.95,
        edgecolor="#ccc",
        handlelength=2.0,
        columnspacing=0.8,
        title="Neighbour-pair scale",
        title_fontsize=FONT_LEGEND,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.09, hspace=0.45, wspace=0.32)

    scales_tag = "all" if len(scale_indices) == n_scales else "-".join(str(i) for i in scale_indices)
    out_path = plots_root / f"id_scaling_full_{token_tag}_{pooling_variant}_{residual_stream_type}_scales{scales_tag}_sliding_4x3.png"
    save_fig(fig, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pile-10k ID-scaling profiles (all GRIDE scales), 4×3 grid."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "intrinsic_dimension" / DATASET_TAG,
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=PROJECT_ROOT / "plots" / "intrinsic_dimension" / DATASET_TAG,
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--prompt-format", choices=["wrapped", "plain", "raw"], default="raw")
    parser.add_argument("--remove-assistant-tokens", action="store_true", default=False)
    parser.add_argument(
        "--residual-stream-type", type=str, default="output_layer",
    )
    parser.add_argument(
        "--range-max", type=int, default=RANGE_MAX_DEFAULT,
        help="range_max used during computation; used as fallback for n_scales.",
    )
    parser.add_argument(
        "--scales", type=int, nargs="+", default=None, metavar="IDX",
        help=(
            "0-based scale indices to plot (e.g. --scales 0 2 4 8). "
            "Default: all scales present in the data."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    token_tag = f"tokens_{args.min_tokens}_{args.max_tokens}"
    # Build the pooling variant the same way the compute script does
    pv_parts = ["last"]
    if args.prompt_format in ("plain", "raw"):
        pv_parts.append(args.prompt_format)
    if args.remove_assistant_tokens:
        pv_parts.append("no_asst")
    pooling_variant = "_".join(pv_parts)

    print("=" * 72)
    print("  plot_pile10k_id_scaling_full.py")
    print(f"  results : {args.results_root}")
    print(f"  plots   : {args.plots_root}")
    print(f"  tokens  : {token_tag}")
    print(f"  variant : {pooling_variant}")
    print(f"  layer   : {args.residual_stream_type}")
    print(f"  models  : {len(MODEL_PAIRS)}")
    print("=" * 72)

    print("\n[0/2] Availability check ...")
    summarize_availability(
        args.results_root, pooling_variant, token_tag, args.residual_stream_type
    )

    n_scales = determine_n_scales(
        args.results_root, pooling_variant, token_tag,
        args.residual_stream_type, args.range_max,
    )
    all_labels = make_scale_labels(n_scales)
    print(f"  Detected n_scales = {n_scales}  ({all_labels})")

    scale_indices = args.scales
    if scale_indices is not None:
        scale_indices = [i for i in scale_indices if 0 <= i < n_scales]
        print(f"  Plotting scales : {[all_labels[i] for i in scale_indices]}")
    else:
        print(f"  Plotting scales : all ({n_scales})")

    print("\n[1/2] Plotting per-layer 4×3 grid ...")
    plot_id_scaling_full(
        args.results_root,
        args.plots_root,
        pooling_variant,
        token_tag,
        args.residual_stream_type,
        n_scales,
        scale_indices=scale_indices,
    )

    print("\n[2/2] Plotting sliding-window 4×3 grid ...")
    plot_id_scaling_full_sliding(
        args.results_root,
        args.plots_root,
        pooling_variant,
        token_tag,
        args.residual_stream_type,
        n_scales,
        scale_indices=scale_indices,
    )

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
