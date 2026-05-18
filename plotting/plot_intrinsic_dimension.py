"""Intrinsic dimension publication figures.

Three figure groups:
  1. summary/all_models_2x2.png
     All models, dataset-averaged, normalized x-axis, 2×2 layout:
     TL=pretrained img, TR=pretrained txt, BL=finetuned img, BR=finetuned txt.
     Legend: model names (bottom of figure).

  2. per_model_avg/<model_slug>.png
     Per-model 1×2: left=txt, right=img.
     Each column: pretrained vs finetuned curves averaged over all datasets.

  3. per_model_detailed/<model_slug>.png
     Per-model 2×2: TL=pretrained img, TR=pretrained txt,
     BL=finetuned img, BR=finetuned txt.
     All 5 datasets plotted; dataset legend in TL subplot (upper right).

Output: PROJECT_ROOT/plots/intrinsic_dimension/
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

# HPC-safe matplotlib config directory setup (must happen before any plt import).
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


# ==============================================================================
# SECTION 1 — CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
RESULTS_DIR = PROJECT_ROOT / "results" / "intrinsic_dimension"
PLOTS_DIR = PROJECT_ROOT / "plots" / "intrinsic_dimension"

POOLING_TAG = "last"
VARIANT_TAG = "unique_images"
LAYER_TYPE = "output_layer"
ALGORITHM = "scaling_gride"
DPI = 200

# ── Model pairs ───────────────────────────────────────────────────────────────
# (finetuned_id, pretrained_id, short_label, output_slug)
MODEL_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    (
        "llava_7b",
        "llava-v1.5-7b-pretrained-lbasile",
        "LLaVA 7B",
        "llava_7b",
    ),
    (
        "llava_13b",
        "llava-v1.5-13b-pretrained-lbasile",
        "LLaVA 13B",
        "llava_13b",
    ),
    (
        "LLaVA-OneVision-1.5-4B-Instruct",
        "LLaVA-OneVision-1.5-4B-Base",
        "LLaVA-OV 4B",
        "onevision_4b",
    ),
    (
        "LLaVA-OneVision-1.5-8B-Instruct",
        "LLaVA-OneVision-1.5-8B-Base",
        "LLaVA-OV 8B",
        "onevision_8b",
    ),
    (
        "InternVL2-8B",
        "InternVL2-8B-Pretrain",
        "InternVL2 8B",
        "internvl2_8b",
    ),
    (
        "cambrian-8b",
        "cambrian-8b-pretrained",
        "Cambrian 8B",
        "cambrian_8b",
    ),
)

# ── Datasets ──────────────────────────────────────────────────────────────────
# (dataset_slug, display_label)
DATASETS: tuple[tuple[str, str], ...] = (
    ("cocoqa_restval_icqa_split01_human", "COCO-QA"),
    ("gqa_val_balanced_noyesno_captioned_synthetic", "GQA"),
    ("vqav2_coco_restval_icq_split01_human", "VQAv2"),
    # ("mmbench_en_dev_captioned_open_ended_synthetic", "MMBench"),
    # ("seed_bench_images_captioned_open_ended_synthetic", "SEED"),
)

# ── Colour-blind-safe palette (Wong 2011) ─────────────────────────────────────
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
)

# Pretrained / finetuned pair colours used in 1×2 avg figures.
COLOR_PRE = "#0072B2"
COLOR_FINE = "#D55E00"

# ── Styling constants ─────────────────────────────────────────────────────────
LINE_WIDTH = 1.45
MARKER_SIZE = 3.0
MARKER_EDGE_WIDTH = 0.0
GRID_ALPHA = 0.18

# ── Figure sizes (inches) ─────────────────────────────────────────────────────
FIGSIZE_2X2 = (8.0, 6.0)
FIGSIZE_1X2 = (7.0, 2.8)
FIGSIZE_SUMMARY = (8.0, 6.5)  # extra height for legend below

# ── Font sizes ────────────────────────────────────────────────────────────────
FONT_SUPTITLE = 11
FONT_TITLE = 10
FONT_AXIS_LABEL = 9
FONT_TICK = 8
FONT_LEGEND = 7

LEGEND_HANDLE_LEN = 2.0
LEGEND_COL_SPACING = 1.0


# ==============================================================================
# SECTION 2 — DATA LOADING
# ==============================================================================


def load_id_curve(dataset: str, model: str, modality: str) -> np.ndarray | None:
    path = (
        RESULTS_DIR
        / dataset
        / POOLING_TAG
        / VARIANT_TAG
        / model
        / LAYER_TYPE
        / f"{modality}_{ALGORITHM}.safetensors"
    )
    if not path.exists():
        return None
    try:
        with safe_open(str(path), framework="numpy") as f:
            if "intrinsic_dimension" not in f.keys():
                return None
            return f.get_tensor("intrinsic_dimension").copy().astype(float)
    except Exception as exc:
        print(f"  ⚠ failed to load {path}: {exc}")
        return None


def average_over_datasets(model: str, modality: str) -> np.ndarray | None:
    curves: list[np.ndarray] = []
    for dataset_slug, _ in DATASETS:
        c = load_id_curve(dataset_slug, model, modality)
        if c is not None:
            curves.append(c)
    if not curves:
        return None
    lengths = {len(c) for c in curves}
    if len(lengths) != 1:
        print(f"  ⚠ {model}/{modality}: mixed curve lengths {sorted(lengths)}, skipping")
        return None
    return np.mean(np.stack(curves, axis=0), axis=0)


# ==============================================================================
# SECTION 3 — PLOT UTILITIES
# ==============================================================================


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


def normalized_depth(n: int) -> np.ndarray:
    if n == 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, n)


def style_axis(
    ax: plt.Axes,
    *,
    title: str,
    x_label: str,
    y_label: str = "Intrinsic dimension",
) -> None:
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=6)
    ax.set_xlabel(x_label, fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_AXIS_LABEL)
    ax.tick_params(
        direction="out", length=3.0, width=0.8, color="#444444", labelsize=FONT_TICK
    )
    ax.set_axisbelow(True)


def mark_missing(ax: plt.Axes) -> None:
    ax.set_facecolor("#f5f5f5")
    ax.text(
        0.5, 0.5, "N/A",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10, color="#aaa", style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  ✓ {path}")


# ==============================================================================
# SECTION 4 — PER-MODEL DETAILED 2×2 FIGURE (ALL DATASETS)
# ==============================================================================
#
# Layout:
#   TL (0,0): pretrained – image    TR (0,1): pretrained – text
#   BL (1,0): finetuned  – image    BR (1,1): finetuned  – text
#
# Dataset legend only in TL subplot (upper right corner).


def plot_per_model_detailed(
    fine_id: str,
    pre_id: str,
    label: str,
    slug: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2, squeeze=False)
    fig.suptitle(label, fontsize=FONT_SUPTITLE, fontweight="bold")

    quadrants: list[tuple[int, int, str, str, str]] = [
        (0, 0, pre_id,  "img", "Pretrained — Image"),
        (0, 1, pre_id,  "txt", "Pretrained — Text"),
        (1, 0, fine_id, "img", "Finetuned — Image"),
        (1, 1, fine_id, "txt", "Finetuned — Text"),
    ]

    for row, col, model, modality, title in quadrants:
        ax = axes[row, col]
        style_axis(ax, title=title, x_label="Layer")
        any_data = False
        for i, (dataset_slug, dataset_label) in enumerate(DATASETS):
            curve = load_id_curve(dataset_slug, model, modality)
            if curve is None:
                continue
            x = np.arange(len(curve))
            ax.plot(
                x,
                curve,
                color=PALETTE[i % len(PALETTE)],
                marker="o",
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=dataset_label,
            )
            any_data = True
        if not any_data:
            mark_missing(ax)

    # Dataset legend — only in TL subplot, upper right.
    axes[0, 0].legend(
        loc="upper right",
        fontsize=FONT_LEGEND,
        handlelength=LEGEND_HANDLE_LEN,
        columnspacing=LEGEND_COL_SPACING,
        framealpha=0.88,
        edgecolor="#ccc",
    )

    fig.tight_layout()
    save_fig(fig, PLOTS_DIR / "per_model_detailed" / f"{slug}.png")


# ==============================================================================
# SECTION 5 — PER-MODEL AVERAGE 1×2 FIGURE (PRETRAINED VS FINETUNED)
# ==============================================================================
#
# Layout:
#   Col 0: txt modality — pretrained curve + finetuned curve (avg over datasets)
#   Col 1: img modality — same


def plot_per_model_avg(
    fine_id: str,
    pre_id: str,
    label: str,
    slug: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_1X2, squeeze=False)
    fig.suptitle(label, fontsize=FONT_SUPTITLE, fontweight="bold")

    for col, modality in enumerate(("txt", "img")):
        ax = axes[0, col]
        mod_title = "Text" if modality == "txt" else "Image"
        style_axis(ax, title=mod_title, x_label="Layer")

        for model_id, model_label, color in (
            (pre_id,  "Pretrained", COLOR_PRE),
            (fine_id, "Finetuned",  COLOR_FINE),
        ):
            curve = average_over_datasets(model_id, modality)
            if curve is None:
                continue
            ax.plot(
                np.arange(len(curve)),
                curve,
                color=color,
                marker="o",
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=model_label,
            )

        ax.legend(
            loc="upper right",
            fontsize=FONT_LEGEND,
            handlelength=LEGEND_HANDLE_LEN,
            framealpha=0.88,
            edgecolor="#ccc",
        )

    fig.tight_layout()
    save_fig(fig, PLOTS_DIR / "per_model_avg" / f"{slug}.png")


# ==============================================================================
# SECTION 6 — SUMMARY 2×2 FIGURE (ALL MODELS, NORMALIZED DEPTH)
# ==============================================================================
#
# Layout:
#   TL (0,0): pretrained – image    TR (0,1): pretrained – text
#   BL (1,0): finetuned  – image    BR (1,1): finetuned  – text
#
# Per model: dataset-averaged curve. X-axis normalized to [0, 1] across models.
# Legend (model names) placed below the figure.


def plot_summary_all_models() -> None:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_SUMMARY, squeeze=False)
    fig.suptitle(
        "Intrinsic Dimension — All Models (dataset average)",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )

    quadrants: list[tuple[int, int, str, str, str]] = [
        (0, 0, "pretrained", "img", "Pretrained — Image"),
        (0, 1, "pretrained", "txt", "Pretrained — Text"),
        (1, 0, "finetuned",  "img", "Finetuned — Image"),
        (1, 1, "finetuned",  "txt", "Finetuned — Text"),
    ]

    # Collect legend handles from the first quadrant only.
    legend_handles: list = []
    legend_labels: list[str] = []

    for row, col, role, modality, title in quadrants:
        ax = axes[row, col]
        style_axis(ax, title=title, x_label="Normalized depth")

        for i, (fine_id, pre_id, model_label, _slug) in enumerate(MODEL_PAIRS):
            model_id = fine_id if role == "finetuned" else pre_id
            curve = average_over_datasets(model_id, modality)
            if curve is None:
                continue
            x = normalized_depth(len(curve))
            (line,) = ax.plot(
                x,
                curve,
                color=PALETTE[i % len(PALETTE)],
                marker="o",
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=model_label,
            )
            if row == 0 and col == 0:
                legend_handles.append(line)
                legend_labels.append(model_label)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#ccc",
        handlelength=LEGEND_HANDLE_LEN,
        columnspacing=LEGEND_COL_SPACING,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    save_fig(fig, PLOTS_DIR / "summary" / "all_models_2x2.png")


# ==============================================================================
# SECTION 7 — FINETUNED MINUS PRETRAINED DIFFERENCE 1×2 FIGURE
# ==============================================================================
#
# Layout:
#   Col 0: img modality — (finetuned - pretrained) per model, normalized depth
#   Col 1: txt modality — same
#
# Per model: diff = avg_over_datasets(finetuned) - avg_over_datasets(pretrained).
# X-axis normalized to [0, 1]. Horizontal dashed zero-line for reference.
# Legend (model names) placed below the figure.


def plot_diff_all_models() -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_1X2, squeeze=False)
    fig.suptitle(
        "Pretrained − Finetuned Intrinsic Dimension (dataset average)",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )

    legend_handles: list = []
    legend_labels: list[str] = []

    for col, modality in enumerate(("img", "txt")):
        ax = axes[0, col]
        mod_title = "Image" if modality == "img" else "Text"
        style_axis(ax, title=mod_title, x_label="Normalized depth", y_label="Δ ID (pretrained − finetuned)")
        ax.axhline(0.0, color="#888", linewidth=0.8, linestyle="--", zorder=0)

        for i, (fine_id, pre_id, model_label, _slug) in enumerate(MODEL_PAIRS):
            fine_curve = average_over_datasets(fine_id, modality)
            pre_curve = average_over_datasets(pre_id, modality)
            if fine_curve is None or pre_curve is None:
                continue
            if len(fine_curve) != len(pre_curve):
                print(f"  ⚠ {model_label}/{modality}: length mismatch, skipping diff")
                continue
            diff = pre_curve - fine_curve
            x = normalized_depth(len(diff))
            (line,) = ax.plot(
                x,
                diff,
                color=PALETTE[i % len(PALETTE)],
                marker="o",
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=model_label,
            )
            if col == 0:
                legend_handles.append(line)
                legend_labels.append(model_label)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#ccc",
        handlelength=LEGEND_HANDLE_LEN,
        columnspacing=LEGEND_COL_SPACING,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_fig(fig, PLOTS_DIR / "summary" / "diff_finetuned_minus_pretrained.png")


# ==============================================================================
# SECTION 8 — MAIN
# ==============================================================================


def main() -> None:
    configure_style()

    print("=" * 72)
    print("  plot_intrinsic_dimension.py")
    print(f"  models   : {len(MODEL_PAIRS)}")
    print(f"  datasets : {len(DATASETS)}")
    print(f"  algorithm: {ALGORITHM}")
    print(f"  output   : {PLOTS_DIR}")
    print("=" * 72)

    print("\n[1/4] Summary — all models, normalized depth ...")
    plot_summary_all_models()

    print("\n[2/4] Diff (finetuned − pretrained), normalized depth ...")
    plot_diff_all_models()

    print("\n[3/4] Per-model average (1×2) ...")
    for fine_id, pre_id, label, slug in MODEL_PAIRS:
        print(f"  {label}")
        plot_per_model_avg(fine_id, pre_id, label, slug)

    print("\n[4/4] Per-model detailed (2×2, all datasets) ...")
    for fine_id, pre_id, label, slug in MODEL_PAIRS:
        print(f"  {label}")
        plot_per_model_detailed(fine_id, pre_id, label, slug)

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
