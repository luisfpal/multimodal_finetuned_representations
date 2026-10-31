"""Publication-quality information-imbalance scatter for Llama-3-90B Instruct.

Single scatter panel, aesthetics faithful to the information-imbalance plots:
seaborn whitegrid, sns.scatterplot, Line2D/Patch legend, tight_layout.
Saves to PDF.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-l11")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from safetensors import safe_open


# ==============================================================================
# SECTION 1 — CONFIGURATION
# ==============================================================================

# Hardcoded path: edit this path for your setup.
PROJECT_ROOT      = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
SIMILARITIES_ROOT = PROJECT_ROOT / "results" / "multimodality_comparison"
OUT_ROOT          = PROJECT_ROOT / "plots" / "llama_vision_similarity" / "last" / "paper"

MODEL_ID    = "Meta-Llama-3-90B-instruct"
MODEL_TITLE = "Llama-3.2-90B-Vision-Instruct"
POOLING_TAG = "last--last"
VARIANT_TAG = "unique_images"

DATASETS: tuple[tuple[str, str], ...] = (
    ("gqa_val_balanced_noyesno_captioned_synthetic",  "GQA"),
    ("mmbench_en_dev_captioned_open_ended_synthetic", "MMBench"),
    ("vqav2_coco_restval_icq_human_split01",          "VQAv2"),
)
DATASET_NAME_TEXT    = ", ".join(label for _d, label in DATASETS)
DATASET_FILENAME_TAG = "_".join(
    "".join(ch for ch in label.lower() if ch.isalnum()) for _d, label in DATASETS
)

SOURCE_ORIENTATION        = ("text", "image")
DISPLAY_ORIENTATION       = ("image", "text")
EXPECTED_VISUAL_LAYERS    = 100
EXPECTED_TEXT_ONLY_LAYERS = 80
FULL_TEXT_LAYERS          = 100

MLLAMA_CROSS_ATTENTION_LAYERS: tuple[int, ...] = (
    3, 8, 13, 18, 23, 28, 33, 38, 43, 48,
    53, 58, 63, 68, 73, 78, 83, 88, 93, 98,
)

MARKER_CONFIG = {
    "text_to_img": {"color": "#007e5d", "marker": "s", "zorder": 2},
    "img_to_text": {"color": "#991499", "marker": "o", "zorder": 2},
}

# Marker sizes scaled for the 100-layer grid (vs. 17–32-layer models in reference)
SCATTER_BASE_SIZE = 14.0   # pt²  (reference uses 80 for ~25-layer models)
SCATTER_FACTOR    = 4.0    # size ratios: 14 / 2.5 / ~0.9

THRESHOLDS = {"small": 0.04, "medium": 0.05, "large": 0.06}


# ==============================================================================
# SECTION 2 — DATA LOADING
# ==============================================================================

def metric_path(dataset: str, metric: str) -> Path:
    return (
        SIMILARITIES_ROOT / dataset / POOLING_TAG / VARIANT_TAG
        / "similarities" / f"{MODEL_ID}_{metric}.safetensors"
    )


def normalize_modality(value: str | None, fallback: str) -> str:
    label = str(value or "").strip().lower()
    return label if label in {"text", "image"} else fallback


def resolve_orientation(metadata: dict[str, str] | None) -> tuple[str, str]:
    if not metadata:
        return SOURCE_ORIENTATION
    row = normalize_modality(metadata.get("row_modality"), SOURCE_ORIENTATION[0])
    col = normalize_modality(metadata.get("col_modality"), SOURCE_ORIENTATION[1])
    if row == col:
        return SOURCE_ORIENTATION
    return row, col


def orient_for_display(matrix: np.ndarray, source_orientation: tuple[str, str]) -> np.ndarray:
    return matrix if source_orientation == DISPLAY_ORIENTATION else matrix.T


def load_matrix(dataset: str, metric: str) -> np.ndarray:
    path = metric_path(dataset, metric)
    if not path.exists():
        raise FileNotFoundError(path)
    with safe_open(path, framework="np", device="cpu") as f:
        if metric not in f.keys():
            raise KeyError(f"Missing key {metric!r} in {path}")
        matrix = f.get_tensor(metric).copy()
        orientation = resolve_orientation(f.metadata())
    matrix = orient_for_display(matrix, orientation)
    expected = (EXPECTED_VISUAL_LAYERS, EXPECTED_TEXT_ONLY_LAYERS)
    if matrix.shape != expected:
        raise ValueError(f"Expected {expected} for {dataset}/{metric}, got {matrix.shape}")
    return matrix


def load_metric_by_dataset(metric: str) -> dict[str, np.ndarray]:
    return {dataset: load_matrix(dataset, metric) for dataset, _label in DATASETS}


def average_metric(matrices: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(np.stack([matrices[d] for d, _label in DATASETS], axis=0), axis=0)


def expand_text_axis_for_mllama(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (EXPECTED_VISUAL_LAYERS, FULL_TEXT_LAYERS):
        return matrix
    expected = (EXPECTED_VISUAL_LAYERS, EXPECTED_TEXT_ONLY_LAYERS)
    if matrix.shape != expected:
        raise ValueError(f"Expected {expected} before expansion, got {matrix.shape}")
    cross_attention = set(MLLAMA_CROSS_ATTENTION_LAYERS)
    columns: list[np.ndarray] = []
    source_col = 0
    for full_layer_idx in range(FULL_TEXT_LAYERS):
        if full_layer_idx in cross_attention:
            columns.append(columns[-1])
        else:
            columns.append(matrix[:, source_col])
            source_col += 1
    return np.stack(columns, axis=1)


# ==============================================================================
# SECTION 3 — SCATTER FORMATTING  (faithful to the information-imbalance plots)
# ==============================================================================

def _get_marker_size(val: float, thresholds: dict, base_size: float, factor: float) -> float:
    if val < thresholds["small"]:
        return base_size
    elif val < thresholds["medium"]:
        return base_size / factor
    return base_size / (factor ** 2)


def _format_scatter(
    imbalance: np.ndarray,
    direction: str,
    thresholds: dict,
    base_size: float,
    factor: float,
    linewidth: float = 0.8,
) -> list[dict]:
    config = MARKER_CONFIG[direction]
    n_visual, n_text = imbalance.shape
    points_by_size: dict[float, dict] = {}
    for i in range(n_visual):
        for j in range(n_text):
            val = imbalance[i, j]
            if val >= thresholds["large"]:
                continue
            size = _get_marker_size(val, thresholds, base_size, factor)
            if size not in points_by_size:
                points_by_size[size] = {"x": [], "y": []}
            points_by_size[size]["x"].append(j + 1)   # text layer on x-axis
            points_by_size[size]["y"].append(i + 1)   # visual layer on y-axis
    datasets = []
    for size, pts in sorted(points_by_size.items(), reverse=True):
        datasets.append({
            "x": np.array(pts["x"]), "y": np.array(pts["y"]),
            "marker": config["marker"], "size": size,
            "linewidth": linewidth, "edgecolor": config["color"],
            "zorder": config["zorder"],
        })
    return datasets


def _plot_scatter(ax: plt.Axes, datasets: list[dict]) -> None:
    for data in datasets:
        sns.scatterplot(
            ax=ax, x=data["x"], y=data["y"],
            marker=data["marker"], s=data["size"],
            facecolors="None", edgecolors=data["edgecolor"],
            linewidths=data["linewidth"], label=None, zorder=data["zorder"],
            rasterized=True,
        )


def _setup_axis(ax: plt.Axes, n_visual: int, n_text: int, thresholds: dict) -> None:
    ticks = np.arange(0, max(n_visual, n_text), 10) + 1  # every 10 layers for 100-layer model
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(0.4, n_text + 1)
    ax.set_ylim(0.4, n_visual + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Layer (text input)", fontsize=15)
    ax.set_ylabel("Layer (multimodal input)", fontsize=15)


# ==============================================================================
# SECTION 4 — LEGEND  (faithful to the information-imbalance plots)
# ==============================================================================

def _make_legend_handles(thresholds: dict) -> list:
    base_size, factor = 80, 5   # fixed reference sizes for legend visual consistency
    dir_kw  = dict(color="w", markerfacecolor="none", linewidth=0)
    size_kw = dict(color="w", markerfacecolor="none", linewidth=0.8)
    return [
        Line2D([0], [0], marker=MARKER_CONFIG["text_to_img"]["marker"],
               markeredgecolor=MARKER_CONFIG["text_to_img"]["color"],
               markersize=8, label="text → image", **dir_kw),
        Line2D([0], [0], marker=MARKER_CONFIG["img_to_text"]["marker"],
               markeredgecolor=MARKER_CONFIG["img_to_text"]["color"],
               markersize=8, label="image → text", **dir_kw),
        Patch(facecolor="none", edgecolor="none", label=" "),
        Line2D([0], [0], marker="o", markeredgecolor="black",
               markersize=np.sqrt(base_size),
               label=f'$\\Delta$ < {thresholds["small"]:.2f}', **size_kw),
        Line2D([0], [0], marker="o", markeredgecolor="black",
               markersize=np.sqrt(base_size / factor),
               label=f'{thresholds["small"]:.2f} ≤ $\\Delta$ < {thresholds["medium"]:.2f}',
               **size_kw),
        Line2D([0], [0], marker="o", markeredgecolor="black",
               markersize=np.sqrt(base_size / factor ** 2),
               label=f'{thresholds["medium"]:.2f} ≤ $\\Delta$ < {thresholds["large"]:.2f}',
               **size_kw),
    ]


# ==============================================================================
# SECTION 5 — FIGURE
# ==============================================================================

def save_figure() -> Path:
    t2i_avg = average_metric(load_metric_by_dataset("information_imbalance"))
    i2t_avg = average_metric(load_metric_by_dataset("information_imbalance_reversed"))
    t2i_exp = expand_text_axis_for_mllama(t2i_avg)
    i2t_exp = expand_text_axis_for_mllama(i2t_avg)

    n_visual, n_text = i2t_exp.shape

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))

    scatter_data = sorted(
        _format_scatter(t2i_exp, "text_to_img", THRESHOLDS, SCATTER_BASE_SIZE, SCATTER_FACTOR) +
        _format_scatter(i2t_exp, "img_to_text", THRESHOLDS, SCATTER_BASE_SIZE, SCATTER_FACTOR),
        key=lambda d: d["zorder"],
    )
    _plot_scatter(ax, scatter_data)
    _setup_axis(ax, n_visual, n_text, THRESHOLDS)

    fig.suptitle(MODEL_TITLE, fontsize=22, y=1.0)

    handles = _make_legend_handles(THRESHOLDS)
    leg1 = fig.legend(
        handles=handles[:2],
        fontsize=15, loc="lower center", bbox_to_anchor=(0.5, -0.07),
        ncol=2, frameon=False, columnspacing=1.5,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=handles[3:],
        fontsize=15, loc="lower center", bbox_to_anchor=(0.5, -0.13),
        ncol=3, frameon=False, columnspacing=1.0,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    out_dir = OUT_ROOT / MODEL_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"information_imbalance_{DATASET_FILENAME_TAG}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==============================================================================
# SECTION 6 — ENTRY POINT
# ==============================================================================

def main() -> None:
    sns.set_style("whitegrid", rc={"axes.edgecolor": ".15",
                                   "xtick.bottom": True, "ytick.left": True})
    out = save_figure()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
