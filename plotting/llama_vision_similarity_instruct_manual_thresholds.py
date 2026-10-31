"""Save one Llama-3-90B Instruct average imbalance overview figure."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-l11")

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

try:
    from scipy.stats import gaussian_kde
except ImportError:  # pragma: no cover - optional dependency fallback
    gaussian_kde = None


PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
SIMILARITIES_ROOT = PROJECT_ROOT / "results" / "multimodality_comparison"
OUT_ROOT = PROJECT_ROOT / "plots" / "llama_vision_similarity" / "last" / "test"

MODEL_ID = "Meta-Llama-3-90B-instruct"
MODEL_TITLE = "Llama-3-90B Instruct"
POOLING_TAG = "last--last"
VARIANT_TAG = "unique_images"

DATASETS: tuple[tuple[str, str], ...] = (
    # ("cocoqa_restval_icqa_human_split01", "COCOQA"),
    ("gqa_val_balanced_noyesno_captioned_synthetic", "GQA"),
    ("mmbench_en_dev_captioned_open_ended_synthetic", "MMBench"),
    # ("seed_bench_images_captioned_open_ended_synthetic", "SEED"),
    ("vqav2_coco_restval_icq_human_split01", "VQAv2"),
)
DATASET_NAME_TEXT = ", ".join(label for _dataset, label in DATASETS)
DATASET_FILENAME_TAG = "_".join(
    "".join(ch for ch in label.lower() if ch.isalnum()) for _dataset, label in DATASETS
)

AVERAGE_IMBALANCE_THRESHOLDS: tuple[float, float, float] = (0.04, 0.05, 0.06)

SOURCE_ORIENTATION = ("text", "image")
DISPLAY_ORIENTATION = ("image", "text")
DPI = 200
EXPECTED_VISUAL_LAYERS = 100
EXPECTED_TEXT_ONLY_LAYERS = 80
FULL_TEXT_LAYERS = 100
MLLAMA_CROSS_ATTENTION_LAYERS: tuple[int, ...] = (
    3,
    8,
    13,
    18,
    23,
    28,
    33,
    38,
    43,
    48,
    53,
    58,
    63,
    68,
    73,
    78,
    83,
    88,
    93,
    98,
)
LAYER_TICKS: tuple[int, ...] = (1, 20, 40, 60, 80, 100)
MARKER_SCALE = 0.63
SINGLE_PANEL_IN = 7.0

TEXT_TO_IMAGE_STYLE = dict(
    marker="s",
    edgecolors="green",
    facecolors="none",
    linewidths=0.8,
)
IMAGE_TO_TEXT_STYLE = dict(
    marker="o",
    edgecolors="purple",
    facecolors="none",
    linewidths=0.8,
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Bitstream Vera Sans",
                "sans-serif",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def metric_path(dataset: str, metric: str) -> Path:
    return (
        SIMILARITIES_ROOT
        / dataset
        / POOLING_TAG
        / VARIANT_TAG
        / "similarities"
        / f"{MODEL_ID}_{metric}.safetensors"
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
    if source_orientation == DISPLAY_ORIENTATION:
        return matrix
    return matrix.T


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
        raise ValueError(
            f"Expected {expected} after orientation for {dataset}/{metric}, got {matrix.shape}"
        )
    return matrix


def load_metric_by_dataset(metric: str) -> dict[str, np.ndarray]:
    return {dataset: load_matrix(dataset, metric) for dataset, _label in DATASETS}


def average_metric(matrices: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(np.stack([matrices[dataset] for dataset, _label in DATASETS], axis=0), axis=0)


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


def threshold_bands(values: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    q1, q2, q3 = sorted(float(value) for value in values)
    pts_per_cell = (SINGLE_PANEL_IN * 72) / FULL_TEXT_LAYERS
    s_max = pts_per_cell**2
    return [(0.0, q1, s_max), (q1, q2, s_max * 0.25), (q2, q3, s_max * 0.05)]


def threshold_values(thresholds: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(hi for _lo, hi, _area in thresholds)  # type: ignore[return-value]


def format_thresholds(thresholds: list[tuple[float, float, float]]) -> str:
    return ", ".join(f"{value:.3f}" for value in threshold_values(thresholds))


def set_layer_ticks(ax: plt.Axes) -> None:
    ax.set_xticks(LAYER_TICKS)
    ax.set_yticks(LAYER_TICKS)


def direction_handles() -> list[mlines.Line2D]:
    kw = dict(linestyle="None", markerfacecolor="none")
    return [
        mlines.Line2D([], [], color="green", marker="s", markersize=8, label="text -> image", **kw),
        mlines.Line2D([], [], color="purple", marker="o", markersize=8, label="image -> text", **kw),
    ]


def size_handles(thresholds: list[tuple[float, float, float]]) -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    for lo, hi, area in thresholds:
        marker_size = max(4.0, float(np.sqrt(area * MARKER_SCALE)))
        label = f"imb. < {hi:.3f}" if lo == 0.0 else f"{lo:.3f} <= imb. < {hi:.3f}"
        handles.append(
            mlines.Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                markerfacecolor="none",
                color="black",
                markersize=marker_size,
                label=label,
            )
        )
    return handles


def draw_imbalance_scatter(
    ax: plt.Axes,
    img_to_text: np.ndarray,
    text_to_img: np.ndarray,
    thresholds: list[tuple[float, float, float]],
) -> None:
    n_visual, n_text = img_to_text.shape
    if text_to_img.shape != img_to_text.shape:
        raise ValueError(f"Direction shape mismatch: {img_to_text.shape} vs {text_to_img.shape}")

    ax.set_xlim(0.5, n_text + 0.5)
    ax.set_ylim(0.5, n_visual + 0.5)
    set_layer_ticks(ax)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.35)

    vis_idx, txt_idx = np.indices((n_visual, n_text))
    x = txt_idx + 1
    y = vis_idx + 1

    for lo, hi, area in thresholds:
        marker_area = area * MARKER_SCALE
        text_to_image_mask = (text_to_img >= lo) & (text_to_img < hi)
        image_to_text_mask = (img_to_text >= lo) & (img_to_text < hi)

        ax.scatter(x[text_to_image_mask], y[text_to_image_mask], s=marker_area, zorder=3, **TEXT_TO_IMAGE_STYLE)
        ax.scatter(x[image_to_text_mask], y[image_to_text_mask], s=marker_area, zorder=4, **IMAGE_TO_TEXT_STYLE)


def draw_distribution(
    ax: plt.Axes,
    pool: np.ndarray,
    thresholds: list[tuple[float, float, float]],
) -> None:
    ax.hist(pool, bins=70, density=True, alpha=0.72, color="#a8c8f0", edgecolor="none")

    if gaussian_kde is not None:
        lo = float(np.percentile(pool, 0.1))
        hi = float(np.percentile(pool, 99.9))
        x_grid = np.linspace(lo, hi, 2048)
        kde_vals = gaussian_kde(pool, bw_method="scott")(x_grid)
        ax.plot(x_grid, kde_vals, color="#1a5fa8", linewidth=1.8)

    colors = ("#d62728", "#e06c00", "#2ca02c")
    handles = []
    for idx, (value, color) in enumerate(zip(threshold_values(thresholds), colors), start=1):
        handles.append(
            ax.axvline(
                value,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
                zorder=5,
                label=f"T{idx} = {value:.3f}",
            )
        )

    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
        edgecolor="#bbb",
        title="Manual thresholds",
        title_fontsize=8.5,
    )
    ax.set_xlabel("information imbalance")
    ax.set_ylabel("density")
    ax.grid(True, linestyle="--", alpha=0.3)


def save_overview() -> Path:
    out_dir = OUT_ROOT / MODEL_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = threshold_bands(AVERAGE_IMBALANCE_THRESHOLDS)
    text_to_img_by_dataset = load_metric_by_dataset("information_imbalance")
    img_to_text_by_dataset = load_metric_by_dataset("information_imbalance_reversed")
    text_to_img_avg = average_metric(text_to_img_by_dataset)
    img_to_text_avg = average_metric(img_to_text_by_dataset)
    pool = np.concatenate([text_to_img_avg.ravel(), img_to_text_avg.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0), gridspec_kw={"width_ratios": [1, 1.05]})
    fig.subplots_adjust(bottom=0.18, wspace=0.27)

    scatter_ax, dist_ax = axes
    draw_imbalance_scatter(
        scatter_ax,
        expand_text_axis_for_mllama(img_to_text_avg),
        expand_text_axis_for_mllama(text_to_img_avg),
        thresholds,
    )
    scatter_ax.set_xlabel("text layer")
    scatter_ax.set_ylabel("visual layer")
    scatter_ax.set_title("Average information imbalance")

    draw_distribution(dist_ax, pool, thresholds)
    dist_ax.set_title("Average imbalance distribution")

    fig.suptitle(
        f"{MODEL_TITLE} - Average imbalance overview\n"
        f"Datasets: {DATASET_NAME_TEXT}\n"
        f"Manual thresholds: {format_thresholds(thresholds)}",
        y=1.04,
        fontsize=13,
    )
    fig.legend(
        handles=size_handles(thresholds),
        loc="lower center",
        bbox_to_anchor=(0.22, 0.01),
        ncol=1,
        fontsize=8,
        framealpha=0.95,
        edgecolor="#bbb",
    )
    fig.legend(
        handles=direction_handles(),
        loc="lower center",
        bbox_to_anchor=(0.42, 0.01),
        ncol=1,
        fontsize=8,
        framealpha=0.95,
        edgecolor="#bbb",
    )

    out_path = out_dir / f"average_information_imbalance_overview_{DATASET_FILENAME_TAG}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    apply_style()
    output = save_overview()
    print(f"Saved overview: {output}")


if __name__ == "__main__":
    main()
