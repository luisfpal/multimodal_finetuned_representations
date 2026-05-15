"""Information-imbalance scatter grids in the tueplots NeurIPS style."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tueplots_style import OUTPUT_DIR, get_rc  # noqa: E402

LUIS_ROOT = Path("/u/dssc/lbasile/scratch/plm/luis")
DATA_DIR = LUIS_ROOT / "data" / "multimodality"
FIGURES_DIR = OUTPUT_DIR / "imbalance_grids" / "last" / "paper"

POOLING_METHOD = "last"
VARIANT_TAG = "unique_images"

PERCENTILES: tuple[int, int, int] = (17, 34, 50)

MODEL_PAIRS: tuple[tuple, ...] = (
    ("llava_7b", "finetuned (LLaVA 7B)", "llava-v1.5-7b-pretrained-lbasile",
     "pretrained (LLaVA 7B)", (5.5, 17.5), (9, 30, 39)),
    ("llava_13b", "finetuned (LLaVA 13B)", "llava-v1.5-13b-pretrained-lbasile",
     "pretrained (LLaVA 13B)", (6.5, 21.5), (9, 32, 43)),
    ("LLaVA-OneVision-1.5-4B-Instruct", "finetuned (OneVision 4B)", "LLaVA-OneVision-1.5-4B-Base",
     "pretrained (OneVision 4B)", (4.5, 13.5), (4, 12, 17)),
    ("LLaVA-OneVision-1.5-8B-Instruct", "finetuned (OneVision 8B)", "LLaVA-OneVision-1.5-8B-Base",
     "pretrained (OneVision 8B)", (5.5, 17.5), (7, 13, 19)),
    ("InternVL2-8B", "finetuned (InternVL2 8B)", "InternVL2-8B-Pretrain",
     "pretrained (InternVL2 8B)", (5.5, 17.5), (10, 26, 41)),
    ("cambrian-8b", "finetuned (Cambrian 8B)", "cambrian-8b-pretrained",
     "pretrained (Cambrian 8B)", (5.5, 17.5), (18, 31, 41)),
)

DATASETS_TO_AVERAGE: tuple[str, ...] = (
    "cocoqa_restval_icqa_split01_human",
    "gqa_val_balanced_noyesno_captioned_synthetic",
    "vqav2_coco_restval_icq_split01_human",
    "mmbench_en_dev_captioned_open_ended_synthetic",
    "seed_bench_images_captioned_open_ended_synthetic",
)

CELL_W = 2.7
CELL_H = 2.7
WSPACE = 0.14
LEFT_M = 0.92
RIGHT_M = 0.25
TOP_M = 0.48
BOTTOM_M = 0.78

STYLE_T2I = dict(marker="s", edgecolors="#228833", facecolors="none", linewidths=0.85)
STYLE_I2T = dict(marker="o", edgecolors="#7b2d8b", facecolors="none", linewidths=0.85)
MARKER_SCALE = 0.63

LEGEND_SIZE_ANCHOR = (0.4, -0.05)
LEGEND_DIR_ANCHOR = (0.6, -0.02)

SOURCE_ORIENTATION = ("text", "image")
DISPLAY_ORIENTATION = ("image", "text")


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower())


def _build_path(dataset: str, model_id: str, metric: str) -> Path:
    return (
        DATA_DIR / dataset / POOLING_METHOD / VARIANT_TAG / "similarities"
        / f"{model_id}_{metric}.json"
    )


def _read_orientation(metadata: dict[str, str] | None) -> tuple[str, str]:
    if not metadata:
        return SOURCE_ORIENTATION
    row = str(metadata.get("row_modality", "")).strip().lower()
    col = str(metadata.get("col_modality", "")).strip().lower()
    if row not in {"text", "image"} or col not in {"text", "image"} or row == col:
        return SOURCE_ORIENTATION
    return (row, col)


def _orient_matrix(matrix: np.ndarray, stored_as: tuple[str, str]) -> np.ndarray:
    if stored_as == DISPLAY_ORIENTATION:
        return matrix
    return matrix.T


def load_matrix(dataset: str, model_id: str, metric: str) -> np.ndarray | None:
    path = _build_path(dataset, model_id, metric)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"WARN: invalid JSON {path}: {exc}")
        return None
    if "matrix" not in payload:
        return None
    raw = np.asarray(payload["matrix"], dtype=float)
    return _orient_matrix(raw, _read_orientation(payload.get("metadata")))


def average_over_datasets(model_id: str, metric: str) -> np.ndarray | None:
    matrices, missing = [], []
    for dataset in DATASETS_TO_AVERAGE:
        m = load_matrix(dataset, model_id, metric)
        if m is None:
            missing.append(dataset)
        else:
            matrices.append(m)
    if missing:
        print(f"WARN: {model_id}/{metric}: {len(missing)}/{len(DATASETS_TO_AVERAGE)} missing")
    if not matrices:
        return None
    return np.mean(np.stack(matrices, axis=0), axis=0)


def load_model_data(model_id: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    return (
        average_over_datasets(model_id, "information_imbalance_reversed"),
        average_over_datasets(model_id, "information_imbalance"),
    )


def compute_pooled_thresholds(
    pre_data: tuple[np.ndarray | None, np.ndarray | None],
    fine_data: tuple[np.ndarray | None, np.ndarray | None],
    percentiles: tuple[int, int, int] = PERCENTILES,
) -> list[tuple[float, float, float]]:
    p1, p2, p3 = percentiles
    n_layers = 32
    for arr in (*pre_data, *fine_data):
        if arr is not None:
            n_layers = arr.shape[0]
            break
    pts_per_cell = min(CELL_W * 72 / n_layers, CELL_H * 72 / n_layers)
    s_max = pts_per_cell ** 2

    pool = np.concatenate([arr.flatten() for arr in (*pre_data, *fine_data) if arr is not None])
    q1 = float(np.percentile(pool, p1))
    q2 = float(np.percentile(pool, p2))
    q3 = float(np.percentile(pool, p3))
    return [(0.0, q1, s_max), (q1, q2, s_max * 0.25), (q2, q3, s_max * 0.05)]


def render_scatter_cell(
    ax: plt.Axes,
    img_to_text: np.ndarray | None,
    text_to_img: np.ndarray | None,
    thresholds: list[tuple[float, float, float]],
    abstraction_band: tuple[float, float] | None,
) -> None:
    n = 32
    for arr in (img_to_text, text_to_img):
        if arr is not None:
            n = arr.shape[0]
            break
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1, 3))
    ax.set_yticks(range(1, n + 1, 3))
    ax.grid(True, linewidth=0.4, color="gray", alpha=0.5)

    if abstraction_band is not None:
        ax.axvspan(*abstraction_band, color="goldenrod", alpha=0.18, zorder=0)
        ax.axhspan(*abstraction_band, color="goldenrod", alpha=0.18, zorder=0)

    vis_idx, txt_idx = np.indices((n, n))
    x = (txt_idx + 1).astype(float)
    y = (vis_idx + 1).astype(float)
    for lo, hi, area in thresholds:
        s = area * MARKER_SCALE
        if text_to_img is not None:
            mask = (text_to_img >= lo) & (text_to_img < hi)
            if mask.any():
                ax.scatter(x[mask], y[mask], s=s, zorder=3, **STYLE_T2I)
        if img_to_text is not None:
            mask = (img_to_text >= lo) & (img_to_text < hi)
            if mask.any():
                ax.scatter(x[mask], y[mask], s=s, zorder=4, **STYLE_I2T)


def make_direction_handles() -> list[mlines.Line2D]:
    kw = dict(linestyle="None", markerfacecolor="none")
    return [
        mlines.Line2D([], [], color="#228833", marker="s", markersize=8,
                      label=r"text $\rightarrow$ image", **kw),
        mlines.Line2D([], [], color="#7b2d8b", marker="o", markersize=8,
                      label=r"image $\rightarrow$ text", **kw),
    ]


def make_size_handles(thresholds: list[tuple[float, float, float]]) -> list[mlines.Line2D]:
    kw = dict(linestyle="None", markerfacecolor="none", color="black", marker="o")
    handles = []
    for lo, hi, area in thresholds:
        ms = float(np.sqrt(area * MARKER_SCALE))
        label = f"imb. < {hi:.3f}" if lo == 0.0 else f"{lo:.3f} $\\leq$ imb. < {hi:.3f}"
        handles.append(mlines.Line2D([], [], markersize=ms, label=label, **kw))
    return handles


def plot_pair(
    fine_id: str,
    fine_label: str,
    pre_id: str,
    pre_label: str,
    abstraction_band: tuple[float, float] | None,
    percentiles: tuple[int, int, int] | None,
) -> None:
    fine_data = load_model_data(fine_id)
    pre_data = load_model_data(pre_id)
    if all(d is None for d in fine_data) or all(d is None for d in pre_data):
        print(f"WARN: skipping {fine_id}: missing data")
        return

    thresholds = compute_pooled_thresholds(
        pre_data, fine_data, percentiles or PERCENTILES,
    )

    fig_w = 2 * CELL_W + WSPACE * CELL_W + LEFT_M + RIGHT_M
    fig_h = CELL_H + TOP_M + BOTTOM_M

    rc = get_rc(ncols=2, **{"figure.figsize": (fig_w, fig_h),
                             "figure.constrained_layout.use": False})
    with plt.rc_context(rc):
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax_pre = fig.add_subplot(gs[0])
        ax_fine = fig.add_subplot(gs[1])

        render_scatter_cell(ax_pre, *pre_data, thresholds, abstraction_band)
        render_scatter_cell(ax_fine, *fine_data, thresholds, abstraction_band)

        ax_pre.set_title(pre_label)
        ax_fine.set_title(fine_label)
        ax_pre.set_xlabel("text layer")
        ax_pre.set_ylabel("visual layer")
        ax_fine.set_xlabel("text layer")

        fig.legend(
            handles=make_size_handles(thresholds),
            loc="lower center", bbox_to_anchor=LEGEND_SIZE_ANCHOR, ncol=1,
            framealpha=0.95, edgecolor="#bbb", handlelength=1.2, borderpad=0.5, handletextpad=0.5,
        )
        fig.legend(
            handles=make_direction_handles(),
            loc="lower center", bbox_to_anchor=LEGEND_DIR_ANCHOR, ncol=1,
            framealpha=0.95, edgecolor="#bbb", handlelength=1.2, borderpad=0.5, handletextpad=0.5,
        )

        fig.subplots_adjust(
            left=LEFT_M / fig_w, right=1 - RIGHT_M / fig_w,
            top=1 - TOP_M / fig_h, bottom=BOTTOM_M / fig_h, wspace=WSPACE,
        )

        out_dir = FIGURES_DIR / f"{_slugify(pre_id)}_vs_{_slugify(fine_id)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "information_imbalance.pdf"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"Saved {out}")


def main() -> None:
    for entry in MODEL_PAIRS:
        fine_id, fine_label, pre_id, pre_label, band, *rest = entry
        percentiles = rest[0] if rest else None
        plot_pair(fine_id, fine_label, pre_id, pre_label, band, percentiles)


if __name__ == "__main__":
    main()
