"""Information imbalance — publication figures for all pretrained/finetuned pairs.

Mirrors heatmap_similarities_plot.py in structure: one figure per model pair,
all pairs in MODEL_PAIRS, all datasets in DATASETS_TO_AVERAGE.

Each figure is 1×2 (pretrained | finetuned), using SHARED thresholds computed
by pooling both models' averaged data. See Section 3 for the mathematical details.

Output:
    plots/imbalance_grids/last/paper/{model_pair_slug}/information_imbalance.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open


# ==============================================================================
# SECTION 1 — CONFIGURATION
# ==============================================================================
# Edit this section only. Everything else adapts automatically.

PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
RESULTS_DIR  = PROJECT_ROOT / "results" / "multimodality_comparison"
PLOTS_DIR    = PROJECT_ROOT / "plots" / "imbalance_grids" / "last" / "paper"

POOLING_METHOD = "last"
VARIANT_TAG    = "unique_images"
DPI            = 200

# ── Threshold percentiles ─────────────────────────────────────────────────────
# P1, P2, P3 define three marker-size buckets. See Section 3 for full explanation.
PERCENTILES: tuple[int, int, int] = (17, 34, 50)

# ── Model pairs: comment/uncomment to include/exclude ────────────────────────
# Format per entry:
#   (finetuned_id, finetuned_label, pretrained_id, pretrained_label, abstraction_band)
#
# abstraction_band = (start_layer, end_layer) for the goldenrod zone.
#   - This marks the embedding/attention → MLP transition in the LLM backbone.
#   - Set to None to omit the band for models where the zone is unknown.
#   - The zone depends on the backbone's architecture; adjust per model.
#
# Layer counts for reference:
#   LLaMA-2 7B  → 32 layers    LLaMA-2 13B → 40 layers
#   LLaMA-3 8B  → 32 layers    Qwen2 0.5B  → 24 layers  (OneVision 4B)
#   InternLM2 7B → 32 layers
MODEL_PAIRS: tuple[tuple[str, str, str, str, tuple[float, float] | None], ...] = (
    (
        "llava_7b",
        "finetuned (LLaVA 1.5 7B)",
        "llava-v1.5-7b-pretrained-lbasile",
        "pretrained (LLaVA 1.5 7B)",
        (5.5, 17.5),   # LLaMA-2 7B: 32 layers
    ),
    (
        "llava_13b",
        "finetuned (LLaVA 1.5 13B)",
        "llava-v1.5-13b-pretrained-lbasile",
        "pretrained (LLaVA 1.5 13B)",
        (6.5, 21.5),   # LLaMA-2 13B: 40 layers — adjust if wrong
    ),
    (
        "LLaVA-OneVision-1.5-4B-Instruct",
        "finetuned (LLaVA-OV 4B)",
        "LLaVA-OneVision-1.5-4B-Base",
        "pretrained (LLaVA-OV 4B)",
        (4.5, 13.5),   # Qwen2 0.5B: 24 layers — adjust if wrong
    ),
    (
        "LLaVA-OneVision-1.5-8B-Instruct",
        "finetuned (LLaVA-OV 8B)",
        "LLaVA-OneVision-1.5-8B-Base",
        "pretrained (LLaVA-OV 8B)",
        (5.5, 17.5),   # LLaMA-3 8B: 32 layers
    ),
    (
        "InternVL2-8B",
        "finetuned (InternVL2 8B)",
        "InternVL2-8B-Pretrain",
        "pretrained (InternVL2 8B)",
        (5.5, 17.5),   # InternLM2 7B: 32 layers — adjust if wrong
    ),
    (
        "cambrian-8b",
        "finetuned (Cambrian 8B)",
        "cambrian-8b-pretrained",
        "pretrained (Cambrian 8B)",
        (5.5, 17.5),   # LLaMA-3 8B: 32 layers
    ),
)

# ── Datasets: comment/uncomment to include/exclude ───────────────────────────
# These are the same 5 datasets used in heatmap_similarities_plot.py.
DATASETS_TO_AVERAGE: tuple[str, ...] = (
    "cocoqa_restval_icqa_split01_human",
    "gqa_val_balanced_noyesno_captioned_synthetic",
    "vqav2_coco_restval_icq_split01_human",
    "mmbench_en_dev_captioned_open_ended_synthetic",
    "seed_bench_images_captioned_open_ended_synthetic",
)

# ── Figure layout (all in inches, except WSPACE which is a fraction) ──────────
CELL_W: float  = 3.5    # width of one subplot panel
CELL_H: float  = 3.5    # height of one subplot panel
WSPACE: float  = 0.12   # horizontal gap between panels (fraction of CELL_W)
LEFT_M: float  = 0.90   # left margin for y-axis label
RIGHT_M: float = 0.25   # right padding
TOP_M: float   = 0.50   # top margin for column titles (no suptitle)
BOTTOM_M: float = 0.85  # bottom margin for legends

# ── Font sizes ────────────────────────────────────────────────────────────────
FONT_COL_TITLE: int  = 9   # column header
FONT_AXIS_LABEL: int = 8   # "text layer" / "visual layer"
FONT_TICKS: int      = 7   # tick numbers
FONT_LEGEND: int     = 8   # legend entries

# ── Scatter visual constants ──────────────────────────────────────────────────
STYLE_TEXT_TO_IMAGE = dict(marker="s", edgecolors="green",  facecolors="none", linewidths=0.85)
STYLE_IMAGE_TO_TEXT = dict(marker="o", edgecolors="purple", facecolors="none", linewidths=0.85)
MARKER_SCALE: float = 0.63

# ── Legend anchor positions (bbox_to_anchor in figure fractions) ─────────────
# loc="lower center" means the legend's lower-center edge sits at this (x, y).
# Increase y to move legends up (closer to the plot). Adjust x to space them out.
LEGEND_SIZE_ANCHOR: tuple[float, float]  = (0.4, -0.03)  # size-bucket legend
LEGEND_DIR_ANCHOR:  tuple[float, float]  = (0.6, 0.0)  # direction legend

# ── Matrix orientation ────────────────────────────────────────────────────────
SOURCE_ORIENTATION  = ("text", "image")   # what the file stores (row, col)
DISPLAY_ORIENTATION = ("image", "text")   # what we need (row=image, col=text)


# ==============================================================================
# SECTION 2 — DATA LOADING
# ==============================================================================


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return re.sub(r"_+", "_", slug)


def _build_path(dataset: str, model_id: str, metric: str) -> Path:
    return (
        RESULTS_DIR
        / dataset
        / POOLING_METHOD
        / VARIANT_TAG
        / "similarities"
        / f"{model_id}_{metric}.safetensors"
    )


def _read_orientation(metadata: dict[str, str] | None) -> tuple[str, str]:
    if not metadata:
        return SOURCE_ORIENTATION
    row = str(metadata.get("row_modality", "")).strip().lower()
    col = str(metadata.get("col_modality", "")).strip().lower()
    if row not in {"text", "image"} or col not in {"text", "image"} or row == col:
        return SOURCE_ORIENTATION
    return (row, col)  # type: ignore[return-value]


def _orient_matrix(matrix: np.ndarray, stored_as: tuple[str, str]) -> np.ndarray:
    """Transpose so rows=image layers, cols=text layers (for y=visual, x=text display)."""
    if stored_as == DISPLAY_ORIENTATION:
        return matrix
    return matrix.T


def load_matrix(dataset: str, model_id: str, metric: str) -> np.ndarray | None:
    path = _build_path(dataset, model_id, metric)
    if not path.exists():
        return None
    try:
        with safe_open(path, framework="np", device="cpu") as f:
            if metric not in f.keys():
                return None
            raw       = f.get_tensor(metric).copy()
            stored_as = _read_orientation(f.metadata())
    except Exception as exc:
        print(f"  ⚠ failed to load {path.name}: {exc}")
        return None
    return _orient_matrix(raw, stored_as)


def average_over_datasets(model_id: str, metric: str) -> np.ndarray | None:
    """Average a metric matrix across DATASETS_TO_AVERAGE; warn on missing ones."""
    matrices: list[np.ndarray] = []
    missing:  list[str]        = []
    for dataset in DATASETS_TO_AVERAGE:
        m = load_matrix(dataset, model_id, metric)
        if m is None:
            missing.append(dataset)
        else:
            matrices.append(m)
    if missing:
        print(f"  ⚠ {model_id}/{metric}: {len(missing)}/{len(DATASETS_TO_AVERAGE)} missing")
    if not matrices:
        return None
    return np.mean(np.stack(matrices, axis=0), axis=0)


def load_model_data(model_id: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load both imbalance directions averaged over datasets.

    Returns (img_to_text, text_to_img) — raw values, no 1-x transform.
    """
    img_to_text = average_over_datasets(model_id, "information_imbalance_reversed")
    text_to_img = average_over_datasets(model_id, "information_imbalance")
    return img_to_text, text_to_img


# ==============================================================================
# SECTION 3 — SHARED THRESHOLD COMPUTATION
# ==============================================================================
#
# Goal: determine marker sizes that are comparable across pretrained and
# finetuned, i.e. a given marker size means the same absolute imbalance value
# in both panels.
#
# This is the EXACT SCATTER ANALOGUE of the heatmap colorbar approach:
#
#   heatmap_similarities_plot.py does:
#       vmin = min(pre.min(), fine.min())   ←  0th percentile of (pre ∪ fine)
#       vmax = max(pre.max(), fine.max())   ← 100th percentile of (pre ∪ fine)
#
#   Here we do the same for interior percentiles:
#       Q₁  = percentile(pre ∪ fine, P₁)   e.g. 20th
#       Q₂  = percentile(pre ∪ fine, P₂)   e.g. 35th
#       Q₃  = percentile(pre ∪ fine, P₃)   e.g. 50th
#
# Pool construction
# ─────────────────
# For each model we have two matrices (text→image and image→text), each 32×32.
# We pool all four:
#
#   pool = flatten(A_pre)   # text→img pretrained
#        + flatten(B_pre)   # img→text pretrained
#        + flatten(A_fine)  # text→img finetuned
#        + flatten(B_fine)  # img→text finetuned
#   size = 4 × 1024 = 4096 values
#
# Pooling both directions gives ONE unified scale for green squares and purple
# circles (same marker size = same imbalance magnitude for both symbols).
#
# Visual interpretation
# ─────────────────────
# Q₁ = "the imbalance value below which P₁% of ALL (both models, both
#        directions) layer pairs fall."
#
# If finetuning collapsed imbalance from [0.3, 0.9] to [0.03, 0.09], then Q₁
# (P20 of the pool) ≈ 0.04 — inside the finetuned range. The pretrained panel
# will be SPARSE (most values exceed Q₃ and are hidden). The finetuned panel
# will be DENSE (most values below Q₁ → large markers). The figure directly
# shows the magnitude of the alignment gain from finetuning.
#
# Marker-size buckets
# ───────────────────
#   [0,  Q₁) → LARGE  markers  (s_max        )
#   [Q₁, Q₂) → MEDIUM markers  (s_max × 0.25 )
#   [Q₂, Q₃) → SMALL  markers  (s_max × 0.05 )
#   [Q₃, ∞ ) → NOT SHOWN
#
# s_max = area that fills one grid cell:  min(CELL_W·72/n, CELL_H·72/n)²  (pt²)


def compute_pooled_thresholds(
    pre_data:  tuple[np.ndarray | None, np.ndarray | None],
    fine_data: tuple[np.ndarray | None, np.ndarray | None],
    percentiles: tuple[int, int, int] = PERCENTILES,
) -> list[tuple[float, float, float]]:
    """
    Compute shared marker-size thresholds by pooling pre and fine data.
    See the long comment block above Section 3 for the full mathematical rationale.

    Returns a list of three (lower_bound, upper_bound, marker_area) tuples.
    marker_area is in matplotlib points² (the s= parameter of ax.scatter).
    """
    p1, p2, p3 = percentiles

    # s_max: marker area that fills exactly one grid cell (no overlap at max density)
    n_layers = 32   # assumed; detected from data if available
    for arr in (*pre_data, *fine_data):
        if arr is not None:
            n_layers = arr.shape[0]
            break
    pts_per_cell = min(CELL_W * 72 / n_layers, CELL_H * 72 / n_layers)
    s_max = pts_per_cell ** 2

    # Pool all four flattened arrays (2 models × 2 directions)
    pool = np.concatenate([
        arr.flatten()
        for arr in (*pre_data, *fine_data)
        if arr is not None
    ])

    q1 = float(np.percentile(pool, p1))
    q2 = float(np.percentile(pool, p2))
    q3 = float(np.percentile(pool, p3))

    return [
        (0.0,  q1,  s_max),          # [0,  Q₁) → LARGE  markers
        (q1,   q2,  s_max * 0.25),   # [Q₁, Q₂) → MEDIUM markers
        (q2,   q3,  s_max * 0.05),   # [Q₂, Q₃) → SMALL  markers
        # values ≥ Q₃ → NOT shown
    ]


# ==============================================================================
# SECTION 4 — CELL RENDERING
# ==============================================================================


def render_scatter_cell(
    ax: plt.Axes,
    img_to_text: np.ndarray | None,
    text_to_img: np.ndarray | None,
    thresholds: list[tuple[float, float, float]],
    abstraction_band: tuple[float, float] | None,
) -> None:
    """
    Draw one imbalance scatter plot into an Axes.

    Arguments:
        ax               : the subplot to draw into
        img_to_text      : 32×32 image→text imbalance (rows=image, cols=text)
        text_to_img      : 32×32 text→image imbalance
        thresholds       : from compute_pooled_thresholds()
        abstraction_band : (start, end) layer range for goldenrod band, or None
    """
    n = 32
    for arr in (img_to_text, text_to_img):
        if arr is not None:
            n = arr.shape[0]
            break

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1, 3))
    ax.set_yticks(range(1, n + 1, 3))
    ax.tick_params(labelsize=FONT_TICKS)
    ax.grid(True, linewidth=0.4, color="gray", alpha=0.5)

    if abstraction_band is not None:
        ax.axvspan(*abstraction_band, color="goldenrod", alpha=0.18, zorder=0)
        ax.axhspan(*abstraction_band, color="goldenrod", alpha=0.18, zorder=0)

    vis_idx, txt_idx = np.indices((n, n))
    X = (txt_idx + 1).astype(float)   # text layer → x coordinate
    Y = (vis_idx + 1).astype(float)   # image layer → y coordinate

    for lo, hi, area in thresholds:
        s = area * MARKER_SCALE
        if text_to_img is not None:
            mask = (text_to_img >= lo) & (text_to_img < hi)
            if mask.any():
                ax.scatter(X[mask], Y[mask], s=s, zorder=3, **STYLE_TEXT_TO_IMAGE)
        if img_to_text is not None:
            mask = (img_to_text >= lo) & (img_to_text < hi)
            if mask.any():
                ax.scatter(X[mask], Y[mask], s=s, zorder=4, **STYLE_IMAGE_TO_TEXT)


def render_missing_cell(ax: plt.Axes) -> None:
    ax.set_facecolor("#ececec")
    ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="#999", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")


# ==============================================================================
# SECTION 5 — LEGEND HANDLES
# ==============================================================================


def make_direction_handles() -> list[mlines.Line2D]:
    kw = dict(linestyle="None", markerfacecolor="none")
    return [
        mlines.Line2D([], [], color="green",  marker="s", markersize=8,
                      label="text → image", **kw),
        mlines.Line2D([], [], color="purple", marker="o", markersize=8,
                      label="image → text", **kw),
    ]


def make_size_handles(thresholds: list[tuple[float, float, float]]) -> list[mlines.Line2D]:
    kw = dict(linestyle="None", markerfacecolor="none", color="black", marker="o")
    handles = []
    for lo, hi, area in thresholds:
        ms = np.sqrt(area * MARKER_SCALE)
        label = f"imb. < {hi:.3f}" if lo == 0.0 else f"{lo:.3f} ≤ imb. < {hi:.3f}"
        handles.append(mlines.Line2D([], [], markersize=ms, label=label, **kw))
    return handles


# ==============================================================================
# SECTION 6 — FIGURE BUILDING
# ==============================================================================


def _compute_geometry() -> dict:
    """Convert inch margins to figure fractions and compute column centers."""
    n_cols       = 2
    wspace_in    = CELL_W * WSPACE
    fig_w        = LEFT_M + n_cols * CELL_W + (n_cols - 1) * wspace_in + RIGHT_M
    fig_h        = TOP_M + CELL_H + BOTTOM_M

    left_frac    = LEFT_M  / fig_w
    right_frac   = 1.0 - RIGHT_M / fig_w
    top_frac     = 1.0 - TOP_M   / fig_h
    bottom_frac  = BOTTOM_M / fig_h

    # x-center of each column in figure fraction (for legend anchoring)
    total_plot   = right_frac - left_frac
    subplot_w    = total_plot / (n_cols + (n_cols - 1) * WSPACE)
    col_centers  = [left_frac + (i + 0.5 + i * WSPACE) * subplot_w for i in range(n_cols)]

    return {
        "fig_w": fig_w, "fig_h": fig_h,
        "left": left_frac, "right": right_frac,
        "top": top_frac,   "bottom": bottom_frac,
        "wspace": WSPACE,  "col_centers": col_centers,
    }


def build_paired_figure(
    pre_id:    str,
    fine_id:   str,
    pre_label: str,
    fine_label: str,
    pre_data:  tuple[np.ndarray | None, np.ndarray | None],
    fine_data: tuple[np.ndarray | None, np.ndarray | None],
    abstraction_band: tuple[float, float] | None,
) -> plt.Figure:
    """
    Build a 1×2 figure (pretrained | finetuned) with shared threshold scale.

    Thresholds are computed by pooling data from both models (see Section 3).
    """
    geo = _compute_geometry()
    thresholds = compute_pooled_thresholds(pre_data, fine_data)

    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(geo["fig_w"], geo["fig_h"]),
        squeeze=False,
    )
    fig.subplots_adjust(
        left   = geo["left"],
        right  = geo["right"],
        top    = geo["top"],
        bottom = geo["bottom"],
        wspace = geo["wspace"],
    )

    for col_idx, (label, data) in enumerate(
        [(pre_label, pre_data), (fine_label, fine_data)]
    ):
        ax = axes[0, col_idx]
        img_to_text, text_to_img = data

        if img_to_text is not None or text_to_img is not None:
            render_scatter_cell(ax, img_to_text, text_to_img, thresholds, abstraction_band)
        else:
            render_missing_cell(ax)

        ax.set_title(label, fontsize=FONT_COL_TITLE, fontweight="bold", pad=6)
        ax.set_xlabel("text layer", fontsize=FONT_AXIS_LABEL, labelpad=4)
        if col_idx == 0:
            ax.set_ylabel("visual layer", fontsize=FONT_AXIS_LABEL, labelpad=4)

    # Size legend (no title text)
    fig.legend(
        handles=make_size_handles(thresholds),
        loc="lower center",
        bbox_to_anchor=LEGEND_SIZE_ANCHOR,
        ncol=1,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#bbb",
        handlelength=1.2,
        borderpad=0.5,
        handletextpad=0.5,
    )

    # Direction legend (no title text)
    fig.legend(
        handles=make_direction_handles(),
        loc="lower center",
        bbox_to_anchor=LEGEND_DIR_ANCHOR,
        ncol=1,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#bbb",
        handlelength=1.2,
        borderpad=0.5,
        handletextpad=0.5,
    )

    return fig


# ==============================================================================
# SECTION 7 — MAIN
# ==============================================================================


def main() -> None:
    print("=" * 72)
    print("  plot_imbalance_paper.py")
    print(f"  datasets   : {len(DATASETS_TO_AVERAGE)}")
    print(f"  model pairs: {len(MODEL_PAIRS)}")
    print(f"  percentiles: P{PERCENTILES[0]} / P{PERCENTILES[1]} / P{PERCENTILES[2]}")
    print(f"  output     : {PLOTS_DIR}")
    print("=" * 72)

    written: list[Path] = []
    skipped: list[str]  = []

    for fine_id, fine_label, pre_id, pre_label, abstraction_band in MODEL_PAIRS:
        pair_slug = f"{_slugify(pre_id)}_vs_{_slugify(fine_id)}"
        print(f"\n{'─' * 64}")
        print(f"  {pre_id}  →  {fine_id}")

        print(f"  loading pretrained ({pre_id}) ...")
        pre_data = load_model_data(pre_id)
        print(f"    img→text: {'✓' if pre_data[0] is not None else '✗'}  "
              f"text→img: {'✓' if pre_data[1] is not None else '✗'}")

        print(f"  loading finetuned  ({fine_id}) ...")
        fine_data = load_model_data(fine_id)
        print(f"    img→text: {'✓' if fine_data[0] is not None else '✗'}  "
              f"text→img: {'✓' if fine_data[1] is not None else '✗'}")

        if all(v is None for v in (*pre_data, *fine_data)):
            print("  skipped: no data for either model")
            skipped.append(pair_slug)
            continue

        fig = build_paired_figure(
            pre_id=pre_id, fine_id=fine_id,
            pre_label=pre_label, fine_label=fine_label,
            pre_data=pre_data, fine_data=fine_data,
            abstraction_band=abstraction_band,
        )

        out_dir = PLOTS_DIR / pair_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        save_kwargs = dict(bbox_inches="tight", pad_inches=0.1, facecolor="white")
        out_png = out_dir / "information_imbalance.png"
        out_pdf = out_dir / "information_imbalance.pdf"
        fig.savefig(out_png, dpi=DPI, **save_kwargs)
        fig.savefig(out_pdf, **save_kwargs)
        plt.close(fig)

        written.extend([out_png, out_pdf])
        print(f"  ✓ {out_png}")
        print(f"  ✓ {out_pdf}")

    print("\n" + "=" * 72)
    print(f"  wrote {len(written)} figure(s)")
    if skipped:
        print(f"  skipped: {', '.join(skipped)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
