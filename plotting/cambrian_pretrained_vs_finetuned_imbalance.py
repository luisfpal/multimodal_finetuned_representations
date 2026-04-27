"""
Cambrian 8B — Pretrained vs Finetuned Information Imbalance
============================================================

Produces two side-by-side figures (pretrained | finetuned) showing the
information imbalance between visual and text residual streams:

  paired_thresholds.png
      Both models share one marker-size scale, computed from their combined
      averaged data. Use this to compare the two models on the same visual scale.

  disjoint_thresholds.png
      Each model has its own marker-size scale, computed from its own data only.
      Use this to see the internal structure of each model without the other
      model pulling the scale.

What is "information imbalance"?
  Given two sets of representations A and B, the information imbalance Δ(A→B)
  measures how much knowing A helps predict B (using k-nearest-neighbours).
  Low values mean A is a good predictor of B.

  We track two directions:
    text → image  (green squares):   knowing text layer helps predict visual layer
    image → text  (purple circles):  knowing visual layer helps predict text layer

  Marker SIZE encodes magnitude — larger marker = stronger asymmetry.
  Only the top fraction of (image_layer, text_layer) pairs is shown (above the
  20th percentile cutoff), to avoid visual clutter.

Data:
  Matrices are averaged over 5 datasets before plotting.
  Thresholds are computed on the averaged data (not per-dataset).

Output:
  plots/imbalance_grids/last/cambrian_pretrained_vs_finetuned/
      paired_thresholds.png
      disjoint_thresholds.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # Non-interactive backend: renders to file, no display needed

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open


# ==============================================================================
# SECTION 1 — CONFIGURATION
# ==============================================================================
# All "magic numbers" in one place. Change here, everything updates.

PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
RESULTS_DIR  = PROJECT_ROOT / "results" / "multimodality_comparison"
PLOTS_DIR    = PROJECT_ROOT / "plots" / "imbalance_grids" / "last" / "cambrian_pretrained_vs_finetuned"

POOLING_METHOD = "last"
VARIANT_TAG    = "unique_images"
DPI            = 200

# ── Models ────────────────────────────────────────────────────────────────────
PRETRAINED_ID = "cambrian-8b-pretrained"
FINETUNED_ID  = "cambrian-8b"

# Labels used in column titles and legends
MODEL_LABELS: dict[str, str] = {
    PRETRAINED_ID: "pretrained\n(Cambrian 8B)",
    FINETUNED_ID:  "finetuned\n(Cambrian 8B)",
}

# ── Datasets ──────────────────────────────────────────────────────────────────
# The default 4-encoder Cambrian model was run with these specific dataset splits.
DATASETS: tuple[str, ...] = (
    "cocoqa_restval_icqa_split01_human",
    "gqa_val_balanced_noyesno_captioned_synthetic",
    "vqav2_coco_restval_icq_split01_human",
    "mmbench_en_dev_captioned_open_ended_synthetic",
    "seed_bench_images_captioned_open_ended_synthetic",
)

# ── Threshold percentiles ─────────────────────────────────────────────────────
PERCENTILES: tuple[int, int, int] = (20, 35, 50)

# ── Figure layout (all in inches, except WSPACE which is a fraction) ──────────
CELL_W: float   = 3.5    # width of one subplot panel
CELL_H: float   = 3.2    # height of one subplot panel
WSPACE: float   = 0.12   # horizontal gap between panels (fraction of CELL_W)
LEFT_M: float   = 0.90   # left margin for y-axis label
RIGHT_M: float  = 0.25   # right padding
TOP_M_PAIRED: float   = 0.50   # top margin — no suptitle, only column titles
TOP_M_DISJOINT: float = 0.50
BOTTOM_M_PAIRED: float   = 1.35  # bottom margin for one row of legends
BOTTOM_M_DISJOINT: float = 1.75  # bottom margin for two rows of legends

# ── Font sizes ────────────────────────────────────────────────────────────────
FONT_COL_TITLE: int  = 9   # column header ("pretrained\n(Cambrian 8B)")
FONT_AXIS_LABEL: int = 8   # "text layer" / "visual layer"
FONT_TICKS: int      = 7   # tick numbers
FONT_LEGEND: int     = 8   # legend entries

# ── Scatter visual constants ──────────────────────────────────────────────────
# Green squares  = text → image direction  (information_imbalance)
# Purple circles = image → text direction  (information_imbalance_reversed)
STYLE_TEXT_TO_IMAGE = dict(marker="s", edgecolors="green",  facecolors="none", linewidths=0.85)
STYLE_IMAGE_TO_TEXT = dict(marker="o", edgecolors="purple", facecolors="none", linewidths=0.85)

# Goldenrod band marks the embedding/attention → MLP transition zone.
# Adjust this per model (Cambrian 8B has 32 layers, zone at layers 5.5–17.5).
ABSTRACTION_BAND: tuple[float, float] = (5.5, 17.5)

# Scale-down factor to prevent markers from fully covering adjacent grid points
MARKER_SCALE: float = 0.63

# ── Matrix orientation ────────────────────────────────────────────────────────
# Safetensors files store matrices as (text_layers × image_layers): row=text, col=image.
# We want (image_layers × text_layers) for display: y-axis=visual, x-axis=text.
# → We check the metadata and transpose when needed.
SOURCE_ORIENTATION  = ("text", "image")   # what the file stores
DISPLAY_ORIENTATION = ("image", "text")   # what we need on screen


# ==============================================================================
# SECTION 2 — DATA LOADING
# ==============================================================================
# These functions load raw .safetensors files and return oriented numpy arrays.


def _build_path(dataset: str, model_id: str, metric: str) -> Path:
    """Return the path to one safetensors file."""
    return (
        RESULTS_DIR
        / dataset
        / POOLING_METHOD
        / VARIANT_TAG
        / "similarities"
        / f"{model_id}_{metric}.safetensors"
    )


def _read_orientation(metadata: dict[str, str] | None) -> tuple[str, str]:
    """
    Read the (row_modality, col_modality) orientation stored in file metadata.
    Falls back to SOURCE_ORIENTATION when metadata is absent or malformed.
    """
    if not metadata:
        return SOURCE_ORIENTATION

    row = str(metadata.get("row_modality", "")).strip().lower()
    col = str(metadata.get("col_modality", "")).strip().lower()

    # Sanity check: both must be valid and distinct
    if row not in {"text", "image"} or col not in {"text", "image"} or row == col:
        return SOURCE_ORIENTATION

    return (row, col)  # type: ignore[return-value]


def _orient_matrix(matrix: np.ndarray, stored_as: tuple[str, str]) -> np.ndarray:
    """
    Transpose the matrix if needed so that rows=image layers, cols=text layers.

    Why transpose?
      The y-axis of the scatter plot represents visual layers (rows of the matrix).
      The x-axis represents text layers (columns of the matrix).
      Files store it the other way, so we flip with .T (transpose).
    """
    if stored_as == DISPLAY_ORIENTATION:
        return matrix        # already (image, text) — no flip needed
    return matrix.T          # flip (text, image) → (image, text)


def load_matrix(dataset: str, model_id: str, metric: str) -> np.ndarray | None:
    """
    Load one 32×32 similarity matrix from disk, returning it display-oriented.
    Returns None if the file doesn't exist or is unreadable.
    """
    path = _build_path(dataset, model_id, metric)
    if not path.exists():
        return None

    try:
        with safe_open(path, framework="np", device="cpu") as f:
            if metric not in f.keys():
                return None
            raw          = f.get_tensor(metric).copy()
            stored_as    = _read_orientation(f.metadata())
    except Exception as exc:
        print(f"  ⚠ failed to load {path.name}: {exc}")
        return None

    return _orient_matrix(raw, stored_as)


def average_over_datasets(model_id: str, metric: str) -> np.ndarray | None:
    """
    Load and element-wise average a metric matrix across all DATASETS.

    Partial averaging: if some datasets are missing, we average over what we have
    and warn. Returns None only if every dataset fails.

    The average is element-wise: each (image_layer, text_layer) cell in the
    output is the mean of that cell across all successfully loaded datasets.
    """
    matrices: list[np.ndarray] = []
    missing:  list[str]        = []

    for dataset in DATASETS:
        m = load_matrix(dataset, model_id, metric)
        if m is None:
            missing.append(dataset)
        else:
            matrices.append(m)

    if missing:
        print(f"  ⚠ {model_id}/{metric}: missing {len(missing)}/{len(DATASETS)} datasets")
    if not matrices:
        return None

    # np.stack → shape (N_datasets, 32, 32); np.mean over axis=0 collapses datasets
    return np.mean(np.stack(matrices, axis=0), axis=0)


def load_model_data(model_id: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load both imbalance directions for one model, averaged over all datasets.

    Returns:
        (img_to_text, text_to_img)
        img_to_text : information_imbalance_reversed  (image → text direction)
        text_to_img : information_imbalance            (text → image direction)

    Note: raw imbalance values are loaded as-is (no 1-x transform).
    The scatter plot encodes the raw magnitude directly.
    """
    img_to_text = average_over_datasets(model_id, "information_imbalance_reversed")
    text_to_img = average_over_datasets(model_id, "information_imbalance")
    return img_to_text, text_to_img


# ==============================================================================
# SECTION 3 — THRESHOLD COMPUTATION
# ==============================================================================
# Thresholds decide which (image_layer, text_layer) pairs are shown and at what size.
#
# The logic (mirrors cambrian_encoder_ablation_plot.py exactly):
#   1. Pool all imbalance values from the data we want to threshold.
#   2. Compute three percentile cutoffs (Q1, Q2, Q3) from PERCENTILES.
#   3. Pairs with imbalance in [0,  Q1) get LARGE  markers — most pairs land here.
#   4. Pairs with imbalance in [Q1, Q2) get MEDIUM markers.
#   5. Pairs with imbalance in [Q2, Q3) get SMALL  markers.
#   6. Pairs with imbalance >= Q3 are NOT shown — the top (100-P3)% are hidden.
#
# Why low imbalance → large markers?
#   Low imbalance = the two layers are SIMILAR (knowing A helps predict B well).
#   Large markers in this dense low-imbalance region highlight where cross-modal
#   alignment is strong. The rare extreme values (very high imbalance) are
#   suppressed to avoid visual clutter.


def compute_marker_thresholds(
    data_arrays: list[np.ndarray],
    percentiles: tuple[int, int, int] = PERCENTILES,
) -> list[tuple[float, float, float]]:
    """
    Compute three (lower_bound, upper_bound, marker_area) buckets.

    Arguments:
        data_arrays : list of 2-D imbalance arrays to pool for threshold computation.
                      For paired mode: both models' arrays.
                      For disjoint mode: one model's arrays only.
        percentiles : (P1, P2, P3) — must be strictly increasing, in [0, 100].

    Returns:
        A list of three tuples: (lo, hi, area)
          lo, hi    : imbalance range for this bucket
          area      : marker area in matplotlib points² (s= parameter in ax.scatter)

    Marker area calculation:
        s_max is the area where a marker just fills one grid cell without
        overlapping its neighbours. We compute it from the physical cell size:
          pts_per_layer = (cell_inches × 72 pts/inch) / 32 layers
          s_max = min(x_pts, y_pts)²
        The smaller two buckets are fractions of s_max.
    """
    p1, p2, p3 = percentiles

    # Physical size of one grid cell in points (72 points = 1 inch)
    pts_per_layer_x = (CELL_W * 72) / 32
    pts_per_layer_y = (CELL_H * 72) / 32
    s_max = min(pts_per_layer_x, pts_per_layer_y) ** 2   # area in points²

    # Pool all values and compute quantile cutoffs
    all_values = np.concatenate([arr.flatten() for arr in data_arrays])
    q1 = float(np.percentile(all_values, p1))
    q2 = float(np.percentile(all_values, p2))
    q3 = float(np.percentile(all_values, p3))

    return [
        (0.0,  q1,  s_max),          # bucket 0: [0,  P1) → LARGE  markers
        (q1,   q2,  s_max * 0.25),   # bucket 1: [P1, P2) → MEDIUM markers
        (q2,   q3,  s_max * 0.05),   # bucket 2: [P2, P3) → SMALL  markers
        # values >= q3 → NOT shown
    ]


# ==============================================================================
# SECTION 4 — CELL RENDERING
# ==============================================================================
# Each function draws into a single matplotlib Axes object (one subplot cell).


def render_scatter_cell(
    ax: plt.Axes,
    img_to_text: np.ndarray | None,
    text_to_img: np.ndarray | None,
    thresholds: list[tuple[float, float, float]],
) -> None:
    """
    Draw one imbalance scatter plot into a matplotlib Axes.

    The grid has 32 × 32 points — one for each (image_layer, text_layer) pair.
    A marker is drawn at (x=text_layer, y=image_layer) only if its imbalance
    value falls within a threshold bucket.

    Arguments:
        ax            : the Axes object to draw into (created by plt.subplots)
        img_to_text   : 32×32 array, image→text direction (information_imbalance_reversed)
        text_to_img   : 32×32 array, text→image direction (information_imbalance)
        thresholds    : from compute_marker_thresholds() — list of (lo, hi, area)
    """
    n_layers = 32
    if img_to_text is not None:
        n_layers = img_to_text.shape[0]
    elif text_to_img is not None:
        n_layers = text_to_img.shape[0]

    # ── Axis setup ────────────────────────────────────────────────────────────
    # set_xlim / set_ylim control which range of values is visible.
    # We use 1-indexed layer numbers (layer 1 = first layer, layer 32 = last).
    # The +/- 0.5 margin ensures the outermost markers aren't clipped.
    ax.set_xlim(0.5, n_layers + 0.5)
    ax.set_ylim(0.5, n_layers + 0.5)

    # Sparse tick marks: one every 3 layers so labels don't crowd each other
    ax.set_xticks(range(1, n_layers + 1, 3))
    ax.set_yticks(range(1, n_layers + 1, 3))
    ax.tick_params(labelsize=FONT_TICKS)

    # Light grid makes it easy to trace specific layer pairs
    ax.grid(True, linewidth=0.4, color="gray", alpha=0.5)

    # ── Goldenrod abstraction band ─────────────────────────────────────────────
    # axvspan shades a vertical stripe across the full y range.
    # axhspan shades a horizontal stripe across the full x range.
    # zorder=0 puts the band behind all markers.
    ax.axvspan(*ABSTRACTION_BAND, color="goldenrod", alpha=0.18, zorder=0)
    ax.axhspan(*ABSTRACTION_BAND, color="goldenrod", alpha=0.18, zorder=0)

    # ── Build coordinate arrays ────────────────────────────────────────────────
    # np.indices((n, n)) returns two arrays of shape (n, n):
    #   vis_idx[i, j] = i  (row index  = visual layer)
    #   txt_idx[i, j] = j  (col index  = text layer)
    # We convert to 1-based layer numbers (+1).
    vis_idx, txt_idx = np.indices((n_layers, n_layers))
    X = (txt_idx + 1).astype(float)   # text layer  → x-axis coordinate
    Y = (vis_idx + 1).astype(float)   # image layer → y-axis coordinate

    # ── Draw markers for each threshold bucket ────────────────────────────────
    # We iterate from the largest bucket (most common) to the smallest.
    # zorder controls drawing order: higher zorder = drawn on top.
    for lo, hi, area in thresholds:
        scaled_area = area * MARKER_SCALE   # shrink slightly to prevent full overlap

        # text → image: green squares
        if text_to_img is not None:
            # Boolean mask: True where imbalance is in [lo, hi)
            mask = (text_to_img >= lo) & (text_to_img < hi)
            if mask.any():
                # X[mask] and Y[mask] select only the (x, y) coordinates where mask=True
                # s= sets the marker area in points². zorder=3 draws above the grid.
                ax.scatter(X[mask], Y[mask], s=scaled_area, zorder=3, **STYLE_TEXT_TO_IMAGE)

        # image → text: purple circles (drawn last = on top of green squares)
        if img_to_text is not None:
            mask = (img_to_text >= lo) & (img_to_text < hi)
            if mask.any():
                ax.scatter(X[mask], Y[mask], s=scaled_area, zorder=4, **STYLE_IMAGE_TO_TEXT)


def render_missing_cell(ax: plt.Axes) -> None:
    """Draw a grey placeholder when data for a model is unavailable."""
    ax.set_facecolor("#ececec")
    ax.text(0.5, 0.5, "N/A",
            transform=ax.transAxes,   # coordinates relative to the axes (0–1)
            ha="center", va="center", fontsize=10, color="#999", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")


# ==============================================================================
# SECTION 5 — LEGEND HANDLES
# ==============================================================================
# Legends in matplotlib work with "handles" — objects that describe what to draw
# in the legend box. We use mlines.Line2D with linestyle="None" to get a
# point-only marker (no connecting line), which matches our scatter style.


def make_direction_handles() -> list[mlines.Line2D]:
    """
    Build legend handles for the two imbalance directions.

    mlines.Line2D([], [], ...) creates a handle with an empty data series —
    it only exists to appear in the legend, not to draw anything on the plot.
    markerfacecolor="none" gives a hollow marker (matching our scatter style).
    """
    kw = dict(linestyle="None", markerfacecolor="none")
    return [
        mlines.Line2D([], [], color="green",  marker="s", markersize=8,
                      label="text → image", **kw),
        mlines.Line2D([], [], color="purple", marker="o", markersize=8,
                      label="image → text", **kw),
    ]


def make_size_handles(
    thresholds: list[tuple[float, float, float]],
    title_prefix: str = "",
) -> list[mlines.Line2D]:
    """
    Build legend handles for the three marker-size buckets.

    np.sqrt(area) converts from scatter's area parameter (points²) to
    markersize (points, i.e. diameter), which is what Line2D uses.
    MARKER_SCALE is applied here too so the legend markers match the plot.

    Arguments:
        thresholds   : from compute_marker_thresholds()
        title_prefix : optional string prepended to each label
    """
    kw = dict(linestyle="None", markerfacecolor="none", color="black", marker="o")
    lo0, hi0, area0 = thresholds[0]
    lo1, hi1, area1 = thresholds[1]
    lo2, hi2, area2 = thresholds[2]

    # Labels mirror the reference script convention:
    #   bucket 0: "imb. < Q1"            (lo=0, so we just say "< hi")
    #   bucket 1: "Q1 <= imb. < Q2"
    #   bucket 2: "Q2 <= imb. < Q3"
    def label(lo: float, hi: float) -> str:
        if lo == 0.0:
            return f"{title_prefix}imb. < {hi:.3f}"
        return f"{title_prefix}{lo:.3f} <= imb. < {hi:.3f}"

    return [
        mlines.Line2D([], [], markersize=np.sqrt(area0 * MARKER_SCALE),
                      label=label(lo0, hi0), **kw),
        mlines.Line2D([], [], markersize=np.sqrt(area1 * MARKER_SCALE),
                      label=label(lo1, hi1), **kw),
        mlines.Line2D([], [], markersize=np.sqrt(area2 * MARKER_SCALE),
                      label=label(lo2, hi2), **kw),
    ]


# ==============================================================================
# SECTION 6 — FIGURE GEOMETRY
# ==============================================================================
# Matplotlib's subplots_adjust() takes fractions of the figure size, not inches.
# We do all the inch math here and convert, so the rest of the code stays clean.


def _compute_geometry(mode: str) -> dict:
    """
    Compute all layout measurements for a 1×2 figure.

    Returns a dict of fractions ready to pass to subplots_adjust(), plus the
    figure size in inches and the x-center of each column (for legend placement).

    The bottom margin is larger for disjoint mode because we need room for
    two separate size legends (one per model) plus the direction legend.
    """
    left_margin   = LEFT_M
    right_margin  = RIGHT_M
    top_margin    = TOP_M_PAIRED   if mode == "paired" else TOP_M_DISJOINT
    bottom_margin = BOTTOM_M_PAIRED if mode == "paired" else BOTTOM_M_DISJOINT
    wspace        = WSPACE

    n_cols = 2
    wspace_inches = CELL_W * wspace
    fig_w = left_margin + n_cols * CELL_W + (n_cols - 1) * wspace_inches + right_margin
    fig_h = top_margin + CELL_H + bottom_margin

    # Convert margins to figure fractions (what subplots_adjust expects)
    left_frac   = left_margin  / fig_w
    right_frac  = 1.0 - right_margin / fig_w
    top_frac    = 1.0 - top_margin   / fig_h
    bottom_frac = bottom_margin / fig_h

    # Column centers in figure fractions (for legend bbox_to_anchor x coordinates).
    # subplot_w_frac = width of one subplot column in figure fraction.
    # With wspace=0.30 and n_cols=2: total = subplot_w*(2 + 1*wspace)
    total_plot_frac = right_frac - left_frac
    subplot_w_frac  = total_plot_frac / (n_cols + (n_cols - 1) * wspace)

    col_centers = [
        left_frac + (i + 0.5 + i * wspace) * subplot_w_frac
        for i in range(n_cols)
    ]

    return {
        "fig_w":       fig_w,
        "fig_h":       fig_h,
        "left":        left_frac,
        "right":       right_frac,
        "top":         top_frac,
        "bottom":      bottom_frac,
        "wspace":      wspace,
        "bottom_frac": bottom_frac,
        "col_centers": col_centers,   # x-centers of col 0 and col 1 in fig fraction
    }


# ==============================================================================
# SECTION 7 — FIGURE BUILDING
# ==============================================================================


def _add_legends_paired(fig: plt.Figure, thresholds: list[tuple[float, float, float]]) -> None:
    """
    Add legends for the paired figure (one shared size legend + direction legend).

    fig.legend() places a legend at the figure level (not per-axis).
    bbox_to_anchor=(x, y) sets the anchor point in figure fraction coordinates.
    loc="lower center" means the legend box's lower-center is placed at that point.
    ncol=N arranges legend entries in N columns.
    """
    # Size legend: left-of-center bottom
    fig.legend(
        handles=make_size_handles(thresholds),
        loc="lower center",
        bbox_to_anchor=(0.22, 0.01),
        ncol=1,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#bbb",
        handlelength=1.2,
        borderpad=0.5,
        handletextpad=0.5,
    )

    # Direction legend: right-of-center bottom
    fig.legend(
        handles=make_direction_handles(),
        loc="lower center",
        bbox_to_anchor=(0.72, 0.01),
        ncol=1,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#bbb",
        handlelength=1.2,
        borderpad=0.5,
        handletextpad=0.5,
    )


def _add_legends_disjoint(
    fig: plt.Figure,
    pre_thresholds: list[tuple[float, float, float]],
    fine_thresholds: list[tuple[float, float, float]],
    col_centers: list[float],
) -> None:
    """
    Add legends for the disjoint figure:
      - One size legend below each column (different thresholds per model)
      - One shared direction legend centered above the size legends

    We add each legend with fig.add_artist() when stacking multiple fig.legend()
    calls, because fig.legend() normally replaces the previous one. add_artist()
    keeps all of them.
    """
    # Pretrained size legend — below left column
    leg_pre = fig.legend(
        handles=make_size_handles(pre_thresholds),
        loc="lower center",
        bbox_to_anchor=(col_centers[0], 0.01),
        ncol=1,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#bbb",
        handlelength=1.2,
        borderpad=0.5,
        handletextpad=0.5,
    )
    fig.add_artist(leg_pre)

    # Finetuned size legend — below right column
    leg_fine = fig.legend(
        handles=make_size_handles(fine_thresholds),
        loc="lower center",
        bbox_to_anchor=(col_centers[1], 0.01),
        ncol=1,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#bbb",
        handlelength=1.2,
        borderpad=0.5,
        handletextpad=0.5,
    )
    fig.add_artist(leg_fine)

    # Direction legend — centered, slightly above the size legends
    fig.legend(
        handles=make_direction_handles(),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.20),
        ncol=2,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        edgecolor="#bbb",
        handlelength=1.2,
        borderpad=0.5,
        columnspacing=1.0,
        handletextpad=0.5,
    )


def build_figure(
    pretrained_data: tuple[np.ndarray | None, np.ndarray | None],
    finetuned_data:  tuple[np.ndarray | None, np.ndarray | None],
    mode: str,
    title: str,
) -> plt.Figure:
    """
    Build a 1×2 figure (pretrained | finetuned) with scatter cells and legends.

    Arguments:
        pretrained_data : (img_to_text, text_to_img) arrays for the pretrained model
        finetuned_data  : (img_to_text, text_to_img) arrays for the finetuned model
        mode            : "paired"   → both cells share one threshold scale
                          "disjoint" → each cell has its own threshold scale
        title           : figure suptitle

    Matplotlib layout concepts used here:
      plt.subplots(nrows, ncols) → creates the Figure and a 2-D array of Axes objects
      fig.subplots_adjust(...)   → controls whitespace (all values are 0–1 fractions)
      fig.suptitle(...)          → title above all subplots
      ax.set_title(...)          → per-subplot column header
    """
    geo = _compute_geometry(mode)

    # ── Create figure and axes ─────────────────────────────────────────────────
    # squeeze=False forces axes to always be a 2-D array shape (nrows, ncols),
    # even when nrows=1. This makes axes[0, c] always valid.
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(geo["fig_w"], geo["fig_h"]),
        squeeze=False,
    )

    # subplots_adjust controls whitespace around and between the subplots.
    # left/right/top/bottom are margins from the figure edges (in figure fraction).
    # wspace is the gap between columns (fraction of average subplot width).
    fig.subplots_adjust(
        left   = geo["left"],
        right  = geo["right"],
        top    = geo["top"],
        bottom = geo["bottom"],
        wspace = geo["wspace"],
    )

    # ── Compute thresholds ────────────────────────────────────────────────────
    # Helper: collect the non-None arrays from one model's data tuple
    def non_none(data: tuple[np.ndarray | None, np.ndarray | None]) -> list[np.ndarray]:
        return [arr for arr in data if arr is not None]

    pre_arrays  = non_none(pretrained_data)
    fine_arrays = non_none(finetuned_data)

    if mode == "paired":
        # Pool both models' data → one shared threshold
        shared = compute_marker_thresholds(pre_arrays + fine_arrays)
        pre_thresholds  = shared
        fine_thresholds = shared
    else:
        # Independent thresholds per model
        pre_thresholds  = compute_marker_thresholds(pre_arrays)
        fine_thresholds = compute_marker_thresholds(fine_arrays)

    # ── Render each cell ──────────────────────────────────────────────────────
    configs = [
        (0, PRETRAINED_ID, pretrained_data, pre_thresholds),
        (1, FINETUNED_ID,  finetuned_data,  fine_thresholds),
    ]

    for col_idx, model_id, (img_to_text, text_to_img), thresholds in configs:
        ax = axes[0, col_idx]

        if img_to_text is not None or text_to_img is not None:
            render_scatter_cell(ax, img_to_text, text_to_img, thresholds)
        else:
            render_missing_cell(ax)

        # Column header (model name, shown above the subplot)
        ax.set_title(MODEL_LABELS[model_id], fontsize=FONT_COL_TITLE, fontweight="bold", pad=6)

        # Axis labels — only left column gets the y-label to avoid repetition
        ax.set_xlabel("text layer", fontsize=FONT_AXIS_LABEL, labelpad=4)
        if col_idx == 0:
            ax.set_ylabel("visual layer", fontsize=FONT_AXIS_LABEL, labelpad=4)

    # ── Legends ───────────────────────────────────────────────────────────────
    if mode == "paired":
        _add_legends_paired(fig, shared)
    else:
        _add_legends_disjoint(fig, pre_thresholds, fine_thresholds, geo["col_centers"])

    return fig


# ==============================================================================
# SECTION 8 — MAIN
# ==============================================================================


def main() -> None:
    print("=" * 72)
    print("  cambrian_pretrained_vs_finetuned_imbalance.py")
    print(f"  datasets   : {len(DATASETS)}")
    print(f"  percentiles: P{PERCENTILES[0]} / P{PERCENTILES[1]} / P{PERCENTILES[2]}")
    print(f"  output     : {PLOTS_DIR}")
    print("=" * 72)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and average data ─────────────────────────────────────────────────
    print(f"\nLoading {PRETRAINED_ID} ...")
    pretrained = load_model_data(PRETRAINED_ID)
    print(f"  img→text : {'✓' if pretrained[0] is not None else '✗'}")
    print(f"  text→img : {'✓' if pretrained[1] is not None else '✗'}")

    print(f"\nLoading {FINETUNED_ID} ...")
    finetuned = load_model_data(FINETUNED_ID)
    print(f"  img→text : {'✓' if finetuned[0] is not None else '✗'}")
    print(f"  text→img : {'✓' if finetuned[1] is not None else '✗'}")

    # ── Figure 1: paired (shared threshold scale) ─────────────────────────────
    print("\nBuilding figure 1 — paired thresholds ...")
    fig_paired = build_figure(
        pretrained_data=pretrained,
        finetuned_data=finetuned,
        mode="paired",
        title="",   # no suptitle — column subtitles carry the labels
    )
    path_paired = PLOTS_DIR / "paired_thresholds.png"
    fig_paired.savefig(path_paired, dpi=DPI, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig_paired)
    print(f"  ✓ {path_paired}")

    # ── Figure 2: disjoint (per-model threshold scale) ────────────────────────
    print("\nBuilding figure 2 — disjoint thresholds ...")
    fig_disjoint = build_figure(
        pretrained_data=pretrained,
        finetuned_data=finetuned,
        mode="disjoint",
        title="",   # no suptitle
    )
    path_disjoint = PLOTS_DIR / "disjoint_thresholds.png"
    fig_disjoint.savefig(path_disjoint, dpi=DPI, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig_disjoint)
    print(f"  ✓ {path_disjoint}")

    print("\n" + "=" * 72)
    print("  Done. 2 figures written.")
    print("=" * 72)


if __name__ == "__main__":
    main()
