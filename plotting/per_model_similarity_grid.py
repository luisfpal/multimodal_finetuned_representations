"""per_model_similarity_grid.py
────────────────────────────────────────────────────────────────────────────────
One figure per model: a (datasets × metrics) grid of independent heatmaps.

Rows   – every dataset that has data for this model, in DATASETS order.
         An "Average" row is appended when ≥ 2 datasets are present.
         Dataset rows with zero data are silently skipped.
Cols   – every metric in METRICS, in the order defined there.
Cells  – each heatmap owns its axis labels, ticks, and a compact colorbar
         (vmin/vmax from that cell's data).  A grey N/A placeholder replaces
         cells where a specific metric file is missing.

Output – PNG (300 dpi) + PDF (vector, lossless zoom) saved to PLOTS_DIR.

Quick-start
    python per_model_similarity_grid.py

Customisation – edit the dicts / constants in SECTION 1.
────────────────────────────────────────────────────────────────────────────────
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from safetensors.torch import load_file

matplotlib.rcParams["font.family"] = "sans-serif"


# ==============================================================================
# SECTION 1  –  CONFIGURATION
# ==============================================================================

RESULTS_DIR = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"
    "results/multimodality_comparison"
)

PLOTS_DIR = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"
    "plots/similarity_grids"
)

# ── models ──────────────────────────────────────────────────────────────────
# key   = exact prefix in the safetensors filename   (e.g. "InternVL2-8B")
# value = display name used as the figure suptitle
# Comment / uncomment to toggle models.
MODELS = OrderedDict(
    [
        ("InternVL2-8B",                      "InternVL2-8B"),
        ("InternVL2-8B-Pretrain",             "InternVL2-8B (Pretrained)"),
        # ("llava_7b", "LLaVA-7B (Finetuned)"),
        # ("llava-v1.5-7b-pretrained-lbasile", "LLaVA-7B (Pretrained)"),
        # ("llava_13b",                         "LLaVA-13B (Finetuned)"),
        # ("llava-v1.5-13b-pretrained-lbasile", "LLaVA-13B (Pretrained)"),
        # ("llava_7b_6_17",                     "LLaVA-7B (6→17)"),
        # ("llava_7b_6_17_complementary",       "LLaVA-7B (6→17 Complementary)"),
        # ("InternVL3_5-8B-Pretrained", "InternVL3.5-8B (Pretrained)"),
        # ("InternVL3_5-8B-Instruct", "InternVL3.5-8B (Instruct)"),
        # ("InternVL3_5-8B-MPO", "InternVL3.5-8B (MPO)"),
        # ("InternVL3_5-8B", "InternVL3.5-8B"),
    ]
)

# ── datasets ────────────────────────────────────────────────────────────────
# key   = subdirectory name under results/multimodality_comparison/
# value = short row label shown on the left of the grid
DATASETS = OrderedDict(
    [
        ("cocoqa_restval_icqa_img_vs_txt", "COCOQA"),
        ("vqav2_coco_restval_icq_img_vs_txt", "VQAv2"),
        ("mme_test_captioned_img_vs_txt", "MME"),
        ("mmbench_en_dev_captioned_img_vs_txt", "MMBench_EN"),
        ("gqa_val_balanced_noyesno_captioned_img_vs_txt", "GQA"),
        # Synthetic captions
        ("cocoqa_restval_icqa_synthetic_img_vs_txt", "COCOQA (Synthetic)"),
        ("vqav2_coco_restval_icq_synthetic_img_vs_txt", "VQAv2 (Synthetic)"),
    ]
)

# ── metrics ─────────────────────────────────────────────────────────────────
# key   = metric name inside the safetensors dict  AND  the `_last_{key}`
#         filename suffix
# value = column header text  (\n renders as two lines)
METRICS = OrderedDict(
    [
        ("neighborhood_overlap", "neighborhood_overlap"),
        ("linear_cka", "linear_cka"),
        ("information_imbalance", "II Complement\n(Text->Image)"),
        ("information_imbalance_reversed", "II Complement\n(Image->Text)"),
        ("information_imbalance_mean", "II Complement\n(Mean)"),
    ]
)

# ── layout knobs ────────────────────────────────────────────────────────────
# All sizes in inches.  Figure dimensions are computed from these.
CELL_W = 2.8  # nominal width of one heatmap (colorbar steals ~5 %)
CELL_H = 2.5  # nominal height of one heatmap row
MARGIN_LEFT = 1.8  # left margin  – row labels live here
MARGIN_TOP = 1.0  # top margin   – column headers + suptitle
MARGIN_RIGHT = 0.6
MARGIN_BOTTOM = 0.7
WSPACE = 0.35  # fraction of cell width used as horizontal gap
HSPACE = 0.30  # fraction of cell height used as vertical gap

# ── typography ──────────────────────────────────────────────────────────────
FONT = dict(
    suptitle=20,
    col_header=11,  # metric name  (column header, top row)
    row_label=13,  # dataset label (left margin)
    axis_label=8,  # "Image Layers" / "Text Layers"
    tick=7,
    cbar=7,
)

# ── rendering ───────────────────────────────────────────────────────────────
CMAP = "viridis"
TICK_STEP = 5  # show every N-th layer index
DPI_PNG = 300


# ==============================================================================
# SECTION 2  –  DATA LOADING
# ==============================================================================


def _load_matrix(model_id: str, metric: str, dataset_dir: str):
    """Load one matrix.  Returns None when file or key is missing."""
    path = (
        RESULTS_DIR
        / dataset_dir
        / "similarities"
        / f"{model_id}_last_{metric}.safetensors"
    )
    if not path.exists():
        return None
    try:
        raw = load_file(str(path))
    except Exception as exc:
        print(f"  ⚠  {path.name}: {exc}")
        return None
    if metric not in raw:
        return None
    val = raw[metric].cpu().numpy()
    if "information_imbalance" in metric:  # stored as distance → complement
        val = 1.0 - val
    return val


def load_all_data() -> dict:
    """
    Walk MODELS × DATASETS × METRICS and load everything that exists.

    Returns
    -------
    {model_id : {dataset_dir : {metric : ndarray}}}
    Only files present on disk appear in the nested dict.
    """
    data: dict = {}
    for model_id in MODELS:
        model_data: dict = {}
        for dataset_dir in DATASETS:
            ds: dict = {}
            for metric in METRICS:
                arr = _load_matrix(model_id, metric, dataset_dir)
                if arr is not None:
                    ds[metric] = arr
            if ds:  # at least one metric loaded
                model_data[dataset_dir] = ds
        if model_data:
            data[model_id] = model_data
            print(f"  ✓  {MODELS[model_id]:45s} – {len(model_data)} dataset(s)")
        else:
            print(f"  ✗  {MODELS[model_id]:45s} – no data, skipped")
    return data


def _averages(model_data: dict) -> dict:
    """
    Per-metric element-wise mean across datasets.
    Only metrics with ≥ 2 same-shape arrays are averaged.

    Returns {metric : ndarray}.
    """
    buckets: dict = {}
    for ds in model_data.values():
        for m, arr in ds.items():
            buckets.setdefault(m, []).append(arr)
    return {
        m: np.mean(np.stack(arrs), axis=0)
        for m, arrs in buckets.items()
        if len(arrs) >= 2 and len({a.shape for a in arrs}) == 1
    }


# ==============================================================================
# SECTION 3  –  PLOTTING PRIMITIVES  (each one touches exactly one Axes)
# ==============================================================================


def _draw_heatmap(ax, matrix):
    """Render heatmap on *ax*.  Returns the ScalarMappable for the colorbar."""
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=CMAP,
        vmin=matrix.min(),
        vmax=matrix.max(),
        interpolation="nearest",
    )
    ticks = list(range(0, matrix.shape[0], TICK_STEP))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Image Layers", fontsize=FONT["axis_label"], labelpad=3)
    ax.set_ylabel("Text Layers", fontsize=FONT["axis_label"], labelpad=3)
    ax.tick_params(labelsize=FONT["tick"], length=3, direction="out")
    return im


def _draw_colorbar(ax, im):
    """Attach a compact colorbar to the right of *ax*."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FONT["cbar"], length=2)
    cb.outline.set_visible(False)
    return cb


def _draw_missing(ax):
    """Style *ax* as a visible 'no data' placeholder."""
    ax.set_facecolor("#ececec")
    ax.text(
        0.5,
        0.5,
        "N/A",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#888888",
        style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")


# ==============================================================================
# SECTION 4  –  FIGURE BUILDER
# ==============================================================================


def _row_specs(model_data: dict, avgs: dict):
    """
    Determine which rows the figure will contain.

    Returns list of (row_id, display_label, {metric: ndarray}).
    """
    rows = []
    for dd in DATASETS:  # config order preserved
        if dd in model_data:
            rows.append((dd, DATASETS[dd], model_data[dd]))
    if avgs and len(model_data) > 1:  # average only when meaningful
        rows.append(("average", "Average", avgs))
    return rows


def build_figure(model_id: str, model_data: dict):
    """
    Assemble the full (rows × metrics) grid figure for one model.
    Returns the Figure, or None when there is nothing to show.
    """
    avgs = _averages(model_data)
    rows = _row_specs(model_data, avgs)
    mkeys = list(METRICS.keys())
    nR, nC = len(rows), len(mkeys)
    if nR == 0:
        return None

    # ── figure size (scales with actual grid dimensions) ────────────────────
    fig_w = MARGIN_LEFT + nC * CELL_W + (nC - 1) * CELL_W * WSPACE + MARGIN_RIGHT
    fig_h = MARGIN_TOP + nR * CELL_H + (nR - 1) * CELL_H * HSPACE + MARGIN_BOTTOM

    fig, axes = plt.subplots(nR, nC, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left=MARGIN_LEFT / fig_w,
        right=1.0 - MARGIN_RIGHT / fig_w,
        top=1.0 - MARGIN_TOP / fig_h,
        bottom=MARGIN_BOTTOM / fig_h,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    # ── populate every cell ─────────────────────────────────────────────────
    for r, (_rid, _lbl, row_data) in enumerate(rows):
        for c, mk in enumerate(mkeys):
            ax = axes[r, c]
            if mk in row_data:
                im = _draw_heatmap(ax, row_data[mk])
                _draw_colorbar(ax, im)
            else:
                _draw_missing(ax)

            if r == 0:  # column header – top row only
                ax.set_title(METRICS[mk], fontsize=FONT["col_header"], pad=8)

    # ── row labels (placed in the left margin via fig.text) ─────────────────
    # One draw() to lock axes geometry; then read positions.
    fig.canvas.draw()
    x_lbl = 0.5 * MARGIN_LEFT / fig_w  # centred in left margin
    for r, (_rid, lbl, _data) in enumerate(rows):
        bb = axes[r, 0].get_position()  # figure-coordinate BBox
        fig.text(
            x_lbl,
            bb.y0 + bb.height / 2,
            lbl,
            fontsize=FONT["row_label"],
            ha="center",
            va="center",
            fontweight="bold",
        )

    # ── suptitle ────────────────────────────────────────────────────────────
    fig.suptitle(MODELS[model_id], fontsize=FONT["suptitle"], fontweight="bold", y=0.98)
    return fig


# ==============================================================================
# SECTION 5  –  MAIN
# ==============================================================================


def main():
    print("=" * 66)
    print("  per_model_similarity_grid  –  loading")
    print("=" * 66)
    data = load_all_data()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 66)
    print("  per_model_similarity_grid  –  generating figures")
    print("=" * 66)

    for model_id in MODELS:
        if model_id not in data:
            continue
        print(f"\n  {MODELS[model_id]}")
        fig = build_figure(model_id, data[model_id])
        if fig is None:
            print("    (nothing to plot)")
            continue

        for ext in ("png", "pdf"):
            out = PLOTS_DIR / f"{model_id}_similarity_grid.{ext}"
            fig.savefig(
                out,
                dpi=DPI_PNG if ext == "png" else None,
                facecolor="white",
                bbox_inches="tight",
            )
            print(f"    ✓  {out}")
        plt.close(fig)

    print("\n  Done.")


if __name__ == "__main__":
    main()
