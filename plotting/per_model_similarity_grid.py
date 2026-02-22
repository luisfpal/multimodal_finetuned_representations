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

Output – PNG (300 dpi) + PDF (vector, lossless zoom) saved to:
           PLOTS_DIR / POOLING_METHOD / [sf-{SEMANTIC_FILTER_TAG}/] {model}_similarity_grid.{ext}

Quick-start
    python per_model_similarity_grid.py

Customisation – edit the constants in SECTION 1 only.
    • Switch pooling variant:  change POOLING_METHOD
    • Add semantic filter run: set  SEMANTIC_FILTER_TAG = "jina_cpu_m1sd"
    • Toggle datasets/models:  comment/uncomment rows in DATASETS / MODELS
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
# SECTION 1  –  CONFIGURATION  (only section you need to edit)
# ==============================================================================

RESULTS_DIR = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"
    "results/multimodality_comparison"
)

PLOTS_DIR_BASE = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"
    "plots/similarity_grids"
)

# ── run variant ──────────────────────────────────────────────────────────────
# Change POOLING_METHOD to switch between run variants.  That is all.
POOLING_METHOD = "mean_context"   # "last" | "mean_context"
SEMANTIC_FILTER_TAG = ""          # "" (no filter) | "jina_cpu_m1sd"

# ── models ──────────────────────────────────────────────────────────────────
# key   = exact prefix used in the safetensors filename  (e.g. "InternVL2-8B")
# value = display name used as the figure suptitle
MODELS = OrderedDict(
    [
        ("InternVL2-8B", "InternVL2-8B"),
        ("InternVL2-8B-Pretrain", "InternVL2-8B (Pretrained)"),
        ("llava_7b", "LLaVA-7B (Finetuned)"),
        ("llava-v1.5-7b-pretrained-lbasile", "LLaVA-7B (Pretrained)"),
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

# ── datasets ─────────────────────────────────────────────────────────────────
# key   = "{dataset}_{caption_source}" folder under results/multimodality_comparison/
#         (no pooling suffix – that lives in POOLING_METHOD above)
# value = short row label shown on the left of the grid
#
# Available dataset_caption_source combos:
#   cocoqa_restval_icqa_human         – COCOQA, human captions
#   cocoqa_restval_icqa_synthetic     – COCOQA, synthetic captions
#   vqav2_coco_restval_icq_human      – VQAv2, human captions
#   vqav2_coco_restval_icq_synthetic  – VQAv2, synthetic captions
#   gqa_val_balanced_noyesno_captioned_synthetic
#   mme_test_captioned_synthetic
#   mmbench_en_dev_captioned_synthetic
DATASETS = OrderedDict(
    [
        ("cocoqa_restval_icqa_human",              "COCOQA (Human)"),
        ("vqav2_coco_restval_icq_human",           "VQAv2 (Human)"),
        ("mme_test_captioned_synthetic",           "MME"),
        ("mmbench_en_dev_captioned_synthetic",     "MMBench"),
        ("gqa_val_balanced_noyesno_captioned_synthetic", "GQA"),
        # ("cocoqa_restval_icqa_synthetic",        "COCOQA (Synthetic)"),
        # ("vqav2_coco_restval_icq_synthetic",     "VQAv2 (Synthetic)"),
    ]
)

# ── metrics ──────────────────────────────────────────────────────────────────
# key   = metric name inside the safetensors dict AND the `_{key}` filename suffix
# value = column header text  (\n renders as two lines)
METRICS = OrderedDict(
    [
        ("neighborhood_overlap", "neighborhood_overlap"),
        ("linear_cka", "linear_cka"),
        ("information_imbalance", "II Complement\n(Text→Image)"),
        ("information_imbalance_reversed", "II Complement\n(Image→Text)"),
        ("information_imbalance_mean", "II Complement\n(Mean)"),
    ]
)

# ── layout knobs ─────────────────────────────────────────────────────────────
CELL_W = 2.8
CELL_H = 2.5
MARGIN_LEFT = 1.8
MARGIN_TOP = 1.0
MARGIN_RIGHT = 0.6
MARGIN_BOTTOM = 0.7
WSPACE = 0.35
HSPACE = 0.30

# ── typography ───────────────────────────────────────────────────────────────
FONT = dict(
    suptitle=20,
    col_header=11,
    row_label=13,
    axis_label=8,
    tick=7,
    cbar=7,
)

# ── rendering ────────────────────────────────────────────────────────────────
CMAP = "viridis"
TICK_STEP = 5
DPI_PNG = 300


# ==============================================================================
# SECTION 2  –  DATA LOADING
# ==============================================================================

def _make_plots_dir() -> Path:
    """Compute the output directory for plots given POOLING_METHOD / SEMANTIC_FILTER_TAG."""
    d = PLOTS_DIR_BASE / POOLING_METHOD
    if SEMANTIC_FILTER_TAG:
        d = d / f"sf-{SEMANTIC_FILTER_TAG}"
    return d


def _make_similarities_dir(dataset_dir: str) -> Path:
    """
    Build the path to the similarities/ folder for a given dataset directory name.

    Structure:
      RESULTS_DIR / {dataset_dir} / {POOLING_METHOD} / [sf-{tag}/] similarities/
    """
    d = RESULTS_DIR / dataset_dir / POOLING_METHOD
    if SEMANTIC_FILTER_TAG:
        d = d / f"sf-{SEMANTIC_FILTER_TAG}"
    return d / "similarities"


def _load_matrix(model_id: str, metric: str, dataset_dir: str):
    """Load one matrix.  Returns None when file or key is missing."""
    # Filename: {model}_{metric}.safetensors  (pooling in directory, not filename)
    path = _make_similarities_dir(dataset_dir) / f"{model_id}_{metric}.safetensors"
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
            if ds:
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
    for dd in DATASETS:
        if dd in model_data:
            rows.append((dd, DATASETS[dd], model_data[dd]))
    if avgs and len(model_data) > 1:
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

    for r, (_rid, _lbl, row_data) in enumerate(rows):
        for c, mk in enumerate(mkeys):
            ax = axes[r, c]
            if mk in row_data:
                im = _draw_heatmap(ax, row_data[mk])
                _draw_colorbar(ax, im)
            else:
                _draw_missing(ax)

            if r == 0:
                ax.set_title(METRICS[mk], fontsize=FONT["col_header"], pad=8)

    # Row labels in left margin
    fig.canvas.draw()
    x_lbl = 0.5 * MARGIN_LEFT / fig_w
    for r, (_rid, lbl, _data) in enumerate(rows):
        bb = axes[r, 0].get_position()
        fig.text(
            x_lbl,
            bb.y0 + bb.height / 2,
            lbl,
            fontsize=FONT["row_label"],
            ha="center",
            va="center",
            fontweight="bold",
        )

    fig.suptitle(MODELS[model_id], fontsize=FONT["suptitle"], fontweight="bold", y=0.98)
    return fig


# ==============================================================================
# SECTION 5  –  MAIN
# ==============================================================================


def main():
    plots_dir = _make_plots_dir()
    sf_info = f" | sf={SEMANTIC_FILTER_TAG}" if SEMANTIC_FILTER_TAG else ""
    print("=" * 66)
    print(f"  per_model_similarity_grid  –  pooling={POOLING_METHOD}{sf_info}")
    print("=" * 66)
    data = load_all_data()

    plots_dir.mkdir(parents=True, exist_ok=True)

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
            out = plots_dir / f"{model_id}_similarity_grid.{ext}"
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
