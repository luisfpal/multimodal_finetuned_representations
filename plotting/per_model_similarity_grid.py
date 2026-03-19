"""per_model_similarity_grid.py
────────────────────────────────────────────────────────────────────────────────
One figure per model per stage: a (datasets × metrics) grid of heatmaps.

Stages  – five dataset groups, each producing a separate figure per model:
            1. cocoqa_gqa    — COCOQA splits + GQA (human + synthetic)
            2. vqav2         — VQAv2 splits  (human + synthetic)
            3. single_base   — MMBench, MME, POPE, SEED  (repeated_images — raw, with duplicates)
            4. single_unique — MMBench, MME, POPE, SEED  (unique_images — deduplicated)
            5. open_ended    — MMBench, SEED formatted as open-ended VQA (no choices list)

Rows    – datasets present for this model in this stage, in STAGES order.
          Optionally appends an "Average" row per stage (SHOW_AVERAGE).
          Rows with zero data are silently skipped.
Cols    – every metric in METRICS, in order.
Cells   – vmin/vmax determined by COLORMAP_NORM (see below).
          Grey "N/A" placeholder when a file is missing.

Output  – PNG (300 dpi) saved to:
            PLOTS_DIR / POOLING_METHOD / [EXPERIMENT_LABEL/] {model}_{stage}.png

Workflow – edit SECTION 1 only:
    1.  Set POOLING_METHOD   ("last" | "mean_context")
    2.  Set COLORMAP_NORM    ("none" | "per_metric" | "cross_model")
    3.  If cross_model: edit CROSS_MODEL_PAIRS (list of (model_id_A, model_id_B))
    4.  Toggle SHOW_AVERAGE
    5.  Comment/uncomment stages or individual datasets within stages
    6.  Run — missing data handled gracefully

Colormap normalization (COLORMAP_NORM):
    "none"        → each cell: independent per-cell [min, max]
    "per_metric"  → each metric column: global [min, max] across ALL enabled
                    datasets for that model; extremes are printed with source
    "cross_model" → like "per_metric" but bounds are shared across each pair
                    in CROSS_MODEL_PAIRS; models not in any pair use per_metric

Note on variants (POOLING_METHOD):
    "last"                    → repeated_images + unique_images available for the 4 benchmarks
    "mean_context"            → unique_images only for benchmarks (repeated_images not computed)
    "mean_image_last_caption" → like mean_context; unique_images only for benchmarks
    "last_image_last_caption" → like mean_context; unique_images only for benchmarks
────────────────────────────────────────────────────────────────────────────────
"""

from collections import OrderedDict
from pathlib import Path
import argparse

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

# ── run variant ───────────────────────────────────────────────────────────────
POOLING_METHODS = [
    "last",
    "mean_context",
    "mean_image_last_caption",
]  # "last" | "mean_context" | "mean_image_last_caption" | "last_image_last_caption"

# Backward-compatible runtime variable used by path helpers.
# It is overwritten inside main() while iterating POOLING_METHODS.
POOLING_METHOD = POOLING_METHODS[0]

# Figure generation mode:
#   "specific"   -> legacy per-stage figures only
#   "aggregated" -> only one figure per pair averaged across all selected datasets
#   "both"       -> generate both sets
FIGURE_MODE = "aggregated"

# Optional sub-directory label (e.g. "baseline", "unique_vs_default").
# Empty string → plots land directly under PLOTS_DIR_BASE / POOLING_METHOD.
EXPERIMENT_LABEL = ""

# ── averaging ─────────────────────────────────────────────────────────────────
# Append an "Average" row at the bottom of each stage (mean within that stage).
SHOW_AVERAGE = False

# ── colormap normalization ────────────────────────────────────────────────────
#   "none"        → per-cell independent [min, max]  (default)
#   "per_metric"  → per-metric global [min, max] across all enabled datasets
#                   for that model; prints where the extremes were found
#   "cross_model" → like "per_metric" but each pair in CROSS_MODEL_PAIRS shares
#                   the same bounds; models not in any pair use "per_metric"
COLORMAP_NORM = "cross_model"  # "none" | "per_metric" | "cross_model"

# Used only when COLORMAP_NORM = "cross_model".
# Each tuple = two model IDs whose colormap bounds are unified per metric.
# Models not appearing in any pair fall back to per_metric normalization.
# Comment out pairs to deactivate them.
CROSS_MODEL_PAIRS: list[tuple[str, str]] = [
    ("llava_7b", "llava-v1.5-7b-pretrained-lbasile"),
    ("llava_13b", "llava-v1.5-13b-pretrained-lbasile"),
    ("InternVL2-8B", "InternVL2-8B-Pretrain"),
    ("LLaVA-OneVision-1.5-8B-stage2-2600", "LLaVA-OneVision-1.5-8B-stage1"),
    ("LLaVA-OneVision-1.5-8B-Instruct", "LLaVA-OneVision-1.5-8B-Base"),
]

# ── models ────────────────────────────────────────────────────────────────────
# key   = prefix in the safetensors filename
# value = display name in the figure suptitle
MODELS = OrderedDict(
    [
        ("InternVL2-8B", "InternVL2-8B"),
        ("InternVL2-8B-Pretrain", "InternVL2-8B (Pretrained)"),
        ("llava_7b", "LLaVA-7B (Finetuned)"),
        ("llava-v1.5-7b-pretrained-lbasile", "LLaVA-7B (Pretrained)"),
        ("llava_13b", "LLaVA-13B (Finetuned)"),
        ("llava-v1.5-13b-pretrained-lbasile", "LLaVA-13B (Pretrained)"),
        ("LLaVA-OneVision-1.5-8B-stage2-2600", "LLaVA-OneVision-8B (Finetuned)"),
        ("LLaVA-OneVision-1.5-8B-stage1", "LLaVA-OneVision-8B (Pretrained)"),
        ("LLaVA-OneVision-1.5-8B-Instruct", "LLaVA-OneVision-8B (Instruct)"),
        ("LLaVA-OneVision-1.5-8B-Base", "LLaVA-OneVision-8B (Base)"),
        # ("llava_7b_6_17",                 "LLaVA-7B (6→17)"),
        # ("llava_7b_6_17_complementary",   "LLaVA-7B (6→17 Complementary)"),
        # ("InternVL3_5-8B-Pretrained",     "InternVL3.5-8B (Pretrained)"),
        # ("InternVL3_5-8B-Instruct",       "InternVL3.5-8B (Instruct)"),
        # ("InternVL3_5-8B-MPO",            "InternVL3.5-8B (MPO)"),
        # ("InternVL3_5-8B",                "InternVL3.5-8B"),
    ]
)

# ── stages ────────────────────────────────────────────────────────────────────
# Each stage → one figure per model.
#
# Format: (stage_key, stage_display_label, [dataset_tuples])
#   stage_key           → used in output filename  e.g. "cocoqa"
#   stage_display_label → appended to model name in suptitle
#   dataset_tuples      → list of (dataset_dir, variant_tag, row_label)
#
# To skip a whole stage: comment out its outer tuple.
# To skip individual datasets within a stage: comment out their inner tuples.
#
# variant_tag: "" = base variant, "unique_images_mc" = unique-images variant
#
# Note: For POOLING_METHOD="mean_context", the base variant of the 4 benchmark
#       datasets (mmbench, mme, pope, seed) does not exist — those rows are
#       silently skipped. Stage 3 (single_base) will produce N/A cells or be
#       empty for those datasets.
STAGES: list[tuple[str, str, list]] = [
    # ── Stage 1: Open-Ended VQA (COCOQA + GQA) ────────────────────────────────
    (
        "cocoqa_gqa",
        "Open-Ended VQA (COCOQA + GQA)",
        [
            ("cocoqa_restval_icqa_split01_human", "unique_images", "COCOQA Split01 (Human)"),
            # ("cocoqa_restval_icqa_split01_synthetic", "unique_images", "COCOQA Split01 (Synthetic)"),
            # ("cocoqa_restval_icqa_split02_human", "unique_images", "COCOQA Split02 (Human)"),
            # ("cocoqa_restval_icqa_split02_synthetic", "unique_images", "COCOQA Split02 (Synthetic)"),
            ("gqa_val_balanced_noyesno_captioned_synthetic", "unique_images", "GQA"),
        ],
    ),
    # ── Stage 2: VQAv2 ───────────────────────────────────────────────────────
    (
        "vqav2",
        "VQAv2",
        [
            ("vqav2_coco_restval_icq_split01_human", "unique_images", "VQAv2 Split01 (Human)"),
            # (
            #     "vqav2_coco_restval_icq_split01_synthetic",
            #     "unique_images",
            #     "VQAv2 Split01 (Synthetic)",
            # ),
            # ("vqav2_coco_restval_icq_split02_human", "unique_images", "VQAv2 Split02 (Human)"),
            # (
            #     "vqav2_coco_restval_icq_split02_synthetic",
            #     "unique_images",
            #     "VQAv2 Split02 (Synthetic)",
            # ),
            # ("vqav2_coco_restval_icq_split03_human", "unique_images", "VQAv2 Split03 (Human)"),
            # (
            #     "vqav2_coco_restval_icq_split03_synthetic",
            #     "unique_images",
            #     "VQAv2 Split03 (Synthetic)",
            # ),
        ],
    ),
    # # ── Stage 3: Benchmarks (Repeated Images) ─────────────────────────────────────
    # (
    #     "single_base",
    #     "Benchmarks (Repeated Images)",
    #     [
    #         ("mmbench_en_dev_captioned_synthetic", "repeated_images", "MMBench"),
    #         ("mme_test_captioned_synthetic", "repeated_images", "MME"),
    #         ("pope_test_captioned_synthetic", "repeated_images", "POPE"),
    #         ("seed_bench_images_captioned_synthetic", "repeated_images", "SEED"),
    #     ],
    # ),
    # # ── Stage 4: Benchmarks (Unique Images) ────────────────────────────────────
    # (
    #     "single_unique",
    #     "Benchmarks (Unique Images)",
    #     [
    #         (
    #             "mmbench_en_dev_captioned_synthetic",
    #             "unique_images",
    #             "MMBench (Unique)",
    #         ),
    #         ("mme_test_captioned_synthetic", "unique_images", "MME (Unique)"),
    #         ("pope_test_captioned_synthetic", "unique_images", "POPE (Unique)"),
    #         (
    #             "seed_bench_images_captioned_synthetic",
    #             "unique_images",
    #             "SEED (Unique)",
    #         ),
    #     ],
    # ),
    # ── Stage 5: Benchmarks (Open-Ended) ───────────────────────────────────────
    # MMBench and SEED reformatted as open-ended VQA: no choices list, open guide text.
    # Dataset prefix: {slug}_open_ended_{caption_source} → separate directory.
    (
        "open_ended",
        "Benchmarks (Open-Ended)",
        [
            (
                "mmbench_en_dev_captioned_open_ended_synthetic",
                "unique_images",
                "MMBench (Open-Ended)",
            ),
            (
                "seed_bench_images_captioned_open_ended_synthetic",
                "unique_images",
                "SEED (Open-Ended)",
            ),
        ],
    ),
]

# ── metrics ───────────────────────────────────────────────────────────────────
METRICS = OrderedDict(
    [
        ("neighborhood_overlap", "neighborhood_overlap"),
        ("linear_cka", "linear_cka"),
        ("information_imbalance", "II Complement\n(Text→Image)"),
        ("information_imbalance_reversed", "II Complement\n(Image→Text)"),
        ("information_imbalance_mean", "II Complement\n(Mean)"),
    ]
)

# ── layout knobs ──────────────────────────────────────────────────────────────
CELL_W = 2.8
CELL_H = 2.5
MARGIN_LEFT = 1.8
MARGIN_TOP = 1.0
MARGIN_RIGHT = 0.6
MARGIN_BOTTOM = 0.7
WSPACE = 0.35
HSPACE = 0.30

# ── typography ────────────────────────────────────────────────────────────────
FONT = dict(
    suptitle=20,
    col_header=11,
    row_label=13,
    axis_label=8,
    tick=7,
    cbar=7,
)

# ── rendering ─────────────────────────────────────────────────────────────────
CMAP = "viridis"
TICK_STEP = 5
DPI_PNG = 300


# ==============================================================================
# SECTION 2  –  DATA LOADING & BOUNDS
# ==============================================================================

# Flat deduplicated list of all (dataset_dir, variant_tag, label) across STAGES.
# Derived from STAGES — do not edit directly.
_ALL_DATASETS: list[tuple[str, str, str]] = []
_seen: set = set()
for _sk, _sl, _ds in STAGES:
    for _entry in _ds:
        _key = (_entry[0], _entry[1])
        if _key not in _seen:
            _seen.add(_key)
            _ALL_DATASETS.append(_entry)
del _sk, _sl, _ds, _entry, _key, _seen


def _make_plots_dir() -> Path:
    d = PLOTS_DIR_BASE / POOLING_METHOD
    if EXPERIMENT_LABEL:
        d = d / EXPERIMENT_LABEL
    return d


def _make_similarities_dir(dataset_dir: str, variant_tag: str = "") -> Path:
    """RESULTS_DIR / dataset_dir / POOLING_METHOD / [variant_tag/] similarities/"""
    d = RESULTS_DIR / dataset_dir / POOLING_METHOD
    if variant_tag:
        d = d / variant_tag
    return d / "similarities"


def _load_matrix(model_id: str, metric: str, dataset_dir: str, variant_tag: str = ""):
    """Load one matrix. Returns None when file or key is missing."""
    path = (
        _make_similarities_dir(dataset_dir, variant_tag)
        / f"{model_id}_{metric}.safetensors"
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
    Walk MODELS × all datasets in STAGES × METRICS and load everything present.

    Returns
    -------
    {model_id : {(dataset_dir, variant_tag) : {metric : ndarray}}}
    """
    data: dict = {}
    for model_id in MODELS:
        model_data: dict = {}
        for dataset_dir, variant_tag, _label in _ALL_DATASETS:
            key = (dataset_dir, variant_tag)
            ds: dict = {}
            for metric in METRICS:
                arr = _load_matrix(model_id, metric, dataset_dir, variant_tag)
                if arr is not None:
                    ds[metric] = arr
            if ds:
                model_data[key] = ds
        if model_data:
            data[model_id] = model_data
            print(f"  ✓  {MODELS[model_id]:45s} – {len(model_data)} dataset(s)")
        else:
            print(f"  ✗  {MODELS[model_id]:45s} – no data, skipped")
    return data


def compute_metric_bounds(data: dict) -> dict:
    """
    Compute per-metric [vmin, vmax] for each model, respecting COLORMAP_NORM.

    Returns
    -------
    {model_id: {metric: (vmin, vmax, min_source, max_source)}}
        min_source / max_source = dataset_dir where the extreme was found.
    Returns {} when COLORMAP_NORM = "none" (no shared bounds needed).

    Logic
    -----
    1. Compute per-metric bounds for every model independently (per_metric).
    2. If COLORMAP_NORM = "cross_model": for each pair in CROSS_MODEL_PAIRS,
       unify the bounds of the two models (take global min/max across both).
       Models not in any pair retain their individual per_metric bounds.
    """
    if COLORMAP_NORM == "none":
        return {}

    bounds: dict = {}

    # Step 1 — per-model, per-metric bounds
    for model_id, model_data in data.items():
        model_bounds: dict = {}
        for metric in METRICS:
            candidates = [
                (float(ds[metric].min()), float(ds[metric].max()), ds_dir)
                for (ds_dir, _var), ds in model_data.items()
                if metric in ds
            ]
            if not candidates:
                continue
            vmin = min(c[0] for c in candidates)
            vmax = max(c[1] for c in candidates)
            min_src = next(c[2] for c in candidates if c[0] == vmin)
            max_src = next(c[2] for c in candidates if c[1] == vmax)
            model_bounds[metric] = (vmin, vmax, min_src, max_src)
        bounds[model_id] = model_bounds

    # Step 2 — unify bounds across each cross-model pair
    if COLORMAP_NORM == "cross_model":
        for m1, m2 in CROSS_MODEL_PAIRS:
            if m1 not in bounds or m2 not in bounds:
                continue
            for metric in METRICS:
                b1 = bounds[m1].get(metric)
                b2 = bounds[m2].get(metric)
                if b1 is None or b2 is None:
                    continue
                shared_vmin = min(b1[0], b2[0])
                shared_vmax = max(b1[1], b2[1])
                # Track which model's dataset drove each extreme
                min_src = f"{m1}:{b1[2]}" if b1[0] <= b2[0] else f"{m2}:{b2[2]}"
                max_src = f"{m1}:{b1[3]}" if b1[1] >= b2[1] else f"{m2}:{b2[3]}"
                shared = (shared_vmin, shared_vmax, min_src, max_src)
                bounds[m1][metric] = shared
                bounds[m2][metric] = shared

    return bounds


def _print_bounds(bounds: dict) -> None:
    """Print colormap bound summary (model → metric → vmin/vmax + source)."""
    if not bounds:
        return
    print(f"\n  Colormap bounds  (COLORMAP_NORM={COLORMAP_NORM!r}):")
    for model_id, model_bounds in bounds.items():
        print(f"\n  ┌ {MODELS[model_id]}")
        for metric, (vmin, vmax, min_src, max_src) in model_bounds.items():
            print(f"  │  {metric:35s}  min={vmin:.4f}  [{min_src}]")
            print(f"  │  {metric:35s}  max={vmax:.4f}  [{max_src}]")
        print("  └")


# ==============================================================================
# SECTION 3  –  PLOTTING PRIMITIVES  (each touches exactly one Axes)
# ==============================================================================


def _draw_heatmap(ax, matrix, vmin=None, vmax=None):
    """Render heatmap on ax.  vmin/vmax=None → matrix's own extremes."""
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=CMAP,
        vmin=float(matrix.min()) if vmin is None else vmin,
        vmax=float(matrix.max()) if vmax is None else vmax,
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FONT["cbar"], length=2)
    cb.outline.set_visible(False)
    return cb


def _draw_missing(ax):
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


def _stage_averages(stage_datasets: list, model_data: dict) -> dict:
    """Element-wise mean across datasets present in this stage."""
    buckets: dict = {}
    for dataset_dir, variant_tag, _ in stage_datasets:
        key = (dataset_dir, variant_tag)
        if key in model_data:
            for m, arr in model_data[key].items():
                buckets.setdefault(m, []).append(arr)
    return {
        m: np.mean(np.stack(arrs), axis=0)
        for m, arrs in buckets.items()
        if len(arrs) >= 2 and len({a.shape for a in arrs}) == 1
    }


def _row_specs(stage_datasets: list, model_data: dict, avgs: dict) -> list:
    """Ordered list of (key, label, data_dict) rows for this stage figure."""
    rows = []
    for dataset_dir, variant_tag, label in stage_datasets:
        key = (dataset_dir, variant_tag)
        if key in model_data:
            rows.append((key, label, model_data[key]))
    if SHOW_AVERAGE and avgs and len(rows) > 1:
        rows.append(("average", "Average", avgs))
    return rows


def aggregate_model_across_all_datasets(model_data: dict) -> dict:
    """Element-wise mean per metric across all selected datasets for one model."""
    buckets: dict = {}
    for ds_metrics in model_data.values():
        for metric, arr in ds_metrics.items():
            buckets.setdefault(metric, []).append(arr)

    aggregated: dict = {}
    for metric, arrays in buckets.items():
        if len(arrays) == 1:
            aggregated[metric] = arrays[0]
            continue
        if len({a.shape for a in arrays}) != 1:
            continue
        aggregated[metric] = np.mean(np.stack(arrays), axis=0)
    return aggregated


def build_figure(
    model_id: str,
    stage_label: str,
    stage_datasets: list,
    model_data: dict,
    model_bounds: dict,
) -> "plt.Figure | None":
    """
    Assemble one (rows × metrics) grid figure for a single (model, stage) pair.

    Parameters
    ----------
    model_id      : key in MODELS
    stage_label   : displayed in suptitle after the model name
    stage_datasets: list of (dataset_dir, variant_tag, row_label) for this stage
    model_data    : {(dataset_dir, variant_tag): {metric: ndarray}}
    model_bounds  : {metric: (vmin, vmax, ...)}; empty dict → per-cell mode
    """
    avgs = _stage_averages(stage_datasets, model_data)
    rows = _row_specs(stage_datasets, model_data, avgs)
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
                vmin, vmax = None, None
                if model_bounds and mk in model_bounds:
                    vmin, vmax = model_bounds[mk][0], model_bounds[mk][1]
                im = _draw_heatmap(ax, row_data[mk], vmin=vmin, vmax=vmax)
                _draw_colorbar(ax, im)
            else:
                _draw_missing(ax)

            if r == 0:
                ax.set_title(METRICS[mk], fontsize=FONT["col_header"], pad=8)

    # Row labels in left margin (computed analytically — avoids OOM from canvas.draw())
    bottom_f = MARGIN_BOTTOM / fig_h
    top_f = 1.0 - MARGIN_TOP / fig_h
    grid_span = top_f - bottom_f
    eff_h = grid_span / (nR + max(nR - 1, 0) * HSPACE)
    x_lbl = 0.5 * MARGIN_LEFT / fig_w
    for r, (_rid, lbl, _data) in enumerate(rows):
        row_from_bottom = nR - 1 - r
        y0 = bottom_f + row_from_bottom * eff_h * (1 + HSPACE)
        fig.text(
            x_lbl,
            y0 + eff_h / 2,
            lbl,
            fontsize=FONT["row_label"],
            ha="center",
            va="center",
            fontweight="bold",
        )

    fig.suptitle(
        f"{MODELS[model_id]}  —  {stage_label}",
        fontsize=FONT["suptitle"],
        fontweight="bold",
        y=0.98,
    )
    return fig


def build_aggregated_pair_figure(
    finetuned_id: str,
    pretrained_id: str,
    aggregated_data: dict,
    bounds: dict,
) -> "plt.Figure | None":
    """Build one 2×metrics figure averaged across all selected datasets for a model pair."""
    row_model_ids = [pretrained_id, finetuned_id]
    mkeys = list(METRICS.keys())
    nR, nC = 2, len(mkeys)

    if not any(aggregated_data.get(mid) for mid in row_model_ids):
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

    for r, model_id in enumerate(row_model_ids):
        row_data = aggregated_data.get(model_id, {})
        row_bounds = bounds.get(model_id, {})
        for c, mk in enumerate(mkeys):
            ax = axes[r, c]
            if mk in row_data:
                vmin, vmax = None, None
                if row_bounds and mk in row_bounds:
                    vmin, vmax = row_bounds[mk][0], row_bounds[mk][1]
                im = _draw_heatmap(ax, row_data[mk], vmin=vmin, vmax=vmax)
                _draw_colorbar(ax, im)
            else:
                _draw_missing(ax)

            if r == 0:
                ax.set_title(METRICS[mk], fontsize=FONT["col_header"], pad=8)

    bottom_f = MARGIN_BOTTOM / fig_h
    top_f = 1.0 - MARGIN_TOP / fig_h
    grid_span = top_f - bottom_f
    eff_h = grid_span / (nR + max(nR - 1, 0) * HSPACE)
    x_lbl = 0.5 * MARGIN_LEFT / fig_w
    for r, model_id in enumerate(row_model_ids):
        row_from_bottom = nR - 1 - r
        y0 = bottom_f + row_from_bottom * eff_h * (1 + HSPACE)
        fig.text(
            x_lbl,
            y0 + eff_h / 2,
            MODELS.get(model_id, model_id),
            fontsize=FONT["row_label"],
            ha="center",
            va="center",
            fontweight="bold",
        )

    fig.suptitle(
        "Aggregated Across Selected Datasets",
        fontsize=FONT["suptitle"],
        fontweight="bold",
        y=0.98,
    )
    return fig


# ==============================================================================
# SECTION 5  –  MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate multimodality similarity heatmaps."
    )
    parser.add_argument(
        "--figure-mode",
        choices=["specific", "aggregated", "both"],
        default=FIGURE_MODE,
        help="specific=stage figures, aggregated=all-datasets average per pair, both=run both",
    )
    parser.add_argument(
        "--pooling-methods",
        nargs="+",
        default=POOLING_METHODS,
        help="Pooling methods to process.",
    )
    args = parser.parse_args()

    norm_info = f"norm={COLORMAP_NORM}"
    if COLORMAP_NORM == "cross_model":
        pairs_str = ", ".join(f"{a}↔{b}" for a, b in CROSS_MODEL_PAIRS)
        norm_info += f" [{pairs_str}]"
    label_info = f" | label={EXPERIMENT_LABEL}" if EXPERIMENT_LABEL else ""

    print("=" * 72)
    print("  per_model_similarity_grid")
    print(f"  mode={args.figure_mode} | {norm_info}{label_info}")
    print("=" * 72)

    global POOLING_METHOD
    for pooling in args.pooling_methods:
        POOLING_METHOD = pooling
        plots_dir = _make_plots_dir()

        print("\n" + "-" * 72)
        print(f"  pooling={POOLING_METHOD}")
        print("-" * 72)

        print("\n  Loading data...")
        data = load_all_data()
        plots_dir.mkdir(parents=True, exist_ok=True)

        if args.figure_mode in {"specific", "both"}:
            print("\n  Computing colormap bounds (specific)...")
            bounds = compute_metric_bounds(data)
            _print_bounds(bounds)

            print("\n  Generating specific-stage figures...")
            for model_id in MODELS:
                if model_id not in data:
                    continue
                model_bounds = bounds.get(model_id, {})

                for stage_key, stage_label, stage_datasets in STAGES:
                    print(f"\n  {MODELS[model_id]}  [{stage_key}]")
                    fig = build_figure(
                        model_id,
                        stage_label,
                        stage_datasets,
                        data[model_id],
                        model_bounds,
                    )
                    if fig is None:
                        print("    (no data for this stage — skipped)")
                        continue

                    out = plots_dir / f"{model_id}_{stage_key}.png"
                    fig.savefig(
                        str(out), dpi=DPI_PNG, facecolor="white", bbox_inches="tight"
                    )
                    print(f"    ✓  {out}")
                    plt.close(fig)

        if args.figure_mode in {"aggregated", "both"}:
            print("\n  Computing aggregated data across all selected datasets...")
            aggregated_data: dict = {}
            for model_id, model_data in data.items():
                aggregated = aggregate_model_across_all_datasets(model_data)
                if aggregated:
                    aggregated_data[model_id] = aggregated

            agg_bounds_input = {
                model_id: {("aggregated_all", ""): metrics}
                for model_id, metrics in aggregated_data.items()
            }
            agg_bounds = compute_metric_bounds(agg_bounds_input)
            _print_bounds(agg_bounds)

            print("\n  Generating aggregated pair figures...")
            for finetuned_id, pretrained_id in CROSS_MODEL_PAIRS:
                if finetuned_id not in aggregated_data and pretrained_id not in aggregated_data:
                    continue

                fig = build_aggregated_pair_figure(
                    finetuned_id,
                    pretrained_id,
                    aggregated_data,
                    agg_bounds,
                )
                if fig is None:
                    print(
                        f"    (no aggregated data for pair {finetuned_id} ↔ {pretrained_id} — skipped)"
                    )
                    continue

                out = plots_dir / f"{finetuned_id}_aggregated_all.png"
                fig.savefig(str(out), dpi=DPI_PNG, facecolor="white", bbox_inches="tight")
                print(f"    ✓  {out}")
                plt.close(fig)

    print("\n  Done.")


if __name__ == "__main__":
    main()
