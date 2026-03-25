"""per_model_linear_regression_r2_grid.py
────────────────────────────────────────────────────────────────────────────────
Dedicated heatmap grids for linear regression R² metrics only.

This script is intentionally independent from per_model_similarity_grid.py but
reuses its active registry (model pairs and uncommented STAGES) so dataset
comment/uncomment behavior stays single-source.

Metrics
-------
- linear_regression_r2          (Text → Image)
- linear_regression_r2_reversed (Image → Text)

Key guarantees
--------------
- Axis orientation is metadata-first (row_modality / col_modality) and display
  is always forced to y=Visual, x=Text.
- Regression backends are never mixed. Each backend (ols/ridge/lasso) is loaded,
  bounded, aggregated, and saved independently.
- Supports both specific (per-stage) and aggregated (all-datasets average) modes.
- Iterates across pooling methods.

Output layout
-------------
plots/linear_regression_r2_grids/<regression_type>/<pooling_method>/
  - <model_id>_<stage_key>.png            (specific mode)
  - <finetuned_id>_aggregated_all.png     (aggregated mode)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from safetensors import safe_open

import per_model_similarity_grid as cfg

matplotlib.rcParams["font.family"] = "sans-serif"


# ==============================================================================
# SECTION 1 – CONFIGURATION
# ==============================================================================

RESULTS_DIR = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"
    "results/multimodality_comparison"
)

PLOTS_DIR_BASE = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"
    "plots/linear_regression_r2_grids"
)

POOLING_METHODS = list(cfg.POOLING_METHODS)
FIGURE_MODE = "both"  # "specific" | "aggregated" | "both"
EXPERIMENT_LABEL = ""
SHOW_AVERAGE = False
COLORMAP_NORM = "cross_model"  # "none" | "per_metric" | "cross_model"

REGRESSION_TYPE_PRIORITY = ["ols"]#, "ridge", "lasso"]

MODEL_PAIRS = list(cfg.MODEL_PAIRS)
CROSS_MODEL_PAIRS = list(cfg.CROSS_MODEL_PAIRS)
MODELS = OrderedDict(cfg.MODELS)
STAGES = list(cfg.STAGES)

METRICS = OrderedDict(
    [
        ("linear_regression_r2", "R² (Text→Image)"),
        ("linear_regression_r2_reversed", "R² (Image→Text)"),
    ]
)

# layout
CELL_W = 2.9
CELL_H = 2.5
MARGIN_LEFT = 1.8
MARGIN_TOP = 1.0
MARGIN_RIGHT = 0.6
MARGIN_BOTTOM = 0.7
WSPACE = 0.35
HSPACE = 0.30

FONT = dict(
    suptitle=20,
    col_header=11,
    row_label=13,
    axis_label=8,
    tick=7,
    cbar=7,
)

CMAP = "viridis"
TICK_STEP = 5
DPI_PNG = 300

EXPECTED_SOURCE_ORIENTATION = ("text", "image")
DISPLAY_ORIENTATION = ("image", "text")


# ==============================================================================
# SECTION 2 – DATA LOADING
# ==============================================================================

_ALL_DATASETS: list[tuple[str, str, str]] = []
_seen: set[tuple[str, str]] = set()
for _stage_key, _stage_label, _datasets in STAGES:
    for _dataset in _datasets:
        _key = (_dataset[0], _dataset[1])
        if _key not in _seen:
            _seen.add(_key)
            _ALL_DATASETS.append(_dataset)
del _stage_key, _stage_label, _datasets, _dataset, _key, _seen


def _make_plots_dir(regression_type: str, pooling_method: str) -> Path:
    output_dir = PLOTS_DIR_BASE / regression_type / pooling_method
    if EXPERIMENT_LABEL:
        output_dir = output_dir / EXPERIMENT_LABEL
    return output_dir


def _make_similarities_dir(
    dataset_dir: str,
    pooling_method: str,
    variant_tag: str = "",
) -> Path:
    output_dir = RESULTS_DIR / dataset_dir / pooling_method
    if variant_tag:
        output_dir = output_dir / variant_tag
    return output_dir / "similarities"


def _normalize_modality_label(value: str, fallback: str) -> str:
    label = str(value).strip().lower()
    if label not in {"text", "image"}:
        return fallback
    return label


def _resolve_matrix_orientation(metadata: dict | None) -> tuple[str, str]:
    if not metadata:
        return EXPECTED_SOURCE_ORIENTATION

    row_mod = _normalize_modality_label(
        metadata.get("row_modality", EXPECTED_SOURCE_ORIENTATION[0]),
        EXPECTED_SOURCE_ORIENTATION[0],
    )
    col_mod = _normalize_modality_label(
        metadata.get("col_modality", EXPECTED_SOURCE_ORIENTATION[1]),
        EXPECTED_SOURCE_ORIENTATION[1],
    )

    if row_mod == col_mod:
        return EXPECTED_SOURCE_ORIENTATION

    return (row_mod, col_mod)


def _orient_for_display(
    matrix: np.ndarray,
    source_orientation: tuple[str, str],
) -> np.ndarray:
    if source_orientation == DISPLAY_ORIENTATION:
        return matrix
    if source_orientation == EXPECTED_SOURCE_ORIENTATION:
        return matrix.T
    return matrix.T


def _metric_filename(model_id: str, metric: str, regression_type: str) -> str:
    return f"{model_id}_{metric}_reg-{regression_type}.safetensors"


def _load_matrix(
    model_id: str,
    metric: str,
    dataset_dir: str,
    pooling_method: str,
    regression_type: str,
    variant_tag: str = "",
) -> np.ndarray | None:
    path = (
        _make_similarities_dir(dataset_dir, pooling_method, variant_tag)
        / _metric_filename(model_id, metric, regression_type)
    )
    if not path.exists():
        return None

    try:
        with safe_open(path, framework="np", device="cpu") as handle:
            if metric not in handle.keys():
                return None
            matrix = handle.get_tensor(metric).copy()
            source_orientation = _resolve_matrix_orientation(handle.metadata())
    except Exception as exc:
        print(f"  ⚠  {path.name}: {exc}")
        return None

    return _orient_for_display(matrix, source_orientation)


def load_all_data(pooling_method: str, regression_type: str) -> dict:
    """Load all available R² matrices for one pooling method and backend."""
    data: dict = {}
    for model_id in MODELS:
        model_data: dict = {}
        for dataset_dir, variant_tag, _label in _ALL_DATASETS:
            key = (dataset_dir, variant_tag)
            dataset_metrics: dict = {}
            for metric in METRICS:
                arr = _load_matrix(
                    model_id=model_id,
                    metric=metric,
                    dataset_dir=dataset_dir,
                    pooling_method=pooling_method,
                    regression_type=regression_type,
                    variant_tag=variant_tag,
                )
                if arr is not None:
                    dataset_metrics[metric] = arr
            if dataset_metrics:
                model_data[key] = dataset_metrics

        if model_data:
            data[model_id] = model_data
            print(f"  ✓  {MODELS[model_id]:45s} – {len(model_data)} dataset(s)")
        else:
            print(f"  ✗  {MODELS[model_id]:45s} – no data, skipped")

    return data


def discover_regression_types(pooling_methods: list[str]) -> list[str]:
    """Discover regression backends present in active datasets/poolings.

    If nothing is found, returns all known backends so plots/report paths stay
    ready and deterministic.
    """
    found: set[str] = set()

    for pooling_method in pooling_methods:
        for dataset_dir, variant_tag, _label in _ALL_DATASETS:
            similarities_dir = _make_similarities_dir(
                dataset_dir=dataset_dir,
                pooling_method=pooling_method,
                variant_tag=variant_tag,
            )
            if not similarities_dir.exists():
                continue

            for metric in METRICS:
                for path in similarities_dir.glob(f"*_{metric}_reg-*.safetensors"):
                    stem = path.stem
                    if "_reg-" not in stem:
                        continue
                    backend = stem.rsplit("_reg-", 1)[1].lower().strip()
                    if backend:
                        found.add(backend)

    ordered = [rt for rt in REGRESSION_TYPE_PRIORITY if rt in found]
    remainder = sorted(rt for rt in found if rt not in REGRESSION_TYPE_PRIORITY)
    discovered = ordered + remainder

    if discovered:
        return discovered

    print(
        "\n  ⚠ No regression backend files discovered in current active dataset scope. "
        "Using default backends for deterministic output paths: "
        f"{REGRESSION_TYPE_PRIORITY}"
    )
    return list(REGRESSION_TYPE_PRIORITY)


def compute_metric_bounds(data: dict) -> dict:
    """Compute per-model/per-metric bounds with optional pair sharing."""
    if COLORMAP_NORM == "none":
        return {}

    bounds: dict = {}

    for model_id, model_data in data.items():
        model_bounds: dict = {}
        for metric in METRICS:
            candidates = [
                (float(ds[metric].min()), float(ds[metric].max()), ds_dir)
                for (ds_dir, _variant), ds in model_data.items()
                if metric in ds
            ]
            if not candidates:
                continue
            vmin = min(item[0] for item in candidates)
            vmax = max(item[1] for item in candidates)
            min_src = next(item[2] for item in candidates if item[0] == vmin)
            max_src = next(item[2] for item in candidates if item[1] == vmax)
            model_bounds[metric] = (vmin, vmax, min_src, max_src)
        bounds[model_id] = model_bounds

    if COLORMAP_NORM == "cross_model":
        for finetuned_id, pretrained_id in CROSS_MODEL_PAIRS:
            if finetuned_id not in bounds or pretrained_id not in bounds:
                continue
            for metric in METRICS:
                b_fine = bounds[finetuned_id].get(metric)
                b_pre = bounds[pretrained_id].get(metric)
                if b_fine is None or b_pre is None:
                    continue

                shared_vmin = min(b_fine[0], b_pre[0])
                shared_vmax = max(b_fine[1], b_pre[1])
                min_src = (
                    f"{finetuned_id}:{b_fine[2]}"
                    if b_fine[0] <= b_pre[0]
                    else f"{pretrained_id}:{b_pre[2]}"
                )
                max_src = (
                    f"{finetuned_id}:{b_fine[3]}"
                    if b_fine[1] >= b_pre[1]
                    else f"{pretrained_id}:{b_pre[3]}"
                )
                shared = (shared_vmin, shared_vmax, min_src, max_src)
                bounds[finetuned_id][metric] = shared
                bounds[pretrained_id][metric] = shared

    return bounds


def _print_bounds(bounds: dict) -> None:
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
# SECTION 3 – PLOTTING PRIMITIVES
# ==============================================================================


def _draw_heatmap(
    ax,
    matrix: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
):
    image = ax.imshow(
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
    ax.set_xlabel("Text Layers", fontsize=FONT["axis_label"], labelpad=3)
    ax.set_ylabel("Visual Layers", fontsize=FONT["axis_label"], labelpad=3)
    ax.tick_params(labelsize=FONT["tick"], length=3, direction="out")
    return image


def _draw_colorbar(ax, image) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = ax.figure.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=FONT["cbar"], length=2)
    cbar.outline.set_visible(False)


def _draw_missing(ax) -> None:
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
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


# ==============================================================================
# SECTION 4 – FIGURE BUILDERS
# ==============================================================================


def _stage_averages(stage_datasets: list, model_data: dict) -> dict:
    buckets: dict = {}
    for dataset_dir, variant_tag, _label in stage_datasets:
        key = (dataset_dir, variant_tag)
        if key not in model_data:
            continue
        for metric, arr in model_data[key].items():
            buckets.setdefault(metric, []).append(arr)

    result: dict = {}
    for metric, arrays in buckets.items():
        if len(arrays) < 2:
            continue
        if len({arr.shape for arr in arrays}) != 1:
            continue
        result[metric] = np.mean(np.stack(arrays), axis=0)
    return result


def _row_specs(stage_datasets: list, model_data: dict, averages: dict) -> list:
    rows = []
    for dataset_dir, variant_tag, label in stage_datasets:
        key = (dataset_dir, variant_tag)
        if key in model_data:
            rows.append((key, label, model_data[key]))

    if SHOW_AVERAGE and averages and len(rows) > 1:
        rows.append(("average", "Average", averages))

    return rows


def aggregate_model_across_all_datasets(model_data: dict) -> dict:
    buckets: dict = {}
    for dataset_metrics in model_data.values():
        for metric, arr in dataset_metrics.items():
            buckets.setdefault(metric, []).append(arr)

    aggregated: dict = {}
    for metric, arrays in buckets.items():
        if len(arrays) == 1:
            aggregated[metric] = arrays[0]
            continue
        if len({arr.shape for arr in arrays}) != 1:
            continue
        aggregated[metric] = np.mean(np.stack(arrays), axis=0)

    return aggregated


def build_stage_figure(
    model_id: str,
    stage_label: str,
    stage_datasets: list,
    model_data: dict,
    model_bounds: dict,
) -> plt.Figure | None:
    averages = _stage_averages(stage_datasets, model_data)
    rows = _row_specs(stage_datasets, model_data, averages)
    metric_keys = list(METRICS.keys())
    n_rows, n_cols = len(rows), len(metric_keys)

    if n_rows == 0:
        return None

    fig_w = MARGIN_LEFT + n_cols * CELL_W + (n_cols - 1) * CELL_W * WSPACE + MARGIN_RIGHT
    fig_h = MARGIN_TOP + n_rows * CELL_H + (n_rows - 1) * CELL_H * HSPACE + MARGIN_BOTTOM

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left=MARGIN_LEFT / fig_w,
        right=1.0 - MARGIN_RIGHT / fig_w,
        top=1.0 - MARGIN_TOP / fig_h,
        bottom=MARGIN_BOTTOM / fig_h,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    for row_idx, (_row_key, _row_label, row_data) in enumerate(rows):
        for col_idx, metric in enumerate(metric_keys):
            ax = axes[row_idx, col_idx]
            if metric in row_data:
                vmin, vmax = None, None
                if model_bounds and metric in model_bounds:
                    vmin, vmax = model_bounds[metric][0], model_bounds[metric][1]
                image = _draw_heatmap(ax, row_data[metric], vmin=vmin, vmax=vmax)
                _draw_colorbar(ax, image)
            else:
                _draw_missing(ax)

            if row_idx == 0:
                ax.set_title(METRICS[metric], fontsize=FONT["col_header"], pad=8)

    bottom_frac = MARGIN_BOTTOM / fig_h
    top_frac = 1.0 - MARGIN_TOP / fig_h
    grid_span = top_frac - bottom_frac
    effective_row_h = grid_span / (n_rows + max(n_rows - 1, 0) * HSPACE)
    label_x = 0.5 * MARGIN_LEFT / fig_w

    for row_idx, (_row_key, row_label, _row_data) in enumerate(rows):
        row_from_bottom = n_rows - 1 - row_idx
        y0 = bottom_frac + row_from_bottom * effective_row_h * (1 + HSPACE)
        fig.text(
            label_x,
            y0 + effective_row_h / 2,
            row_label,
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
) -> plt.Figure | None:
    row_model_ids = [pretrained_id, finetuned_id]
    metric_keys = list(METRICS.keys())
    n_rows, n_cols = 2, len(metric_keys)

    if not any(aggregated_data.get(model_id) for model_id in row_model_ids):
        return None

    fig_w = MARGIN_LEFT + n_cols * CELL_W + (n_cols - 1) * CELL_W * WSPACE + MARGIN_RIGHT
    fig_h = MARGIN_TOP + n_rows * CELL_H + (n_rows - 1) * CELL_H * HSPACE + MARGIN_BOTTOM

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left=MARGIN_LEFT / fig_w,
        right=1.0 - MARGIN_RIGHT / fig_w,
        top=1.0 - MARGIN_TOP / fig_h,
        bottom=MARGIN_BOTTOM / fig_h,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    for row_idx, model_id in enumerate(row_model_ids):
        row_data = aggregated_data.get(model_id, {})
        row_bounds = bounds.get(model_id, {})

        for col_idx, metric in enumerate(metric_keys):
            ax = axes[row_idx, col_idx]
            if metric in row_data:
                vmin, vmax = None, None
                if row_bounds and metric in row_bounds:
                    vmin, vmax = row_bounds[metric][0], row_bounds[metric][1]
                image = _draw_heatmap(ax, row_data[metric], vmin=vmin, vmax=vmax)
                _draw_colorbar(ax, image)
            else:
                _draw_missing(ax)

            if row_idx == 0:
                ax.set_title(METRICS[metric], fontsize=FONT["col_header"], pad=8)

    bottom_frac = MARGIN_BOTTOM / fig_h
    top_frac = 1.0 - MARGIN_TOP / fig_h
    grid_span = top_frac - bottom_frac
    effective_row_h = grid_span / (n_rows + max(n_rows - 1, 0) * HSPACE)
    label_x = 0.5 * MARGIN_LEFT / fig_w

    for row_idx, model_id in enumerate(row_model_ids):
        row_from_bottom = n_rows - 1 - row_idx
        y0 = bottom_frac + row_from_bottom * effective_row_h * (1 + HSPACE)
        fig.text(
            label_x,
            y0 + effective_row_h / 2,
            MODELS.get(model_id, model_id),
            fontsize=FONT["row_label"],
            ha="center",
            va="center",
            fontweight="bold",
        )

    fig.suptitle(
        "Linear Regression R² — Aggregated Across Selected Datasets",
        fontsize=FONT["suptitle"],
        fontweight="bold",
        y=0.98,
    )
    return fig


# ==============================================================================
# SECTION 5 – MAIN
# ==============================================================================


def _parse_regression_types(raw_types: list[str], pooling_methods: list[str]) -> list[str]:
    normalized = [item.lower().strip() for item in raw_types if item and item.strip()]
    if not normalized or normalized == ["auto"]:
        return discover_regression_types(pooling_methods)

    if any(item == "auto" for item in normalized):
        raise ValueError("'auto' cannot be combined with explicit regression types.")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in normalized:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dedicated linear regression R² heatmaps (metadata-oriented, "
            "backend-separated)."
        )
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
    parser.add_argument(
        "--regression-types",
        nargs="+",
        default=["auto"],
        help=(
            "Regression backend(s) to plot. Use explicit values (e.g. ols ridge), "
            "or 'auto' to discover from existing files."
        ),
    )

    args = parser.parse_args()

    regression_types = _parse_regression_types(args.regression_types, args.pooling_methods)

    norm_info = f"norm={COLORMAP_NORM}"
    if COLORMAP_NORM == "cross_model":
        pairs_str = ", ".join(f"{a}↔{b}" for a, b in CROSS_MODEL_PAIRS)
        norm_info += f" [{pairs_str}]"
    label_info = f" | label={EXPERIMENT_LABEL}" if EXPERIMENT_LABEL else ""

    print("=" * 90)
    print("  per_model_linear_regression_r2_grid")
    print(
        f"  mode={args.figure_mode} | {norm_info}{label_info} "
        f"| regression_types={regression_types}"
    )
    print("=" * 90)

    for regression_type in regression_types:
        print("\n" + "#" * 90)
        print(f"  regression_type={regression_type}")
        print("#" * 90)

        for pooling_method in args.pooling_methods:
            plots_dir = _make_plots_dir(regression_type, pooling_method)
            plots_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "-" * 90)
            print(f"  pooling={pooling_method}")
            print("-" * 90)

            print("\n  Loading data...")
            data = load_all_data(pooling_method, regression_type)

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
                        fig = build_stage_figure(
                            model_id=model_id,
                            stage_label=stage_label,
                            stage_datasets=stage_datasets,
                            model_data=data[model_id],
                            model_bounds=model_bounds,
                        )
                        if fig is None:
                            print("    (no data for this stage — skipped)")
                            continue

                        out = plots_dir / f"{model_id}_{stage_key}.png"
                        fig.savefig(
                            str(out),
                            dpi=DPI_PNG,
                            facecolor="white",
                            bbox_inches="tight",
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
                    if (
                        finetuned_id not in aggregated_data
                        and pretrained_id not in aggregated_data
                    ):
                        continue

                    fig = build_aggregated_pair_figure(
                        finetuned_id=finetuned_id,
                        pretrained_id=pretrained_id,
                        aggregated_data=aggregated_data,
                        bounds=agg_bounds,
                    )
                    if fig is None:
                        print(
                            "    (no aggregated data for pair "
                            f"{finetuned_id} ↔ {pretrained_id} — skipped)"
                        )
                        continue

                    out = plots_dir / f"{finetuned_id}_aggregated_all.png"
                    fig.savefig(
                        str(out),
                        dpi=DPI_PNG,
                        facecolor="white",
                        bbox_inches="tight",
                    )
                    print(f"    ✓  {out}")
                    plt.close(fig)

    print("\n  Done.")


if __name__ == "__main__":
    main()
