"""Plot representations-classification outputs.

Layer metrics are read from safetensors and plotted as decoder-layer curves.
Scalar run metrics are read from run_metrics.jsonl and plotted as cross-model bars.
Aggregate figures can additionally average curves across datasets for each model.
Missing data is skipped; corrupt or conflicting data is reported and skipped.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

HOME_DIR = Path.home()
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = HOME_DIR / "scratch" / "matplotlib"
    try:
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    except OSError:
        pass

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_RESULTS_BASE = (
    HOME_DIR
    / "multimodal_finetuned_representations"
    / "results"
    / "representations_classification"
)
DEFAULT_PLOTS_DIR = (
    HOME_DIR
    / "multimodal_finetuned_representations"
    / "plots"
    / "representations_classification"
)

CONFIG_RESULTS_BASE = DEFAULT_RESULTS_BASE
CONFIG_PLOTS_DIR = DEFAULT_PLOTS_DIR

CONFIG_DATASETS: list[tuple[str, str]] | None = [
    ("scienceqa", "test"),
    ("mmmu", "test"),
    ("mini_imagenet", "validation"),
    ("imagenet_short_text_100_classes_x_100_samples", "train"),
    ("food101", "validation"),
    ("sun397", "test"),
    ("caltech101", "train"),
]
CONFIG_AUTO_DISCOVER_DATASETS = False
CONFIG_AUTO_DISCOVER_POOLING_METHODS = False
CONFIG_POOLING_METHODS: list[str] | None = ["last"]
CONFIG_POOLING_METHODS_FILTER: list[str] | None = None

CONFIG_CLASS_FRACTIONS: list[float | None] | None = [
    0.3,
    # None,
    # 0.1,
    # 0.5,
    # 0.7,
]
CONFIG_MAXK_VALUES: list[int | None] | None = None
CONFIG_N_CHOICES: list[int | None] = [10, 20]
CONFIG_SUBSAMPLE_SIZES: list[int | None] = [
    None,
    1000,
]
CONFIG_CLUSTER_VARIANTS: list[bool] = [
    False,
    # True,
]
CONFIG_LABEL_MODES: list[str] = ["class_name"]

CONFIG_PROMPT_MODES: list[str] = [
    "describe",
    "continuation",
    "animal_mcq_filter",
]

DATASET_PROMPT_MODES: dict[str, list[str]] = {
    "scienceqa": ["describe"],
    "mmmu": ["describe"],
    "mini_imagenet": [
        "describe",
        "continuation",
        "identify",
        "category_id",
        "mcq",
        "mcq_nocat",
        "mcq_nocat_filter",
    ],
    "imagenet_short_text_100_classes_x_100_samples": [
        "describe",
        "continuation",
        "animal_id",
        "animal_mcq",
        "animal_mcq_filter",
    ],
    "food101": [
        "generic_mcq",
        "generic_mcq_filter",
    ],
    "sun397": [
        "generic_mcq",
        "generic_mcq_filter",
    ],
    "caltech101": [
        "generic_mcq",
        "generic_mcq_top_filter",
    ],
}

_PLOT_MCQ_MODES: frozenset[str] = frozenset(
    {
        "mcq",
        "mcq_nocat",
        "mcq_nocat_filter",
        "animal_mcq",
        "animal_mcq_filter",
        "generic_mcq",
        "generic_mcq_filter",
        "generic_mcq_top_filter",
    }
)
_PLOT_FILTER_MODES: frozenset[str] = frozenset(
    {
        "mcq_nocat_filter",
        "animal_mcq_filter",
        "generic_mcq_filter",
        "generic_mcq_top_filter",
    }
)
_SYNTHETIC_MCQ_DATASETS: frozenset[str] = frozenset(
    {
        "mini_imagenet",
        "imagenet_short_text_100_classes_x_100_samples",
        "food101",
        "sun397",
        "caltech101",
    }
)
_FIXED_DATASETS: frozenset[str] = frozenset({"scienceqa", "mmmu"})

CONFIG_GLOBAL_FIGURE = False
CONFIG_INDIVIDUAL_METRIC_FIGURES = True
CONFIG_SCALAR_METRICS: list[str] = ["accuracy", "clip_label_overlap"]
CONFIG_PLOT_DPI = 150

LAYER_COLUMN_SPECS: dict[str, tuple[str, str, str]] = {
    "logistic_regression_f1": (
        "logistic_regression",
        "f1_weighted",
        "LR - F1 (weighted)",
    ),
    "logistic_regression_f1_train": (
        "logistic_regression",
        "f1_weighted_train",
        "LR - F1 Train (weighted)",
    ),
    "logistic_regression_mcc": ("logistic_regression", "mcc", "LR - MCC"),
    "knn_f1": ("knn", "knn_f1_weighted", "KNN - F1 (weighted)"),
    "knn_mcc": ("knn", "knn_mcc", "KNN - MCC"),
    "label_overlap": ("label_overlap", "label_overlap", "Label Overlap"),
    "homogeneity": ("homogeneity", "homogeneity_score", "Homogeneity"),
    "adjusted_rand": ("homogeneity", "adjusted_rand_score", "Adjusted Rand Index"),
    "n_clusters_ratio": ("homogeneity", "n_clusters_ratio", "N Clusters / N Labels"),
    "clip_neighborhood_overlap": (
        "clip_neighborhood_overlap",
        "clip_neighborhood_overlap",
        "CLIP Neighborhood Overlap",
    ),
}

CONFIG_LAYER_COLUMNS: list[str] = [
    "label_overlap",
    "clip_neighborhood_overlap",
    # "logistic_regression_f1",
    # "logistic_regression_f1_train",
    # "logistic_regression_mcc",
    # "knn_f1",
    # "knn_mcc",
    # "homogeneity",
    # "adjusted_rand",
    # "n_clusters_ratio",
]
CONFIG_DATASET_MEAN_FIGURES = True
CONFIG_DATASET_MEAN_FAMILIES: list[str] = [
    "mcq_option_width",
    "mcq_filter_classes",
]
CONFIG_DATASET_MEAN_N_CHOICES: list[int] = [10, 20]

# Dataset-mean aggregate uncertainty bands.
# Set CONFIG_DATASET_MEAN_STD_BAND=True to save *_std_band.png figures with
# mean ± 1 standard deviation shading across datasets for each model.
CONFIG_DATASET_MEAN_STD_BAND = True
CONFIG_DATASET_MEAN_STD_BAND_ALPHA = 0.18

SCALAR_METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "clip_label_overlap": "CLIP Label Overlap",
}

_MCC_COLUMNS = {"logistic_regression_mcc", "knn_mcc", "adjusted_rand"}
_AUTOSCALE_COLUMNS = {"n_clusters_ratio"}

ROW1_MODELS = [
    "lbasile/llava_7b",
    "l11p/llava-v1.5-7b-pretrained-lbasile",
]
ROW2_MODELS = [
    "lbasile/llava_13b",
    "l11p/llava-v1.5-13b-pretrained-lbasile",
]
ROW3_MODELS = [
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
    "lmms-lab/LLaVA-OneVision-1.5-4B-Base",
]
ROW4_MODELS = [
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    "lmms-lab/LLaVA-OneVision-1.5-8B-Base",
]
ROW5_MODELS = [
    "OpenGVLab/InternVL2-8B",
    "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain",
]
ROW6_MODELS = [
    "nyu-visionx/cambrian-8b",
    "lbasile/cambrian-8b-pretrained",
]

ROWS: list[tuple[list[str], str]] = [
    (ROW1_MODELS, "LLaVA 1.5 7B family"),
    (ROW2_MODELS, "LLaVA 1.5 13B family"),
    (ROW3_MODELS, "OV 4B family"),
    (ROW4_MODELS, "OV 8B family"),
    (ROW5_MODELS, "InternVL2 8B family"),
    (ROW6_MODELS, "Cambrian 8B family"),
]

DATASET_MEAN_PROMPT_MAP: dict[str, list[tuple[str, str, str]]] = {
    "mcq_option_width": [
        ("mini_imagenet", "validation", "mcq_nocat"),
        ("imagenet_short_text_100_classes_x_100_samples", "train", "animal_mcq"),
        ("food101", "validation", "generic_mcq"),
        ("sun397", "test", "generic_mcq"),
        ("caltech101", "train", "generic_mcq"),
    ],
    "mcq_filter_classes": [
        ("mini_imagenet", "validation", "mcq_nocat_filter"),
        ("imagenet_short_text_100_classes_x_100_samples", "train", "animal_mcq_filter"),
        ("food101", "validation", "generic_mcq_filter"),
        ("sun397", "test", "generic_mcq_filter"),
        ("caltech101", "train", "generic_mcq_top_filter"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate layerwise and scalar representations-classification plots."
    )
    parser.add_argument("--results-base", type=Path, default=CONFIG_RESULTS_BASE)
    parser.add_argument("--plots-dir", type=Path, default=CONFIG_PLOTS_DIR)
    parser.add_argument(
        "--pooling-methods", type=str, nargs="+", default=CONFIG_POOLING_METHODS
    )
    parser.add_argument(
        "--auto-discover-datasets",
        action=argparse.BooleanOptionalAction,
        default=CONFIG_AUTO_DISCOVER_DATASETS,
    )
    parser.add_argument(
        "--auto-discover-pooling-methods",
        action=argparse.BooleanOptionalAction,
        default=CONFIG_AUTO_DISCOVER_POOLING_METHODS,
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=CONFIG_LAYER_COLUMNS,
        choices=list(LAYER_COLUMN_SPECS.keys()),
        help="Layerwise columns to include. Missing files are skipped.",
    )
    parser.add_argument(
        "--scalar-metrics",
        type=str,
        nargs="+",
        default=CONFIG_SCALAR_METRICS,
        choices=list(SCALAR_METRIC_LABELS.keys()),
        help="Scalar JSONL metrics to plot as cross-model bar charts.",
    )
    parser.add_argument("--icl-shots", type=int, default=0)
    parser.add_argument("--class-fraction", type=float, default=None)
    parser.add_argument("--maxk", type=int, default=None)
    parser.add_argument("--n-choices", type=int, default=None)
    parser.add_argument("--subsample-size", type=int, default=None)
    parser.add_argument(
        "--dataset-mean-std-band",
        action=argparse.BooleanOptionalAction,
        default=CONFIG_DATASET_MEAN_STD_BAND,
        help=(
            "For dataset-mean aggregate figures, optionally draw a shaded "
            "mean ± 1 standard deviation band across datasets for each model."
        ),
    )
    parser.add_argument(
        "--dataset-mean-std-alpha",
        type=float,
        default=CONFIG_DATASET_MEAN_STD_BAND_ALPHA,
        help="Alpha value for the optional dataset-mean standard-deviation band.",
    )
    return parser.parse_args()


def make_model_slug(model_name: str) -> str:
    return Path(model_name).name


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _all_models() -> list[str]:
    models: list[str] = []
    for row_models, _ in ROWS:
        models.extend(row_models)
    return models


def _configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fontsize": 8,
        }
    )


def _build_color_map(models: list[str]) -> dict[str, tuple[float, float, float, float]]:
    palette = plt.colormaps["tab20"]
    return {
        make_model_slug(model_name): palette((index % 20 + 0.5) / 20)
        for index, model_name in enumerate(models)
    }


_CANDIDATE_SPLITS = ("test", "validation", "train")


def _discover_datasets(results_base: Path) -> list[tuple[str, str]]:
    if not results_base.exists():
        return []
    pairs: list[tuple[str, str]] = []
    for candidate in sorted(results_base.iterdir()):
        if not candidate.is_dir():
            continue
        for split in _CANDIDATE_SPLITS:
            if (candidate / split).exists():
                pairs.append((candidate.name, split))
                break
    return pairs


def _discover_pooling_methods(
    results_base: Path,
    dataset_slug: str,
    split: str,
) -> list[str]:
    split_dir = results_base / dataset_slug / split
    if not split_dir.exists():
        return []
    pooling_methods: set[str] = set()
    for model_dir in split_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for pooling_dir in model_dir.iterdir():
            if pooling_dir.is_dir():
                pooling_methods.add(pooling_dir.name)
    return sorted(pooling_methods)


def _resolve_datasets(
    args: argparse.Namespace, results_base: Path
) -> list[tuple[str, str]]:
    if CONFIG_DATASETS is not None:
        return CONFIG_DATASETS
    if not args.auto_discover_datasets:
        print("No datasets configured. Set CONFIG_DATASETS or enable auto-discovery.")
        return []
    return _discover_datasets(results_base)


def _resolve_poolings(
    args: argparse.Namespace,
    results_base: Path,
    dataset_slug: str,
    dataset_split: str,
) -> list[str]:
    if args.pooling_methods:
        return args.pooling_methods
    if not args.auto_discover_pooling_methods:
        return []
    return _discover_pooling_methods(results_base, dataset_slug, dataset_split)


def _settings_label(
    dataset_slug: str,
    split: str,
    pooling_method: str,
    suffix: str,
    label_overlap_suffix: str = "",
) -> str:
    setting = suffix.lstrip("_") if suffix else "describe/default"
    lo = f" | lo={label_overlap_suffix.lstrip('_')}" if label_overlap_suffix else ""
    return f"dataset={dataset_slug} | split={split} | pooling={pooling_method} | {setting}{lo}"


def _format_param_value(value: float | int) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _setup_segments(
    *,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
) -> list[str]:
    segments: list[str] = []
    if icl_shots > 0:
        segments.append(f"{icl_shots}shot")
    if n_choices is not None:
        segments.append(f"n{n_choices}")
    if subsample_size is not None:
        segments.append(f"ss{subsample_size}")
    segments.append(f"label_{label_mode}")
    segments.append("cluster_labels" if cluster_labels else "raw_labels")
    return segments


def _metric_param_segments(
    *,
    class_fraction: float | None,
    maxk: int | None,
) -> list[str]:
    segments: list[str] = []
    if class_fraction is not None:
        segments.append(f"cf{_format_param_value(class_fraction)}")
    if maxk is not None:
        segments.append(f"mk{maxk}")
    return segments or ["default"]


def _build_suffix(
    *,
    dataset_slug: str,
    prompt_mode: str,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
) -> str:
    icl_suffix = f"_{icl_shots}shot" if icl_shots > 0 else ""
    n_suffix = f"_n{n_choices}" if n_choices is not None else ""
    ss_suffix = f"_ss{subsample_size}" if subsample_size is not None else ""
    lbl_suffix = "_lbl_letter" if label_mode == "option_letter" else ""
    cluster_suffix = "_cluster" if cluster_labels else ""
    return (
        icl_suffix
        + _prompt_suffix(dataset_slug, prompt_mode)
        + n_suffix
        + ss_suffix
        + lbl_suffix
        + cluster_suffix
    )


def _plot_output_path(
    *,
    plots_dir: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    prompt_mode: str,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    metric_name: str,
    class_fraction: float | None,
    maxk: int | None,
    file_name: str | None = None,
) -> Path:
    return (
        plots_dir
        / dataset_slug
        / split
        / pooling_method
        / prompt_mode
        / Path(
            *_setup_segments(
                icl_shots=icl_shots,
                n_choices=n_choices,
                subsample_size=subsample_size,
                label_mode=label_mode,
                cluster_labels=cluster_labels,
            )
        )
        / metric_name
        / Path(
            *_metric_param_segments(class_fraction=class_fraction, maxk=maxk)
        )
        / f"{file_name or metric_name}.png"
    )


def _aggregate_output_path(
    *,
    plots_dir: Path,
    family: str,
    pooling_method: str,
    metric_name: str,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    class_fraction: float | None,
    maxk: int | None,
    file_name: str | None = None,
) -> Path:
    return (
        plots_dir
        / "aggregates"
        / "dataset_means"
        / family
        / pooling_method
        / metric_name
        / Path(
            *_setup_segments(
                icl_shots=icl_shots,
                n_choices=n_choices,
                subsample_size=subsample_size,
                label_mode=label_mode,
                cluster_labels=cluster_labels,
            )
        )
        / Path(
            *_metric_param_segments(class_fraction=class_fraction, maxk=maxk)
        )
        / f"{file_name or metric_name}.png"
    )


def _load_layer_values(
    *,
    results_base: Path,
    dataset_slug: str,
    split: str,
    model_slug: str,
    pooling_method: str,
    file_name: str,
    tensor_key: str,
    suffix: str,
) -> np.ndarray | None:
    file_path = (
        results_base
        / dataset_slug
        / split
        / model_slug
        / pooling_method
        / f"{file_name}{suffix}.safetensors"
    )
    if not file_path.exists():
        return None
    try:
        with safe_open(str(file_path), framework="numpy") as handle:
            if tensor_key not in handle.keys():
                _warn(f"Tensor key {tensor_key!r} missing in {file_path}")
                return None
            values = handle.get_tensor(tensor_key)
    except Exception as exc:
        _warn(f"Could not load {file_path}: {exc}")
        return None
    if values.size == 0:
        _warn(f"Tensor key {tensor_key!r} is empty in {file_path}")
        return None
    return values


LayerData = dict[str, list[tuple[str, str, np.ndarray]]]


def _collect_layer_data(
    *,
    results_base: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    selected_columns: list[str],
    suffix: str,
    label_overlap_suffix: str,
) -> LayerData:
    data_by_column: LayerData = {}
    for column_name in selected_columns:
        file_name, tensor_key, _ = LAYER_COLUMN_SPECS[column_name]
        effective_suffix = suffix
        if file_name == "label_overlap":
            effective_suffix += label_overlap_suffix
        column_data: list[tuple[str, str, np.ndarray]] = []
        for models, row_label in ROWS:
            for model_name in models:
                model_slug = make_model_slug(model_name)
                values = _load_layer_values(
                    results_base=results_base,
                    dataset_slug=dataset_slug,
                    split=split,
                    model_slug=model_slug,
                    pooling_method=pooling_method,
                    file_name=file_name,
                    tensor_key=tensor_key,
                    suffix=effective_suffix,
                )
                if values is not None:
                    column_data.append((row_label, model_slug, values))
        if column_data:
            data_by_column[column_name] = column_data
    return data_by_column


def _row_has_column_data(
    data_by_column: LayerData,
    row_label: str,
) -> bool:
    return any(
        any(item_row == row_label for item_row, _, _ in column_data)
        for column_data in data_by_column.values()
    )


DatasetMeanCurve = tuple[np.ndarray, np.ndarray, int]


def _strict_mean_std_curve(
    curves: list[np.ndarray],
    *,
    context: str,
) -> DatasetMeanCurve | None:
    if not curves:
        return None
    lengths = {len(curve) for curve in curves}
    if len(lengths) != 1:
        _warn(f"Cannot average curves with different lengths for {context}: {sorted(lengths)}")
        return None
    stacked = np.stack(curves, axis=0)
    mean_curve = np.mean(stacked, axis=0)
    if len(curves) > 1:
        std_curve = np.std(stacked, axis=0, ddof=1)
    else:
        std_curve = np.zeros_like(mean_curve)
    return mean_curve, std_curve, len(curves)


def _plot_layer_axis(
    ax,
    *,
    column_name: str,
    row_label: str,
    column_data: list[tuple[str, str, np.ndarray]],
    color_map: dict[str, tuple[float, float, float, float]],
    show_legend: bool,
) -> None:
    _, _, y_label = LAYER_COLUMN_SPECS[column_name]
    plotted = False
    for item_row, model_slug, values in column_data:
        if item_row != row_label:
            continue
        layers = np.arange(len(values))
        ax.plot(
            layers,
            values,
            marker="o",
            markersize=3,
            linewidth=2,
            color=color_map[model_slug],
            alpha=0.95,
            label=model_slug,
        )
        plotted = True

    if not plotted:
        ax.set_visible(False)
        return

    ax.set_title(y_label, fontsize=12)
    ax.set_xlabel("Decoder Layer", fontsize=10)
    ax.set_ylabel(f"{y_label} ({row_label})", fontsize=10)
    if column_name not in _AUTOSCALE_COLUMNS:
        ax.set_ylim(-1.0, 1.0) if column_name in _MCC_COLUMNS else ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=9)
    if plotted and show_legend:
        ax.legend(fontsize=8, loc="best")


def _generate_layer_grid(
    *,
    results_base: Path,
    plots_dir: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    prompt_mode: str,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    selected_columns: list[str],
    suffix: str,
    label_overlap_suffix: str,
    class_fraction: float | None,
    maxk: int | None,
) -> bool:
    data_by_column = _collect_layer_data(
        results_base=results_base,
        dataset_slug=dataset_slug,
        split=split,
        pooling_method=pooling_method,
        selected_columns=selected_columns,
        suffix=suffix,
        label_overlap_suffix=label_overlap_suffix,
    )
    if not data_by_column:
        return False

    active_columns = list(data_by_column)
    active_rows = [
        (models, row_label)
        for models, row_label in ROWS
        if _row_has_column_data(data_by_column, row_label)
    ]
    if not active_rows:
        return False

    fig, axes = plt.subplots(
        len(active_rows),
        len(active_columns),
        figsize=(7 * len(active_columns), 5.5 * len(active_rows)),
        squeeze=False,
    )
    for row_idx, (models, row_label) in enumerate(active_rows):
        color_map = _build_color_map(models)
        for col_idx, column_name in enumerate(active_columns):
            _plot_layer_axis(
                axes[row_idx, col_idx],
                column_name=column_name,
                row_label=row_label,
                column_data=data_by_column[column_name],
                color_map=color_map,
                show_legend=col_idx == 0,
            )

    fig.suptitle(
        "Representations Classification Metrics | "
        + _settings_label(
            dataset_slug, split, pooling_method, suffix, label_overlap_suffix
        ),
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = _plot_output_path(
        plots_dir=plots_dir,
        dataset_slug=dataset_slug,
        split=split,
        pooling_method=pooling_method,
        prompt_mode=prompt_mode,
        icl_shots=icl_shots,
        n_choices=n_choices,
        subsample_size=subsample_size,
        label_mode=label_mode,
        cluster_labels=cluster_labels,
        metric_name="global_layer_metrics",
        class_fraction=class_fraction,
        maxk=maxk,
        file_name="representations_classification",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=CONFIG_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


def _generate_individual_layer_plots(
    *,
    results_base: Path,
    plots_dir: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    prompt_mode: str,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    selected_columns: list[str],
    suffix: str,
    label_overlap_suffix: str,
    class_fraction: float | None,
    maxk: int | None,
) -> int:
    data_by_column = _collect_layer_data(
        results_base=results_base,
        dataset_slug=dataset_slug,
        split=split,
        pooling_method=pooling_method,
        selected_columns=selected_columns,
        suffix=suffix,
        label_overlap_suffix=label_overlap_suffix,
    )
    generated = 0
    for column_name, column_data in data_by_column.items():
        active_rows = [
            (models, row_label)
            for models, row_label in ROWS
            if any(item_row == row_label for item_row, _, _ in column_data)
        ]
        if not active_rows:
            continue
        _, _, y_label = LAYER_COLUMN_SPECS[column_name]
        fig, axes = plt.subplots(
            1,
            len(active_rows),
            figsize=(7 * len(active_rows), 5.5),
            squeeze=False,
        )
        for row_idx, (models, row_label) in enumerate(active_rows):
            color_map = _build_color_map(models)
            _plot_layer_axis(
                axes[0, row_idx],
                column_name=column_name,
                row_label=row_label,
                column_data=column_data,
                color_map=color_map,
                show_legend=True,
            )
            axes[0, row_idx].set_title(row_label, fontsize=12)
            axes[0, row_idx].set_ylabel(y_label, fontsize=10)

        fig.suptitle(
            y_label
            + " | "
            + _settings_label(
                dataset_slug, split, pooling_method, suffix, label_overlap_suffix
            ),
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        file_name, _, _ = LAYER_COLUMN_SPECS[column_name]
        metric_class_fraction = (
            class_fraction if file_name == "label_overlap" else None
        )
        metric_maxk = maxk if file_name == "label_overlap" else None
        out_path = _plot_output_path(
            plots_dir=plots_dir,
            dataset_slug=dataset_slug,
            split=split,
            pooling_method=pooling_method,
            prompt_mode=prompt_mode,
            icl_shots=icl_shots,
            n_choices=n_choices,
            subsample_size=subsample_size,
            label_mode=label_mode,
            cluster_labels=cluster_labels,
            metric_name=column_name,
            class_fraction=metric_class_fraction,
            maxk=metric_maxk,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=CONFIG_PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        generated += 1
    return generated


def _run_metrics_path(
    results_base: Path,
    dataset_slug: str,
    split: str,
    model_slug: str,
) -> Path:
    return results_base / dataset_slug / split / model_slug / "run_metrics.jsonl"


MetricRecord = tuple[int, dict[str, Any]]


def _read_run_metric_records(path: Path) -> list[MetricRecord]:
    if not path.exists():
        return []
    records: list[MetricRecord] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _warn(f"Malformed JSON in {path}:{line_number}: {exc}")
                continue
            if not isinstance(record, dict):
                _warn(f"Non-object JSON record in {path}:{line_number}")
                continue
            records.append((line_number, record))
    return records


def _same_optional_int(record_value: Any, expected: int | None) -> bool:
    if expected is None:
        return record_value is None
    try:
        return int(record_value) == expected
    except (TypeError, ValueError):
        return False


def _same_optional_float(record_value: Any, expected: float | None) -> bool:
    if expected is None:
        return record_value is None
    try:
        return np.isclose(float(record_value), expected)
    except (TypeError, ValueError):
        return False


def _record_matches_scalar_setup(
    record: dict[str, Any],
    *,
    metric: str,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    prompt_mode: str,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    class_fraction: float | None,
    maxk: int | None,
) -> bool:
    if record.get("metric") != metric:
        return False
    if record.get("dataset_slug") != dataset_slug or record.get("split") != split:
        return False
    if record.get("pooling_method") != pooling_method:
        return False
    if record.get("prompt_mode") != prompt_mode:
        return False
    if record.get("label_mode") != label_mode:
        return False
    if bool(record.get("cluster_labels", False)) != cluster_labels:
        return False
    if not _same_optional_int(record.get("n_choices"), n_choices):
        return False
    if not _same_optional_int(record.get("subsample_size"), subsample_size):
        return False
    if metric != "clip_label_overlap":
        return True
    if not _same_optional_float(record.get("class_fraction"), class_fraction):
        return False
    if maxk is None:
        return record.get("maxk") in (None, 30, "30")
    return _same_optional_int(record.get("maxk"), maxk)


def _select_scalar_value(
    records: list[MetricRecord],
    *,
    metric: str,
    model_slug: str,
    path: Path,
) -> float | None:
    if not records:
        return None
    for line_number, record in reversed(records):
        try:
            return float(record["value"])
        except (KeyError, TypeError, ValueError):
            _warn(
                f"Skipping newest malformed {metric} record for {model_slug} "
                f"in {path}:{line_number}; falling back to older matches."
            )
    _warn(
        f"No numeric {metric} value found for {model_slug} in matching records from {path}"
    )
    return None


def _collect_scalar_values(
    *,
    results_base: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    metric: str,
    prompt_mode: str,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    class_fraction: float | None,
    maxk: int | None,
) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for model_name in _all_models():
        model_slug = make_model_slug(model_name)
        path = _run_metrics_path(results_base, dataset_slug, split, model_slug)
        matches = [
            (line_number, record)
            for line_number, record in _read_run_metric_records(path)
            if _record_matches_scalar_setup(
                record,
                metric=metric,
                dataset_slug=dataset_slug,
                split=split,
                pooling_method=pooling_method,
                prompt_mode=prompt_mode,
                n_choices=n_choices,
                subsample_size=subsample_size,
                label_mode=label_mode,
                cluster_labels=cluster_labels,
                class_fraction=class_fraction,
                maxk=maxk,
            )
        ]
        value = _select_scalar_value(
            matches, metric=metric, model_slug=model_slug, path=path
        )
        if value is not None:
            values.append((model_slug, value))
    return values


def _dataset_mean_components(
    family: str,
    available_dataset_pairs: set[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    return [
        component
        for component in DATASET_MEAN_PROMPT_MAP.get(family, [])
        if (component[0], component[1]) in available_dataset_pairs
    ]


def _collect_dataset_mean_model_curves(
    *,
    results_base: Path,
    family: str,
    available_dataset_pairs: set[tuple[str, str]],
    pooling_method: str,
    column_name: str,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    class_fraction: float | None,
    maxk: int | None,
) -> dict[str, DatasetMeanCurve]:
    components = _dataset_mean_components(family, available_dataset_pairs)
    if not components:
        return {}

    file_name, tensor_key, _ = LAYER_COLUMN_SPECS[column_name]
    label_overlap_suffix = (
        _label_overlap_suffix(class_fraction, maxk)
        if file_name == "label_overlap"
        else ""
    )
    component_hits = {
        f"{dataset_slug}:{split}:{prompt_mode}": 0
        for dataset_slug, split, prompt_mode in components
    }
    model_mean_curves: dict[str, DatasetMeanCurve] = {}

    for model_name in _all_models():
        model_slug = make_model_slug(model_name)
        curves: list[np.ndarray] = []
        component_names: list[str] = []
        for dataset_slug, split, prompt_mode in components:
            suffix = _build_suffix(
                dataset_slug=dataset_slug,
                prompt_mode=prompt_mode,
                icl_shots=icl_shots,
                n_choices=n_choices,
                subsample_size=subsample_size,
                label_mode=label_mode,
                cluster_labels=cluster_labels,
            )
            values = _load_layer_values(
                results_base=results_base,
                dataset_slug=dataset_slug,
                split=split,
                model_slug=model_slug,
                pooling_method=pooling_method,
                file_name=file_name,
                tensor_key=tensor_key,
                suffix=suffix + label_overlap_suffix,
            )
            if values is None:
                continue
            curves.append(values)
            component_key = f"{dataset_slug}:{split}:{prompt_mode}"
            component_names.append(component_key)
            component_hits[component_key] += 1

        mean_curve = _strict_mean_std_curve(
            curves,
            context=(
                f"dataset-mean {column_name} for {model_slug} "
                f"(family={family}, components={component_names})"
            ),
        )
        if mean_curve is not None:
            model_mean_curves[model_slug] = mean_curve

    for component_key, hits in component_hits.items():
        if hits == 0:
            _warn(
                f"No usable {column_name} data for dataset-mean component {component_key} "
                f"(family={family}, pooling={pooling_method}, n_choices={n_choices}, "
                f"subsample_size={subsample_size}, label_mode={label_mode}, "
                f"cluster_labels={cluster_labels})"
            )

    return model_mean_curves


def _generate_dataset_mean_figure(
    *,
    results_base: Path,
    plots_dir: Path,
    available_dataset_pairs: set[tuple[str, str]],
    family: str,
    pooling_method: str,
    column_name: str,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    class_fraction: float | None,
    maxk: int | None,
    show_std_band: bool,
    std_band_alpha: float,
) -> bool:
    model_mean_curves = _collect_dataset_mean_model_curves(
        results_base=results_base,
        family=family,
        available_dataset_pairs=available_dataset_pairs,
        pooling_method=pooling_method,
        column_name=column_name,
        icl_shots=icl_shots,
        n_choices=n_choices,
        subsample_size=subsample_size,
        label_mode=label_mode,
        cluster_labels=cluster_labels,
        class_fraction=class_fraction,
        maxk=maxk,
    )
    if not model_mean_curves:
        return False

    active_rows = [
        (models, row_label)
        for models, row_label in ROWS
        if any(make_model_slug(model_name) in model_mean_curves for model_name in models)
    ]
    if not active_rows:
        return False

    _, _, y_label = LAYER_COLUMN_SPECS[column_name]
    fig, axes = plt.subplots(
        1,
        len(active_rows),
        figsize=(7 * len(active_rows), 5.5),
        squeeze=False,
    )
    for row_idx, (models, row_label) in enumerate(active_rows):
        color_map = _build_color_map(models)
        row_curves = [
            (make_model_slug(model_name), model_mean_curves[make_model_slug(model_name)])
            for model_name in models
            if make_model_slug(model_name) in model_mean_curves
        ]
        if not row_curves:
            axes[0, row_idx].set_visible(False)
            continue
        for model_slug, (values, std_values, n_components) in row_curves:
            layers = np.arange(len(values))
            color = color_map[model_slug]
            if show_std_band and n_components > 1:
                axes[0, row_idx].fill_between(
                    layers,
                    values - std_values,
                    values + std_values,
                    color=color,
                    alpha=std_band_alpha,
                    linewidth=0,
                )
            axes[0, row_idx].plot(
                layers,
                values,
                marker="o",
                markersize=3,
                linewidth=2,
                color=color,
                alpha=0.95,
                label=model_slug,
            )
        axes[0, row_idx].set_title(row_label, fontsize=12)
        axes[0, row_idx].set_xlabel("Decoder Layer", fontsize=10)
        axes[0, row_idx].set_ylabel(y_label, fontsize=10)
        if column_name not in _AUTOSCALE_COLUMNS:
            axes[0, row_idx].set_ylim(-1.0, 1.0) if column_name in _MCC_COLUMNS else axes[0, row_idx].set_ylim(0.0, 1.0)
        axes[0, row_idx].tick_params(axis="both", which="major", labelsize=9)
        axes[0, row_idx].legend(fontsize=8, loc="best")

    components = _dataset_mean_components(family, available_dataset_pairs)
    component_label = ", ".join(
        f"{dataset_slug}/{prompt_mode}"
        for dataset_slug, _, prompt_mode in components
    )
    fig.suptitle(
        f"{y_label} | dataset-means={family} | pooling={pooling_method} | "
        f"n_choices={n_choices} | subsample={subsample_size} | label={label_mode} | "
        f"cluster={cluster_labels} | sources={component_label}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    file_name, _, _ = LAYER_COLUMN_SPECS[column_name]
    metric_class_fraction = class_fraction if file_name == "label_overlap" else None
    metric_maxk = maxk if file_name == "label_overlap" else None
    out_path = _aggregate_output_path(
        plots_dir=plots_dir,
        family=family,
        pooling_method=pooling_method,
        metric_name=column_name,
        icl_shots=icl_shots,
        n_choices=n_choices,
        subsample_size=subsample_size,
        label_mode=label_mode,
        cluster_labels=cluster_labels,
        class_fraction=metric_class_fraction,
        maxk=metric_maxk,
        file_name=f"{column_name}_std_band" if show_std_band else None,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=CONFIG_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


def _generate_scalar_plot(
    *,
    results_base: Path,
    plots_dir: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    metric: str,
    prompt_mode: str,
    icl_shots: int,
    n_choices: int | None,
    subsample_size: int | None,
    label_mode: str,
    cluster_labels: bool,
    suffix: str,
    class_fraction: float | None,
    maxk: int | None,
    label_overlap_suffix: str,
) -> bool:
    values = _collect_scalar_values(
        results_base=results_base,
        dataset_slug=dataset_slug,
        split=split,
        pooling_method=pooling_method,
        metric=metric,
        prompt_mode=prompt_mode,
        n_choices=n_choices,
        subsample_size=subsample_size,
        label_mode=label_mode,
        cluster_labels=cluster_labels,
        class_fraction=class_fraction,
        maxk=maxk,
    )
    if not values:
        return False

    model_slugs = [model_slug for model_slug, _ in values]
    metric_values = [value for _, value in values]
    color_map = _build_color_map(_all_models())
    colors = [color_map.get(model_slug, "#4c72b0") for model_slug in model_slugs]

    fig_width = max(9, 0.75 * len(model_slugs))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    positions = np.arange(len(model_slugs))
    ax.bar(positions, metric_values, color=colors, width=0.72)
    ax.set_xticks(positions)
    ax.set_xticklabels(model_slugs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(SCALAR_METRIC_LABELS[metric])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        SCALAR_METRIC_LABELS[metric]
        + " | "
        + _settings_label(
            dataset_slug, split, pooling_method, suffix, label_overlap_suffix
        ),
        fontsize=13,
        fontweight="bold",
    )
    for pos, value in zip(positions, metric_values):
        ax.text(
            pos, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8
    )
    fig.tight_layout()

    out_path = _plot_output_path(
        plots_dir=plots_dir,
        dataset_slug=dataset_slug,
        split=split,
        pooling_method=pooling_method,
        prompt_mode=prompt_mode,
        icl_shots=icl_shots,
        n_choices=n_choices,
        subsample_size=subsample_size,
        label_mode=label_mode,
        cluster_labels=cluster_labels,
        metric_name=metric,
        class_fraction=class_fraction,
        maxk=maxk,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=CONFIG_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


def _prompt_suffix(dataset_slug: str, prompt_mode: str) -> str:
    is_imagenet_animals = "imagenet_short_text" in dataset_slug
    is_generic = any(ds in dataset_slug for ds in ("food101", "sun397", "caltech101"))
    if prompt_mode != "describe" or is_imagenet_animals or is_generic:
        return f"_{prompt_mode}"
    return ""


def _is_synthetic_mcq_setup(dataset_slug: str, prompt_mode: str) -> bool:
    return dataset_slug in _SYNTHETIC_MCQ_DATASETS and prompt_mode in _PLOT_MCQ_MODES


def _label_overlap_suffix(class_fraction: float | None, maxk: int | None) -> str:
    parts: list[str] = []
    if class_fraction is not None:
        parts.append(f"cf{class_fraction}")
    if maxk is not None:
        parts.append(f"mk{maxk}")
    return "_" + "_".join(parts) if parts else ""


def main() -> None:
    args = parse_args()
    results_base = args.results_base.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()
    _configure_plot_style()

    datasets = _resolve_datasets(args, results_base)
    if not datasets:
        print("No datasets found.")
        return

    class_fractions = (
        CONFIG_CLASS_FRACTIONS
        if CONFIG_CLASS_FRACTIONS is not None
        else ([args.class_fraction] if args.class_fraction is not None else [None])
    )
    maxk_values = (
        CONFIG_MAXK_VALUES
        if CONFIG_MAXK_VALUES is not None
        else ([args.maxk] if args.maxk is not None else [None])
    )
    n_choices_list = (
        CONFIG_N_CHOICES
        if CONFIG_N_CHOICES is not None
        else ([args.n_choices] if args.n_choices is not None else [None])
    )
    subsample_sizes = (
        CONFIG_SUBSAMPLE_SIZES
        if CONFIG_SUBSAMPLE_SIZES is not None
        else ([args.subsample_size] if args.subsample_size is not None else [None])
    )
    label_modes = CONFIG_LABEL_MODES if CONFIG_LABEL_MODES else ["class_name"]

    total_generated = 0
    available_dataset_pairs = set(datasets)
    all_pooling_methods: set[str] = set()
    for dataset_slug, dataset_split in datasets:
        pooling_methods = _resolve_poolings(
            args, results_base, dataset_slug, dataset_split
        )
        if CONFIG_POOLING_METHODS_FILTER is not None:
            pooling_methods = [
                p for p in pooling_methods if p in CONFIG_POOLING_METHODS_FILTER
            ]
        if not pooling_methods:
            continue
        all_pooling_methods.update(pooling_methods)

        for prompt_mode in DATASET_PROMPT_MODES.get(dataset_slug, CONFIG_PROMPT_MODES):
            is_mcq = prompt_mode in _PLOT_MCQ_MODES
            is_synthetic_mcq = _is_synthetic_mcq_setup(dataset_slug, prompt_mode)
            is_filter = prompt_mode in _PLOT_FILTER_MODES
            local_n_list = n_choices_list if is_mcq else [None]
            local_ss_list = (
                [None]
                if is_filter or dataset_slug in _FIXED_DATASETS
                else subsample_sizes
            )
            local_label_modes = label_modes if is_mcq else ["class_name"]

            for n_choices in local_n_list:
                for subsample in local_ss_list:
                    for label_mode in local_label_modes:
                        for cluster in CONFIG_CLUSTER_VARIANTS:
                            suffix = _build_suffix(
                                dataset_slug=dataset_slug,
                                prompt_mode=prompt_mode,
                                icl_shots=args.icl_shots,
                                n_choices=n_choices if is_mcq else None,
                                subsample_size=subsample,
                                label_mode=label_mode,
                                cluster_labels=cluster,
                            )

                            one_time_layer_columns = [
                                c
                                for c in args.columns
                                if LAYER_COLUMN_SPECS[c][0] != "label_overlap"
                            ]
                            label_overlap_columns = [
                                c
                                for c in args.columns
                                if LAYER_COLUMN_SPECS[c][0] == "label_overlap"
                            ]

                            for pooling_method in pooling_methods:
                                if one_time_layer_columns:
                                    layer_kwargs = dict(
                                        results_base=results_base,
                                        plots_dir=plots_dir,
                                        dataset_slug=dataset_slug,
                                        split=dataset_split,
                                        pooling_method=pooling_method,
                                        prompt_mode=prompt_mode,
                                        icl_shots=args.icl_shots,
                                        n_choices=n_choices,
                                        subsample_size=subsample,
                                        label_mode=label_mode,
                                        cluster_labels=cluster,
                                        selected_columns=one_time_layer_columns,
                                        suffix=suffix,
                                        label_overlap_suffix="",
                                        class_fraction=None,
                                        maxk=None,
                                    )
                                    if CONFIG_GLOBAL_FIGURE:
                                        total_generated += int(
                                            _generate_layer_grid(**layer_kwargs)
                                        )
                                    if CONFIG_INDIVIDUAL_METRIC_FIGURES:
                                        total_generated += (
                                            _generate_individual_layer_plots(
                                                **layer_kwargs
                                            )
                                        )

                                for class_fraction in class_fractions:
                                    for maxk in maxk_values:
                                        lo_suffix = _label_overlap_suffix(
                                            class_fraction, maxk
                                        )
                                        if label_overlap_columns:
                                            layer_kwargs = dict(
                                                results_base=results_base,
                                                plots_dir=plots_dir,
                                                dataset_slug=dataset_slug,
                                                split=dataset_split,
                                                pooling_method=pooling_method,
                                                prompt_mode=prompt_mode,
                                                icl_shots=args.icl_shots,
                                                n_choices=n_choices,
                                                subsample_size=subsample,
                                                label_mode=label_mode,
                                                cluster_labels=cluster,
                                                selected_columns=label_overlap_columns,
                                                suffix=suffix,
                                                label_overlap_suffix=lo_suffix,
                                                class_fraction=class_fraction,
                                                maxk=maxk,
                                            )
                                            if CONFIG_GLOBAL_FIGURE:
                                                total_generated += int(
                                                    _generate_layer_grid(**layer_kwargs)
                                                )
                                            if CONFIG_INDIVIDUAL_METRIC_FIGURES:
                                                total_generated += (
                                                    _generate_individual_layer_plots(
                                                        **layer_kwargs
                                                    )
                                                )

                                        if "clip_label_overlap" in args.scalar_metrics:
                                            total_generated += int(
                                                _generate_scalar_plot(
                                                    results_base=results_base,
                                                    plots_dir=plots_dir,
                                                    dataset_slug=dataset_slug,
                                                    split=dataset_split,
                                                    pooling_method=pooling_method,
                                                    metric="clip_label_overlap",
                                                    prompt_mode=prompt_mode,
                                                    icl_shots=args.icl_shots,
                                                    n_choices=n_choices,
                                                    subsample_size=subsample,
                                                    label_mode=label_mode,
                                                    cluster_labels=cluster,
                                                    suffix=suffix,
                                                    class_fraction=class_fraction,
                                                    maxk=maxk,
                                                    label_overlap_suffix=lo_suffix,
                                                )
                                            )

                                if (
                                    "accuracy" in args.scalar_metrics
                                    and is_synthetic_mcq
                                ):
                                    total_generated += int(
                                        _generate_scalar_plot(
                                            results_base=results_base,
                                            plots_dir=plots_dir,
                                            dataset_slug=dataset_slug,
                                            split=dataset_split,
                                            pooling_method=pooling_method,
                                            metric="accuracy",
                                            prompt_mode=prompt_mode,
                                            icl_shots=args.icl_shots,
                                            n_choices=n_choices,
                                            subsample_size=subsample,
                                            label_mode=label_mode,
                                            cluster_labels=cluster,
                                            suffix=suffix,
                                            class_fraction=None,
                                            maxk=None,
                                            label_overlap_suffix="",
                                        )
                                    )

    if CONFIG_DATASET_MEAN_FIGURES and all_pooling_methods:
        aggregate_n_choices = CONFIG_DATASET_MEAN_N_CHOICES or [
            n for n in n_choices_list if n is not None
        ]
        label_overlap_columns = [
            column_name
            for column_name in args.columns
            if LAYER_COLUMN_SPECS[column_name][0] == "label_overlap"
        ]
        other_columns = [
            column_name
            for column_name in args.columns
            if LAYER_COLUMN_SPECS[column_name][0] != "label_overlap"
        ]

        for family in CONFIG_DATASET_MEAN_FAMILIES:
            if family not in DATASET_MEAN_PROMPT_MAP:
                _warn(f"Unknown dataset-mean family {family!r}; skipping")
                continue
            family_subsample_sizes = [None] if family == "mcq_filter_classes" else subsample_sizes
            for pooling_method in sorted(all_pooling_methods):
                for n_choices in aggregate_n_choices:
                    for subsample in family_subsample_sizes:
                        for label_mode in label_modes:
                            for cluster in CONFIG_CLUSTER_VARIANTS:
                                for column_name in other_columns:
                                    total_generated += int(
                                        _generate_dataset_mean_figure(
                                            results_base=results_base,
                                            plots_dir=plots_dir,
                                            available_dataset_pairs=available_dataset_pairs,
                                            family=family,
                                            pooling_method=pooling_method,
                                            column_name=column_name,
                                            icl_shots=args.icl_shots,
                                            n_choices=n_choices,
                                            subsample_size=subsample,
                                            label_mode=label_mode,
                                            cluster_labels=cluster,
                                            class_fraction=None,
                                            maxk=None,
                                            show_std_band=args.dataset_mean_std_band,
                                            std_band_alpha=args.dataset_mean_std_alpha,
                                        )
                                    )
                                for class_fraction in class_fractions:
                                    for maxk in maxk_values:
                                        for column_name in label_overlap_columns:
                                            total_generated += int(
                                                _generate_dataset_mean_figure(
                                                    results_base=results_base,
                                                    plots_dir=plots_dir,
                                                    available_dataset_pairs=available_dataset_pairs,
                                                    family=family,
                                                    pooling_method=pooling_method,
                                                    column_name=column_name,
                                                    icl_shots=args.icl_shots,
                                                    n_choices=n_choices,
                                                    subsample_size=subsample,
                                                    label_mode=label_mode,
                                                    cluster_labels=cluster,
                                                    class_fraction=class_fraction,
                                                    maxk=maxk,
                                                    show_std_band=args.dataset_mean_std_band,
                                                    std_band_alpha=args.dataset_mean_std_alpha,
                                                )
                                            )

    if total_generated == 0:
        print("No plots generated. Verify results-base, split, and metric settings.")


if __name__ == "__main__":
    main()
