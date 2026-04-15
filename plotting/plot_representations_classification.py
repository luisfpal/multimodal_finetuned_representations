"""Plot representations classification results with adaptive column layout.

Rows correspond to model families (7B, 13B, OV 4B). Columns are selectable metric/score
combinations. Figure size adapts to the number of selected columns and active rows.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file

HOME_DIR = Path.home()
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
CONFIG_SPLIT = "test"
CONFIG_DATASET_SLUGS: list[str] | None = None
CONFIG_POOLING_METHODS: list[str] | None = None
CONFIG_AUTO_DISCOVER_DATASETS = True
CONFIG_AUTO_DISCOVER_POOLING_METHODS = True

# (file_name, tensor_key, y_axis_label)
COLUMN_SPECS: dict[str, tuple[str, str, str]] = {
    "logistic_regression_f1":       ("logistic_regression", "f1_weighted",       "LR — F1 (weighted)"),
    # "logistic_regression_f1_train": ("logistic_regression", "f1_weighted_train", "LR — F1 Train (weighted)"),
    # "logistic_regression_mcc":      ("logistic_regression", "mcc",               "LR — MCC"),
    # "knn_f1":                       ("knn",                 "knn_f1_weighted",   "KNN — F1 (weighted)"),
    # "knn_mcc":                       ("knn",                 "knn_mcc",           "KNN — MCC"),
    "label_overlap":                ("label_overlap",       "label_overlap",      "Label Overlap"),
    "homogeneity":                  ("homogeneity",         "homogeneity_score",  "Homogeneity"),
    "adjusted_rand":                ("homogeneity",         "adjusted_rand_score","Adjusted Rand Index"),
    "n_clusters_ratio":             ("homogeneity",         "n_clusters_ratio",   "N Clusters / N Labels"),
}

# MCC and ARI range [-1, 1]; all others [0, 1]
_MCC_COLUMNS = {"logistic_regression_mcc", "knn_mcc", "adjusted_rand"}
# No fixed y-axis — let matplotlib auto-scale
_AUTOSCALE_COLUMNS = {"n_clusters_ratio"}

ROW1_MODELS = [
    "lbasile/llava_7b",
    "l11p/llava-v1.5-7b-pretrained-lbasile",
    # "lbasile/llava_7b_frozen",
    # "lbasile/llava_7b_0_16",
    # "lbasile/llava_7b_16_32",
    # "lbasile/llava_7b_6_16",
    # "lbasile/llava_7b_6_17_complementary",
]
ROW2_MODELS = [
    "lbasile/llava_13b",
    "l11p/llava-v1.5-13b-pretrained-lbasile",
    # "lbasile/llava_13b_6_21",
    # "lbasile/llava_13b_6_21_complementary",
]
ROW3_MODELS = [
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
    "lmms-lab/LLaVA-OneVision-1.5-4B-Base",
]

ROW4_MODELS = [
    "OpenGVLab/InternVL2-8B",
    "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain",
]

ROW5_MODELS = [
    "nyu-visionx/cambrian-8b",
    "lbasile/cambrian-8b-pretrained",
]

# Each entry: (model_list, row_label). Rows with no models are skipped automatically.
ROWS: list[tuple[list[str], str]] = [
    (ROW1_MODELS, "LLaVA 1.5 7B family"),
    (ROW2_MODELS, "LLaVA 1.5 13B family"),
    (ROW3_MODELS, "OV 4B family"),
    (ROW4_MODELS, "InternVL2 8B family"),
    (ROW5_MODELS, "Cambrian 8B family"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate representations-classification plots. "
            "Default behavior is driven by editable CONFIG_* globals in this file."
        )
    )
    parser.add_argument(
        "--results-base",
        type=Path,
        default=CONFIG_RESULTS_BASE,
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=CONFIG_PLOTS_DIR,
    )
    parser.add_argument(
        "--dataset-slugs",
        type=str,
        nargs="+",
        default=CONFIG_DATASET_SLUGS,
    )
    parser.add_argument("--split", type=str, default=CONFIG_SPLIT)
    parser.add_argument(
        "--pooling-methods",
        type=str,
        nargs="+",
        default=CONFIG_POOLING_METHODS,
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
        default=list(COLUMN_SPECS.keys()),
        choices=list(COLUMN_SPECS.keys()),
        help="Columns to include. Each column is one metric/score combination.",
    )
    parser.add_argument(
        "--icl-shots",
        type=int,
        default=0,
        help=(
            "If > 0, read metric files with the _{n}shot suffix "
            "(e.g. logistic_regression_3shot.safetensors). "
            "The output plot filename also carries the suffix."
        ),
    )
    return parser.parse_args()


def make_model_slug(model_name: str) -> str:
    return Path(model_name).name


def _discover_dataset_slugs(results_base: Path, split: str) -> list[str]:
    if not results_base.exists():
        return []
    return [
        candidate.name
        for candidate in sorted(results_base.iterdir())
        if candidate.is_dir() and (candidate / split).exists()
    ]


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


def _resolve_datasets(args: argparse.Namespace, results_base: Path) -> list[str]:
    if args.dataset_slugs:
        return args.dataset_slugs
    if not args.auto_discover_datasets:
        print("No datasets configured. Set CONFIG_DATASET_SLUGS or enable auto-discovery.")
        return []
    return _discover_dataset_slugs(results_base=results_base, split=args.split)


def _resolve_poolings(
    args: argparse.Namespace,
    results_base: Path,
    dataset_slug: str,
) -> list[str]:
    if args.pooling_methods:
        return args.pooling_methods
    if not args.auto_discover_pooling_methods:
        print(
            "No pooling methods configured. "
            "Set CONFIG_POOLING_METHODS or enable auto-discovery."
        )
        return []
    return _discover_pooling_methods(
        results_base=results_base,
        dataset_slug=dataset_slug,
        split=args.split,
    )


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
    palette = plt.cm.get_cmap("tab10", len(models))
    return {
        make_model_slug(model_name): palette(index)
        for index, model_name in enumerate(models)
    }


def load_metric_values(
    *,
    results_base: Path,
    dataset_slug: str,
    split: str,
    model_slug: str,
    pooling_method: str,
    file_name: str,
    tensor_key: str,
    icl_suffix: str = "",
) -> np.ndarray | None:
    file_path = (
        results_base
        / dataset_slug
        / split
        / model_slug
        / pooling_method
        / f"{file_name}{icl_suffix}.safetensors"
    )

    if not file_path.exists():
        return None

    try:
        data = load_file(str(file_path))
    except Exception:
        return None

    if tensor_key not in data:
        return None

    return data[tensor_key].cpu().numpy()


def _plot_row(
    *,
    ax,
    models: list[str],
    column_name: str,
    results_base: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    row_name: str,
    color_map: dict[str, tuple[float, float, float, float]],
    show_legend: bool,
    icl_suffix: str = "",
) -> bool:
    file_name, tensor_key, y_label = COLUMN_SPECS[column_name]
    has_data = False

    for model_name in models:
        model_slug = make_model_slug(model_name)
        values = load_metric_values(
            results_base=results_base,
            dataset_slug=dataset_slug,
            split=split,
            model_slug=model_slug,
            pooling_method=pooling_method,
            file_name=file_name,
            tensor_key=tensor_key,
            icl_suffix=icl_suffix,
        )
        if values is None:
            continue

        has_data = True
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

    ax.set_title(y_label, fontsize=12)
    ax.set_xlabel("Decoder Layer", fontsize=10)
    ax.set_ylabel(f"{y_label} ({row_name})", fontsize=10)
    if column_name not in _AUTOSCALE_COLUMNS:
        ax.set_ylim(-1.0, 1.0) if column_name in _MCC_COLUMNS else ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=9)

    if has_data and show_legend:
        ax.legend(fontsize=8, loc="best")

    return has_data


def _generate_plot_for_combo(
    *,
    results_base: Path,
    plots_dir: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    selected_columns: list[str],
    icl_suffix: str = "",
) -> bool:
    n_cols = len(selected_columns)
    active_rows = [(models, label) for models, label in ROWS if models]
    n_rows = len(active_rows)
    color_maps = [_build_color_map(models) for models, _ in active_rows]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    any_data = False
    for col_idx, col_name in enumerate(selected_columns):
        for row_idx, ((models, row_label), color_map) in enumerate(
            zip(active_rows, color_maps)
        ):
            has_data = _plot_row(
                ax=axes[row_idx, col_idx],
                models=models,
                column_name=col_name,
                results_base=results_base,
                dataset_slug=dataset_slug,
                split=split,
                pooling_method=pooling_method,
                row_name=row_label,
                color_map=color_map,
                show_legend=col_idx == 0,
                icl_suffix=icl_suffix,
            )
            any_data = any_data or has_data

    icl_label = icl_suffix.lstrip("_") if icl_suffix else "0shot"
    fig.suptitle(
        (
            "Representations Classification Metrics | "
            f"dataset={dataset_slug} | split={split} | pooling={pooling_method} | {icl_label}"
        ),
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if not any_data:
        plt.close(fig)
        return False

    plots_dir.mkdir(parents=True, exist_ok=True)
    cols_tag = "_".join(selected_columns)
    out_path = (
        plots_dir
        / f"representations_classification_{dataset_slug}_{split}_{pooling_method}{icl_suffix}_{cols_tag}.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


def main() -> None:
    args = parse_args()
    results_base = args.results_base.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()

    _configure_plot_style()

    dataset_slugs = _resolve_datasets(args=args, results_base=results_base)
    if not dataset_slugs:
        return

    total_generated = 0
    for dataset_slug in dataset_slugs:
        pooling_methods = _resolve_poolings(
            args=args,
            results_base=results_base,
            dataset_slug=dataset_slug,
        )
        if not pooling_methods:
            continue

        icl_suffix = f"_{args.icl_shots}shot" if args.icl_shots > 0 else ""
        for pooling_method in pooling_methods:
            did_generate = _generate_plot_for_combo(
                results_base=results_base,
                plots_dir=plots_dir,
                dataset_slug=dataset_slug,
                split=args.split,
                pooling_method=pooling_method,
                selected_columns=args.columns,
                icl_suffix=icl_suffix,
            )
            total_generated += int(did_generate)

    if total_generated == 0:
        print("No plots generated. Verify results-base, split, and available metric files.")


if __name__ == "__main__":
    main()
