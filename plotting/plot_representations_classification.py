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

# Each entry: (dataset_slug, split). Split is intrinsic to the dataset — not a global default.
# Set to None to auto-discover all datasets under results_base (split auto-detected).
CONFIG_DATASETS: list[tuple[str, str]] | None = [
    ("mmmu", "test"),
    ("mini_imagenet", "validation"),
]

CONFIG_POOLING_METHODS: list[str] | None = None
CONFIG_AUTO_DISCOVER_DATASETS = True
CONFIG_AUTO_DISCOVER_POOLING_METHODS = True

# Prompt modes to plot — each produces a separate figure.
# "describe"      → no filename suffix (default free-description prompt)
# "continuation"  → _continuation suffix (seeded assistant-prefix prompt)
CONFIG_PROMPT_MODES: list[str] = [
    "describe",
    "continuation",
]

# Whitelist of pooling methods to include. None = all auto-discovered methods.
CONFIG_POOLING_METHODS_FILTER: list[str] | None = [
    "last",
    # "mean_image",
]

# Figure modes — at least one must be True.
# Global : one figure per combo (rows=model families, cols=metrics).
# Individual: one figure per metric per combo (1 row, cols=model families),
#             saved to a per-combo subfolder under plots_dir.
CONFIG_GLOBAL_FIGURE = True
CONFIG_INDIVIDUAL_METRIC_FIGURES = False

# DPI for saved figures. 150 ≈ 4× smaller than 300, still crisp on screen.
# Raise to 200–300 for print-quality output.
CONFIG_PLOT_DPI = 150

# (file_name, tensor_key, y_axis_label)
COLUMN_SPECS: dict[str, tuple[str, str, str]] = {
    "logistic_regression_f1":       ("logistic_regression", "f1_weighted",       "LR — F1 (weighted)"),
    "logistic_regression_f1_train": ("logistic_regression", "f1_weighted_train", "LR — F1 Train (weighted)"),
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


def _resolve_datasets(args: argparse.Namespace, results_base: Path) -> list[tuple[str, str]]:
    if CONFIG_DATASETS is not None:
        return CONFIG_DATASETS
    if not args.auto_discover_datasets:
        print("No datasets configured. Set CONFIG_DATASETS or enable auto-discovery.")
        return []
    return _discover_datasets(results_base=results_base)


def _resolve_poolings(
    args: argparse.Namespace,
    results_base: Path,
    dataset_slug: str,
    dataset_split: str,
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
        split=dataset_split,
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
    # tab10 has 10 equally-spaced slots; pick by center of each slot (i+0.5)/10
    palette = plt.colormaps["tab10"]
    return {
        make_model_slug(model_name): palette((index % 10 + 0.5) / 10)
        for index, model_name in enumerate(models)
    }


def _settings_label(
    dataset_slug: str, split: str, pooling_method: str, icl_suffix: str
) -> str:
    icl_label = icl_suffix.lstrip("_") if icl_suffix else "0shot"
    return f"dataset={dataset_slug} | split={split} | pooling={pooling_method} | {icl_label}"


def _combo_tag(
    dataset_slug: str, split: str, pooling_method: str, icl_suffix: str
) -> str:
    return f"{dataset_slug}_{split}_{pooling_method}{icl_suffix}"


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

    fig.suptitle(
        "Representations Classification Metrics | "
        + _settings_label(dataset_slug, split, pooling_method, icl_suffix),
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if not any_data:
        plt.close(fig)
        return False

    plots_dir.mkdir(parents=True, exist_ok=True)
    tag = _combo_tag(dataset_slug, split, pooling_method, icl_suffix)
    out_path = plots_dir / f"representations_classification_{tag}.png"
    fig.savefig(out_path, dpi=CONFIG_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


def _generate_individual_metric_plots(
    *,
    results_base: Path,
    plots_dir: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    selected_columns: list[str],
    icl_suffix: str = "",
) -> int:
    """One figure per metric: 1 row × n_families columns, saved to a per-combo subfolder."""
    active_rows = [(models, label) for models, label in ROWS if models]
    color_maps = [_build_color_map(models) for models, _ in active_rows]
    n_families = len(active_rows)

    combo_dir = plots_dir / _combo_tag(dataset_slug, split, pooling_method, icl_suffix)
    generated = 0

    for col_name in selected_columns:
        _, _, y_label = COLUMN_SPECS[col_name]
        fig, axes = plt.subplots(1, n_families, figsize=(7 * n_families, 6), squeeze=False)

        any_data = False
        for fam_idx, ((models, row_label), color_map) in enumerate(
            zip(active_rows, color_maps)
        ):
            has_data = _plot_row(
                ax=axes[0, fam_idx],
                models=models,
                column_name=col_name,
                results_base=results_base,
                dataset_slug=dataset_slug,
                split=split,
                pooling_method=pooling_method,
                row_name=row_label,
                color_map=color_map,
                show_legend=True,
                icl_suffix=icl_suffix,
            )
            # Override titles set by _plot_row: subplot title = family, y-label = metric only
            axes[0, fam_idx].set_title(row_label, fontsize=12)
            axes[0, fam_idx].set_ylabel(y_label, fontsize=10)
            any_data = any_data or has_data

        fig.suptitle(
            y_label + " | " + _settings_label(dataset_slug, split, pooling_method, icl_suffix),
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if not any_data:
            plt.close(fig)
            continue

        combo_dir.mkdir(parents=True, exist_ok=True)
        tag = _combo_tag(dataset_slug, split, pooling_method, icl_suffix)
        out_path = combo_dir / f"{tag}_{col_name}.png"
        fig.savefig(out_path, dpi=CONFIG_PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        generated += 1

    return generated


def main() -> None:
    args = parse_args()
    results_base = args.results_base.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()

    _configure_plot_style()

    datasets = _resolve_datasets(args=args, results_base=results_base)
    if not datasets:
        return

    icl_suffix = f"_{args.icl_shots}shot" if args.icl_shots > 0 else ""

    total_generated = 0
    for dataset_slug, dataset_split in datasets:
        pooling_methods = _resolve_poolings(
            args=args,
            results_base=results_base,
            dataset_slug=dataset_slug,
            dataset_split=dataset_split,
        )
        if CONFIG_POOLING_METHODS_FILTER is not None:
            pooling_methods = [p for p in pooling_methods if p in CONFIG_POOLING_METHODS_FILTER]
        if not pooling_methods:
            continue

        for prompt_mode in CONFIG_PROMPT_MODES:
            prompt_suffix = f"_{prompt_mode}" if prompt_mode != "describe" else ""
            combined_suffix = icl_suffix + prompt_suffix
            for pooling_method in pooling_methods:
                kwargs = dict(
                    results_base=results_base,
                    plots_dir=plots_dir,
                    dataset_slug=dataset_slug,
                    split=dataset_split,
                    pooling_method=pooling_method,
                    selected_columns=args.columns,
                    icl_suffix=combined_suffix,
                )
                if CONFIG_GLOBAL_FIGURE:
                    total_generated += int(_generate_plot_for_combo(**kwargs))
                if CONFIG_INDIVIDUAL_METRIC_FIGURES:
                    total_generated += _generate_individual_metric_plots(**kwargs)

    if total_generated == 0:
        print("No plots generated. Verify results-base, split, and available metric files.")


if __name__ == "__main__":
    main()
