"""Plot 2x3 metric comparison for representations classification results.

Columns correspond to metrics: logistic_regression, knn, label_overlap.
Row 1 includes selected 7B model family variants.
Row 2 includes selected 13B model family variants.
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

METRICS = ["logistic_regression", "label_overlap", "knn"]
METRIC_KEYS = {
    "logistic_regression": "f1_weighted",
    "knn": "knn_f1_weighted",
    "label_overlap": "label_overlap",
}
METRIC_SCORE_LABELS = {
    "logistic_regression": "F1 (weighted)",
    "knn": "F1 (weighted)",
    "label_overlap": "Label Overlap",
}

ROW1_MODELS = [
    "lbasile/llava_7b",
    "lbasile/llava_7b_frozen",
    "lbasile/llava_7b_0_16",
    "lbasile/llava_7b_16_32",
    "lbasile/llava_7b_6_16",
    "lbasile/llava_7b_6_17_complementary",
    "l11p/llava-v1.5-7b-pretrained-lbasile",
]
ROW2_MODELS = [
    "lbasile/llava_13b_6_21",
    "lbasile/llava_13b_6_21_complementary",
    "lbasile/llava_13b",
    "l11p/llava-v1.5-13b-pretrained-lbasile",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-base",
        type=Path,
        default=DEFAULT_RESULTS_BASE,
        help="Root directory containing results/representations_classification.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory where the 2x3 figure is saved.",
    )
    parser.add_argument("--dataset-slug", type=str, default="ScienceQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--pooling-method", type=str, default="last")
    return parser.parse_args()


def make_model_slug(model_name: str) -> str:
    return Path(model_name).name


def load_metric_values(
    *,
    results_base: Path,
    dataset_slug: str,
    split: str,
    model_slug: str,
    pooling_method: str,
    metric_name: str,
) -> np.ndarray | None:
    file_path = (
        results_base
        / dataset_slug
        / split
        / model_slug
        / pooling_method
        / f"{metric_name}.safetensors"
    )

    if not file_path.exists():
        print(f"  Missing: {file_path}")
        return None

    try:
        data = load_file(str(file_path))
    except Exception as exc:
        print(f"  Could not load {file_path}: {exc}")
        return None

    key = METRIC_KEYS[metric_name]
    if key not in data:
        print(f"  Missing key '{key}' in {file_path}")
        return None

    return data[key].cpu().numpy()


def _plot_row(
    *,
    ax,
    models: list[str],
    metric_name: str,
    results_base: Path,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    row_name: str,
) -> None:
    has_data = False

    for model_name in models:
        model_slug = make_model_slug(model_name)
        values = load_metric_values(
            results_base=results_base,
            dataset_slug=dataset_slug,
            split=split,
            model_slug=model_slug,
            pooling_method=pooling_method,
            metric_name=metric_name,
        )
        if values is None:
            continue

        has_data = True
        layers = np.arange(len(values))
        ax.plot(layers, values, marker="o", markersize=3, label=model_slug)

    ax.set_title(metric_name.replace("_", " "), fontsize=12)
    ax.set_xlabel("Decoder Layer", fontsize=10)
    ax.set_ylabel(f"{METRIC_SCORE_LABELS[metric_name]} ({row_name})", fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    if has_data:
        ax.legend(fontsize=8, loc="best")


def main() -> None:
    args = parse_args()
    results_base = args.results_base.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()

    fig, axes = plt.subplots(2, 3, figsize=(21, 12), squeeze=False)

    for col_idx, metric_name in enumerate(METRICS):
        _plot_row(
            ax=axes[0, col_idx],
            models=ROW1_MODELS,
            metric_name=metric_name,
            results_base=results_base,
            dataset_slug=args.dataset_slug,
            split=args.split,
            pooling_method=args.pooling_method,
            row_name="7B family",
        )
        _plot_row(
            ax=axes[1, col_idx],
            models=ROW2_MODELS,
            metric_name=metric_name,
            results_base=results_base,
            dataset_slug=args.dataset_slug,
            split=args.split,
            pooling_method=args.pooling_method,
            row_name="13B family",
        )

    fig.suptitle(
        (
            "Representations Classification Metrics | "
            f"dataset={args.dataset_slug} | split={args.split} | pooling={args.pooling_method}"
        ),
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        plots_dir
        / f"representations_classification_2x3_{args.dataset_slug}_{args.split}_{args.pooling_method}.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
