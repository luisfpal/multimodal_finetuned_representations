"""Paper-ready filtered-MCQ label-overlap profiles.

This script is intentionally narrow: it produces the NeurIPS-style figures for
filtered synthetic MCQ experiments, using last-token pooling and label-overlap
curves.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

HOME_DIR = Path.home()
if "MPLCONFIGDIR" not in os.environ:
    for mpl_config_dir in (
        HOME_DIR / "scratch" / "matplotlib",
        Path("/dev/shm") / f"matplotlib-{os.environ.get('USER', 'user')}",
    ):
        try:
            mpl_config_dir.mkdir(parents=True, exist_ok=True)
            probe = mpl_config_dir / ".write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError:
            continue
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
        break

import matplotlib.pyplot as plt  # noqa: E402


# ==============================================================================
# SECTION 1 - CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
RESULTS_DIR = PROJECT_ROOT / "results" / "representations_classification"
PLOTS_DIR = PROJECT_ROOT / "plots" / "representations_classification" / "paper_figures"

POOLING_METHOD = "last"
FAMILY = "mcq_filter_classes"
N_CHOICES = (10,)
DPI_PNG = 200
CLASS_FRACTION = 0.3
MAXK: int | None = None
STD_ALPHA = 0.18

# Figure sizes, in inches.
FIGSIZE_INDIVIDUAL_MEAN = (3.35, 2.15)
FIGSIZE_ALL_MODELS = (4.9, 3.35)
FIGSIZE_DATASETS_BY_MODEL = (3.8, 3.0)

# Font sizes.
FONT_SIZE_TITLE = 12
FONT_SIZE_AXIS_LABEL = 10
FONT_SIZE_TICK_LABEL = 9
FONT_SIZE_LEGEND_LABEL = 8

# Line and marker styling.
LINE_WIDTH = 1.45
MARKER_SIZE_INDIVIDUAL = 3.0
MARKER_SIZE_ALL_MODELS = 2.6
MARKER_SIZE_DATASET = 2.6
MARKER_EDGE_WIDTH = 0.0
TICK_LENGTH = 3.0
TICK_WIDTH = 0.8
GRID_ALPHA = 0.18
LEGEND_HANDLE_LENGTH = 2.2
LEGEND_COLUMN_SPACING = 1.1
TITLE_PAD = 8

MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("lbasile/llava_7b", "LLaVA-7B"),
    ("lbasile/llava_13b", "LLaVA-13B"),
    ("lmms-lab/LLaVA-OneVision-1.5-4B-Instruct", "LLaVA-OneVision-4B"),
    ("lmms-lab/LLaVA-OneVision-1.5-8B-Instruct", "LLaVA-OneVision-8B"),
    ("OpenGVLab/InternVL2-8B", "InternVL2-8B"),
    ("nyu-visionx/cambrian-8b", "Cambrian-8B"),
)

DATASET_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("mini_imagenet", "validation", "mcq_nocat_filter", "mini-ImageNet"),
    (
        "imagenet_short_text_100_classes_x_100_samples",
        "train",
        "animal_mcq_filter",
        "ImageNet-short-text",
    ),
    ("food101", "validation", "generic_mcq_filter", "Food101"),
    ("sun397", "test", "generic_mcq_filter", "SUN397"),
    ("caltech101", "train", "generic_mcq_top_filter", "Caltech101"),
    ("oxford_flowers102", "test", "generic_mcq_top_filter", "Oxford Flowers 102"),
    ("places365", "train", "generic_mcq_filter", "Places365"),
    ("dtd", "train", "generic_mcq_filter", "DTD"),
)

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str

    @property
    def slug(self) -> str:
        return Path(self.name).name


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    split: str
    prompt_mode: str
    label: str


@dataclass(frozen=True)
class ModelCurves:
    model: ModelSpec
    dataset_curves: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray


# ==============================================================================
# SECTION 2 - DATA LOADING
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready filtered-MCQ label-overlap figures."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    return parser.parse_args()


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _prompt_suffix(dataset_slug: str, prompt_mode: str) -> str:
    is_imagenet_animals = "imagenet_short_text" in dataset_slug
    is_generic = any(
        ds in dataset_slug
        for ds in ("food101", "sun397", "caltech101", "oxford_flowers102", "places365", "dtd")
    )
    if prompt_mode != "describe" or is_imagenet_animals or is_generic:
        return f"_{prompt_mode}"
    return ""


def _label_overlap_suffix() -> str:
    parts: list[str] = []
    if CLASS_FRACTION is not None:
        parts.append(f"cf{CLASS_FRACTION}")
    if MAXK is not None:
        parts.append(f"mk{MAXK}")
    return "_" + "_".join(parts) if parts else ""


def _file_suffix(dataset: DatasetSpec, n_choices: int) -> str:
    return f"{_prompt_suffix(dataset.slug, dataset.prompt_mode)}_n{n_choices}{_label_overlap_suffix()}"


def _label_overlap_path(
    *,
    results_dir: Path,
    dataset: DatasetSpec,
    model: ModelSpec,
    n_choices: int,
) -> Path:
    return (
        results_dir
        / dataset.slug
        / dataset.split
        / model.slug
        / POOLING_METHOD
        / f"label_overlap{_file_suffix(dataset, n_choices)}.safetensors"
    )


def _load_label_overlap(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with safe_open(str(path), framework="numpy") as handle:
            if "label_overlap" not in handle.keys():
                _warn(f"Missing tensor key 'label_overlap' in {path}")
                return None
            values = handle.get_tensor("label_overlap").copy()
    except Exception as exc:
        _warn(f"Could not load {path}: {exc}")
        return None
    if values.size == 0:
        _warn(f"Empty label_overlap tensor in {path}")
        return None
    return np.asarray(values, dtype=float)


def _strict_mean_std(
    curves: list[np.ndarray],
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    lengths = {len(curve) for curve in curves}
    if len(lengths) != 1:
        _warn(f"Cannot average curves with different lengths for {context}: {sorted(lengths)}")
        return None
    stacked = np.stack(curves, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0, ddof=1)


def collect_model_curves(
    *,
    results_dir: Path,
    n_choices: int,
    models: list[ModelSpec],
    datasets: list[DatasetSpec],
) -> list[ModelCurves]:
    collected: list[ModelCurves] = []
    for model in models:
        dataset_curves: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for dataset in datasets:
            path = _label_overlap_path(
                results_dir=results_dir,
                dataset=dataset,
                model=model,
                n_choices=n_choices,
            )
            values = _load_label_overlap(path)
            if values is None:
                missing.append(f"{dataset.label} ({dataset.prompt_mode})")
                continue
            dataset_curves[dataset.label] = values

        if missing:
            _warn(
                f"{model.label} / n={n_choices}: missing {len(missing)} dataset(s); "
                "plotting available datasets."
            )
            for item in missing:
                print(f"  - {item}")

        if not dataset_curves:
            _warn(f"{model.label} / n={n_choices}: no dataset curves found; skipping model.")
            continue

        mean_std = _strict_mean_std(
            list(dataset_curves.values()),
            context=f"{model.label}, n={n_choices}",
        )
        if mean_std is None:
            continue
        mean, std = mean_std
        collected.append(
            ModelCurves(
                model=model,
                dataset_curves=dataset_curves,
                mean=mean,
                std=std,
            )
        )
    return collected


# ==============================================================================
# SECTION 3 - PLOTTING
# ==============================================================================


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Bitstream Vera Sans",
                "sans-serif",
            ],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_AXIS_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK_LABEL,
            "ytick.labelsize": FONT_SIZE_TICK_LABEL,
            "legend.fontsize": FONT_SIZE_LEGEND_LABEL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "--",
            "lines.linewidth": LINE_WIDTH,
            "savefig.facecolor": "white",
        }
    )


def _style_axis(ax, *, title: str | None, x_label: str = "Layer") -> None:
    if title:
        ax.set_title(title, pad=TITLE_PAD, fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel(x_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Label overlap", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        color="#444444",
        labelsize=FONT_SIZE_TICK_LABEL,
    )
    ax.set_axisbelow(True)


def _save_figure(fig, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_stem.parent / f"{output_stem.name}.png",
        output_stem.parent / f"{output_stem.name}.pdf",
    ]
    fig.savefig(outputs[0], dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    for output in outputs:
        print(f"Saved: {output}")
    return outputs


def _normalized_depth(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, len(values))


def plot_individual_mean(
    *,
    output_dir: Path,
    curves: ModelCurves,
    with_std: bool,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE_INDIVIDUAL_MEAN)
    layers = np.arange(len(curves.mean))
    color = PALETTE[0]
    if with_std:
        ax.fill_between(
            layers,
            np.clip(curves.mean - curves.std, 0.0, 1.0),
            np.clip(curves.mean + curves.std, 0.0, 1.0),
            color=color,
            alpha=STD_ALPHA,
            linewidth=0,
        )
    ax.plot(
        layers,
        curves.mean,
        color=color,
        marker="o",
        markersize=MARKER_SIZE_INDIVIDUAL,
        markeredgewidth=MARKER_EDGE_WIDTH,
    )
    _style_axis(ax, title=curves.model.label)
    fig.tight_layout()

    variant = "with_std" if with_std else "no_std"
    suffix = "_std" if with_std else ""
    return _save_figure(fig, output_dir / variant / f"{curves.model.slug}{suffix}")


def plot_all_models(
    *,
    output_dir: Path,
    model_curves: list[ModelCurves],
) -> list[Path] | None:
    if not model_curves:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_ALL_MODELS)
    for index, curves in enumerate(model_curves):
        x = _normalized_depth(curves.mean)
        ax.plot(
            x,
            curves.mean,
            color=PALETTE[index % len(PALETTE)],
            marker="o",
            markersize=MARKER_SIZE_ALL_MODELS,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=curves.model.label,
        )
    _style_axis(ax, title=None, x_label="Normalized depth")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND_LABEL,
        handlelength=LEGEND_HANDLE_LENGTH,
        columnspacing=LEGEND_COLUMN_SPACING,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir / "all_models")


def plot_datasets_by_model(
    *,
    output_dir: Path,
    curves: ModelCurves,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE_DATASETS_BY_MODEL)
    for index, (dataset_label, values) in enumerate(curves.dataset_curves.items()):
        layers = np.arange(len(values))
        ax.plot(
            layers,
            values,
            color=PALETTE[index % len(PALETTE)],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE_DATASET,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=dataset_label,
        )
    _style_axis(ax, title=curves.model.label, x_label="Layer")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND_LABEL,
        handlelength=LEGEND_HANDLE_LENGTH,
        columnspacing=LEGEND_COLUMN_SPACING,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir / curves.model.slug)


# ==============================================================================
# SECTION 4 - MANIFEST AND MAIN
# ==============================================================================


def write_manifest(
    *,
    output_dir: Path,
    models: list[ModelSpec],
    datasets: list[DatasetSpec],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Representations Classification Paper Figures",
        "",
        f"- Family: `{FAMILY}`",
        "- Metric: `label_overlap`",
        f"- Pooling: `{POOLING_METHOD}`",
        f"- Class fraction: `{CLASS_FRACTION}`",
        f"- N choices: {', '.join(f'`{n}`' for n in N_CHOICES)}",
        "",
        "## Models",
        "",
    ]
    lines.extend(f"- {model.label}: `{model.slug}`" for model in models)
    lines.extend(["", "## Dataset Prompt Profiles", ""])
    lines.extend(
        (
            f"- {dataset.label}: `{dataset.slug}` / `{dataset.split}` / "
            f"`{dataset.prompt_mode}`"
        )
        for dataset in datasets
    )
    lines.extend(
        [
            "",
            "## Figure Groups",
            "",
            "- `individual_mean/no_std`: one mean profile per model.",
            "- `individual_mean/with_std`: one mean profile per model with mean ± 1 std across datasets.",
            "- `all_models`: all model mean profiles in one normalized-depth figure.",
            "- `datasets_by_model`: all dataset profiles for each model.",
            "",
        ]
    )
    manifest = output_dir / "manifest.md"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {manifest}")
    return manifest


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()
    models = [ModelSpec(*spec) for spec in MODEL_SPECS]
    datasets = [DatasetSpec(*spec) for spec in DATASET_SPECS]

    configure_publication_style()

    print("=" * 72)
    print("  plot_representations_classification_paper.py")
    print(f"  family={FAMILY} | metric=label_overlap | pooling={POOLING_METHOD}")
    print(f"  n_choices={', '.join(str(n) for n in N_CHOICES)}")
    print(f"  models={len(models)} | datasets={len(datasets)}")
    print(f"  results={results_dir}")
    print(f"  output={plots_dir}")
    print("=" * 72)

    write_manifest(output_dir=plots_dir, models=models, datasets=datasets)

    written = 0
    for n_choices in N_CHOICES:
        print(f"\nn={n_choices}")
        model_curves = collect_model_curves(
            results_dir=results_dir,
            n_choices=n_choices,
            models=models,
            datasets=datasets,
        )
        if not model_curves:
            _warn(f"No complete model curves found for n={n_choices}; skipping figures.")
            continue

        n_dir = plots_dir / FAMILY / f"n{n_choices}"
        for curves in model_curves:
            written += len(
                plot_individual_mean(
                    output_dir=n_dir / "individual_mean",
                    curves=curves,
                    with_std=False,
                )
            )
            written += len(
                plot_individual_mean(
                    output_dir=n_dir / "individual_mean",
                    curves=curves,
                    with_std=True,
                )
            )
            written += len(
                plot_datasets_by_model(
                    output_dir=n_dir / "datasets_by_model",
                    curves=curves,
                )
            )

        outputs = plot_all_models(output_dir=n_dir / "all_models", model_curves=model_curves)
        if outputs is not None:
            written += len(outputs)

    print(f"\nFinished. Wrote {written} figure file(s).")


if __name__ == "__main__":
    main()
