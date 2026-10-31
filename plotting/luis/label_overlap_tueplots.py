"""Label-overlap plots in the tueplots NeurIPS style."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tueplots_style import OUTPUT_DIR, get_rc  # noqa: E402

# Hardcoded path: edit this path for your setup.
LUIS_ROOT = Path("/u/dssc/lbasile/scratch/plm/luis")
DATA_DIR = LUIS_ROOT / "data" / "label_overlap"
FIGURES_DIR = OUTPUT_DIR / "label_overlap"

POOLING_METHOD = "last"
FAMILY = "mcq_filter_classes"

N_CHOICES = (10, 20)
CLASS_FRACTION = 0.3
MAXK: int | None = None
STD_ALPHA = 0.18

MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("lbasile/llava_7b", "LLaVA-7B"),
    ("lbasile/llava_13b", "LLaVA-13B"),
    ("nyu-visionx/cambrian-8b", "Cambrian-8B"),
    ("lmms-lab/LLaVA-OneVision-1.5-8B-Instruct", "OneVision-8B"),
    ("lmms-lab/LLaVA-OneVision-1.5-4B-Instruct", "OneVision-4B"),
    ("OpenGVLab/InternVL2-8B", "InternVL2-8B"),
)

# Each entry: (slug, split, prompt_mode, label, max_n_choices)
# max_n_choices=None means all N_CHOICES are available.
DATASET_SPECS: tuple[tuple, ...] = (
    ("mini_imagenet", "validation", "mcq_nocat_filter", "mini-ImageNet", None),
    ("food101", "validation", "generic_mcq_filter", "Food101", None),
    ("sun397", "test", "generic_mcq_filter", "SUN397", None),
    ("caltech101", "train", "generic_mcq_top_filter", "Caltech101", None),
    ("dtd", "train", "generic_mcq_filter", "DTD", 10),
    ("oxford_flowers102", "test", "generic_mcq_top_filter", "Flowers102", 10),
    ("places365", "train", "generic_mcq_filter", "Places365", 10),
)

MODEL_COLORS: dict[str, str] = {
    "LLaVA-7B":     "#E09F3E",
    "LLaVA-13B":    "#E9CF3A",
    "InternVL2-8B": "#99a88c",
    "OneVision-4B": "#457b9d",
    "OneVision-8B": "#335c67",
    "Cambrian-8B":  "#9e2a2b",
}

PALETTE = tuple(MODEL_COLORS.values())

GROUP_ONE = ("LLaVA-7B", "OneVision-4B", "InternVL2-8B")
GROUP_TWO = ("LLaVA-13B", "OneVision-8B", "Cambrian-8B")


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
    max_n_choices: int | None = None


@dataclass(frozen=True)
class ModelCurves:
    model: ModelSpec
    dataset_curves: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray


def _prompt_suffix(dataset_slug: str, prompt_mode: str) -> str:
    is_imagenet_animals = "imagenet_short_text" in dataset_slug
    is_generic = any(ds in dataset_slug for ds in ("food101", "sun397", "caltech101"))
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


def _label_overlap_path(*, data_dir: Path, dataset: DatasetSpec, model: ModelSpec, n_choices: int) -> Path:
    return data_dir / dataset.slug / dataset.split / model.slug / f"label_overlap{_file_suffix(dataset, n_choices)}.json"


def _load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"WARN: invalid JSON {path}: {exc}")
        return None
    values = payload.get("label_overlap")
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    return arr if arr.size else None


def _strict_mean_std(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    lengths = {len(c) for c in curves}
    if len(lengths) != 1:
        print(f"⚠ mismatched curve lengths: {sorted(lengths)}")
        return None
    stacked = np.stack(curves, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0, ddof=1)


def collect_model_curves(*, data_dir: Path, n_choices: int,
                         models: list[ModelSpec], datasets: list[DatasetSpec]) -> list[ModelCurves]:
    collected = []
    active_datasets = [d for d in datasets if d.max_n_choices is None or n_choices <= d.max_n_choices]
    for model in models:
        dataset_curves = {}
        missing = []
        for dataset in active_datasets:
            path = _label_overlap_path(data_dir=data_dir, dataset=dataset, model=model, n_choices=n_choices)
            values = _load(path)
            if values is None:
                missing.append(f"{dataset.label} ({dataset.prompt_mode})")
                continue
            dataset_curves[dataset.label] = values
        if missing:
            print(f"WARN: {model.label} / n={n_choices}: missing {len(missing)} dataset(s)")
            continue
        mean_std = _strict_mean_std(list(dataset_curves.values()))
        if mean_std is None:
            continue
        mean, std = mean_std
        collected.append(ModelCurves(model=model, dataset_curves=dataset_curves, mean=mean, std=std))
    return collected


def _style_axis(ax, *, x_label: str = "Layer") -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel("Label overlap")
    ax.set_ylim(0.0, 1.1)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)


def _save(fig, output_stem: Path) -> Path:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    out = output_stem.parent / f"{output_stem.name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def _normalized_depth(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, len(values))


def plot_individual_mean(*, output_dir: Path, curves: ModelCurves, with_std: bool) -> Path:
    with plt.rc_context(get_rc(ncols=1)):
        fig, ax = plt.subplots()
        layers = np.arange(len(curves.mean))
        color = PALETTE[0]
        if with_std:
            ax.fill_between(
                layers,
                np.clip(curves.mean - curves.std, 0.0, 1.0),
                np.clip(curves.mean + curves.std, 0.0, 1.0),
                color=color, alpha=STD_ALPHA, linewidth=0,
            )
        ax.plot(layers, curves.mean, color=color, marker="o", markersize=4, linewidth=1.5)
        _style_axis(ax)
        variant = "with_std" if with_std else "no_std"
        suffix = "_std" if with_std else ""
        return _save(fig, output_dir / variant / f"{curves.model.slug}{suffix}")


def _patch_legend(ax, model_curves: list[ModelCurves], ncol: int = 3, anchor: tuple = (0.5, -0.22)) -> None:
    handles = [
        mpatches.Patch(facecolor=MODEL_COLORS.get(c.model.label, "#333333"), label=c.model.label)
        for c in model_curves
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=anchor, ncol=ncol, frameon=False)


def plot_all_models_group(*, output_dir: Path, model_curves: list[ModelCurves], group_label: str) -> Path | None:
    if not model_curves:
        return None
    with plt.rc_context(get_rc(ncols=1)):
        fig, ax = plt.subplots()
        for curves in model_curves:
            sampled = curves.mean[1::2]
            x = _normalized_depth(sampled)
            color = MODEL_COLORS.get(curves.model.label, "#333333")
            ax.plot(x, sampled, color=color,
                    marker="o", markersize=5, linewidth=2, label=curves.model.label)
        _style_axis(ax, x_label="Normalized depth")
        _patch_legend(ax, model_curves, ncol=3, anchor=(0.5, -0.22))
        return _save(fig, output_dir / f"all_models_{group_label}")


def plot_all_models_combined(*, output_dir: Path, model_curves: list[ModelCurves]) -> Path | None:
    if not model_curves:
        return None
    with plt.rc_context(get_rc(ncols=1, **{"figure.figsize": (5.0, 4.4)})):
        fig, ax = plt.subplots()
        for curves in model_curves:
            sampled = curves.mean[1::2]
            x = _normalized_depth(sampled)
            color = MODEL_COLORS.get(curves.model.label, "#333333")
            ax.plot(x, sampled, color=color,
                    marker="o", markersize=5, linewidth=2, label=curves.model.label)
        _style_axis(ax, x_label="Normalized depth")
        _patch_legend(ax, model_curves, ncol=3, anchor=(0.5, -0.28))
        return _save(fig, output_dir / "all_models_combined")


def plot_datasets_by_model(*, output_dir: Path, curves: ModelCurves) -> Path:
    PLT= ('#f94144', '#f8961e', '#f9c74f', '#90be6d', '#469583', '#557CAF', '#0077B6')
    with plt.rc_context(get_rc(ncols=1)):
        fig, ax = plt.subplots()
        items = list(curves.dataset_curves.items())
        for index, (dataset_label, values) in enumerate(items):
            sampled = values[1::2]
            layers = np.arange(len(values))[1::2]
            ax.plot(layers, sampled, color=PLT[index % len(PLT)],
                    linewidth=2, marker="o", markersize=5, label=dataset_label)
        _style_axis(ax)
        handles = [
            mpatches.Patch(facecolor=PLT[i % len(PLT)], label=lbl)
            for i, (lbl, _) in enumerate(items)
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
        return _save(fig, output_dir / curves.model.slug)


def plot_datasets_by_model_grid(*, output_dir: Path, all_curves: list[ModelCurves]) -> Path | None:
    PLT = ('#f94144', '#f8961e', '#f9c74f', '#90be6d', '#469583', '#557CAF', '#0077B6')
    if not all_curves:
        return None
    with plt.rc_context(get_rc(ncols=3, nrows=2, **{"figure.figsize": (12, 7)})):
        fig, axes = plt.subplots(2, 3)
        axes_flat = axes.flatten()
        for ax, curves in zip(axes_flat, all_curves):
            items = list(curves.dataset_curves.items())
            for index, (dataset_label, values) in enumerate(items):
                sampled = values[1::2]
                layers = np.arange(len(values))[1::2]
                ax.plot(layers, sampled, color=PLT[index % len(PLT)],
                        linewidth=2, marker="o", markersize=5)
            _style_axis(ax)
            ax.set_title(curves.model.label)
            ax.set_ylabel("")
        for ax in axes[:, 0]:
            ax.set_ylabel("Label overlap")
        for ax in axes_flat[len(all_curves):]:
            ax.set_visible(False)

        items = list(all_curves[0].dataset_curves.items())
        handles = [
            mpatches.Patch(facecolor=PLT[i % len(PLT)], label=lbl)
            for i, (lbl, _) in enumerate(items)
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=4, frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        return _save(fig, output_dir / "datasets_by_model_grid")


def _filter_group(model_curves: list[ModelCurves], labels: tuple[str, ...]) -> list[ModelCurves]:
    label_set = set(labels)
    return [c for c in model_curves if c.model.label in label_set]


def main() -> None:
    models = [ModelSpec(*spec) for spec in MODEL_SPECS]
    datasets = [DatasetSpec(*spec) for spec in DATASET_SPECS]
    written = 0
    for n_choices in N_CHOICES:
        model_curves = collect_model_curves(
            data_dir=DATA_DIR, n_choices=n_choices, models=models, datasets=datasets,
        )
        if not model_curves:
            print(f"WARN: no complete model curves for n={n_choices}")
            continue
        n_dir = FIGURES_DIR / FAMILY / f"n{n_choices}"
        for curves in model_curves:
            if plot_individual_mean(output_dir=n_dir / "individual_mean", curves=curves, with_std=False):
                written += 1
            if plot_individual_mean(output_dir=n_dir / "individual_mean", curves=curves, with_std=True):
                written += 1
            if plot_datasets_by_model(output_dir=n_dir / "datasets_by_model", curves=curves):
                written += 1
        if plot_all_models_group(output_dir=n_dir / "all_models",
                                 model_curves=_filter_group(model_curves, GROUP_ONE),
                                 group_label="group_1"):
            written += 1
        if plot_all_models_group(output_dir=n_dir / "all_models",
                                 model_curves=_filter_group(model_curves, GROUP_TWO),
                                 group_label="group_2"):
            written += 1
        if plot_all_models_combined(output_dir=n_dir / "all_models",
                                    model_curves=model_curves):
            written += 1
        if plot_datasets_by_model_grid(output_dir=n_dir / "datasets_by_model",
                                       all_curves=model_curves):
            written += 1
    print(f"Wrote {written} figure file(s).")


if __name__ == "__main__":
    main()
