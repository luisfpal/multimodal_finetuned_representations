"""Label-overlap profiles for the LLaMA Vision 3.2 experiment.

Mirrors the structure of the label-overlap paper plots but
targets a single model (Meta-Llama-3-90B-instruct) so we can inspect LLaMA
Vision results independently without touching the reference figures.

Produces three figure groups:
  individual_mean/no_std  — mean profile across datasets
  individual_mean/with_std — mean ± 1 std band
  datasets_by_model       — per-dataset curves overlaid

Usage:
  python plotting/plot_llama_vision_representations_classification.py
  python plotting/plot_llama_vision_representations_classification.py \\
    --results-dir /path/to/results --plots-dir /path/to/plots
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open

HOME_DIR = Path.home()
if "MPLCONFIGDIR" not in os.environ:
    for _mpl_dir in (
        HOME_DIR / "scratch" / "matplotlib",
        Path("/dev/shm") / f"matplotlib-{os.environ.get('USER', 'user')}",
    ):
        try:
            _mpl_dir.mkdir(parents=True, exist_ok=True)
            (_mpl_dir / ".write_test").write_text("", encoding="utf-8")
            (_mpl_dir / ".write_test").unlink()
        except OSError:
            continue
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
        break

import matplotlib.pyplot as plt  # noqa: E402


# ==============================================================================
# SECTION 1 — CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
RESULTS_DIR = PROJECT_ROOT / "results" / "representations_classification"
PLOTS_DIR = PROJECT_ROOT / "plots" / "representations_classification" / "llama_vision"

POOLING_METHOD = "last"
N_CHOICES = 10
CLASS_FRACTION = 0.3
MAXK: int | None = None

DPI_PNG = 200
STD_ALPHA = 0.18

FIGSIZE_MEAN = (3.35, 2.15)
FIGSIZE_DATASETS = (4.2, 3.0)

FONT_SIZE_TITLE = 12
FONT_SIZE_AXIS_LABEL = 10
FONT_SIZE_TICK_LABEL = 9
FONT_SIZE_LEGEND_LABEL = 8

LINE_WIDTH = 1.45
MARKER_SIZE = 3.0
MARKER_EDGE_WIDTH = 0.0
TICK_LENGTH = 3.0
TICK_WIDTH = 0.8
GRID_ALPHA = 0.18
LEGEND_HANDLE_LENGTH = 2.2
LEGEND_COLUMN_SPACING = 1.1
TITLE_PAD = 8

# Single model
MODEL_SLUG = "Meta-Llama-3-90B-instruct"
MODEL_LABEL = "LLaMA-3.2-Vision-90B-Instruct"

# Same 7 datasets as the LLaMA Vision shell script.
# Tuple: (dataset_slug, split, prompt_mode, display_label)
DATASET_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("mini_imagenet",      "validation", "mcq_nocat_filter",      "mini-ImageNet"),
    ("food101",            "validation", "generic_mcq_filter",     "Food101"),
    ("sun397",             "test",       "generic_mcq_filter",     "SUN397"),
    ("caltech101",         "train",      "generic_mcq_top_filter", "Caltech101"),
    ("oxford_flowers102",  "test",       "generic_mcq_top_filter", "Oxford Flowers 102"),
    ("places365",          "train",      "generic_mcq_filter",     "Places365"),
    ("dtd",                "train",      "generic_mcq_filter",     "DTD"),
)

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
)


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    split: str
    prompt_mode: str
    label: str


# ==============================================================================
# SECTION 2 — DATA LOADING
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LLaMA Vision label-overlap profiles."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument(
        "--model-slug",
        type=str,
        default=MODEL_SLUG,
        help=f"Model directory slug under results/ (default: {MODEL_SLUG})",
    )
    return parser.parse_args()


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}")


def _file_suffix(dataset: DatasetSpec) -> str:
    """Matches the suffix logic in llama_vision_representations_classification.py."""
    parts: list[str] = []
    if dataset.prompt_mode != "describe":
        parts.append(dataset.prompt_mode)
    parts_str = "_" + "_".join(parts) if parts else ""
    n_str = f"_n{N_CHOICES}"
    lo_parts: list[str] = []
    if CLASS_FRACTION is not None:
        lo_parts.append(f"cf{CLASS_FRACTION}")
    if MAXK is not None:
        lo_parts.append(f"mk{MAXK}")
    lo_str = ("_" + "_".join(lo_parts)) if lo_parts else ""
    return f"{parts_str}{n_str}{lo_str}"


def _label_overlap_path(
    results_dir: Path,
    model_slug: str,
    dataset: DatasetSpec,
) -> Path:
    return (
        results_dir
        / dataset.slug
        / dataset.split
        / model_slug
        / POOLING_METHOD
        / f"label_overlap{_file_suffix(dataset)}.safetensors"
    )


def _load_curve(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with safe_open(str(path), framework="numpy") as h:
            if "label_overlap" not in h.keys():
                _warn(f"Missing 'label_overlap' key in {path}")
                return None
            values = h.get_tensor("label_overlap").copy()
    except Exception as exc:
        _warn(f"Could not load {path}: {exc}")
        return None
    if values.size == 0:
        _warn(f"Empty tensor in {path}")
        return None
    return np.asarray(values, dtype=float)


def load_all_curves(
    results_dir: Path,
    model_slug: str,
    datasets: list[DatasetSpec],
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for ds in datasets:
        path = _label_overlap_path(results_dir, model_slug, ds)
        values = _load_curve(path)
        if values is None:
            _warn(f"Missing: {path}")
        else:
            curves[ds.label] = values
    return curves


def mean_std(curves: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    arrays = list(curves.values())
    if not arrays:
        return None
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        _warn(f"Curves have different lengths {sorted(lengths)}; cannot average.")
        return None
    stacked = np.stack(arrays, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0, ddof=1 if len(arrays) > 1 else 0)


# ==============================================================================
# SECTION 3 — PLOTTING
# ==============================================================================


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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
    })


def _style_axis(ax: plt.Axes, *, title: str | None, x_label: str = "Layer") -> None:
    if title:
        ax.set_title(title, pad=TITLE_PAD, fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel(x_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Label overlap", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(
        direction="out", length=TICK_LENGTH, width=TICK_WIDTH,
        color="#444444", labelsize=FONT_SIZE_TICK_LABEL,
    )
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        stem.parent / f"{stem.name}.png",
        stem.parent / f"{stem.name}.pdf",
    ]
    fig.savefig(outputs[0], dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    for p in outputs:
        print(f"  Saved: {p}")
    return outputs


def plot_mean(
    *,
    output_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    model_label: str,
    with_std: bool,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE_MEAN)
    x = np.arange(len(mean))
    color = PALETTE[0]
    if with_std:
        ax.fill_between(
            x,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color, alpha=STD_ALPHA, linewidth=0,
        )
    ax.plot(x, mean, color=color, marker="o", markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH)
    _style_axis(ax, title=model_label)
    fig.tight_layout()
    variant = "with_std" if with_std else "no_std"
    suffix = "_std" if with_std else ""
    return _save(fig, output_dir / variant / f"{MODEL_SLUG}{suffix}")


def plot_datasets(
    *,
    output_dir: Path,
    curves: dict[str, np.ndarray],
    model_label: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE_DATASETS)
    for idx, (label, values) in enumerate(curves.items()):
        ax.plot(
            np.arange(len(values)),
            values,
            color=PALETTE[idx % len(PALETTE)],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=label,
        )
    _style_axis(ax, title=model_label)
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
    return _save(fig, output_dir / MODEL_SLUG)


# ==============================================================================
# SECTION 4 — MAIN
# ==============================================================================


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()
    model_slug = args.model_slug

    datasets = [DatasetSpec(*spec) for spec in DATASET_SPECS]

    configure_style()

    print("=" * 72)
    print("  plot_llama_vision_representations_classification.py")
    print(f"  model:    {model_slug}")
    print(f"  datasets: {len(datasets)}")
    print(f"  n={N_CHOICES} | cf={CLASS_FRACTION} | pooling={POOLING_METHOD}")
    print(f"  results:  {results_dir}")
    print(f"  output:   {plots_dir}")
    print("=" * 72)

    curves = load_all_curves(results_dir, model_slug, datasets)
    if not curves:
        print("No data found. Check that the experiment has been run.")
        return

    print(f"\nLoaded {len(curves)}/{len(datasets)} dataset curves.")
    for label, arr in curves.items():
        print(f"  {label}: {len(arr)} layers")

    ms = mean_std(curves)
    n_dir = plots_dir / f"n{N_CHOICES}"

    written = 0
    if ms is not None:
        mean, std = ms
        written += len(plot_mean(
            output_dir=n_dir / "individual_mean",
            mean=mean, std=std,
            model_label=MODEL_LABEL, with_std=False,
        ))
        written += len(plot_mean(
            output_dir=n_dir / "individual_mean",
            mean=mean, std=std,
            model_label=MODEL_LABEL, with_std=True,
        ))
    else:
        print("Skipping mean figures (curves have different lengths).")

    written += len(plot_datasets(
        output_dir=n_dir / "datasets_by_model",
        curves=curves,
        model_label=MODEL_LABEL,
    ))

    print(f"\nFinished. Wrote {written} file(s).")


if __name__ == "__main__":
    main()
