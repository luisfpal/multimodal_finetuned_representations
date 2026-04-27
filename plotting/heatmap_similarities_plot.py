"""Paper-ready paired pretrained/finetuned multimodality heatmaps.

This script intentionally keeps the original two-panel heatmap aesthetic from
the legacy version, but reads the current multimodality result layout:

    results/multimodality_comparison/<dataset>/last/unique_images/similarities/
        <model_id>_<metric>.safetensors
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open


# ==============================================================================
# SECTION 1 - CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
RESULTS_DIR = PROJECT_ROOT / "results" / "multimodality_comparison"
PLOTS_DIR = PROJECT_ROOT / "plots" / "similarity_grids"

POOLING_METHOD = "last"
VARIANT_TAG = "unique_images"
DPI_PNG = 200

METRICS: tuple[str, ...] = (
    "linear_cka",
    "neighborhood_overlap",
)

DATASETS_TO_AVERAGE: tuple[tuple[str, str], ...] = (
    ("cocoqa_restval_icqa_split01_human", "COCOQA Split01 (Human)"),
    ("gqa_val_balanced_noyesno_captioned_synthetic", "GQA"),
    ("vqav2_coco_restval_icq_split01_human", "VQAv2 Split01 (Human)"),
    ("mmbench_en_dev_captioned_open_ended_synthetic", "MMBench (Open-Ended)"),
    ("seed_bench_images_captioned_open_ended_synthetic", "SEED (Open-Ended)"),
)

# Tuple format:
#   (finetuned_id, finetuned_label, pretrained_id, pretrained_label)
MODEL_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    (
        "llava_7b",
        "finetuned",
        "llava-v1.5-7b-pretrained-lbasile",
        "pretrained",
    ),
    (
        "llava_13b",
        "finetuned",
        "llava-v1.5-13b-pretrained-lbasile",
        "pretrained",
    ),
    (
        "LLaVA-OneVision-1.5-4B-Instruct",
        "finetuned",
        "LLaVA-OneVision-1.5-4B-Base",
        "pretrained",
    ),
    (
        "LLaVA-OneVision-1.5-8B-Instruct",
        "finetuned",
        "LLaVA-OneVision-1.5-8B-Base",
        "pretrained",
    ),
    (
        "InternVL2-8B",
        "finetuned",
        "InternVL2-8B-Pretrain",
        "pretrained",
    ),
    (
        "cambrian-8b",
        "finetuned",
        "cambrian-8b-pretrained",
        "pretrained",
    ),
)

EXPECTED_SOURCE_ORIENTATION = ("text", "image")
DISPLAY_ORIENTATION = ("image", "text")


@dataclass(frozen=True)
class ModelPair:
    finetuned_id: str
    finetuned_label: str
    pretrained_id: str
    pretrained_label: str

    @property
    def slug(self) -> str:
        return f"{_slugify(self.pretrained_id)}_vs_{_slugify(self.finetuned_id)}"


# ==============================================================================
# SECTION 2 - DATA LOADING
# ==============================================================================


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return re.sub(r"_+", "_", slug)


def _similarities_path(dataset_dir: str, model_id: str, metric: str) -> Path:
    return (
        RESULTS_DIR
        / dataset_dir
        / POOLING_METHOD
        / VARIANT_TAG
        / "similarities"
        / f"{model_id}_{metric}.safetensors"
    )


def _normalize_modality_label(value: str | None, fallback: str) -> str:
    label = str(value or "").strip().lower()
    return label if label in {"text", "image"} else fallback


def _resolve_matrix_orientation(metadata: dict[str, str] | None) -> tuple[str, str]:
    if not metadata:
        return EXPECTED_SOURCE_ORIENTATION

    row_modality = _normalize_modality_label(
        metadata.get("row_modality"), EXPECTED_SOURCE_ORIENTATION[0]
    )
    col_modality = _normalize_modality_label(
        metadata.get("col_modality"), EXPECTED_SOURCE_ORIENTATION[1]
    )
    if row_modality == col_modality:
        return EXPECTED_SOURCE_ORIENTATION
    return row_modality, col_modality


def _orient_for_display(
    matrix: np.ndarray,
    source_orientation: tuple[str, str],
) -> np.ndarray:
    if source_orientation == DISPLAY_ORIENTATION:
        return matrix
    if source_orientation == EXPECTED_SOURCE_ORIENTATION:
        return matrix.T
    return matrix.T


def load_matrix(dataset_dir: str, model_id: str, metric: str) -> np.ndarray | None:
    path = _similarities_path(dataset_dir, model_id, metric)
    if not path.exists():
        return None

    try:
        with safe_open(path, framework="np", device="cpu") as f:
            if metric not in f.keys():
                print(f"  ⚠ missing key {metric!r}: {path}")
                return None
            matrix = f.get_tensor(metric).copy()
            source_orientation = _resolve_matrix_orientation(f.metadata())
    except Exception as exc:
        print(f"  ⚠ failed to load {path}: {exc}")
        return None

    matrix = _orient_for_display(matrix, source_orientation)
    if "information_imbalance" in metric:
        matrix = 1.0 - matrix
    return matrix


def average_over_datasets(model_id: str, metric: str) -> np.ndarray | None:
    matrices: list[np.ndarray] = []
    missing: list[str] = []

    for dataset_dir, _label in DATASETS_TO_AVERAGE:
        matrix = load_matrix(dataset_dir, model_id, metric)
        if matrix is None:
            missing.append(dataset_dir)
        else:
            matrices.append(matrix)

    if missing:
        print(f"  ⚠ {model_id} / {metric}: missing {len(missing)} dataset(s)")
        for dataset_dir in missing:
            print(f"     - {dataset_dir}")
        return None

    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        print(f"  ⚠ {model_id} / {metric}: shape mismatch {sorted(shapes)}")
        return None

    return np.mean(np.stack(matrices, axis=0), axis=0)


# ==============================================================================
# SECTION 3 - PLOTTING
# ==============================================================================


def _apply_publication_style(
    fontsize_title: int,
    fontsize_label: int,
    fontsize_ticks: int,
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.titlesize"] = fontsize_title
    plt.rcParams["axes.labelsize"] = fontsize_label
    plt.rcParams["xtick.labelsize"] = fontsize_ticks
    plt.rcParams["ytick.labelsize"] = fontsize_ticks
    plt.rcParams["figure.titlesize"] = fontsize_title + 2
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _layer_ticks(num_layers: int) -> range:
    return range(0, num_layers, 5)


def plot_heatmap_comparison(
    matrix_pretrained: np.ndarray,
    matrix_finetuned: np.ndarray,
    metric: str,
    pair: ModelPair,
    output_dir: Path,
    *,
    fontsize_title: int = 22,
    fontsize_label: int = 20,
    fontsize_ticks: int = 18,
    fontsize_cbar: int = 18,
    trim_whitespace: bool = True,
) -> list[Path]:
    """Generate the original-style side-by-side heatmap comparison."""
    _apply_publication_style(fontsize_title, fontsize_label, fontsize_ticks)

    vmin = float(min(matrix_pretrained.min(), matrix_finetuned.min()))
    vmax = float(max(matrix_pretrained.max(), matrix_finetuned.max()))

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1, 1],
        wspace=0.1,
        left=0.05,
        right=0.92,
        top=0.85,
        bottom=0.10,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    im1 = ax1.imshow(
        matrix_pretrained,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    ax2.imshow(
        matrix_finetuned,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )

    for ax, title, matrix in (
        (ax1, pair.pretrained_label, matrix_pretrained),
        (ax2, pair.finetuned_label, matrix_finetuned),
    ):
        ax.set_title(
            title,
            pad=16,
            color="#222222",
            fontsize=fontsize_title,
        )
        ax.set_xlabel("text input", labelpad=12, color="#333333", fontsize=fontsize_label)
        ax.set_ylabel("visual input", labelpad=12, color="#333333", fontsize=fontsize_label)
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=6,
            width=1.5,
            color="#444444",
            labelsize=fontsize_ticks,
        )
        ax.set_xticks(_layer_ticks(matrix.shape[1]))
        ax.set_yticks(_layer_ticks(matrix.shape[0]))
        ax.grid(False)

    fig.canvas.draw()
    pos = ax2.get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(im1, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0, labelsize=fontsize_cbar)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename_stem = f"{metric}_average_{len(DATASETS_TO_AVERAGE)}datasets"

    save_kwargs = {"facecolor": "white"}
    if trim_whitespace:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = 0.1

    outputs = [
        output_dir / f"{filename_stem}.png",
        output_dir / f"{filename_stem}.pdf",
    ]
    fig.savefig(outputs[0], dpi=DPI_PNG, **save_kwargs)
    fig.savefig(outputs[1], **save_kwargs)
    plt.close(fig)
    return outputs


# ==============================================================================
# SECTION 4 - MAIN
# ==============================================================================


def main() -> None:
    pairs = [ModelPair(*pair) for pair in MODEL_PAIRS]
    paper_dir = PLOTS_DIR / POOLING_METHOD / "paper"

    print("=" * 72)
    print("  heatmap_similarities_plot.py")
    print(f"  pooling={POOLING_METHOD} | variant={VARIANT_TAG}")
    print(f"  datasets={len(DATASETS_TO_AVERAGE)} | metrics={', '.join(METRICS)}")
    print(f"  output={paper_dir}")
    print("=" * 72)

    written: list[Path] = []
    skipped: list[str] = []

    for pair in pairs:
        pair_dir = paper_dir / pair.slug
        print(f"\n{pair.pretrained_id}  ->  {pair.finetuned_id}")

        for metric in METRICS:
            print(f"  metric={metric}")
            pretrained = average_over_datasets(pair.pretrained_id, metric)
            finetuned = average_over_datasets(pair.finetuned_id, metric)

            if pretrained is None or finetuned is None:
                skipped.append(f"{pair.slug}/{metric}")
                print("    skipped: incomplete data")
                continue

            if pretrained.shape != finetuned.shape:
                skipped.append(f"{pair.slug}/{metric}")
                print(
                    "    skipped: pretrained/finetuned shape mismatch "
                    f"{pretrained.shape} vs {finetuned.shape}"
                )
                continue

            outputs = plot_heatmap_comparison(
                pretrained,
                finetuned,
                metric,
                pair,
                pair_dir,
            )
            written.extend(outputs)
            for output in outputs:
                print(f"    ✓ {output}")

    print("\n" + "=" * 72)
    print(f"  wrote {len(written)} file(s)")
    print(f"  skipped {len(skipped)} plot(s)")
    for item in skipped:
        print(f"    - {item}")
    print("=" * 72)


if __name__ == "__main__":
    main()
