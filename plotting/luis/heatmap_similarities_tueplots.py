"""Heatmap-similarity plots in the tueplots NeurIPS style."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tueplots_style import OUTPUT_DIR, get_rc  # noqa: E402

LUIS_ROOT = Path("/u/dssc/lbasile/scratch/plm/luis")
DATA_DIR = LUIS_ROOT / "data" / "multimodality"
FIGURES_DIR = OUTPUT_DIR / "heatmap_similarities"

POOLING_METHOD = "last"
VARIANT_TAG = "unique_images"

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

MODEL_PAIRS: tuple[tuple[str, str, str, str, str], ...] = (
    ("llava_7b", "finetuned", "llava-v1.5-7b-pretrained-lbasile", "pretrained", "LLaVA-7B"),
    ("llava_13b", "finetuned", "llava-v1.5-13b-pretrained-lbasile", "pretrained", "LLaVA-13B"),
    ("LLaVA-OneVision-1.5-4B-Instruct", "finetuned", "LLaVA-OneVision-1.5-4B-Base", "pretrained", "OneVision-4B"),
    ("LLaVA-OneVision-1.5-8B-Instruct", "finetuned", "LLaVA-OneVision-1.5-8B-Base", "pretrained", "OneVision-8B"),
    ("InternVL2-8B", "finetuned", "InternVL2-8B-Pretrain", "pretrained", "InternVL2-8B"),
    ("cambrian-8b", "finetuned", "cambrian-8b-pretrained", "pretrained", "Cambrian-8B"),
)

EXPECTED_SOURCE_ORIENTATION = ("text", "image")
DISPLAY_ORIENTATION = ("image", "text")


@dataclass(frozen=True)
class ModelPair:
    finetuned_id: str
    finetuned_label: str
    pretrained_id: str
    pretrained_label: str
    model_title: str

    @property
    def slug(self) -> str:
        return f"{_slugify(self.pretrained_id)}_vs_{_slugify(self.finetuned_id)}"


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower())


def _similarities_path(dataset_dir: str, model_id: str, metric: str) -> Path:
    return (
        DATA_DIR / dataset_dir / POOLING_METHOD / VARIANT_TAG / "similarities"
        / f"{model_id}_{metric}.json"
    )


def _normalize_modality_label(value: str | None, fallback: str) -> str:
    label = str(value or "").strip().lower()
    return label if label in {"text", "image"} else fallback


def _resolve_orientation(metadata: dict[str, str] | None) -> tuple[str, str]:
    if not metadata:
        return EXPECTED_SOURCE_ORIENTATION
    row = _normalize_modality_label(metadata.get("row_modality"), EXPECTED_SOURCE_ORIENTATION[0])
    col = _normalize_modality_label(metadata.get("col_modality"), EXPECTED_SOURCE_ORIENTATION[1])
    if row == col:
        return EXPECTED_SOURCE_ORIENTATION
    return row, col


def _orient(matrix: np.ndarray, source: tuple[str, str]) -> np.ndarray:
    if source == DISPLAY_ORIENTATION:
        return matrix
    return matrix.T


def load_matrix(dataset_dir: str, model_id: str, metric: str) -> np.ndarray | None:
    path = _similarities_path(dataset_dir, model_id, metric)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"WARN: invalid JSON {path}: {exc}")
        return None
    matrix = np.asarray(payload.get("matrix"), dtype=float)
    return _orient(matrix, _resolve_orientation(payload.get("metadata") or {}))


def average_over_datasets(model_id: str, metric: str) -> np.ndarray | None:
    matrices, missing = [], []
    for dataset_dir, _label in DATASETS_TO_AVERAGE:
        m = load_matrix(dataset_dir, model_id, metric)
        if m is None:
            missing.append(dataset_dir)
        else:
            matrices.append(m)
    if missing:
        print(f"WARN: {model_id} / {metric}: missing {len(missing)} dataset(s)")
        for d in missing:
            print(f"  - {d}")
        return None
    shapes = {m.shape for m in matrices}
    if len(shapes) != 1:
        print(f"WARN: {model_id} / {metric}: shape mismatch {sorted(shapes)}")
        return None
    return np.mean(np.stack(matrices, axis=0), axis=0)


def _layer_ticks(num_layers: int) -> np.ndarray:
    return np.arange(0, num_layers, 4) + 1


def plot_heatmap_comparison(
    matrix_pretrained: np.ndarray,
    matrix_finetuned: np.ndarray,
    metric: str,
    pair: ModelPair,
    output_dir: Path,
) -> Path:
    vmin = float(min(matrix_pretrained.min(), matrix_finetuned.min()))
    vmax = float(max(matrix_pretrained.max(), matrix_finetuned.max()))

    rc = get_rc(ncols=2, **{"figure.figsize": (8.8, 5.2), "figure.constrained_layout.use": False})
    with plt.rc_context(rc):
        fig = plt.figure()
        gs = gridspec.GridSpec(
            1, 2, width_ratios=[1, 1], wspace=0.08,
            left=0.08, right=0.93, top=0.88, bottom=0.20,
        )
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        im1 = ax1.imshow(matrix_pretrained, cmap="Blues", vmin=vmin, vmax=vmax,
                         origin="lower", interpolation="nearest")
        ax2.imshow(matrix_finetuned, cmap="Blues", vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest")

        for ax, subtitle, matrix in (
            (ax1, "Pretrained", matrix_pretrained),
            (ax2, "Finetuned", matrix_finetuned),
        ):
            ax.set_title(subtitle, fontsize=16)
            ax.set_xlabel("Layer (text input)", fontsize=15)
            ax.set_xticks(_layer_ticks(matrix.shape[1]))
            ax.set_yticks(_layer_ticks(matrix.shape[0]))
            ax.grid(False)

        ax1.set_ylabel("Layer (multimodal input)", fontsize=15)
        ax2.set_ylabel("")

        fig.suptitle(pair.model_title, fontsize=22, y=1.0)

        fig.canvas.draw()
        pos = ax2.get_position()
        cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.02, pos.height])
        cbar = fig.colorbar(im1, cax=cax)
        cbar.outline.set_visible(False)

        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{metric}_average_{len(DATASETS_TO_AVERAGE)}datasets.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
        return out


def main() -> None:
    pairs = [ModelPair(*p) for p in MODEL_PAIRS]
    paper_dir = FIGURES_DIR / POOLING_METHOD / "paper"
    skipped = []
    for pair in pairs:
        pair_dir = paper_dir / pair.slug
        for metric in METRICS:
            pre = average_over_datasets(pair.pretrained_id, metric)
            fine = average_over_datasets(pair.finetuned_id, metric)
            if pre is None or fine is None or pre.shape != fine.shape:
                skipped.append(f"{pair.slug}/{metric}")
                continue
            plot_heatmap_comparison(pre, fine, metric, pair, pair_dir)
    if skipped:
        print(f"Skipped {len(skipped)} plot(s):")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
