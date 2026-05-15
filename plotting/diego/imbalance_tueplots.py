"""Information-imbalance scatter grids in the tueplots NeurIPS style.

Adapted from diego/plot_imbalance2.py. Produces one two-panel figure per
model pair (pretrained left, finetuned right), averaging over tasks with
last-token open-ended NPY data.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tueplots_style import OUTPUT_DIR, get_rc  # noqa: E402

DATA_ROOT = Path("/u/dssc/lbasile/scratch/plm/diego/imbalance")
OUT_DIR = OUTPUT_DIR / "imbalance_grids"

TOKEN_TYPE = "last_token"
TASKS = ["vqav2", "cocoqa", "gqa", "mmbench", "seed"]
BACKGROUND_COLOR = "goldenrod"

MARKER_CONFIG = {
    "text_to_img": {"color": "#007e5d", "marker": "s", "zorder": 2},
    "img_to_text": {"color": "#991499", "marker": "o", "zorder": 2},
}

MODEL_PAIRS: tuple[tuple, ...] = (
    (
        "llava_7b", "llava-v1.5-7b-pretrained-lbasile", "LLaVA-7B",
        [5.5, 16.5, 5.5, 16.5],
        {"scatter_base_size": 80, "scatter_factor": 5,
         "thresholds": {"small": 0.025, "medium": 0.05, "large": 0.07}},
    ),
    (
        "llava_13b", "llava-v1.5-13b-pretrained-lbasile", "LLaVA-13B",
        [5.5, 16.5, 5.5, 16.5],
        {"scatter_base_size": 45, "scatter_factor": 5,
         "thresholds": {"small": 0.02, "medium": 0.04, "large": 0.06}},
    ),
    (
        "LLaVA-OneVision-1.5-4B-Instruct", "LLaVA-OneVision-1.5-4B-Base", "OneVision-4B",
        [7.5, 25.5, 7.5, 25.5],
        {"scatter_base_size": 60, "scatter_factor": 5,
         "thresholds": {"small": 0.04, "medium": 0.065, "large": 0.09}},
    ),
    (
        "LLaVA-OneVision-1.5-8B-Instruct", "LLaVA-OneVision-1.5-8B-Base", "OneVision-8B",
        [7.5, 25.5, 7.5, 25.5],
        {"scatter_base_size": 60, "scatter_factor": 5,
         "thresholds": {"small": 0.06, "medium": 0.09, "large": 0.12}},
    ),
    (
        "InternVL2-8B", "InternVL2-8B-Pretrain", "InternVL2-8B",
        [5.5, 20.5, 5.5, 20.5],
        {"scatter_base_size": 60, "scatter_factor": 5,
         "thresholds": {"small": 0.02, "medium": 0.04, "large": 0.06}},
    ),
    (
        "cambrian-8b", "cambrian-8b-pretrained", "Cambrian-8B",
        [4.5, 16.5, 4.5, 16.5],
        {"scatter_base_size": 80, "scatter_factor": 5,
         "thresholds": {"small": 0.025, "medium": 0.05, "large": 0.07}},
    ),
)


def _slugify(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower())


# ── data loading ──────────────────────────────────────────────────────────────

def load_model_data(model_id: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    model_dir = DATA_ROOT / model_id
    if not model_dir.is_dir():
        print(f"WARN: {model_id}: directory not found")
        return None, None

    it_list, ti_list = [], []
    for task in TASKS:
        task_dir = model_dir / task / TOKEN_TYPE
        for prefix in ("ft", "pre"):
            p_it = task_dir / f"{prefix}_ii_img_txt_dedup_open_ended.npy"
            p_ti = task_dir / f"{prefix}_ii_txt_img_dedup_open_ended.npy"
            if p_it.exists() and p_ti.exists():
                it_list.append(np.load(p_it))
                ti_list.append(np.load(p_ti))
                break
        else:
            print(f"WARN: {model_id}/{task}: data not found")

    if not it_list:
        return None, None
    return np.stack(it_list).mean(axis=0), np.stack(ti_list).mean(axis=0)


# ── scatter formatting (faithful to original) ─────────────────────────────────

def _get_marker_size(val: float, thresholds: dict, base_size: float, factor: float) -> float:
    if val < thresholds["small"]:
        return base_size
    elif val < thresholds["medium"]:
        return base_size / factor
    return base_size / (factor ** 2)


def _format_scatter(
    imbalance: np.ndarray, direction: str, thresholds: dict,
    base_size: float, factor: float, linewidth: float = 0.8,
) -> list[dict]:
    config = MARKER_CONFIG[direction]
    n = imbalance.shape[0]
    points_by_size: dict[float, dict] = {}
    for i in range(n):
        for j in range(n):
            val = imbalance[i, j]
            if val >= thresholds["large"]:
                continue
            size = _get_marker_size(val, thresholds, base_size, factor)
            if size not in points_by_size:
                points_by_size[size] = {"x": [], "y": []}
            points_by_size[size]["x"].append(i)
            points_by_size[size]["y"].append(j)
    datasets = []
    for size, pts in sorted(points_by_size.items(), reverse=True):
        datasets.append({
            "x": np.array(pts["x"]), "y": np.array(pts["y"]),
            "marker": config["marker"], "size": size,
            "linewidth": linewidth, "edgecolor": config["color"],
            "zorder": config["zorder"],
        })
    return datasets


def _plot_scatter(ax: plt.Axes, datasets: list[dict]) -> None:
    for data in datasets:
        sns.scatterplot(
            ax=ax, x=data["x"], y=data["y"],
            marker=data["marker"], s=data["size"],
            facecolors="None", edgecolors=data["edgecolor"],
            linewidths=data["linewidth"], label=None, zorder=data["zorder"],
        )


def _setup_axis(
    ax: plt.Axes, title: str, n_layers: int,
    highlight_region: list | None, thresholds: dict,
) -> None:
    if highlight_region is not None:
        xmin, xmax, ymin, ymax = highlight_region
        ax.axvspan(xmin, xmax, alpha=0.10, color=BACKGROUND_COLOR, zorder=0)
        ax.axhspan(ymin, ymax, alpha=0.10, color=BACKGROUND_COLOR, zorder=0)
        ax.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            alpha=0.20, facecolor=BACKGROUND_COLOR, edgecolor="none", zorder=0,
        ))
    ticks = np.arange(0, n_layers, 4) + 1
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(0.4, n_layers + 1)
    ax.set_ylim(0.4, n_layers + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("text", fontsize=15)
    ax.set_ylabel("visual", fontsize=15)
    ax.set_title(title, fontsize=16)


# ── legend ────────────────────────────────────────────────────────────────────

def _make_legend_handles(thresholds: dict) -> list:
    base_size, factor = 80, 5
    dir_kw = dict(color="w", markerfacecolor="none", linewidth=0)
    size_kw = dict(color="w", markerfacecolor="none", linewidth=0.8)
    # Row 1: direction markers (2 items)
    # Row 2: size markers (3 items, padded to same ncol with a blank)
    # Rendered with ncol=3 so row 1 gets a blank at the end, row 2 fills all 3.
    return [
        Line2D([0], [0], marker=MARKER_CONFIG["text_to_img"]["marker"],
               markeredgecolor=MARKER_CONFIG["text_to_img"]["color"],
               markersize=8, label="text → image", **dir_kw),
        Line2D([0], [0], marker=MARKER_CONFIG["img_to_text"]["marker"],
               markeredgecolor=MARKER_CONFIG["img_to_text"]["color"],
               markersize=8, label="image → text", **dir_kw),
        Patch(facecolor="none", edgecolor="none", label=" "),
        Line2D([0], [0], marker="o", markeredgecolor="black",
               markersize=np.sqrt(base_size),
               label=f'$\\Delta$ < {thresholds["small"]:.3f}', **size_kw),
        Line2D([0], [0], marker="o", markeredgecolor="black",
               markersize=np.sqrt(base_size / factor),
               label=f'{thresholds["small"]:.3f} ≤ $\\Delta$ < {thresholds["medium"]:.3f}', **size_kw),
        Line2D([0], [0], marker="o", markeredgecolor="black",
               markersize=np.sqrt(base_size / factor ** 2),
               label=f'{thresholds["medium"]:.3f} ≤ $\\Delta$ < {thresholds["large"]:.3f}', **size_kw),
    ]


# ── per-pair plot ─────────────────────────────────────────────────────────────

def plot_pair(
    fine_id: str,
    pre_id: str,
    model_title: str,
    highlight_region: list | None,
    style: dict,
) -> None:
    fine_it, fine_ti = load_model_data(fine_id)
    pre_it,  pre_ti  = load_model_data(pre_id)
    if fine_it is None or pre_it is None:
        print(f"WARN: skipping {model_title}")
        return

    base_size  = style["scatter_base_size"]
    factor     = style["scatter_factor"]
    thresholds = style["thresholds"]

    rc = get_rc(ncols=2, **{"figure.figsize": (8.8, 5.2),
                             "figure.constrained_layout.use": False})
    with plt.rc_context(rc):
        fig, (ax_pre, ax_fine) = plt.subplots(1, 2, figsize=(8.8, 5.2))

        for ax, img_text, text_img, subtitle in (
            (ax_pre,  pre_it,  pre_ti,  "Pretrained"),
            (ax_fine, fine_it, fine_ti, "Finetuned"),
        ):
            scatter_data = sorted(
                _format_scatter(text_img, "text_to_img", thresholds, base_size, factor) +
                _format_scatter(img_text, "img_to_text", thresholds, base_size, factor),
                key=lambda d: d["zorder"],
            )
            _plot_scatter(ax, scatter_data)
            _setup_axis(ax, subtitle, img_text.shape[0], highlight_region, thresholds)
            ax.set_xlabel("Layer (text input)", fontsize=15)

        ax_pre.set_ylabel("Layer (multimodal input)", fontsize=15)
        ax_fine.set_ylabel("")

        fig.suptitle(model_title, fontsize=22, y=1.0)

        handles = _make_legend_handles(thresholds)
        leg1 = fig.legend(
            handles=handles[:2],
            fontsize=15, loc="lower center", bbox_to_anchor=(0.5, -0.13),
            ncol=2, frameon=False, columnspacing=1.5,
        )
        fig.add_artist(leg1)
        fig.legend(
            handles=handles[3:],
            fontsize=15, loc="lower center", bbox_to_anchor=(0.5, -0.20),
            ncol=3, frameon=False, columnspacing=1.0,
        )

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)

        out_dir = OUT_DIR / f"{_slugify(pre_id)}_vs_{_slugify(fine_id)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "information_imbalance.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def main() -> None:
    sns.set_style("whitegrid", rc={"axes.edgecolor": ".15",
                                   "xtick.bottom": True, "ytick.left": True})
    for fine_id, pre_id, model_title, region, style in MODEL_PAIRS:
        plot_pair(fine_id, pre_id, model_title, region, style)


if __name__ == "__main__":
    main()
