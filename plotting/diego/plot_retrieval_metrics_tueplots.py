"""Plot retrieval metric bar grids for multiple VLMs using tueplots styling."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tueplots_style import get_rc  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "diego" / "embedding_similarity" / "retrieval_metrics"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tueplots" / "diego"

PART = "res2"       # embedding stream: "res2" or "norm1"
SCORE_KEY = "mean_percentile_rank"
METRICS = ["euclidean", "cosine"]

MODELS = [
    ("llava_7b",          "LLaVA-7B"),
    ("llava_13b",         "LLaVA-13B"),
    ("cambrian_8b",       "Cambrian-8B"),
    ("onevision_4b_hf",   "OneVision-4B"),
    ("onevision_8b_hf",   "OneVision-8B"),
    ("internvl_8b_hf",    "InternVL2-8B"),
]

DATASET_PATHS = {
    "coco_detail23k":    ("coco_retrieval_23k_detailed", "retrieval"),
    "coco_karpathy":     ("coco_karpathy_test",          "retrieval"),
    "winoground":        ("winoground_test",              "winoground"),
}

METRIC_PALETTE = ('#f94144', '#457b9d')
DIRECTION_LINESTYLE = {
    "imgavg_to_textavg": "-",
    "textavg_to_imgavg": "--",
    "last_to_last": "-",
}


def _load_paired(subdir, json_prefix, mode, model_slug):
    path = RESULTS_ROOT / subdir / model_slug / f"{json_prefix}_{mode}.json"
    if not path.exists():
        print(f"Missing: {path}")
        return None
    return json.loads(path.read_text())["results"]


def _panel_paired(ax, model_slug, model_label, subdir, json_prefix, mode, show_xlabel):
    results = _load_paired(subdir, json_prefix, mode, model_slug)
    if results is None:
        ax.set_title(model_label)
        return

    directions = {"last_vs_last": ["last_to_last"],
                  "imgavg_vs_textavg": ["imgavg_to_textavg", "textavg_to_imgavg"]}[mode]

    for m_idx, metric in enumerate(METRICS):
        data = results[metric][PART]
        sorted_layers = sorted(data[directions[0]].keys(), key=int)
        x = np.array([int(k) for k in sorted_layers])
        # average across directions
        ys = np.array([[data[d][k][SCORE_KEY] for k in sorted_layers] for d in directions], dtype=float)
        y = ys.mean(axis=0)
        ax.plot(x[1::2], y[1::2], color=METRIC_PALETTE[m_idx], linewidth=2, marker="o", markersize=5)

    ax.set_title(model_label)
    if show_xlabel:
        ax.set_xlabel("Layer")
    ax.grid(True, alpha=0.3)


def _load_winoground(subdir, json_prefix, mode, model_slug):
    path = RESULTS_ROOT / subdir / model_slug / f"{json_prefix}_{mode}.json"
    if not path.exists():
        print(f"Missing: {path}")
        return None
    return json.loads(path.read_text())["results"]


WINO_SCORES = ["text_score", "image_score", "group_score"]
WINO_PALETTE = ('#f94144', '#f8961e', '#90be6d')


def _panel_winoground(ax, model_slug, model_label, subdir, json_prefix, mode, show_xlabel):
    results = _load_winoground(subdir, json_prefix, mode, model_slug)
    if results is None:
        ax.set_title(model_label)
        return

    metric = METRICS[0]
    data = results[metric][PART]
    sorted_layers = sorted(data.keys(), key=int)
    x = np.array([int(k) for k in sorted_layers])

    for s_idx, score in enumerate(WINO_SCORES):
        y = np.array([data[k][score] for k in sorted_layers], dtype=float)
        ax.plot(x[1::2], y[1::2], color=WINO_PALETTE[s_idx], linewidth=2, marker="o", markersize=5)

    ax.axhline(0.25, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(1.0 / 6.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(model_label)
    if show_xlabel:
        ax.set_xlabel("Layer")
    ax.grid(True, alpha=0.3)


def _make_grid(panel_fn, legend_handles, ylabel, save_name, **panel_kwargs):
    with plt.rc_context(get_rc(ncols=3, nrows=2, **{"figure.figsize": (12, 7)})):
        fig, axes = plt.subplots(2, 3)
        axes_flat = axes.flatten()

        for i, (ax, (slug, label)) in enumerate(zip(axes_flat, MODELS)):
            panel_fn(ax, slug, label, show_xlabel=(i >= 3), **panel_kwargs)

        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)

        ncol = len(legend_handles)
        fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=ncol, frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / save_name
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


def plot_retrieval_grid(dataset="coco_detail23k", mode="imgavg_vs_textavg"):
    subdir, json_prefix = DATASET_PATHS[dataset]
    legend_handles = [
        mpatches.Patch(facecolor=METRIC_PALETTE[i], label=m)
        for i, m in enumerate(METRICS)
    ]
    _make_grid(
        _panel_paired,
        legend_handles,
        ylabel="Mean percentile rank",
        save_name=f"retrieval_{dataset}_{mode}.pdf",
        subdir=subdir,
        json_prefix=json_prefix,
        mode=mode,
    )


def plot_winoground_grid(mode="imgavg_vs_textavg"):
    subdir, json_prefix = DATASET_PATHS["winoground"]
    legend_handles = [
        mpatches.Patch(facecolor=WINO_PALETTE[i], label=s)
        for i, s in enumerate(WINO_SCORES)
    ]
    _make_grid(
        _panel_winoground,
        legend_handles,
        ylabel="Score",
        save_name=f"winoground_{mode}.pdf",
        subdir=subdir,
        json_prefix=json_prefix,
        mode=mode,
    )


def main():
    plot_retrieval_grid(dataset="coco_detail23k", mode="imgavg_vs_textavg")
    plot_retrieval_grid(dataset="coco_detail23k", mode="last_vs_last")
    plot_winoground_grid(mode="imgavg_vs_textavg")


if __name__ == "__main__":
    main()
