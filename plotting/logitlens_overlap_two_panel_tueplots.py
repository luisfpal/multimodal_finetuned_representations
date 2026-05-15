"""Two-panel: multi-model logit-lens (MC) on the left, label-overlap group on the right."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "luis"))
from _tueplots_style import OUTPUT_DIR, get_rc, out_path  # noqa: E402
from logitlens_plots_tueplots import _gather_profiles  # noqa: E402
from label_overlap_tueplots import (  # noqa: E402
    DATA_DIR as LO_DATA_DIR,
    DATASET_SPECS,
    GROUP_ONE,
    MODEL_COLORS,
    MODEL_SPECS,
    DatasetSpec,
    ModelCurves,
    ModelSpec,
    _filter_group,
    _normalized_depth,
    collect_model_curves,
)


LOGITLENS_PANEL_MODELS = [
    ("llava_7b", "LLaVA-7B"),
    ("LLaVA-OneVision-1.5-4B-Instruct", "OneVision-4B"),
    ("InternVL2-8B", "InternVL2-8B"),
]


def _logitlens_relative_panel(ax, models):
    from matplotlib.lines import Line2D

    for slug, label in models:
        color = MODEL_COLORS.get(label, "#333333")
        mean_mc_ll, mean_mc_lp, _mean_qa_ll = _gather_profiles(slug)
        if mean_mc_ll is not None:
            x = _normalized_depth(mean_mc_ll[1::2])
            ax.plot(x, mean_mc_ll[1::2], color=color, marker="o", markersize=6,
                    linewidth=2.3, label=label)
        if mean_mc_lp is not None:
            x = _normalized_depth(mean_mc_lp[1::2])
            ax.plot(x, mean_mc_lp[1::2], color=color, marker="s", markersize=6,
                    linewidth=2.3, linestyle="--")

    ax.set_ylim(-0.11, 1.05)
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)

    style_handles = [
        Line2D([0], [0], color="black", marker="o", markersize=5, linewidth=2, label="Logit Lens"),
        Line2D([0], [0], color="black", marker="s", markersize=5, linewidth=2, linestyle="--", label="Linear Probe"),
    ]
    ax.legend(handles=style_handles, loc="lower right", bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=12)


def _all_models_panel(ax, model_curves):
    for curves in model_curves:
        sampled = curves.mean[1::2]
        x = _normalized_depth(sampled)
        color = MODEL_COLORS.get(curves.model.label, "#333333")
        ax.plot(x, sampled, color=color,
                marker="o", markersize=6, linewidth=2.3, label=curves.model.label)
    ax.set_ylim(-0.11, 1.05)
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Label overlap")
    ax.grid(True, alpha=0.3)


def plot_logitlens_overlap_two_panel(save_path=None, n_choices: int = 10):
    models = [ModelSpec(*spec) for spec in MODEL_SPECS]
    datasets = [DatasetSpec(*spec) for spec in DATASET_SPECS]
    model_curves = collect_model_curves(
        data_dir=LO_DATA_DIR, n_choices=n_choices, models=models, datasets=datasets,
    )
    overlap_curves = _filter_group(model_curves, GROUP_ONE)

    rc = get_rc(ncols=2, **{"figure.figsize": (11, 4.0)})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2)
        _all_models_panel(axes[0], overlap_curves)
        _logitlens_relative_panel(axes[1], LOGITLENS_PANEL_MODELS)

        patch_handles = [
            mpatches.Patch(facecolor=MODEL_COLORS.get(c.model.label, "#333333"), label=c.model.label)
            for c in overlap_curves
        ]
        fig.legend(
            handles=patch_handles,
            loc="lower center", bbox_to_anchor=(0.5, -0.08),
            ncol=len(patch_handles), frameon=False, fontsize=12,
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.22, wspace=0.25)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved {save_path}")
        plt.close(fig)


def main():
    plot_logitlens_overlap_two_panel(save_path=out_path("logitlens_overlap_two_panel.pdf"))


if __name__ == "__main__":
    main()
