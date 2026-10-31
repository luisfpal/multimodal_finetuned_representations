"""Plot logit lens accuracy profiles across layers for multiple VLMs and datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file

from _tueplots_style import get_rc, OUTPUT_DIR, out_path


# Hardcoded path: edit this path for your setup.
RESULTS_DIR = Path("/u/dssc/lbasile/scratch/plm/results/logitlens_evaluation")

DATASETS = [
    ("l11p/COCOQA-restval-ICQA", "images_qa"),
    ("l11p/VQAv2-COCO-restval-ICQ", "images_qa"),
    ("lmms-lab/ScienceQA", "images_qa"),
    ("l11p/MME-test-captioned", "images_qa"),
    ("l11p/MMBench_EN-dev-captioned", "images_qa"),
    ("l11p/GQA-val-balanced-noyesno-captioned", "images_qa"),
    ("lmms-lab/POPE", "images_qa"),
    ("Lin-Chen/MMStar", "images_qa"),
    ("vikhyatk/CountBenchQA", "images_qa"),
]

GROUP_MC = {
    "l11p/MME-test-captioned",
    "l11p/MMBench_EN-dev-captioned",
    "lmms-lab/POPE",
    "lmms-lab/ScienceQA", "images_qa"
    # "lmms-lab/ai2d"
    #"Lin-Chen/MMStar",
}
GROUP_QA = {
    "l11p/GQA-val-balanced-noyesno-captioned",
    "l11p/COCOQA-restval-ICQA",
    "l11p/VQAv2-COCO-restval-ICQ",
    "vikhyatk/CountBenchQA",
}

BASELINES = {
    "scienceqa": 0.2,
    "mmbench_en_dev_captioned": 0.25,
    "mme_test_captioned": 0.5,
    "cocoqa_restval_icqa": 0.0,
    "vqav2_coco_restval_icq": 0.0,
    "gqa_val_balanced_noyesno_captioned": 0.0,
    "ai2d": 0.25,
    "pope": 0.5,
    "mmstar": 0.25,
    "countbenchqa": 0.0,
}

MODELS = [
    ("llava_7b", "LLaVA-7B", (5.5, 16.5)),
    ("llava_13b", "LLaVA-13B", None),
    ("cambrian-8b", "Cambrian-8B", None),
    ("LLaVA-OneVision-1.5-8B-Instruct", "LLaVA-OneVision-1.5-8B", None),
    ("LLaVA-OneVision-1.5-4B-Instruct", "LLaVA-OneVision-1.5-4B", (7.5, 26.5)),
    ("InternVL2-8B", "InternVL2-8B", None),
]


def _load_accuracies(model_slug, dataset_slug, subset, mode):
    path = RESULTS_DIR / f"{model_slug}_{dataset_slug}_{subset}_{mode}.safetensors"
    if path.exists():
        return load_file(str(path))["accuracies"]
    if mode == "logit_lens":
        legacy = RESULTS_DIR / f"{model_slug}_{dataset_slug}_{subset}.safetensors"
        if legacy.exists():
            return load_file(str(legacy))["accuracies"]
    return None


def _normalize(arr, baseline):
    if arr is None:
        return None
    max_val = float(np.max(arr))
    denom = max_val - baseline if max_val > baseline else 1.0
    return (arr - baseline) / denom


def _avg(arrays):
    if not arrays:
        return None
    min_len = min(len(a) for a in arrays)
    clipped = [a[:min_len] for a in arrays]
    return np.mean(clipped, axis=0)


def _gather_profiles(model_slug):
    mc_ll, mc_lp, qa_ll = [], [], []
    for dataset_name, subset in DATASETS:
        ds_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
        baseline = BASELINES.get(ds_slug, 0.0)
        norm_ll = _normalize(_load_accuracies(model_slug, ds_slug, subset, "logit_lens"), baseline)
        norm_lp = _normalize(_load_accuracies(model_slug, ds_slug, subset, "linear_probe"), baseline)
        if dataset_name in GROUP_MC:
            if norm_ll is not None:
                mc_ll.append(norm_ll)
            if norm_lp is not None:
                mc_lp.append(norm_lp)
        if dataset_name in GROUP_QA:
            if norm_ll is not None:
                qa_ll.append(norm_ll)
    return _avg(mc_ll), _avg(mc_lp), _avg(qa_ll)


def _logitlens_panel(ax, model_slug, abstraction, title=None, show_legend=True):
    mean_mc_ll, mean_mc_lp, mean_qa_ll = _gather_profiles(model_slug)

    if abstraction is not None:
        ax.axvspan(abstraction[0], abstraction[1], alpha=0.2, color="goldenrod",
                   zorder=0, label="abstraction phase")

    if mean_mc_ll is not None:
        x = np.arange(len(mean_mc_ll))[1::2]
        ax.plot(x, mean_mc_ll[1::2], color="darkslategray", marker="o", markersize=4,
                linewidth=1, label="multiple choice (LL)")
    if mean_mc_lp is not None:
        x = np.arange(len(mean_mc_lp))[1::2]
        ax.plot(x, mean_mc_lp[1::2], color="darkslategray", marker="s", markersize=4,
                linewidth=1, linestyle="--", label="multiple choice (LP)")
    if mean_qa_ll is not None:
        x = np.arange(len(mean_qa_ll))[1::2]
        ax.plot(x, mean_qa_ll[1::2], color="darkred", marker="s", markersize=4,
                linewidth=1, label="open ended")

    lengths = [len(p) for p in (mean_mc_ll, mean_mc_lp, mean_qa_ll) if p is not None]
    n_layers = max(lengths) if lengths else 32
    step = 4 if n_layers <= 36 else 8
    ax.set_xticks(list(range(0, n_layers + 1, step)))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Layers")
    ax.set_ylabel("Performance")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)


def plot_logitlens_per_model(save_dir=None):
    if save_dir is None:
        save_dir = OUTPUT_DIR
    rc = get_rc(
        ncols=1,
        **{
            "figure.figsize": (5.5, 4),
            "figure.constrained_layout.use": True,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        },
    )

    out_dir = Path(save_dir)
    for model_slug, label, abstraction in MODELS:
        with plt.rc_context(rc):
            fig, ax = plt.subplots()
            _logitlens_panel(ax, model_slug, abstraction, title=label)
            save_path = out_dir / f"logitlens_{model_slug}.pdf"
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved {save_path}")
            plt.close(fig)


def plot_logitlens_two_panel(save_path=None):
    rc = get_rc(
        ncols=2,
        **{
            "figure.figsize": (11, 4.2),
            "figure.constrained_layout.use": False,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        },
    )

    panels = [
        ("llava_7b", "LLaVA-7B", (5.5, 16.5)),
        ("LLaVA-OneVision-1.5-4B-Instruct", "LLaVA-OneVision-1.5-4B", (7.5, 26.5)),
    ]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2)
        for ax, (slug, label, abstraction) in zip(axes, panels):
            _logitlens_panel(ax, slug, abstraction, title=label, show_legend=False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.22, wspace=0.22)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved {save_path}")
        plt.close(fig)


MC_PALETTE = ('#f94144', '#f8961e', '#f9c74f', '#90be6d')
OE_PALETTE = ('#469583', '#557CAF', '#0077B6', '#9e2a2b')

MC_DATASETS = [
    # ("ai2d",                              "AI2D",      "images_qa"),
    ("scienceqa",                         "ScienceQA", "images_qa"),
    ("mme_test_captioned",                "MME",       "images_qa"),
    ("mmbench_en_dev_captioned",          "MMBench",   "images_qa"),
    ("pope",                              "POPE",      "images_qa"),
    # ("mmstar",                            "MMStar",    "images_qa"),
]

OE_DATASETS = [
    ("gqa_val_balanced_noyesno_captioned", "GQA",        "images_qa"),
    ("cocoqa_restval_icqa",                "COCOQA",     "images_qa"),
    ("vqav2_coco_restval_icq",             "VQAv2",      "images_qa"),
    ("countbenchqa",                       "CountBench", "images_qa"),
]

MODELS_GRID = [
    ("llava_7b",                        "LLaVA-7B"),
    ("llava_13b",                       "LLaVA-13B"),
    ("cambrian-8b",                     "Cambrian-8B"),
    ("LLaVA-OneVision-1.5-4B-Instruct", "OneVision-4B"),
    ("LLaVA-OneVision-1.5-8B-Instruct", "OneVision-8B"),
    ("InternVL2-8B",                    "InternVL2-8B"),
]


def _gather_per_dataset(model_slug, dataset_list):
    result = {}
    for ds_slug, ds_label, subset in dataset_list:
        baseline = BASELINES.get(ds_slug, 0.0)
        ll = _normalize(_load_accuracies(model_slug, ds_slug, subset, "logit_lens"), baseline)
        lp = _normalize(_load_accuracies(model_slug, ds_slug, subset, "linear_probe"), baseline)
        result[ds_label] = {"ll": ll, "lp": lp}
    return result


def _logitlens_grid_panel(ax, model_slug, model_label, dataset_list, palette, mode, ylim, show_xlabel=True):
    per_ds = _gather_per_dataset(model_slug, dataset_list)
    all_lens = []
    for idx, (_, ds_label, _) in enumerate(dataset_list):
        color = palette[idx % len(palette)]
        data = per_ds[ds_label]
        arr = data["ll"] if mode == "ll" else data["lp"]
        if arr is not None:
            x = np.arange(len(arr))[1::2]
            ax.plot(x, arr[1::2], color=color, linewidth=2, marker="o", markersize=5)
            all_lens.append(len(arr))

    n_layers = max(all_lens) if all_lens else 32
    step = 4 if n_layers <= 36 else 8
    ax.set_xticks(list(range(0, n_layers + 1, step)))
    ax.set_ylim(*ylim)
    if show_xlabel:
        ax.set_xlabel("Layers")
    ax.set_title(model_label)
    ax.grid(True, alpha=0.3)


def _make_logitlens_grid(dataset_list, palette, mode, ylim, save_path):
    import matplotlib.patches as mpatches

    n_ds = len(dataset_list)

    with plt.rc_context(get_rc(ncols=3, nrows=2, **{"figure.figsize": (12, 7)})):
        fig, axes = plt.subplots(2, 3)
        axes_flat = axes.flatten()

        for i, (ax, (model_slug, model_label)) in enumerate(zip(axes_flat, MODELS_GRID)):
            show_xlabel = i >= 3
            _logitlens_grid_panel(ax, model_slug, model_label, dataset_list, palette, mode, ylim,
                                  show_xlabel=show_xlabel)

        for ax in axes[:, 0]:
            ax.set_ylabel("Accuracy")

        handles = [
            mpatches.Patch(facecolor=palette[i % len(palette)], label=ds_label)
            for i, (_, ds_label, _) in enumerate(dataset_list)
        ]
        ncol = (n_ds + 1) // 2
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.04), ncol=ncol, frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close(fig)


def plot_logitlens_grid_lp_mc(save_path=None):
    if save_path is None:
        save_path = out_path("logitlens_grid_lp_mc.pdf")
    _make_logitlens_grid(MC_DATASETS, MC_PALETTE, "lp", (-0.37, 1.05), save_path)


def plot_logitlens_grid_ll_mc(save_path=None):
    if save_path is None:
        save_path = out_path("logitlens_grid_ll_mc.pdf")
    _make_logitlens_grid(MC_DATASETS, MC_PALETTE, "ll", (-0.30, 1.05), save_path)


def plot_logitlens_grid_ll_oe(save_path=None):
    if save_path is None:
        save_path = out_path("logitlens_grid_ll_oe.pdf")
    _make_logitlens_grid(OE_DATASETS, OE_PALETTE, "ll", (-0.05, 1.05), save_path)


def main():
    plot_logitlens_per_model()
    plot_logitlens_two_panel(save_path=out_path("logitlens_two_panel.pdf"))
    plot_logitlens_grid_lp_mc()
    plot_logitlens_grid_ll_mc()
    plot_logitlens_grid_ll_oe()


if __name__ == "__main__":
    main()
