import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

alpha = 0.9
GRIDE_SCALE = 0  # which GRIDE scale index to use (out of 6)
USE_AVG_TOKEN = False  # set True to use avg_token instead of last_token

sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

base_path = "results/intrinsic_dim"

models = {
    "LLaVA-7B": {
        "pre": "llava-v1.5-7b-pretrained-lbasile",
        "ft": "llava_7b",
    },
    "LLaVA-13B": {
        "pre": "llava-v1.5-13b-pretrained-lbasile",
        "ft": "llava_13b",
    },
    "InternVL2-8B": {
        "pre": "InternVL2-8B-Pretrain",
        "ft": "InternVL2-8B",
    },
}

datasets = ["vqav2", "cocoqa", "gqa", "mme", "mmbench"]

# txt file suffix differs per dataset
txt_suffix = {
    "vqav2": "human_txt",
    "cocoqa": "human_txt",
    "gqa": "synthetic_txt",
    "mme": "synthetic_txt",
    "mmbench": "synthetic_txt",
}


def load_ids(model_dir, dataset, ft_mode, data_type):
    """Load ID array (num_layers, num_scales) if file exists, else None."""
    if data_type == "txt":
        suffix = txt_suffix[dataset]
    else:
        suffix = data_type
    token_dir = "avg_token" if USE_AVG_TOKEN else "last_token"
    fpath = f"{base_path}/{model_dir}/{dataset}/{token_dir}/{ft_mode}_id_{suffix}.npy"
    if os.path.exists(fpath):
        return np.load(fpath)[:, GRIDE_SCALE]
    return None


def collect_mean_std(model_dir, ft_mode, data_type, subset=None):
    """
    Returns (layers, mean, std) averaged over all available datasets
    (or a given subset) for the given model_dir, ft_mode and data_type.
    """
    ds_list = subset if subset is not None else datasets
    arrays = []
    for ds in ds_list:
        ids = load_ids(model_dir, ds, ft_mode, data_type)
        if ids is not None:
            arrays.append(ids)
    if not arrays:
        return None, None, None
    arr = np.stack(arrays, axis=0)  # (n_datasets, n_layers)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    layers = np.arange(arr.shape[1])
    return layers, mean, std


dataset_groups = {
    "gqa / cocoqa / vqav2": ["gqa", "cocoqa", "vqav2"],
    "mme / mmbench": ["mme", "mmbench"],
}
group_markers = {"gqa / cocoqa / vqav2": "o", "mme / mmbench": "s"}


colors = {"pre": "#4878CF", "ft": "#D65F5F"}
labels = {"pre": "Pretrained", "ft": "Finetuned"}
linestyles = {"pre": "--", "ft": "-"}

# Goldenrod axvspan extent per model (layer range of "abstraction phase")
axvspan_ranges = {
    "LLaVA-7B": (5, 16),
    "LLaVA-13B": (5, 20),
    "InternVL2-8B": (5, 16),
}

dataset_colors = sns.color_palette("tab10", n_colors=len(datasets))
dataset_color_map = dict(zip(datasets, dataset_colors))

os.makedirs("figures", exist_ok=True)


for model_label, dirs in models.items():
    # ── Figure 1: average over datasets ──────────────────────────────────
    fig = plt.figure(figsize=(10, 4.2))
    fig.suptitle(model_label, fontsize=13, fontweight="bold")
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    vspan_lo, vspan_hi = axvspan_ranges[model_label]

    for col, data_type in enumerate(["img", "txt"]):
        ax = fig.add_subplot(gs[col])
        panel_title = (
            "Image representations" if data_type == "img" else "Text representations"
        )
        ax.set_title(panel_title)
        ax.axvspan(vspan_lo, vspan_hi, alpha=0.15, color="goldenrod", zorder=0)

        for ft_mode in ["pre", "ft"]:
            model_dir = dirs[ft_mode]
            layers, mean, std = collect_mean_std(model_dir, ft_mode, data_type)
            if layers is None:
                continue

            ax.plot(
                layers,
                mean,
                color=colors[ft_mode],
                linestyle=linestyles[ft_mode],
                marker="o",
                markersize=4,
                label=labels[ft_mode],
                alpha=alpha,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Intrinsic dimension")
        ax.legend()

    out_path = f"figures/id_{model_label.replace(' ', '_').lower()}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")

    # ── Figure 2: all datasets individually ──────────────────────────────
    fig2 = plt.figure(figsize=(12, 5))
    fig2.suptitle(f"{model_label} — per dataset", fontsize=13, fontweight="bold")
    gs2 = GridSpec(1, 2, figure=fig2, wspace=0.3)

    for col, data_type in enumerate(["img", "txt"]):
        ax2 = fig2.add_subplot(gs2[col])
        panel_title = (
            "Image representations" if data_type == "img" else "Text representations"
        )
        ax2.set_title(panel_title)
        ax2.axvspan(vspan_lo, vspan_hi, alpha=0.15, color="goldenrod", zorder=0)

        for ds in datasets:
            color = dataset_color_map[ds]
            for ft_mode in ["pre", "ft"]:
                model_dir = dirs[ft_mode]
                ids = load_ids(model_dir, ds, ft_mode, data_type)
                if ids is None:
                    continue
                layers = np.arange(len(ids))
                ax2.plot(
                    layers,
                    ids,
                    color=color,
                    linestyle=linestyles[ft_mode],
                    marker="o",
                    markersize=4,
                    alpha=alpha,
                    label=f"{ds} ({labels[ft_mode]})",
                )

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Intrinsic dimension")
        ax2.legend(fontsize=7, ncol=2)

    out_path2 = f"figures/id_{model_label.replace(' ', '_').lower()}_per_dataset.pdf"
    fig2.savefig(out_path2, bbox_inches="tight")
    print(f"Saved {out_path2}")

    # ── Figure 3: grouped averages (mme+mmbench vs gqa+cocoqa+vqav2) ────
    fig3 = plt.figure(figsize=(10, 4.2))
    fig3.suptitle(f"{model_label} — grouped datasets", fontsize=13, fontweight="bold")
    gs3 = GridSpec(1, 2, figure=fig3, wspace=0.3)

    for col, data_type in enumerate(["img", "txt"]):
        ax3 = fig3.add_subplot(gs3[col])
        panel_title = (
            "Image representations" if data_type == "img" else "Text representations"
        )
        ax3.set_title(panel_title)
        ax3.axvspan(vspan_lo, vspan_hi, alpha=0.15, color="goldenrod", zorder=0)

        for group_name, group_ds in dataset_groups.items():
            marker = group_markers[group_name]
            for ft_mode in ["pre", "ft"]:
                model_dir = dirs[ft_mode]
                layers, mean, std = collect_mean_std(
                    model_dir, ft_mode, data_type, subset=group_ds
                )
                if layers is None:
                    continue
                ax3.errorbar(
                    layers,
                    mean,
                    yerr=std,
                    color=colors[ft_mode],
                    linestyle=linestyles[ft_mode],
                    marker=marker,
                    markersize=6,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    alpha=alpha,
                    label=f"{group_name} ({labels[ft_mode]})",
                )

        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Intrinsic dimension")
        ax3.legend(fontsize=8)

    out_path3 = f"figures/id_{model_label.replace(' ', '_').lower()}_grouped.pdf"
    fig3.savefig(out_path3, bbox_inches="tight")
    print(f"Saved {out_path3}")

plt.show()
