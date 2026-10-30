"""Plot overlap similarity heatmaps for embedding similarity results across models and layers."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OVERLAPS_DIR = "results/embedding_similarity/overlaps"
PARENT_DIR = "results/embedding_similarity"

MODELS = [
    "onevision_4b_hf",
    "onevision_8b_hf",
    "internvl_8b_hf",
    "llava_7b",
    "llava_13b",
]
ROWS = [
    ("coco_sim90", "coco sim90"),
    ("coco_sim85", "coco sim85"),
]

PROFILES = ["imgavg"]


def load_panel(row_key, model, layer_kind, use_repr=False):
    """Return dict {profile: list} or None if not available."""
    prefix = "overlap_repr" if use_repr else "overlap"
    if row_key == "imagenet":
        path = os.path.join(
            OVERLAPS_DIR,
            f"{prefix}_imagenet_animals_10_{layer_kind}_{model}.json",
        )
    else:
        # row_key like "coco_sim90" -> "{prefix}_coco_image_question_{layer_kind}_sim90_{model}.json"
        sim = row_key.split("_")[-1]  # "sim90" or "sim85"
        path = os.path.join(
            OVERLAPS_DIR,
            f"{prefix}_coco_image_question_{layer_kind}_{sim}_{model}.json",
        )
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def draw_panel(ax, data, title):
    if data is None:
        ax.set_title(f"{title}\n(missing)", fontsize=10)
        ax.axis("off")
        return
    for profile in PROFILES:
        if profile not in data:
            continue
        values = data[profile]
        if len(values) and isinstance(values[0], list):
            values = [v[0] for v in values]
        sns.lineplot(ax=ax, x=np.arange(len(values)), y=values, label=profile)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("layers")
    ax.set_ylabel("overlap")
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def first_legend(axes):
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            return h, l
    return None, None


def make_figure(layer_kind, use_repr=False):
    sns.set_style(
        "whitegrid",
        rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
    )

    n_rows, n_cols = len(ROWS), len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))

    for r, (row_key, row_label) in enumerate(ROWS):
        for c, model in enumerate(MODELS):
            data = load_panel(row_key, model, layer_kind, use_repr=use_repr)
            draw_panel(axes[r, c], data, f"{row_label} — {model}")

    handles, labels = first_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(PROFILES),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    kind_label = "repr overlap" if use_repr else "label overlap"
    fig.suptitle(f"{kind_label} — {layer_kind}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


fig_norm1 = make_figure("norm1")
fig_res2 = make_figure("res2")
fig_norm1_repr = make_figure("norm1", use_repr=True)
fig_res2_repr = make_figure("res2", use_repr=True)
plt.show()


# ***********************


# model = MODELS[0]
# sim = "sim90"
# layer_kind = "res2"

# data = json.load(
#     open(f"{OVERLAPS_DIR}/overlap_coco_image_question_{layer_kind}_{sim}_{model}.json")
# )


# data["imgeoi"]
# sns.set_style(
#     "whitegrid",
#     rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
# )
# n_rows, n_cols = len(ROWS), len(MODELS)
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
