"""Plot overlap metrics across layers for two representations of a model."""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

model = "llava_13b_hf"
xmin = 6
xmax = 20
embedder = "qwen8b"
representation = "norm1"


sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 3)

for index, representation in enumerate(["res2", "norm1"]):
    i = index * 3
    filepath = f"results/embedding_similarity/{model}/coco_image_question/{representation}_{embedder}_{model}_all_metrics.json"
    with open(filepath, "r") as f:
        data = json.load(f)

    n_layers = len(data["metrics"]["overlap__k_5"])
    x = np.arange(0, n_layers)
    xticks = np.arange(0, n_layers, 4)

    ax = fig.add_subplot(gs[0 + i])
    for k in [5, 10, 20]:
        sns.lineplot(
            x=x,
            y=list(data["metrics"][f"overlap__k_{k}"].values()),
            marker="o",
            label=k,
        )
    ax.set_xticks(xticks)
    ax.axvspan(xmin, xmax, alpha=0.10, color="goldenrod", zorder=0)
    ax.legend()
    ax.set_ylabel(f"overlap_{representation}", fontsize=12)
    ax.set_xlabel("layers")

    # imbalance
    ax = fig.add_subplot(gs[1 + i])
    imbalances = list(data["metrics"]["imbalance"].values())
    llava_to_qwen = np.array([imb["llava_to_qwen"] for imb in imbalances])
    qwen_to_llava = np.array([imb["qwen_to_llava"] for imb in imbalances])
    sns.lineplot(
        x=x, y=(llava_to_qwen + qwen_to_llava) / 2, marker="o", label="imbalance"
    )
    ax.set_xticks(xticks)

    ax.axvspan(xmin, xmax, alpha=0.10, color="goldenrod", zorder=0)
    ax.legend()
    ax.set_ylabel(f"imbalance_{representation}", fontsize=12)
    ax.set_xlabel("layers")

    # cka
    ax = fig.add_subplot(gs[2 + i])
    linear_cka = list(data["metrics"]["linear_cka"].values())
    sns.lineplot(x=x, y=linear_cka, marker="o", label="linear_cka")
    # for sigma in [0.2]:
    #     gaussian_cka = list(data["metrics"][f"gaussian_cka__sigma_{sigma}"].values())
    #     sns.lineplot(x=x, y=gaussian_cka, marker="o", label=f"gaussina sigma {sigma}")
    ax.set_xticks(xticks)
    ax.axvspan(xmin, xmax, alpha=0.10, color="goldenrod", zorder=0)
    ax.legend()
    ax.set_ylabel(f"cka_{representation}", fontsize=12)
    ax.set_xlabel("layers")

gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
fig.suptitle(
    f"{model}-{embedder} embedding repr similarity (COCO captions)", fontsize=16
)
