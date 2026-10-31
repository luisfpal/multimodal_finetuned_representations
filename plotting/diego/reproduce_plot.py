import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# *************************************************************************
# Configuration
# *************************************************************************
INPUT_FILE = "aggregated_plot_data.json"
OUTPUT_DIR = "."
SHOW_STD = True

# Allow passing input file as argument
# if len(sys.argv) > 1:
#     INPUT_FILE = sys.argv[1]

# Set seaborn style (Exact match from original notebook)
sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

# *************************************************************************


def plot_model(model_slug, data):
    """Generate normalized aggregated plots for a given model."""

    layers = data["layers"]
    limits = data["limits"]

    # Helper to convert list back to numpy array if exists
    def to_np(d, key):
        return np.array(d[key]) if d and d[key] else None

    mean_mc_ll = to_np(data["mc_ll"], "mean")
    std_mc_ll = to_np(data["mc_ll"], "std")

    mean_mc_lp = to_np(data["mc_lp"], "mean")
    std_mc_lp = to_np(data["mc_lp"], "std")

    mean_qa_ll = to_np(data["qa_ll"], "mean")
    std_qa_ll = to_np(data["qa_ll"], "std")

    # Plotting
    fig_combined = plt.figure(figsize=(4.5, 4))
    gs_combined = GridSpec(1, 1)
    ax_combined = fig_combined.add_subplot(gs_combined[0])

    # Abstraction phase
    ax_combined.axvspan(
        limits[0],
        limits[1],
        alpha=0.15,
        color="goldenrod",
        zorder=0,
        label="abstraction phase",
    )

    # Plot Multiple Choice (LL)
    if mean_mc_ll is not None:
        sns.lineplot(
            x=range(len(mean_mc_ll))[1::2],
            y=mean_mc_ll[1::2],
            label="multiple choice (LL)",
            color="darkslategray",
            marker="o",
            markersize=4,
            linewidth=1,
            ax=ax_combined,
        )
        if SHOW_STD and std_mc_ll is not None:
            ax_combined.fill_between(
                range(len(mean_mc_ll)),
                mean_mc_ll - std_mc_ll / 2,
                mean_mc_ll + std_mc_ll / 2,
                color="darkslategray",
                alpha=0.2,
            )

    # Plot Multiple Choice (LP)
    if mean_mc_lp is not None:
        sns.lineplot(
            x=range(len(mean_mc_lp))[1::2],
            y=mean_mc_lp[1::2],
            label="multiple choice (LP)",
            color="darkslategray",
            linestyle="--",
            marker="s",
            markersize=4,
            linewidth=1,
            ax=ax_combined,
        )
        if SHOW_STD and std_mc_lp is not None:
            ax_combined.fill_between(
                range(len(mean_mc_lp)),
                mean_mc_lp - std_mc_lp / 2,
                mean_mc_lp + std_mc_lp / 2,
                color="darkslategray",
                alpha=0.2,
            )

    # Plot Open Ended (QA)
    if mean_qa_ll is not None:
        sns.lineplot(
            x=range(len(mean_qa_ll))[1::2],
            y=mean_qa_ll[1::2],
            label="open ended",
            color="darkred",
            marker="s",
            markersize=4,
            linewidth=1,
            ax=ax_combined,
        )
        if SHOW_STD and std_qa_ll is not None:
            ax_combined.fill_between(
                range(len(mean_qa_ll)),
                mean_qa_ll - std_qa_ll / 2,
                mean_qa_ll + std_qa_ll / 2,
                color="darkred",
                alpha=0.2,
            )

    ax_combined.set_xticks(np.array(list(range(0, layers + 1, 4))))
    # ax_combined.set_ylim(-0.05, 1.05)
    ax_combined.set_ylabel("performance ", fontsize=14)
    ax_combined.set_xlabel("layers", fontsize=12)
    ax_combined.legend(fontsize=10, loc="best")
    ax_combined.set_title("probing analysis")
    # Title commented out in original, keeping it that way or adding back if needed
    # ax_combined.set_title(f"Aggregated: {label}", fontsize=15)
    ax_combined.grid(True, alpha=0.4)

    gs_combined.tight_layout(fig_combined)

    out_name = f"aggregated_normalized_{model_slug}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")


# *************************************************************************
# Load data
# *************************************************************************

# if not os.path.exists(INPUT_FILE):
#     print(f"Error: Data file '{INPUT_FILE}' not found.")
#     sys.exit(1)


with open(f"./{INPUT_FILE}", "r") as f:
    all_data = json.load(f)

# *************************************************************************
# Generate plots
# *************************************************************************

for model_slug, data in all_data.items():
    plot_model(model_slug, data)
