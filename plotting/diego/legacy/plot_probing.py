#!/usr/bin/env python3
"""Plot probing results across layers using seaborn.

Tasks are grouped according to Conneau et al. (2018):
"What you can cram into a single vector: Probing sentence embeddings for linguistic properties"
https://aclanthology.org/P18-1198.pdf
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# *************************************************************************
alpha = 0.9
sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

# Task groupings from Conneau et al. (2018)
TASK_CATEGORIES = {
    # Surface information - simple properties, no linguistic knowledge needed
    "SentLen": "Surface",
    "WC": "Surface",
    # Syntactic information - sensitivity to syntactic properties
    "BShift": "Syntactic",
    "TreeDepth": "Syntactic",
    "TopConst": "Syntactic",
    # Semantic information - semantic properties
    # "Tense": "Semantic",
    # "SubjNum": "Semantic",
    # "ObjNum": "Semantic",
    "SOMO": "Semantic",
    "CoordInv": "Semantic",
}

CATEGORY_ORDER = ["Surface", "Syntactic", "Semantic"]

# Color palettes per category
CATEGORY_COLORS = {
    "Surface": sns.color_palette("Blues_d", 4)[1:],
    "Syntactic": sns.color_palette("Greens_d", 5)[1:],
    "Semantic": sns.color_palette("Reds_d", 7)[1:],
}


def extract_layer_number(layer_name: str) -> int | str:
    """Extract layer number from layer name."""
    if layer_name == "model.norm":
        return 32  # Place final norm after all layers
    match = re.search(r"model\.layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    return -1


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all JSON result files and return a DataFrame."""
    data = []

    for json_file in results_dir.glob("*_accuracy.json"):
        # Extract task name from filename
        task_name = json_file.stem.replace("_accuracy", "").replace("_scaler", "")

        # Get category from mapping
        category = TASK_CATEGORIES.get(task_name, "Unknown")

        with open(json_file, "r") as f:
            results = json.load(f)

        for layer_name, values in results.items():
            layer_num = extract_layer_number(layer_name)

            # Add individual runs for error bands
            for i, val in enumerate(values):
                data.append(
                    {
                        "Task": task_name,
                        "Category": category,
                        "Layer": layer_num,
                        "Accuracy": val,
                        "Run": i,
                    }
                )

    return pd.DataFrame(data)


results_dir = Path(__file__).parent / "results" / "probing_tasks"
plots_dir = Path(__file__).parent / "results" / "plots"

# Load data
df = load_results(results_dir)

# Filter out unknown categories and duplicates (e.g., scaler variants)
df = df[df["Category"] != "Unknown"]
df = df.drop_duplicates(subset=["Task", "Layer", "Run"])

# Create figure with GridSpec for each category
fig = plt.figure(figsize=(12, 4))
gs = GridSpec(1, 3, figure=fig)

layer_labels = [str(i) for i in range(32)] + ["norm"]

for idx, category in enumerate(CATEGORY_ORDER):
    ax = fig.add_subplot(gs[idx])

    # Add abstraction phase shading
    ax.axvspan(
        6, 17, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
    )

    cat_df = df[df["Category"] == category]

    if cat_df.empty:
        ax.set_visible(False)
        continue

    # Get tasks in this category
    tasks_in_cat = cat_df["Task"].unique()
    n_tasks = len(tasks_in_cat)
    colors = CATEGORY_COLORS.get(category, sns.color_palette("husl", n_tasks))

    # Create color mapping for tasks
    task_colors = {task: colors[i % len(colors)] for i, task in enumerate(tasks_in_cat)}

    # Plot with seaborn lineplot
    sns.lineplot(
        data=cat_df,
        x="Layer",
        y="Accuracy",
        hue="Task",
        palette=task_colors,
        marker="o",
        linewidth=1,
        ax=ax,
        errorbar=("sd", 1),
        alpha=alpha,
    )

    # Customize x-axis
    ax.set_xticks(range(0, 33, 4))
    ax.set_xticklabels([str(i) for i in range(0, 33, 4)], fontsize=10)
    ax.set_xlabel("layers", fontsize=12)

    # Customize y-axis
    if idx == 0:
        ax.set_ylabel("accuracy", fontsize=14)
    else:
        ax.set_ylabel("")

    ax.set_ylim(0.25, 1.05)

    # Title for subplot
    ax.set_title(
        f"{category} tasks",
        fontsize=15,
    )

    # Legend
    ax.legend(
        title="",
        fontsize=9,
        loc="best",
        framealpha=0.9,
    )

    # Add grid
    ax.grid(True, alpha=0.4)

# Tight layout
gs.tight_layout(fig)

# Save figure
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / "probing_results.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Plot saved to: {output_path}")
