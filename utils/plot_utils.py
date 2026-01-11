import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from typing import Optional, Tuple, Union


def plot_similarity_measure_matrix(
    ax,
    matrix_similarities: Union[pd.DataFrame, torch.Tensor],
    title: str = "",
    xlabel: str = "Heads",
    ylabel: str = "Layers",
    percentage_threshold: Optional[float] = None,
    top_p_percent: Optional[float] = None,
    bottom_p_percent: Optional[float] = None,
    measure: str = "neighborhood_overlap",
    max_xticks: Optional[int] = None,
    max_yticks: Optional[int] = None,
    annot_font_size: int = 8,
    colorbar_labelsize: int = 14,
    show_annotations: bool = True,
    add_title: bool = True,
    colorbar_label: Optional[str] = None,
) -> None:
    """
    Create a similarity measure heatmap on a provided matplotlib axis.
    This is the core plotting function for similarity matrices.

    Args:
        ax: Matplotlib axis object to draw on
        matrix_similarities: Similarity data (DataFrame or Tensor)
        title: Title for the heatmap
        xlabel: Label for x-axis (default: "Heads")
        ylabel: Label for y-axis (default: "Layers")
        percentage_threshold: Threshold for highlighting extreme values
        top_p_percent: Percentage of top values to highlight in red
        bottom_p_percent: Percentage of bottom values to highlight in orange
        measure: Name of the similarity measure being plotted
        max_xticks: Maximum number of x-axis ticks to show (None = show all)
        max_yticks: Maximum number of y-axis ticks to show (None = show all)
        annot_font_size: Font size for cell annotations
        colorbar_labelsize: Font size for colorbar tick labels
        show_annotations: Whether to show annotations on the heatmap
        add_title: Whether to add a title to the plot
        colorbar_label: Optional label for the colorbar
    """
    # Convert torch tensor to DataFrame if needed
    if isinstance(matrix_similarities, torch.Tensor):
        # Convert tensor to numpy and then to DataFrame
        tensor_np = matrix_similarities.detach().cpu().numpy()
        matrix_similarities = pd.DataFrame(tensor_np)
    elif not isinstance(matrix_similarities, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame or torch.Tensor for 'matrix_similarities'."
        )

    # Input validation
    for name, val in [
        ("percentage_threshold", percentage_threshold),
        ("top_p_percent", top_p_percent),
        ("bottom_p_percent", bottom_p_percent),
    ]:
        if val is not None and not (0 <= val <= 1):
            raise ValueError(f"{name} must be between 0 and 1")

    # ---------------------- BASE METRICS ----------------------
    max_val_global = matrix_similarities.max().max()
    min_val_global = matrix_similarities.min().min()
    print("=" * 50)
    print(f"\nMax {measure}: {max_val_global:.2f}, Min {measure}: {min_val_global:.2f}")

    # ------------------ THRESHOLD BOUNDARIES ------------------
    if percentage_threshold is not None:
        max_layer = matrix_similarities.max(axis=1)
        min_layer = matrix_similarities.min(axis=1)
        range_layer = max_layer - min_layer

        upper_bound = max_layer - (range_layer * percentage_threshold)
        lower_bound = min_layer + (range_layer * percentage_threshold)

        threshold_mask = (matrix_similarities >= upper_bound.values[:, None]) | (
            matrix_similarities <= lower_bound.values[:, None]
        )

        min_upper_bound = upper_bound.min()
        max_lower_bound = lower_bound.max()
        print(
            f"Min upper bound: {min_upper_bound:.4f}, Max lower bound: {max_lower_bound:.4f}"
        )

        # Percentage of values in the upper and lower bounds
        num_values_in_upper_bound = (
            (matrix_similarities >= upper_bound.values[:, None]).sum().sum()
        )
        num_values_in_lower_bound = (
            (matrix_similarities <= lower_bound.values[:, None]).sum().sum()
        )
        percentage_in_upper_bound = num_values_in_upper_bound / matrix_similarities.size
        percentage_in_lower_bound = num_values_in_lower_bound / matrix_similarities.size
        print(
            f"% in upper bound (white): {percentage_in_upper_bound * 100:.2f}%, # values: {num_values_in_upper_bound}"
        )
        print(
            f"% in lower bound (white): {percentage_in_lower_bound * 100:.2f}%, # values: {num_values_in_lower_bound}"
        )
        total_values_in_bounds = num_values_in_upper_bound + num_values_in_lower_bound
        print(
            f"% in bounds (white): {total_values_in_bounds / matrix_similarities.size * 100:.2f}%, # values: {total_values_in_bounds}"
        )
    else:
        threshold_mask = pd.DataFrame(
            False,
            index=matrix_similarities.index,
            columns=matrix_similarities.columns,
        )  # all are valid

    # Get valid (non-NaN) values for percentile calculations
    valid_values = matrix_similarities.values.flatten()
    valid_values = valid_values[~np.isnan(valid_values)]
    total_values = matrix_similarities.size
    valid_count = len(valid_values)

    # ---------- TOP p% ----------
    top_p_mask = pd.DataFrame(
        False,
        index=matrix_similarities.index,
        columns=matrix_similarities.columns,
    )
    if top_p_percent is not None and top_p_percent > 0 and valid_count > 0:
        # Use percentile instead of partition for more robust calculation
        top_cutoff = np.percentile(valid_values, (1 - top_p_percent) * 100)
        top_p_mask = matrix_similarities >= top_cutoff
        actual_top_percentage = top_p_mask.sum().sum() / total_values
        print(f"Top {top_p_percent * 100:.2f}% cutoff value: {top_cutoff:.4f}")
        print(
            f"% identified as top p%: {actual_top_percentage * 100:.2f}%, # values: {top_p_mask.sum().sum()}"
        )

    # ---------- BOTTOM p% ----------
    bottom_p_mask = pd.DataFrame(
        False,
        index=matrix_similarities.index,
        columns=matrix_similarities.columns,
    )
    if bottom_p_percent is not None and bottom_p_percent > 0 and valid_count > 0:
        # Use percentile instead of partition for more robust calculation
        bottom_cutoff = np.percentile(valid_values, bottom_p_percent * 100)
        bottom_p_mask = matrix_similarities <= bottom_cutoff
        actual_bottom_percentage = bottom_p_mask.sum().sum() / total_values
        print(f"Bottom {bottom_p_percent * 100:.2f}% cutoff value: {bottom_cutoff:.4f}")
        print(
            f"% identified as bottom p%: {actual_bottom_percentage * 100:.2f}%, # values: {bottom_p_mask.sum().sum()}"
        )

    # ------------------ FINAL MASKS ------------------
    final_annotation_mask = threshold_mask  # Only annotate if outside threshold
    final_red_mask = (
        top_p_mask & threshold_mask
    )  # Only color red if top and outside threshold
    final_orange_mask = (
        bottom_p_mask & threshold_mask
    )  # Only color orange if bottom and outside threshold

    # Final percentage of values in the final masks
    total_values_in_red_mask = final_red_mask.sum().sum()
    total_values_in_orange_mask = final_orange_mask.sum().sum()
    final_red_percentage = total_values_in_red_mask / total_values
    final_orange_percentage = total_values_in_orange_mask / total_values
    total_values_in_red_and_orange_mask = (
        total_values_in_red_mask + total_values_in_orange_mask
    )
    print(
        f"% in top p% (red): {final_red_percentage * 100:.2f}%, # values: {total_values_in_red_mask}"
    )
    print(
        f"% in bottom p% (orange): {final_orange_percentage * 100:.2f}%, # values: {total_values_in_orange_mask}"
    )
    print(
        f"% in top and bottom p% (red and orange): {total_values_in_red_and_orange_mask / total_values * 100:.2f}%, # values: {total_values_in_red_and_orange_mask}"
    )
    print("=" * 50)

    # ------------------ ANNOTATION LABELS ------------------
    if show_annotations:
        annot_labels = matrix_similarities.where(final_annotation_mask).map(
            lambda x: f"{x:.2f}"
        )
        annot_labels = annot_labels.replace("nan", "")
    else:
        annot_labels = None

    # ------------------ HEATMAP CREATION ------------------
    heatmap_kwargs = {
        "data": matrix_similarities,
        "fmt": "",  # we already formatted the numbers in annot_labels
        "cmap": "viridis",
        "cbar_kws": {
            "pad": 0.01,
        },
    }

    if show_annotations:
        heatmap_kwargs.update(
            {
                "annot": annot_labels,
                "annot_kws": {
                    "size": annot_font_size,
                    "weight": "bold",
                },  # Use parameter for font size
            }
        )

    if min_val_global > 0.0:
        heatmap_kwargs["vmax"] = 1

    # Create heatmap on the provided axis
    sns.heatmap(**heatmap_kwargs, ax=ax)

    # Configure colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=colorbar_labelsize)

    # Set colorbar label if provided
    if colorbar_label:
        colorbar.set_label(colorbar_label, fontsize=colorbar_labelsize)

    # Set titles and labels
    if add_title:
        ax.set_title(
            f"{title}" if title else f"[{measure}]", fontsize=14, fontweight="bold"
        )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # ------------------ SMART TICK CONFIGURATION ------------------
    n_cols = len(matrix_similarities.columns)
    n_rows = len(matrix_similarities.index)

    # Determine which x-ticks to show
    if max_xticks is not None and max_xticks <= n_cols:
        # Show equally spaced ticks up to max_xticks
        x_tick_indices = np.linspace(0, n_cols - 1, max_xticks, dtype=int)
        x_tick_labels = [matrix_similarities.columns[i] for i in x_tick_indices]
    else:
        # Show all ticks
        x_tick_indices = range(n_cols)
        x_tick_labels = matrix_similarities.columns

    # Determine which y-ticks to show
    if max_yticks is not None and max_yticks <= n_rows:
        # Show equally spaced ticks up to max_yticks
        y_tick_indices = np.linspace(0, n_rows - 1, max_yticks, dtype=int)
        y_tick_labels = [matrix_similarities.index[i] for i in y_tick_indices]
    else:
        # Show all ticks
        y_tick_indices = range(n_rows)
        y_tick_labels = matrix_similarities.index

    # Set centered ticks and labels
    ax.set_xticks([i + 0.5 for i in x_tick_indices])  # Center ticks in cells
    ax.set_yticks([i + 0.5 for i in y_tick_indices])  # Center ticks in cells
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_tick_labels, rotation=0)
    ax.tick_params(axis="both", labelsize=14)

    # ------------------ MANUAL CELL ANNOTATION ------------------
    if show_annotations:
        for i in range(matrix_similarities.shape[0]):
            for j in range(matrix_similarities.shape[1]):
                label = annot_labels.iat[i, j]
                if label:
                    color = (
                        "red"
                        if final_red_mask.iat[i, j]
                        else "darkorange"
                        if final_orange_mask.iat[i, j]
                        else "white"
                    )
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        label,
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=annot_font_size,  # Use parameter for consistency
                        weight="bold",
                    )

    # Add background color and style
    ax.set_facecolor("#F8F8F8")
