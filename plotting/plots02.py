import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from typing import List, Optional, Union
import traceback
from functools import partial
import torch
import numpy as np
from pathlib import Path
import re
from utils import plot_similarity_measure_matrix
from config_results_loader import ResultsLoader

# Constants for commonly used values
DEFAULT_MEASURES = ["neighborhood_overlap"]  # Only neighborhood_overlap for thesis
DEFAULT_POOLING_METHODS = ["last"]  # Only last pooling for thesis
PLOTS_DIR = Path(__file__).parent.parent / "plots" / "thesis"

# Dataset name mapping for thesis display
DATASET_DISPLAY_NAMES = {
    "cocoqa_txt": "Text QA",
    "cocoqa_img": "Visual QA",
    "coco_captioning": "Captioning",
    "cocoqa_txt_minus_cocoqa_img": "cocovqa_txt_minus_cocovqa_mmd",
}

# Transplantation method display names
TRANSPLANTATION_DISPLAY_NAMES = {
    "two_parts_s2": "incremental",
    "sliding_window_ws2_s2": "sliding_window",
    "sliding_window_ws1_s2": "sliding_window",
}

# Measure display names
MEASURE_DISPLAY_NAMES = {
    "neighborhood_overlap": "Neighborhood Overlap",
    "linear_cka": "Linear CKA",
}

# Model display names
MODEL_DISPLAY_NAMES = {
    "pretrained": "Pretrained Model (PM)",
    "finetuned": "Finetuned Model (FM)",
    "pretrained-finetuned": "Pretrained - Finetuned",
}

# Dataset measure display names
DATASET_MEASURE_DISPLAY_NAMES = {
    "dataset_entropy": "Dataset Entropy",
    "intrinsic_dimension": "ID",
}


def safe_tensor_to_numpy(
    data: Union[torch.Tensor, np.ndarray, list, float, int],
) -> np.ndarray:
    """
    Safely convert various data types to numpy array.

    Args:
        data: Input data that could be a tensor, numpy array, or other numeric type

    Returns:
        numpy.ndarray: Converted numpy array
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    else:
        return np.array([data])


def save_plot_if_requested(save_plots: bool, filename: str):
    """Helper function to save or show plot based on save_plots flag."""
    if save_plots:
        PLOTS_DIR.mkdir(exist_ok=True)
        plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_display_name(dataset_name: str) -> str:
    """Get the display name for a dataset."""
    return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)


def plot_similarity_measures_matrix(
    results_loader: ResultsLoader,
    dataset: str,
    stream_type: str,
    pooling: str,
    measures: List[str],
    subplot_width: int = 10,
    plot_height: int = 8,
    percentage_threshold: float = 0.08,
    top_p_percent: float = 0.03,
    bottom_p_percent: float = 0.03,
    max_xticks: Optional[int] = None,
    max_yticks: Optional[int] = None,
    annot_font_size: int = 10,
    colorbar_labelsize: int = 12,
    wspace: float = 0.0,
    save_plots: bool = False,
    show_annotations: bool = True,
    add_title: bool = True,
    title: Optional[str] = None,  # If None, the title will be the dataset name
    colorbar_label: Optional[str] = None,
):
    """
    Create side-by-side heatmaps for multiple similarity measures.

    Args:
        results_loader: ResultsLoader instance with loaded data
        dataset: Dataset name (e.g., "cocoqaqa_img")
        stream_type: Stream type (e.g., "heads_projection")
        pooling: Pooling method (e.g., "last")
        measures: List of measure names to plot
        subplot_width: Width of each subplot
        plot_height: Height of the plot
        percentage_threshold: Threshold for percentage of values to be shown
        top_p_percent: Percentage of top values to be shown
        bottom_p_percent: Percentage of bottom values to be shown
        max_xticks: Maximum number of x-axis ticks to show (None = show all)
        max_yticks: Maximum number of y-axis ticks to show (None = show all)
        annot_font_size: Font size for annotations
        colorbar_labelsize: Font size for colorbar tick labels
        wspace: Width of the space between subplots
        save_plots: Whether to save plots instead of showing them
        show_annotations: Whether to show annotations on the heatmap
        add_title: Whether to add titles to the plots
        title: Optional title for the plots
        colorbar_label: Optional label for the colorbar
    """
    num_measures = len(measures)

    # Calculate figure size: width scales with number of measures
    fig_width = num_measures * subplot_width

    # Create the main figure with subplots
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_measures,
        figsize=(fig_width, plot_height),
    )

    # Handle single subplot case (axes is not a list when ncols=1)
    if num_measures == 1:
        axes = [axes]

    # Create heatmap for each measure using the core plotting function
    for idx, measure in enumerate(measures):
        # Get similarity data for this measure
        matrix_similarities = results_loader.get_similarity(
            dataset=dataset,
            stream_type=stream_type,
            pooling=pooling,
            measure=measure,
        )

        # Create heatmap directly on the subplot axis
        plot_similarity_measure_matrix(
            ax=axes[idx],
            matrix_similarities=matrix_similarities,
            xlabel="Heads",
            ylabel="Layers" if idx == 0 else "",  # Only first subplot gets y-label
            measure=MEASURE_DISPLAY_NAMES[measure],
            max_xticks=max_xticks,
            max_yticks=max_yticks,
            percentage_threshold=percentage_threshold,
            top_p_percent=top_p_percent,
            bottom_p_percent=bottom_p_percent,
            annot_font_size=annot_font_size,
            colorbar_labelsize=colorbar_labelsize,
            show_annotations=show_annotations,
            add_title=add_title,
            title=title,  # Pass the title parameter
            colorbar_label=colorbar_label,
        )

        # Clean up y-axis elements for all columns except the first
        if idx > 0:
            axes[idx].set_ylabel("")
            axes[idx].set_yticklabels([])
            axes[idx].tick_params(left=False)  # Remove y-axis ticks

    # Configure overall figure layout with reduced spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace)  # Reduced wspace for tighter layout

    filename = (
        f"similarity_matrix_{dataset}_{stream_type}_{pooling}_{'_'.join(measures)}.png"
    )
    save_plot_if_requested(save_plots, filename)


def plot_all_similarity_measures_matrices(
    results_loader: ResultsLoader,
    save_plots: bool = False,
    show_annotations: bool = True,
):
    """Plot similarity measures matrices for heads projection data."""
    # Configuration for heads projection matrices
    datasets = [
        # "cocoqa_txt",
        # "cocoqa_img",
        "cocoqa_txt_minus_cocoqa_img",
    ]
    pooling_methods = ["last"]  # Only last pooling for thesis

    # Plot individual matrices
    template_plot = partial(
        plot_similarity_measures_matrix,
        results_loader=results_loader,
        measures=DEFAULT_MEASURES,
        max_xticks=10,
        max_yticks=10,
        percentage_threshold=0.05,
        top_p_percent=0.05,
        bottom_p_percent=0.05,
        annot_font_size=7,
        colorbar_labelsize=14,
        subplot_width=10,
        plot_height=8,
        wspace=-0.05,
        save_plots=save_plots,
        show_annotations=show_annotations,
        add_title=True,
        title=r"$\Delta$ NO (Text QA - Visual QA)",  # LaTeX title
        colorbar_label="Neighborhood Overlap Difference",
    )

    for dataset in datasets:
        for pooling in pooling_methods:
            template_plot(
                dataset=dataset,
                stream_type="heads_projection",
                pooling=pooling,
            )

    # Plot combined 1x2 matrix for cocoqa_txt and cocoqa_img
    # plot_combined_similarity_matrices(results_loader, save_plots, show_annotations)


def plot_combined_similarity_matrices(
    results_loader: ResultsLoader,
    save_plots: bool = False,
    show_annotations: bool = True,
):
    """Plot combined 2x1 similarity matrices for cocoqa_txt and cocoqa_img."""

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "subtitle": 14,  # Subplot titles
        "axis_label": 14,  # X and Y axis labels
        "tick_label": 14,  # Tick labels on axes
        "annotation": 7,  # Any text annotations
    }

    datasets = ["cocoqa_txt", "cocoqa_img"]
    measure = "neighborhood_overlap"

    # Create 2x1 subplot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Get similarity data
        matrix_similarities = results_loader.get_similarity(
            dataset=dataset,
            stream_type="heads_projection",
            pooling="last",
            measure=measure,
        )

        # Create heatmap
        plot_similarity_measure_matrix(
            ax=ax,
            matrix_similarities=matrix_similarities,
            xlabel="Heads",
            ylabel="Layers",
            measure=MEASURE_DISPLAY_NAMES[measure],
            max_xticks=10,
            max_yticks=10,
            percentage_threshold=0.08,
            top_p_percent=0.03,
            bottom_p_percent=0.03,
            annot_font_size=FONT_SIZES["annotation"],
            colorbar_labelsize=14,
            show_annotations=show_annotations,
            colorbar_label=MEASURE_DISPLAY_NAMES[measure] if idx == 1 else None,
        )

        # Fix the fonr sizes of the subplot
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])
        ax.set_xlabel("Heads", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Layers", fontsize=FONT_SIZES["axis_label"])

        # Set title for each subplot
        display_name = get_display_name(dataset)
        ax.set_title(display_name, fontsize=FONT_SIZES["subtitle"], fontweight="bold")

        # Remove y-axis elements for right subplot
        if idx == 1:
            ax.set_yticklabels([])
            ax.set_ylabel("")
            ax.tick_params(left=False)

        # Remove the colorbar for the left subplot
        if idx == 0:
            ax.collections[0].colorbar.remove()

    plt.tight_layout()
    # plt.subplots_adjust(wspace=-0.02)

    filename = f"similarity_matrix_combined_{measure}.png"
    save_plot_if_requested(save_plots, filename)


def template_ax_plot(
    ax: plt.Axes,
    data,
    **kwargs,
):
    """
    Plot data on a single axis.
    """
    # Ensure data is a numpy array
    data = safe_tensor_to_numpy(data)

    # For 1D data (mean heads projection), plot as a line
    if data.ndim == 1:
        ax.plot(range(len(data)), data, **kwargs)
    else:
        print(f"⚠️  Unexpected data shape: {data.shape}")


def template_plot_similarity_measures(
    results_loader: ResultsLoader,
    stream_type: str,
    pooling_methods: List[str],
    measures: List[str],
    save_plots: bool = False,
):
    """
    Template plot function for similarity measures with multiple pooling methods.
    """
    from matplotlib.colors import TABLEAU_COLORS
    from matplotlib.lines import Line2D

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "axis_label": 14,  # X and Y axis labels
        "tick_label": 14,  # Tick labels on axes
        "legend": 14,  # Legend text
        "subtitle": 14,  # Subplot titles
    }

    # Set style for better plots
    plt.style.use("default")
    hex_colors = list(TABLEAU_COLORS.values())

    height = 6
    width = 6 * len(measures)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(measures), figsize=(width, height), sharey=True
    )

    # Handle single subplot case (axes is not a list when ncols=1)
    if len(measures) == 1:
        axes = [axes]

    datasets = results_loader.config["datasets"].copy()
    datasets.remove("coco_captioning")

    # Collect all data to determine Y limits
    min_all_data_values = []

    # Iterate through each measure (subplot)
    for m_idx, measure in enumerate(measures):
        ax = axes[m_idx]

        ax.set_title(
            MEASURE_DISPLAY_NAMES[measure],
            fontsize=FONT_SIZES["subtitle"],
            fontweight="bold",
        )

        # Plot each dataset (only last pooling method)
        for d_idx, dataset in enumerate(datasets):
            try:
                data = results_loader.get_similarity(
                    dataset=dataset,
                    stream_type=stream_type,
                    pooling="last",  # Only last pooling
                    measure=measure,  # Use the original measure key, not display name
                )
                if isinstance(data, torch.Tensor):
                    data = safe_tensor_to_numpy(data)

                # Collect data for Y limits calculation
                min_all_data_values.append(data.min())

                # Set styling: same color for dataset
                color = hex_colors[d_idx % len(hex_colors)]

                template_ax_plot(
                    ax,
                    data,
                    color=color,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    linestyle="-",
                )

            except Exception as e:
                print(f"⚠️  Failed to plot {dataset}/{stream_type}_last/{measure}: {e}")

        # Configure subplot
        ax.grid(True, alpha=0.3)

        # Configure axis labels with proper font sizes
        if m_idx == 0:
            ax.set_ylabel("Similarity", fontsize=FONT_SIZES["axis_label"])
        else:
            # Remove Y-axis elements for non-first subplots
            ax.tick_params(left=False)
            ax.set_ylabel("")  # Set an empty string as ylabel

        ax.set_xlabel("Layer", fontsize=FONT_SIZES["axis_label"])

        # Configure tick labels
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    # Set Y limits: min = 0 or minimum data value, max = 1
    if min_all_data_values:
        y_min = min(0, min(min_all_data_values))
        y_max = 1.0
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    # Add legend only to the first subplot
    first_ax = axes[0]

    # Create custom legend for datasets (colors)
    dataset_legend_elements = []
    for d_idx, dataset in enumerate(datasets):
        color = hex_colors[d_idx % len(hex_colors)]
        display_name = get_display_name(dataset)
        dataset_legend_elements.append(
            Line2D([0], [0], color=color, linewidth=3, label=display_name)
        )

    # Add dataset legend with proper font sizes
    first_ax.legend(
        handles=dataset_legend_elements,
        loc="upper left",
        fontsize=FONT_SIZES["legend"],
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.0, 1),
        ncols=1,
    )

    plt.tight_layout()

    filename = f"similarity_measures_{stream_type}_{'_'.join(pooling_methods)}_{'_'.join(measures)}.png"
    save_plot_if_requested(save_plots, filename)


def plot_similarity_measures_for_stream_type(
    results_loader: ResultsLoader,
    stream_type: str,
    pooling_methods: List[str] = None,
    measures: List[str] = None,
    save_plots: bool = False,
):
    """
    Generic function to plot similarity measures for any stream type.

    Args:
        results_loader: ResultsLoader instance with loaded data
        stream_type: Stream type (e.g., "output_layer", "post_mlp", "mean_heads_projection")
        pooling_methods: List of pooling methods (defaults to ["last"])
        measures: List of measures (defaults to ["neighborhood_overlap"])
        save_plots: Whether to save plots instead of showing them
    """
    if pooling_methods is None:
        pooling_methods = DEFAULT_POOLING_METHODS
    if measures is None:
        measures = DEFAULT_MEASURES

    print(f"\n📊 Plotting {stream_type} with last pooling method...")
    template_plot_similarity_measures(
        results_loader=results_loader,
        stream_type=stream_type,
        pooling_methods=pooling_methods,
        measures=measures,
        save_plots=save_plots,
    )


def plot_all_output_layer_similarity_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """Plot similarity measures for output layer data."""
    plot_similarity_measures_for_stream_type(
        results_loader=results_loader,
        stream_type="output_layer",
        save_plots=save_plots,
        measures=["neighborhood_overlap", "linear_cka"],
    )


def plot_all_post_mlp_similarity_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """Plot similarity measures for post-mlp data."""
    plot_similarity_measures_for_stream_type(
        results_loader=results_loader,
        stream_type="post_mlp",
        save_plots=save_plots,
        measures=["neighborhood_overlap", "linear_cka"],
    )


def plot_all_mean_heads_projection_similarity_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """Plot similarity measures for mean heads projection data."""
    plot_similarity_measures_for_stream_type(
        results_loader=results_loader,
        stream_type="mean_heads_projection",
        save_plots=save_plots,
        measures=["neighborhood_overlap", "linear_cka"],
    )


def template_plot_model_data_measures(
    results_loader: ResultsLoader,
    dataset_measure: str,
    stream_type: str,
    pooling_methods: List[str],
    save_plots: bool = False,
):
    """
    Template plot function for model data measures.
    Creates subplots for each model, showing dataset measures across layers.
    """
    from matplotlib.colors import TABLEAU_COLORS
    from matplotlib.lines import Line2D

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "subtitle": 12,  # Subplot titles
        "axis_label": 12,  # X and Y axis labels
        "tick_label": 12,  # Tick labels on axes
        "legend": 12,  # Legend text
    }

    # Set style for better plots
    plt.style.use("default")
    hex_colors = list(TABLEAU_COLORS.values())

    if "model_data_measures" not in results_loader.data:
        print("⚠️  No model data measures loaded")
        return

    model_data_measures = results_loader.data["model_data_measures"]

    # Get all available models from all datasets
    all_models = set()
    datasets = []
    for dataset_name, dataset_data in model_data_measures.items():
        datasets.append(dataset_name)
        all_models.update(dataset_data.keys())

    all_models = sorted(list(all_models))
    all_models.remove("pretrained-finetuned")

    # Create subplots: 1 row, num_models columns
    height = 6
    width = 5 * len(all_models)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(all_models), figsize=(width, height), sharey=True
    )

    # Handle single subplot case
    if len(all_models) == 1:
        axes = [axes]

    # y limits
    y_min = float("inf")
    y_max = float("-inf")

    # Plot for each model (subplot)
    for m_idx, model in enumerate(all_models):
        ax = axes[m_idx]

        # Plot each dataset (only last pooling method)
        for d_idx, dataset in enumerate(datasets):
            # Check if this model exists in this dataset
            if model not in model_data_measures[dataset]:
                continue

            model_data = model_data_measures[dataset][model]

            # Check if the measure type exists
            if dataset_measure not in model_data:
                continue

            measure_data = model_data[dataset_measure]

            # Handle different pooling methods for different measure types
            if dataset_measure == "prompt_entropy":
                # Prompt entropy uses "none" as pooling method
                key = f"{stream_type}_none"
            else:
                # Construct the key: stream_type_pooling (only last)
                key = f"{stream_type}_last"

            # For intrinsic dimension, we need to handle additional parameters
            if dataset_measure == "intrinsic_dimension":
                # Look for keys that start with our prefix (e.g., "output_layer_last")
                matching_keys = [k for k in measure_data.keys() if k.startswith(key)]
                if not matching_keys:
                    continue
                # Use the first matching key (they should all have the same data anyway)
                actual_key = matching_keys[0]
            else:
                actual_key = key
                if actual_key not in measure_data:
                    continue

            try:
                # Get the data - it's stored as a dict with one item usually
                data_dict = measure_data[actual_key]
                if isinstance(data_dict, dict) and len(data_dict) > 0:
                    # Get the first (and usually only) tensor from the dict
                    data = next(iter(data_dict.values()))
                else:
                    data = data_dict

                if isinstance(data, torch.Tensor):
                    data = safe_tensor_to_numpy(data)

                # Collect data for Y limits calculation
                y_min = min(y_min, data.min())
                y_max = max(y_max, data.max())

                # Set styling: same color for dataset
                color = hex_colors[d_idx % len(hex_colors)]

                # Plot the data
                ax.plot(
                    range(len(data)),
                    data,
                    color=color,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    linestyle="-",
                )

            except Exception as e:
                print(
                    f"⚠️  Failed to plot {dataset}/{model}/{dataset_measure}/{actual_key}: {e}"
                )

        # Configure subplot
        ax.set_title(f"{MODEL_DISPLAY_NAMES[model]}", fontsize=FONT_SIZES["subtitle"])
        ax.grid(True, alpha=0.3)

        ax.set_xlabel("Layer", fontsize=FONT_SIZES["axis_label"])

        # Configure tick labels
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    # Configure Y-axis labels and limits for all subplots
    for idx in range(len(axes)):
        if idx == 0:
            axes[idx].set_ylabel(
                f"{DATASET_MEASURE_DISPLAY_NAMES[dataset_measure]}",
                fontsize=FONT_SIZES["axis_label"],
            )
        else:
            # Remove Y-axis elements for non-first subplots
            axes[idx].tick_params(left=False)
            axes[idx].set_ylabel("")

    # Set Y limits for all subplots
    if y_min != float("inf") and y_max != float("-inf"):
        if dataset_measure == "intrinsic_dimension":
            y_min_plot = 0
            y_max_plot = y_max * 1.02
        elif dataset_measure == "dataset_entropy":
            y_min_plot = y_min * 0.95
            y_max_plot = y_max * 1.02
        else:
            y_min_plot = 0
            y_max_plot = 1

        # Apply Y limits to all subplots
        for ax in axes:
            ax.set_ylim(y_min_plot, y_max_plot)

    # Add dataset legend
    dataset_legend_elements = []
    for d_idx, dataset in enumerate(datasets):
        color = hex_colors[d_idx % len(hex_colors)]
        display_name = get_display_name(dataset)
        dataset_legend_elements.append(
            Line2D([0], [0], color=color, linewidth=3, label=display_name)
        )

    # Add dataset legend to the second subplot (if it exists) in lower right
    if len(axes) > 1:
        # Add legend to second subplot in lower right
        axes[1].legend(
            handles=dataset_legend_elements,
            fontsize=FONT_SIZES["legend"],
            loc="lower right",
            frameon=True,
            fancybox=True,
            shadow=True,
        )
    else:
        # If only one subplot, add legend to the figure
        fig.legend(
            handles=dataset_legend_elements,
            fontsize=FONT_SIZES["legend"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncols=len(dataset_legend_elements),
        )

    # Adjust subplot spacing for better horizontal distribution
    plt.subplots_adjust(wspace=0.05)

    filename = f"model_data_measures_{dataset_measure}_{stream_type}_{'_'.join(pooling_methods)}.png"
    save_plot_if_requested(save_plots, filename)


def plot_all_model_data_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """Plot all model data measures for different stream types and dataset measures."""
    # Define configurations for different combinations (only output_layer and last pooling)
    configs = [
        ("dataset_entropy", "output_layer", DEFAULT_POOLING_METHODS),
        ("intrinsic_dimension", "output_layer", DEFAULT_POOLING_METHODS),
    ]

    for dataset_measure, stream_type, pooling_methods in configs:
        print(f"🖼️ Plotting {dataset_measure} for {stream_type} with {pooling_methods}")
        template_plot_model_data_measures(
            results_loader=results_loader,
            dataset_measure=dataset_measure,
            stream_type=stream_type,
            pooling_methods=pooling_methods,
            save_plots=save_plots,
        )


def plot_modalities_similarities(
    results_loader: ResultsLoader,
    stream_types: List[str],
    similarity_type: str = "cosine_similarity",
    save_plots: bool = False,
):
    """
    Plot modalities similarities as line plots.

    Args:
        results_loader: ResultsLoader instance with loaded data
        stream_types: List of stream types to plot (e.g., ['output_layer'])
        similarity_type: Type of similarity to plot ("cosine_similarity", "homogeneity_score_cosine")
        save_plots: Whether to save plots instead of showing them
    """
    print(f"\n📊 Plotting modalities similarities: {similarity_type}")

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "subtitle": 14,  # Subplot titles
        "axis_label": 14,  # X and Y axis labels
        "tick_label": 14,  # Tick labels on axes
        "legend": 14,  # Legend text
    }

    if "modalities_similarities" not in results_loader.data:
        print("⚠️  No modalities similarities data loaded")
        return

    modalities_data = results_loader.data["modalities_similarities"]

    # Get all available models and datasets
    all_models = list(modalities_data.keys())
    all_datasets = set()

    for model_data in modalities_data.values():
        for dataset in model_data.keys():
            all_datasets.add(dataset)

    all_datasets = sorted(list(all_datasets))

    # Create subplots based on number of datasets
    if len(all_datasets) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Define colors for models
    model_colors = {"finetuned": "red", "pretrained": "blue"}

    # y limits
    y_min = float("inf")
    y_max = float("-inf")

    # Plot data for each dataset (subplot)
    for d_idx, dataset in enumerate(all_datasets):
        ax = axes[d_idx]

        # Plot each model (only output_layer stream type)
        for model in all_models:
            try:
                similarity_data = results_loader.get_modalities_similarity(
                    model, dataset, "output_layer", similarity_type
                )

                if isinstance(similarity_data, torch.Tensor):
                    similarity_data = safe_tensor_to_numpy(similarity_data)

                # Collect data for Y limits
                if similarity_data.ndim == 1:
                    y_min = min(y_min, similarity_data.min())
                    y_max = max(y_max, similarity_data.max())
                elif similarity_data.ndim == 2:
                    y_min = min(y_min, similarity_data[:, 0].min())
                    y_max = max(y_max, similarity_data[:, 0].max())

                # Create line plot
                layers = range(len(similarity_data))
                color = model_colors.get(model, "gray")

                ax.plot(
                    layers,
                    similarity_data
                    if similarity_data.ndim == 1
                    else similarity_data[:, 0],
                    color=color,
                    linestyle="-",
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    label=model,
                )

                # if similarity_data.ndim == 2:
                #     ax.fill_between(
                #         layers,
                #         similarity_data[:, 0] - similarity_data[:, 1],
                #         similarity_data[:, 0] + similarity_data[:, 1],
                #         alpha=0.25,
                #         color=color,
                #     )

            except KeyError:
                # Data not available for this combination
                print(
                    f"🖼️ {model} for {dataset} with {similarity_type} is not available"
                )
                # traceback.print_exc()
                continue

        # Configure subplot
        display_name = get_display_name(dataset)
        ax.set_title(display_name, fontsize=FONT_SIZES["subtitle"], fontweight="bold")
        ax.set_xlabel("Layer", fontsize=FONT_SIZES["axis_label"])
        if d_idx == 0:  # Only first subplot gets y-label
            if similarity_type == "cosine_similarity":
                y_label = "Cosine Similarity"
            elif similarity_type == "homogeneity_score_cosine":
                y_label = "Homogeneity Score"
            else:
                y_label = "Similarity"
            ax.set_ylabel(y_label, fontsize=FONT_SIZES["axis_label"])
        else:
            # Remove Y-axis elements for non-first subplots
            ax.tick_params(left=False)
            ax.set_ylabel("")

        ax.grid(True, alpha=0.3)

        # Configure tick labels
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    # Set Y limits based on collected data
    if y_min != float("inf") and y_max != float("-inf"):
        # Calculate Y limits with padding: min * 0.95, max * 1.1
        y_min_plot = y_min * 0.90
        y_max_plot = y_max * 1.1

        # Since axes are shared, only need to set limits on the first axis
        axes[0].set_ylim(y_min_plot, y_max_plot)

    # Add legend only to the first subplot
    if len(all_datasets) > 0:
        # Create custom legend for models (colors)
        from matplotlib.lines import Line2D

        model_legend_elements = []
        for model, color in model_colors.items():
            model_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=3,
                    label=MODEL_DISPLAY_NAMES[model]
                    .replace(" (PM)", "")
                    .replace(" (FM)", ""),
                )
            )

        # Set legend location based on similarity type
        kwargs = {}
        if similarity_type == "cosine_similarity":
            kwargs["loc"] = "upper left"
            kwargs["bbox_to_anchor"] = (0.0, 1)
        elif similarity_type == "homogeneity_score_cosine":
            kwargs["loc"] = "lower left"
            kwargs["bbox_to_anchor"] = (0.02, 0.02)

        # Add model legend
        axes[1].legend(
            handles=model_legend_elements,
            fontsize=FONT_SIZES["legend"],
            frameon=True,
            fancybox=True,
            shadow=True,
            **kwargs,
            ncols=1,
        )

    plt.tight_layout()

    filename = f"modalities_similarities_{similarity_type}_{'_'.join(stream_types)}.png"
    save_plot_if_requested(save_plots, filename)


def plot_transplanting_layers_qa_benchmarking(
    results_loader: ResultsLoader,
    transplantation_method: str,
    save_plots: bool = False,
    exclude_fm_for_sliding_window: bool = True,
):
    """
    Plot benchmarking and transplanting layers.

    Args:
        results_loader: ResultsLoader instance with loaded data
        transplantation_method: Either 'sliding_window_ws2_s2' or 'two_parts_s2'
    """

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "subtitle": 14,  # Subplot titles
        "axis_label": 14,  # X and Y axis labels
        "tick_label": 14,  # Tick labels on axes
        "legend": 14,  # Legend text
        "annotation": 10,  # Any text annotations
    }

    # Load transplanting data for both datasets
    datasets = ["cocoqa_txt", "cocoqa_img"]

    # Validate transplantation method
    if "transplanting_layers_qa_benchmarking" not in results_loader.data:
        print("⚠️  No transplanting layers benchmarking data loaded")
        return

    print(f"\n📊 Plotting transplanting layers benchmarking: {transplantation_method}")

    # Max y-value for all datasets
    max_y_value = 0

    # Create figure with 1x2 subplots (shared y-axis)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    # Colors
    colors = {
        "PM": "dodgerblue",
        "FM": "crimson",
        "Layers": "skyblue",
    }

    # Process each dataset
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        try:
            # Get transplant data for this dataset and method
            data_df = results_loader.get_transplant(dataset, transplantation_method)

            # Extract layer data (excluding mm_model and mm_pretrained_connector)
            layer_prefix = (
                "start_layer_"
                if "sliding_window" in transplantation_method
                else "split_layer_"
            )

            # Get layer entries - the layer names are in the first column (Unnamed: 0)
            layer_names_column = data_df.columns[0]  # "Unnamed: 0"
            layer_rows_mask = data_df[layer_names_column].str.startswith(layer_prefix)
            layer_data = data_df[layer_rows_mask]

            layer_numbers = []
            layer_accuracies = []

            for _, row in layer_data.iterrows():
                # Extract layer number from string like "start_layer_4" or "split_layer_7"
                layer_name = row[layer_names_column]
                layer_num = int(layer_name.split("_")[-1])
                accuracy = row["accuracy"]

                layer_numbers.append(layer_num)
                layer_accuracies.append(accuracy)

            # Sort by layer number
            sorted_data = sorted(zip(layer_numbers, layer_accuracies))
            layer_numbers, layer_accuracies = zip(*sorted_data)

            # Get f model and pretrained model accuracies
            fm_row = data_df[data_df[layer_names_column] == "mm_model"]
            pm_row = data_df[data_df[layer_names_column] == "mm_pretrained_connector"]

            fm_accuracy = fm_row["accuracy"].iloc[0]
            pm_accuracy = pm_row["accuracy"].iloc[0]

            # Check if this is a sliding_window method to conditionally exclude FM
            is_sliding_window = "sliding_window" in transplantation_method

            # Prepare data for plotting - rearrange order: PM first, then layers, then FM
            if is_sliding_window and exclude_fm_for_sliding_window:
                # For sliding_window methods, exclude FM data if flag is set
                x_positions = list(range(len(layer_numbers) + 1))  # +1 for PM only
                all_accuracies = [pm_accuracy] + list(
                    layer_accuracies
                )  # PM first, then layers
                x_labels = ["PM"] + [
                    str(num) for num in layer_numbers
                ]  # PM first, then layers
            else:
                # For other methods or if flag is False, include both FM and PM
                x_positions = list(range(len(layer_numbers) + 2))  # +2 for PM and FM
                all_accuracies = (
                    [pm_accuracy] + list(layer_accuracies) + [fm_accuracy]
                )  # PM first, then layers, then FM
                x_labels = (
                    ["PM"] + [str(num) for num in layer_numbers] + ["FM"]
                )  # PM first, then layers, then FM

            max_y_value = max(max_y_value, max(all_accuracies))

            # Create bar plot
            bars = ax.bar(x_positions, all_accuracies, alpha=0.7, edgecolor="black")

            # Color coding: PM in red (first), layers in blue, FM in green (if present)
            for i, bar in enumerate(bars):
                if x_labels[i] == "PM":
                    bar.set_color(colors["PM"])  # PM is first
                elif x_labels[i] == "FM":
                    bar.set_color(colors["FM"])  # FM is last (if present)
                else:
                    bar.set_color(colors["Layers"])  # Layers are in the middle

            # Customize subplot
            display_name = get_display_name(dataset)
            ax.set_xlabel(
                "Layer Number / Model Type", fontsize=FONT_SIZES["axis_label"]
            )
            if idx == 0:  # Only first subplot gets y-label
                ax.set_ylabel("Accuracy", fontsize=FONT_SIZES["axis_label"])

            ax.set_title(
                display_name, fontsize=FONT_SIZES["subtitle"], fontweight="bold"
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 10 else 0)

            ax.grid(True, alpha=0.3, axis="y")

            # Configure tick labels
            ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

            # Add value labels on top of bars
            for i, (pos, acc) in enumerate(zip(x_positions, all_accuracies)):
                text_offset = 0.01  # Normal offset

                ax.text(
                    pos,
                    acc + text_offset,
                    f"{acc:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["annotation"],
                )

        except Exception as e:
            print(f"⚠️  Failed to plot {dataset}/{transplantation_method}: {e}")
            ax.text(
                0.5,
                0.5,
                f"No data for\n{dataset}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color="red",
            )
            ax.set_title(
                display_name, fontsize=FONT_SIZES["subtitle"], fontweight="bold"
            )

    # Set Y limits for all subplots
    if is_sliding_window and exclude_fm_for_sliding_window:
        axes[0].set_ylim(0, max_y_value * 1.15)  # 15% padding
    else:
        axes[0].set_ylim(0, 1)  # Normal limits

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors["PM"], label="Pretrained Model (PM)"),
        Patch(facecolor=colors["Layers"], label="Layer Patching"),
    ]

    if is_sliding_window:
        if not exclude_fm_for_sliding_window:
            legend_elements.append(
                Patch(facecolor=colors["FM"], label="Finetuned Model (FM)")
            )
    else:
        legend_elements.append(
            Patch(facecolor=colors["FM"], label="Finetuned Model (FM)")
        )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(legend_elements),
        fontsize=FONT_SIZES["legend"],
    )

    # Remove Y-axis elements for non-first subplots
    axes[1].tick_params(left=False)
    axes[1].set_ylabel("")  # Set an empty string as ylabel

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for legend

    filename = f"transplanting_layers_qa_benchmarking_{transplantation_method}.png"
    save_plot_if_requested(save_plots, filename)


def plot_all_transplanting_layers_qa_benchmarking(
    results_loader: ResultsLoader,
    save_plots: bool = False,
    exclude_fm_for_sliding_window: bool = True,
):
    """
    Plot all available transplanting layers benchmarking experiments.
    """
    print("\n📊 Plotting all transplanting layers benchmarking experiments...")

    # Available transplantation methods
    methods = [
        "two_parts_s2",
        "sliding_window_ws1_s2",
    ]

    for method in methods:
        plot_transplanting_layers_qa_benchmarking(
            results_loader,
            method,
            save_plots=save_plots,
            exclude_fm_for_sliding_window=exclude_fm_for_sliding_window,
        )


def _extract_max_tokens_from_directory(directory_name: str) -> int:
    """
    Extract max_new_tokens value from directory name.

    Args:
        directory_name: Directory name (e.g., 'transplanting_layers_caption_benchmarking_max_new_tokens_20')

    Returns:
        int: Number of max_new_tokens

    Raises:
        ValueError: If max_new_tokens value cannot be extracted from directory name
    """

    # Extract integer after "max_new_tokens_"
    match = re.search(r"max_new_tokens_(\d+)", directory_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(
            f"Could not extract max_new_tokens value from directory name: {directory_name}"
        )


def _extract_caption_benchmarking_data(data_df, method, metric_name):
    """
    Extract and validate caption benchmarking data for plotting.

    Args:
        data_df: DataFrame containing caption benchmarking results
        method: Transplantation method name
        metric_name: Name of the metric to extract

    Returns:
        tuple: (layer_numbers, layer_metrics, fm_metric, pm_metric)

    Raises:
        ValueError: If data structure is invalid
    """
    # Validate data structure
    if data_df.empty:
        raise ValueError(f"Empty dataframe for {method}")

    if metric_name not in data_df.columns:
        raise ValueError(
            f"Metric '{metric_name}' not found in data columns: {list(data_df.columns)}"
        )

    # Extract layer data (excluding baseline models)
    layer_prefix = "start_layer_" if "sliding_window" in method else "split_layer_"

    # Get layer entries - the layer names are in the first column
    layer_names_column = data_df.columns[0]
    layer_rows_mask = data_df[layer_names_column].str.startswith(layer_prefix)
    layer_data = data_df[layer_rows_mask]

    if layer_data.empty:
        raise ValueError(f"No layer data found for prefix '{layer_prefix}' in {method}")

    layer_numbers = []
    layer_metrics = []

    for _, row in layer_data.iterrows():
        layer_name = row[layer_names_column]
        layer_num = int(layer_name.split("_")[-1])
        metric_value = row[metric_name]

        layer_numbers.append(layer_num)
        layer_metrics.append(metric_value)

    # Sort by layer number
    sorted_data = sorted(zip(layer_numbers, layer_metrics))
    layer_numbers, layer_metrics = zip(*sorted_data)

    # Get baseline model metrics
    fm_row = data_df[data_df[layer_names_column] == "mm_model"]
    pm_row = data_df[data_df[layer_names_column] == "mm_pretrained_connector"]

    # For sliding_window methods, we don't need FM data
    if "sliding_window" in method:
        if pm_row.empty:
            raise ValueError(f"Missing pretrained model data in {method}")
        pm_metric = pm_row[metric_name].iloc[0]
        return layer_numbers, layer_metrics, None, pm_metric
    else:
        # For other methods, we need both FM and PM data
        if fm_row.empty or pm_row.empty:
            raise ValueError(f"Missing baseline model data in {method}")
        fm_metric = fm_row[metric_name].iloc[0]
        pm_metric = pm_row[metric_name].iloc[0]
        return layer_numbers, layer_metrics, fm_metric, pm_metric


def plot_caption_benchmarking_for_directory(
    results_loader: ResultsLoader,
    metric_name: str,
    directory_name: str,
    save_plots: bool = False,
    exclude_fm_for_sliding_window: bool = True,
):
    """
    Plot caption benchmarking results for a specific metric and directory.

    Args:
        results_loader: ResultsLoader instance with loaded data
        metric_name: Metric to plot ('CIDEr', 'CLIP-S', 'RefCLIP-S')
        directory_name: Directory name (e.g., 'transplanting_layers_caption_benchmarking')
        save_plots: Whether to save plots instead of showing them
        exclude_fm_for_sliding_window: If True, exclude FM data for sliding_window methods
    """

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "subtitle": 14,  # Subplot titles
        "axis_label": 14,  # X and Y axis labels
        "tick_label": 14,  # Tick labels on axes
        "legend": 14,  # Legend text
        "annotation": 10,  # Any text annotations
    }

    if "transplanting_layers_caption_benchmarking" not in results_loader.data:
        print("⚠️  No transplanting layers caption benchmarking data loaded")
        return

    print(f"\n📊 Plotting caption benchmarking: {metric_name} from {directory_name}")

    # Available transplantation methods
    transplantation_methods = [
        "two_parts_s2",
        "sliding_window_ws1_s2",
    ]
    dataset_name = "coco_captioning"

    # Colors
    colors = {
        "PM": "dodgerblue",
        "FM": "crimson",
        "Layers": "skyblue",
    }

    # Create figure with 1x2 subplots (shared y-axis)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    # Collect all metric values to determine Y limits
    all_metric_values = []

    # Process each transplantation method
    for idx, method in enumerate(transplantation_methods):
        ax = axes[idx]

        try:
            # Get caption data for this method from specific directory
            data_df = results_loader.get_caption_benchmarking(
                dataset_name, method, directory_name
            )

            # Extract and validate data using helper function
            layer_numbers, layer_metrics, fm_metric, pm_metric = (
                _extract_caption_benchmarking_data(data_df, method, metric_name)
            )

            # Check if this is a sliding_window method to conditionally exclude FM
            is_sliding_window = "sliding_window" in method

            # Prepare data for plotting
            if is_sliding_window and exclude_fm_for_sliding_window:
                # For sliding_window methods, exclude FM data if flag is set
                x_positions = list(range(len(layer_numbers) + 1))  # +1 for PM only
                all_metrics = list(layer_metrics) + [pm_metric]
                x_labels = [str(num) for num in layer_numbers] + ["PM"]
            else:
                # For other methods or if flag is False, include both FM and PM
                x_positions = list(range(len(layer_numbers) + 2))  # +2 for FM and PM
                all_metrics = list(layer_metrics) + [fm_metric, pm_metric]
                x_labels = [str(num) for num in layer_numbers] + ["FM", "PM"]

            # Collect all values for Y limits
            all_metric_values.extend(all_metrics)

            # Create bar plot
            bars = ax.bar(x_positions, all_metrics, alpha=0.7, edgecolor="black")

            # Color coding: layers in blue, FM in green, PM in red
            for i, bar in enumerate(bars):
                if i < len(layer_numbers):
                    bar.set_color(colors["Layers"])
                elif x_labels[i] == "FM":
                    bar.set_color(colors["FM"])
                else:  # PM
                    bar.set_color(colors["PM"])

            # Customize subplot
            ax.set_xlabel(
                "Layer Number / Model Type", fontsize=FONT_SIZES["axis_label"]
            )
            if idx == 0:  # Only first subplot gets y-label
                ax.set_ylabel(metric_name, fontsize=FONT_SIZES["axis_label"])

            # Use display names for titles
            display_name = TRANSPLANTATION_DISPLAY_NAMES.get(method, method)
            ax.set_title(
                display_name, fontsize=FONT_SIZES["subtitle"], fontweight="bold"
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 10 else 0)
            ax.grid(True, alpha=0.3, axis="y")

            # Configure tick labels
            ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

            # Add value labels on top of bars
            for i, (pos, val) in enumerate(zip(x_positions, all_metrics)):
                # Format value: round to 0 if very small, otherwise 2 decimals
                if abs(val) < 0.01:
                    display_val = "0"
                else:
                    display_val = f"{val:.2f}"

                # Adjust text positioning based on whether FM is excluded
                if is_sliding_window and exclude_fm_for_sliding_window:
                    # For sliding_window without FM, position text closer to bars
                    text_offset = -0.05  # Smaller offset
                    print("sliding_window without FM")
                else:
                    # For methods with FM or when flag is False, use normal offset
                    text_offset = 0.01  # Normal offset

                ax.text(
                    pos,
                    val + text_offset,
                    display_val,
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["annotation"],
                )

        except Exception as e:
            print(
                f"⚠️  Failed to plot {method}/{metric_name} from {directory_name}: {e}"
            )
            ax.text(
                0.5,
                0.5,
                f"No data for\n{method}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color="red",
            )
            display_name = TRANSPLANTATION_DISPLAY_NAMES.get(method, method)
            ax.set_title(display_name, fontsize=14, fontweight="bold")

    # Set Y limits based on collected data and whether FM is excluded
    if all_metric_values:
        y_max = max(all_metric_values)

        # Check if any method is sliding_window to conditionally adjust padding
        has_sliding_window = any(
            "sliding_window" in method for method in transplantation_methods
        )

        if has_sliding_window and exclude_fm_for_sliding_window:
            # For sliding_window without FM, use smaller padding to keep bars closer
            y_max_adjusted = y_max * 1.05  # Reduced padding (5% instead of 10%)
        else:
            # For methods with FM or when flag is False, use normal padding
            y_max_adjusted = y_max * 1.1  # Normal padding (10%)

        for ax in axes:
            ax.set_ylim(0, y_max_adjusted)

    # Extract max_new_tokens from directory name
    max_tokens = _extract_max_tokens_from_directory(directory_name)

    # Add legend - conditionally exclude FM if any method is sliding_window and flag is set
    from matplotlib.patches import Patch

    # Check if any method is sliding_window to conditionally exclude FM from legend
    has_sliding_window = any(
        "sliding_window" in method for method in transplantation_methods
    )

    legend_elements = [
        Patch(facecolor=colors["Layers"], label="Layer Patching"),
        Patch(facecolor=colors["PM"], label="Pretrained Model (PM)"),
    ]

    # Only add FM to legend if no sliding_window methods are present or if flag is False
    if not (has_sliding_window and exclude_fm_for_sliding_window):
        legend_elements.insert(
            1, Patch(facecolor=colors["FM"], label="Finetuned Model (FM)")
        )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(legend_elements),
        fontsize=FONT_SIZES["legend"],
    )

    # Remove Y-axis elements for non-first subplots
    axes[1].tick_params(left=False)
    axes[1].set_ylabel("")  # Set an empty string as ylabel

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Create filename with token information
    max_tokens = _extract_max_tokens_from_directory(directory_name)
    token_suffix = f"_tokens{max_tokens}"
    filename = f"transplanting_layers_caption_benchmarking_{metric_name.lower()}{token_suffix}.png"
    save_plot_if_requested(save_plots, filename)


def plot_transplanting_layers_caption_benchmarking(
    results_loader: ResultsLoader,
    metric_name: str,
    save_plots: bool = False,
    exclude_fm_for_sliding_window: bool = True,
):
    """
    Plot caption benchmarking results for a specific metric (default directory).

    Args:
        results_loader: ResultsLoader instance with loaded data
        metric_name: Metric to plot ('CIDEr' or 'SPICE')
        save_plots: Whether to save plots instead of showing them
        exclude_fm_for_sliding_window: If True, exclude FM data for sliding_window methods
    """
    # Use default directory for backward compatibility
    default_directory = "transplanting_layers_caption_benchmarking"
    plot_caption_benchmarking_for_directory(
        results_loader,
        metric_name,
        default_directory,
        save_plots,
        exclude_fm_for_sliding_window,
    )


def _print_caption_benchmarking_summary(results_loader):
    """
    Print a summary of available caption benchmarking data.

    Args:
        results_loader: ResultsLoader instance with loaded data
    """
    if "transplanting_layers_caption_benchmarking" not in results_loader.data:
        print("⚠️  No caption benchmarking data available")
        return

    print("\n📋 Caption Benchmarking Data Summary:")
    caption_data = results_loader.data["transplanting_layers_caption_benchmarking"]

    for dataset_name, experiments in caption_data.items():
        print(f"  📊 {dataset_name}:")
        for experiment_name in experiments.keys():
            print(f"    - {experiment_name}")


def plot_combined_caption_benchmarking(
    results_loader: ResultsLoader,
    directory_name: str,
    save_plots: bool = False,
    exclude_fm_for_sliding_window: bool = True,
):
    """
    Plot combined caption benchmarking results with CIDEr and CLIP-S in subplots.
    Creates two figures: one for incremental data and one for sliding_window data.

    Args:
        results_loader: ResultsLoader instance with loaded data
        directory_name: Directory name containing the data
        save_plots: Whether to save plots instead of showing them
        exclude_fm_for_sliding_window: If True, exclude FM data for sliding_window methods
    """
    if "transplanting_layers_caption_benchmarking" not in results_loader.data:
        print("⚠️  No transplanting layers caption benchmarking data loaded")
        return

    print(f"\n📊 Plotting combined caption benchmarking from {directory_name}")

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "subtitle": 14,  # Subplot titles
        "axis_label": 14,  # X and Y axis labels
        "tick_label": 14,  # Tick labels on axes
        "legend": 14,  # Legend text
        "annotation": 10,  # Any text annotations
    }

    # Define transplantation methods and their corresponding data types
    method_groups = {
        "incremental": ["two_parts_s2"],
        "sliding_window": ["sliding_window_ws1_s2"],
    }

    dataset_name = "coco_captioning"
    metrics = ["CIDEr", "CLIP-S"]

    # Colors
    colors = {
        "PM": "dodgerblue",
        "FM": "crimson",
        "Layers": "skyblue",
    }

    # Create one figure for each data type (incremental and sliding_window)
    for data_type, methods in method_groups.items():
        # Create figure with 1x2 subplots (no shared y-axis since metrics are different)
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Collect all metric values for each subplot to determine Y limits
        all_metric_values = {metric: [] for metric in metrics}

        # Process each metric in its corresponding subplot
        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Process each transplantation method for this data type
            for method in methods:
                try:
                    # Get caption data for this method from specific directory
                    data_df = results_loader.get_caption_benchmarking(
                        dataset_name, method, directory_name
                    )

                    # Extract and validate data using helper function
                    layer_numbers, layer_metrics, fm_metric, pm_metric = (
                        _extract_caption_benchmarking_data(data_df, method, metric)
                    )

                    # Check if this is a sliding_window method to conditionally exclude FM
                    is_sliding_window = "sliding_window" in method

                    # Prepare data for plotting - rearrange order: PM first, then layers, then FM
                    if is_sliding_window and exclude_fm_for_sliding_window:
                        # For sliding_window methods, exclude FM data if flag is set
                        x_positions = list(
                            range(len(layer_numbers) + 1)
                        )  # +1 for PM only
                        all_metrics = [pm_metric] + list(
                            layer_metrics
                        )  # PM first, then layers
                        x_labels = ["PM"] + [
                            str(num) for num in layer_numbers
                        ]  # PM first, then layers
                    else:
                        # For other methods or if flag is False, include both FM and PM
                        x_positions = list(
                            range(len(layer_numbers) + 2)
                        )  # +2 for PM and FM
                        all_metrics = (
                            [pm_metric] + list(layer_metrics) + [fm_metric]
                        )  # PM first, then layers, then FM
                        x_labels = (
                            ["PM"] + [str(num) for num in layer_numbers] + ["FM"]
                        )  # PM first, then layers, then FM

                    # Collect all values for Y limits
                    all_metric_values[metric].extend(all_metrics)

                    # Create bar plot
                    bars = ax.bar(
                        x_positions, all_metrics, alpha=0.7, edgecolor="black"
                    )

                    # Color coding: PM in red (first), layers in blue, FM in green (if present)
                    for i, bar in enumerate(bars):
                        if x_labels[i] == "PM":
                            bar.set_color(colors["PM"])  # PM is first
                        elif x_labels[i] == "FM":
                            bar.set_color(colors["FM"])  # FM is last (if present)
                        else:
                            bar.set_color(colors["Layers"])  # Layers are in the middle

                    # Customize subplot
                    ax.set_xlabel(
                        "Layer Number / Model Type", fontsize=FONT_SIZES["axis_label"]
                    )
                    ax.set_ylabel(metric, fontsize=FONT_SIZES["axis_label"])

                    # Set title based on data type
                    # ax.set_title(
                    #     data_type, fontsize=FONT_SIZES["subtitle"], fontweight="bold"
                    # )
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(
                        x_labels, rotation=45 if len(x_labels) > 10 else 0
                    )
                    ax.grid(True, alpha=0.3, axis="y")

                    # Configure tick labels
                    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

                    # Add value labels on top of bars
                    for i, (pos, val) in enumerate(zip(x_positions, all_metrics)):
                        # Format value: round to 0 if very small, otherwise 2 decimals
                        if abs(val) < 0.01:
                            display_val = "0"
                        else:
                            display_val = f"{val:.2f}"

                        # Adjust text positioning based on whether FM is excluded
                        if is_sliding_window and exclude_fm_for_sliding_window:
                            # For sliding_window without FM, position text closer to bars
                            text_offset = 0.0
                        else:
                            # For methods with FM or when flag is False, use normal offset
                            text_offset = 0.01  # Normal offset

                        ax.text(
                            pos,
                            val + text_offset,
                            display_val,
                            ha="center",
                            va="bottom",
                            fontsize=FONT_SIZES["annotation"],
                        )

                except Exception as e:
                    print(
                        f"⚠️  Failed to plot {method}/{metric} from {directory_name}: {e}"
                    )
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for\n{method}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=14,
                        color="red",
                    )
                    ax.set_title(data_type, fontsize=14, fontweight="bold")

        # Set Y limits for each subplot based on collected data and whether FM is excluded
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            if all_metric_values[metric]:
                y_max = max(all_metric_values[metric])

                # Check if this is a sliding_window data type to conditionally adjust padding
                is_sliding_window_data = data_type == "sliding_window"

                if is_sliding_window_data and exclude_fm_for_sliding_window:
                    # For sliding_window without FM, use smaller padding to keep bars closer
                    y_max_adjusted = y_max * 1.15  # Reduced padding (5% instead of 10%)
                else:
                    # For methods with FM or when flag is False, use normal padding
                    y_max_adjusted = y_max * 1.1  # Normal padding (10%)

                ax.set_ylim(0, y_max_adjusted)

        # Extract max_new_tokens from directory name
        max_tokens = _extract_max_tokens_from_directory(directory_name)

        # Add legend - conditionally exclude FM for sliding_window data type
        from matplotlib.patches import Patch

        # Check if this is a sliding_window data type to conditionally exclude FM from legend
        is_sliding_window_data = data_type == "sliding_window"

        legend_elements = [
            Patch(facecolor=colors["PM"], label="Pretrained Model (PM)"),
            Patch(facecolor=colors["Layers"], label="Layer Patching"),
        ]

        if is_sliding_window_data:
            if not exclude_fm_for_sliding_window:
                legend_elements.append(
                    Patch(facecolor=colors["FM"], label="Finetuned Model (FM)")
                )
        else:
            legend_elements.append(
                Patch(facecolor=colors["FM"], label="Finetuned Model (FM)")
            )

        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=len(legend_elements),
            fontsize=FONT_SIZES["legend"],
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Create filename with token information
        max_tokens = _extract_max_tokens_from_directory(directory_name)
        token_suffix = f"_tokens{max_tokens}"
        filename = f"transplanting_layers_caption_benchmarking_combined_{data_type}{token_suffix}.png"
        save_plot_if_requested(save_plots, filename)


def plot_all_transplanting_layers_caption_benchmarking(
    results_loader: ResultsLoader,
    save_plots: bool = False,
    exclude_fm_for_sliding_window: bool = True,
):
    """
    Plot all available caption benchmarking experiments for all metrics.

    Args:
        results_loader: ResultsLoader instance with loaded data
        save_plots: Whether to save plots instead of showing them
        exclude_fm_for_sliding_window: If True, exclude FM data for sliding_window methods
    """
    print("\n📊 Plotting all transplanting layers caption benchmarking experiments...")

    # Print summary of available data
    _print_caption_benchmarking_summary(results_loader)

    # Available metrics - plot each metric separately
    metrics = [
        "CIDEr",
        "CLIP-S",
    ]

    # Available directories from config
    caption_directories = results_loader.config.get("caption_benchmarking", {}).get(
        "directories", []
    )

    # for metric in metrics:
    #     for directory in caption_directories:
    #         plot_caption_benchmarking_for_directory(
    #             results_loader, metric, directory, save_plots=save_plots
    #         )

    # Generate combined plots with CIDEr and CLIP-S in subplots
    for directory in caption_directories:
        plot_combined_caption_benchmarking(
            results_loader,
            directory,
            save_plots=save_plots,
            exclude_fm_for_sliding_window=exclude_fm_for_sliding_window,
        )


def plot_similarities_difference_measures(
    results_loader: ResultsLoader,
    stream_type: str,
    save_plots: bool = False,
):
    """Plot the similarity difference measures for the output layer."""
    # from matplotlib.colors import TABLEAU_COLORS

    # hex_colors = list(TABLEAU_COLORS.values())
    hex_colors = ["black", "cyan"]

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "axis_label": 14,  # X and Y axis labels
        "tick_label": 14,  # Tick labels on axes
        "legend": 14,  # Legend text
        "title": 14,  # Title
    }
    # Set style for better plots
    plt.style.use("default")

    # Initialize figure and axes
    height = 6
    width = 6
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))

    # Collect all data to determine Y limits
    min_value = float("inf")
    max_value = float("-inf")

    # Default values
    dataset_name = "cocoqa_txt_minus_cocoqa_img"
    pooling = "last"
    measures = ["neighborhood_overlap", "linear_cka"]

    try:
        for m_idx, measure in enumerate(measures):
            data = results_loader.get_similarity(
                dataset=dataset_name,
                stream_type=stream_type,
                pooling=pooling,
                measure=measure,
            )

            if isinstance(data, torch.Tensor):
                data = safe_tensor_to_numpy(data)

            min_value = min(min_value, data.min())
            max_value = max(max_value, data.max())

            ax.plot(
                range(len(data)),
                data,
                # color=hex_colors[(m_idx + 2) % len(hex_colors)],
                color=hex_colors[m_idx],
                linestyle="--",
                marker="o",
                label=MEASURE_DISPLAY_NAMES[measure],
            )

        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.grid(True, alpha=0.3)

        # Set title and labels
        ax.set_title(
            "Text QA - Visual QA", fontsize=FONT_SIZES["title"], fontweight="bold"
        )
        ax.set_ylabel(
            "Similarity Difference",
            fontsize=FONT_SIZES["axis_label"],
        )
        ax.set_xlabel("Layer", fontsize=FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])
        ax.set_ylim(
            min_value * (0.95 if min_value > 0 else 1.15),
            max_value * 1.05,
        )

        plt.tight_layout()

        # Create filename and save plot
        filename = f"similarity_difference_measures_{stream_type}_{pooling}.png"
        save_plot_if_requested(save_plots, filename)

    except Exception as e:
        print(f"⚠️  Failed to plot {dataset_name}/{stream_type}_last: {e}")


def plot_all_similarities_difference_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """Plot all similarities difference measures for the output layer."""
    for stream_type in ["output_layer", "mean_heads_projection"]:
        plot_similarities_difference_measures(
            results_loader,
            stream_type,
            save_plots,
        )


def plot_model_data_measures_differences(
    results_loader: ResultsLoader,
    dataset_measure: str,
    save_plots: bool = False,
):
    """
    Plot model data measures differences (pretrained - finetuned) for a specific measure type.

    Args:
        results_loader: ResultsLoader instance containing the data
        dataset_measure: Type of measure to plot (e.g., 'dataset_entropy', 'intrinsic_dimension')
        save_plots: Whether to save the generated plots
    """
    from matplotlib.colors import TABLEAU_COLORS

    # Define font sizes for thesis-quality plots
    FONT_SIZES = {
        "subtitle": 14,  # Subplot titles
        "axis_label": 14,  # X and Y axis labels
        "tick_label": 14,  # Tick labels on axes
        "legend": 14,  # Legend text
    }

    # Set style for better plots
    plt.style.use("default")
    hex_colors = list(TABLEAU_COLORS.values())

    # Default model and data key for differences
    models_data = "pretrained-finetuned"
    data_key = "output_layer_last"

    # Initialize figure and axes
    height = 6
    width = 5
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))

    # y limits
    y_min = float("inf")
    y_max = float("-inf")

    try:
        if "model_data_measures" not in results_loader.data:
            print("⚠️  No model data measures loaded")
            return

        model_data_measures = results_loader.data["model_data_measures"]

        datasets = []
        for dataset_name, dataset_data in model_data_measures.items():
            datasets.append(dataset_name)

        # Plot each dataset
        for d_idx, dataset in enumerate(datasets):
            # Get model data for the current dataset
            model_data = model_data_measures[dataset][models_data][dataset_measure]

            # Handle different measure types with appropriate data keys
            if dataset_measure == "intrinsic_dimension":
                # Intrinsic dimension has additional parameters in the key
                matching_keys = [k for k in model_data.keys() if k.startswith(data_key)]
                if not matching_keys:
                    continue  # Skip if no matching keys found
                actual_key = matching_keys[0]
            else:
                actual_key = data_key
                if actual_key not in model_data:
                    continue  # Skip if key doesn't exist

            try:
                # Extract data from the nested structure
                data_dict = model_data[actual_key]
                if isinstance(data_dict, dict) and len(data_dict) > 0:
                    # Get the first (and usually only) tensor from the dict
                    data = next(iter(data_dict.values()))
                else:
                    data = data_dict

                # Convert PyTorch tensor to numpy if needed
                if isinstance(data, torch.Tensor):
                    data = safe_tensor_to_numpy(data)

                # Collect data for Y limits calculation
                y_min = min(y_min, data.min())
                y_max = max(y_max, data.max())

                # Set styling: same color for dataset across all plots
                color = hex_colors[d_idx % len(hex_colors)]

                # Plot the difference data with dashed lines
                ax.plot(
                    range(len(data)),
                    data,
                    color=color,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    linestyle="--",
                    label=get_display_name(dataset),  # Add label for legend
                )

            except Exception as e:
                print(
                    f"⚠️  Failed to plot {dataset}/{models_data}/{dataset_measure}/{actual_key}: {e}"
                )

        # Configure plot appearance and labels
        ax.set_title(
            f"{MODEL_DISPLAY_NAMES[models_data]}", fontsize=FONT_SIZES["subtitle"]
        )
        ax.grid(True, alpha=0.3)

        # Set Y-axis label with proper measure name
        ax.set_ylabel(
            f"{DATASET_MEASURE_DISPLAY_NAMES[dataset_measure]} Difference",
            fontsize=FONT_SIZES["axis_label"],
        )

        ax.set_xlabel("Layer", fontsize=FONT_SIZES["axis_label"])

        # Configure tick labels
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

        # Set Y-axis limits with proper padding
        if y_min != float("inf") and y_max != float("-inf"):
            ax.set_ylim(
                y_min * (0.95 if y_min > 0 else 1.15),
                y_max * 1.05,
            )

        # Ensure proper layout
        plt.tight_layout()

        # Save plot if requested
        filename = f"model_data_measures_differences_{dataset_measure}.png"
        save_plot_if_requested(save_plots, filename)
    except Exception as e:
        print(f"⚠️  Failed to plot {dataset_measure}: {e}")
        traceback.print_exc()


def plot_all_model_data_measures_differences(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """Plot all model data measures differences for different stream types and dataset measures."""

    for dataset_measure in ["dataset_entropy", "intrinsic_dimension"]:
        plot_model_data_measures_differences(
            results_loader,
            dataset_measure,
            save_plots,
        )


def inspect_similarity_results(results: ResultsLoader):
    """Inspect the results."""
    import pandas as pd

    # ===============================
    # Similarities
    # ===============================
    # neighborhood overlap and linear_cka of all the models
    # for the cocoqa_txt, cocoqa_img, cocoqa_txt_minus_cocoqa_img
    # residual stream: output_layer
    # pooling: last
    similarities_dict = {
        "cocoqa_txt": {
            "neighborhood_overlap": results.get_similarity(
                dataset="cocoqa_txt",
                stream_type="output_layer",
                pooling="last",
                measure="neighborhood_overlap",
            ).numpy(),
            "linear_cka": results.get_similarity(
                dataset="cocoqa_txt",
                stream_type="output_layer",
                pooling="last",
                measure="linear_cka",
            ).numpy(),
        },
        "cocoqa_img": {
            "neighborhood_overlap": results.get_similarity(
                dataset="cocoqa_img",
                stream_type="output_layer",
                pooling="last",
                measure="neighborhood_overlap",
            ).numpy(),
            "linear_cka": results.get_similarity(
                dataset="cocoqa_img",
                stream_type="output_layer",
                pooling="last",
                measure="linear_cka",
            ).numpy(),
        },
        "cocoqa_txt_minus_cocoqa_img": {
            "neighborhood_overlap": results.get_similarity(
                dataset="cocoqa_txt_minus_cocoqa_img",
                stream_type="output_layer",
                pooling="last",
                measure="neighborhood_overlap",
            ).numpy(),
            "linear_cka": results.get_similarity(
                dataset="cocoqa_txt_minus_cocoqa_img",
                stream_type="output_layer",
                pooling="last",
                measure="linear_cka",
            ).numpy(),
        },
    }
    cocoqa_txt_similarities_pd = pd.DataFrame(similarities_dict["cocoqa_txt"])
    cocoqa_img_similarities_pd = pd.DataFrame(similarities_dict["cocoqa_img"])
    cocoqa_txt_minus_cocoqa_img_similarities_pd = pd.DataFrame(
        similarities_dict["cocoqa_txt_minus_cocoqa_img"]
    )
    print("cocoqa_txt:")
    print(cocoqa_txt_similarities_pd)
    print("cocoqa_img:")
    print(cocoqa_img_similarities_pd)
    print("cocoqa_txt_minus_cocoqa_img:")
    print(cocoqa_txt_minus_cocoqa_img_similarities_pd)


def inspect_intrinsic_dimension(results: ResultsLoader):
    """Inspect the intrinsic dimension."""
    import pandas as pd

    models = ["pretrained", "finetuned", "pretrained-finetuned"]
    datasets = ["cocoqa_txt", "cocoqa_img", "coco_captioning"]
    intrinsic_dimension_dict = {}
    for model in models:
        intrinsic_dimension_dict[model] = {}
        for dataset in datasets:
            tmp = results.get_intrinsic_dimension(
                dataset=dataset,
                model=model,
                stream_type="output_layer",
                pooling="last",
            )
            if isinstance(tmp, dict):
                tmp = tmp["layers_intrinsic_dimension"]
            tmp = tmp.numpy()
            intrinsic_dimension_dict[model][dataset] = tmp

    for model in models:
        print(f"{model}:")
        df = pd.DataFrame(intrinsic_dimension_dict[model])
        print(df)
        print("\n")


def inspect_dataset_entropy(results: ResultsLoader):
    """Inspect the dataset entropy."""
    import pandas as pd

    models = ["pretrained", "finetuned", "pretrained-finetuned"]
    datasets = ["cocoqa_txt", "cocoqa_img", "coco_captioning"]
    dataset_entropy_dict = {}
    for model in models:
        dataset_entropy_dict[model] = {}
        for dataset in datasets:
            tmp = results.get_entropy(
                dataset=dataset,
                model=model,
                stream_type="output_layer",
                pooling="last",
            )
            if isinstance(tmp, dict):
                tmp = tmp["layers_dataset_entropy"]
            tmp = tmp.numpy()
            dataset_entropy_dict[model][dataset] = tmp

    # Print the dataset entropy for each model
    for model in models:
        print(f"{model}:")
        df = pd.DataFrame(dataset_entropy_dict[model])
        print(df)
        print("\n")


def main(save_plots: bool = False):
    """Main function to demonstrate similarity measures heatmap."""
    # Initialize results loader and load all data
    results = ResultsLoader()
    results.load_all()
    results.print_summary()

    # Get the dataset names from the config
    dataset_names = results.config["datasets"]

    if ("cocoqa_txt" in dataset_names) and ("cocoqa_img" in dataset_names):
        # Create similarities difference dataset using renamed method
        results.create_similarities_difference_dataset(
            source_dataset1="cocoqa_txt",
            source_dataset2="cocoqa_img",
            target_dataset="cocoqa_txt_minus_cocoqa_img",
            verbose=False,
        )

        # Compute mean heads projections at layers using renamed method
        results.compute_mean_heads_projections_at_layers()

        # Create model measures differences for each dataset
        print("\n🔄 Creating model measures differences...")

        # Create model measures differences using renamed method
        results.create_dataset_measures_models_difference(
            dataset_name="cocoqa_txt",
            model1="pretrained",
            model2="finetuned",
        )
        results.create_dataset_measures_models_difference(
            dataset_name="cocoqa_img",
            model1="pretrained",
            model2="finetuned",
        )

    if "coco_captioning" in dataset_names:
        results.create_dataset_measures_models_difference(
            dataset_name="coco_captioning",
            model1="pretrained",
            model2="finetuned",
        )

    results.print_summary()

    # ! results loaded and processed correctly: don't touch unless necessary
    print("🎉 All processing completed!")

    # Generate thesis plots
    plot_all_output_layer_similarity_measures(
        results,
        save_plots=save_plots,
    )
    plot_all_mean_heads_projection_similarity_measures(
        results,
        save_plots=save_plots,
    )
    plot_all_post_mlp_similarity_measures(results, save_plots=save_plots)
    plot_all_similarities_difference_measures(
        results,
        save_plots=save_plots,
    )
    plot_all_similarity_measures_matrices(
        results, save_plots=save_plots, show_annotations=True
    )
    plot_all_model_data_measures(results, save_plots=save_plots)
    plot_all_model_data_measures_differences(
        results,
        save_plots=save_plots,
    )
    plot_modalities_similarities(
        results,
        ["output_layer"],
        "cosine_similarity",
        save_plots=save_plots,
    )
    plot_modalities_similarities(
        results,
        ["output_layer"],
        "homogeneity_score_cosine",
        save_plots=save_plots,
    )
    plot_all_transplanting_layers_qa_benchmarking(
        results,
        save_plots=save_plots,
        exclude_fm_for_sliding_window=True,
    )
    plot_all_transplanting_layers_caption_benchmarking(
        results,
        save_plots=save_plots,
        exclude_fm_for_sliding_window=True,
    )

    # ===============================
    # Results Inspection
    # ===============================
    # inspect_similarity_results(results)
    # inspect_intrinsic_dimension(results)
    # inspect_dataset_entropy(results)

    # ===============================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multimodal analysis plots")
    parser.add_argument(
        "--save", action="store_true", help="Save plots instead of showing them"
    )
    args = parser.parse_args()

    main(save_plots=args.save)
