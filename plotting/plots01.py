import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from typing import List, Optional, Union
from functools import partial
import torch
import numpy as np
from pathlib import Path
import re
from utils import plot_similarity_measure_matrix
from config_results_loader import ResultsLoader

# Constants for commonly used values
DEFAULT_MEASURES = ["neighborhood_overlap", "linear_cka"]
DEFAULT_POOLING_METHODS = ["last", "mean"]
PROMPT_ENTROPY_POOLING = ["none"]
PLOTS_DIR = Path(__file__).parent.parent / "plots"


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


def plot_similarity_measures_matrix(
    results_loader: ResultsLoader,
    dataset: str,
    stream_type: str,
    pooling: str,
    measures: List[str],
    subplot_width: int = 8,
    plot_height: int = 10,
    percentage_threshold: float = 0.1,
    top_p_percent: float = 0.05,
    bottom_p_percent: float = 0.03,
    max_xticks: Optional[int] = None,
    max_yticks: Optional[int] = None,
    annot_font_size: int = 8,
    colorbar_labelsize: int = 12,
    wspace: float = 0.0,
    save_plots: bool = False,
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
            measure=measure,
            max_xticks=max_xticks,
            max_yticks=max_yticks,
            percentage_threshold=percentage_threshold,
            top_p_percent=top_p_percent,
            bottom_p_percent=bottom_p_percent,
            annot_font_size=annot_font_size,
            colorbar_labelsize=colorbar_labelsize,
        )

        # Clean up y-axis elements for all columns except the first
        if idx > 0:
            axes[idx].set_ylabel("")
            axes[idx].set_yticklabels([])
            axes[idx].tick_params(left=False)  # Remove y-axis ticks

    # Configure overall figure layout with reduced spacing
    plt.suptitle(f"{dataset} - {stream_type} - {pooling}", fontsize=16, y=0.97)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=wspace)  # Reduced wspace for tighter layout

    filename = (
        f"similarity_matrix_{dataset}_{stream_type}_{pooling}_{'_'.join(measures)}.png"
    )
    save_plot_if_requested(save_plots, filename)


def plot_all_similarity_measures_matrices(
    results_loader: ResultsLoader, save_plots: bool = False
):
    """Plot similarity measures matrices for heads projection data."""
    # Configuration for heads projection matrices
    datasets = ["cocoqa_txt", "cocoqa_img", "cocoqa_txt_minus_cocoqa_img"]
    pooling_methods = ["last", "mean"]

    template_plot = partial(
        plot_similarity_measures_matrix,
        results_loader=results_loader,
        measures=DEFAULT_MEASURES,
        max_xticks=10,
        max_yticks=10,
        percentage_threshold=0.08,
        top_p_percent=0.03,
        bottom_p_percent=0.03,
        annot_font_size=6,
        colorbar_labelsize=14,
        subplot_width=10,
        plot_height=8,
        wspace=-0.05,
        save_plots=save_plots,
    )

    for dataset in datasets:
        for pooling in pooling_methods:
            template_plot(
                dataset=dataset,
                stream_type="heads_projection",
                pooling=pooling,
            )


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
        pooling_methods: List of pooling methods (defaults to ["last", "mean"])
        measures: List of measures (defaults to ["neighborhood_overlap", "linear_cka"])
        save_plots: Whether to save plots instead of showing them
    """
    if pooling_methods is None:
        pooling_methods = DEFAULT_POOLING_METHODS
    if measures is None:
        measures = DEFAULT_MEASURES

    print(f"\n📊 Plotting {stream_type} with combined pooling methods...")
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
    )


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

    # Set style for better plots
    plt.style.use("default")
    hex_colors = list(TABLEAU_COLORS.values())

    # Define markers and linestyles for different pooling methods
    pooling_styles = {
        "last": {"marker": "o", "linestyle": "-"},
        "mean": {"marker": "s", "linestyle": "--"},
        "none": {
            "marker": "^",
            "linestyle": ":",
        },  # Triangle marker, dotted line for "none"
    }

    height = 8
    width = 6 * len(measures)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(measures), figsize=(width, height), sharey=True
    )

    # Handle single subplot case (axes is not a list when ncols=1)
    if len(measures) == 1:
        axes = [axes]

    datasets = results_loader.config["datasets"].copy()
    datasets.append("cocoqa_txt_minus_cocoqa_img")
    datasets.remove("coco_captioning")

    # Collect all data to determine Y limits
    min_all_data_values = []

    # Iterate through each measure (subplot)
    for m_idx, measure in enumerate(measures):
        ax = axes[m_idx]

        # Plot each combination of dataset and pooling method
        for d_idx, dataset in enumerate(datasets):
            for pooling in pooling_methods:
                try:
                    data = results_loader.get_similarity(
                        dataset=dataset,
                        stream_type=stream_type,
                        pooling=pooling,
                        measure=measure,
                    )
                    if isinstance(data, torch.Tensor):
                        data = safe_tensor_to_numpy(data)

                    # Collect data for Y limits calculation
                    min_all_data_values.append(data.min())

                    # Set styling: same color for dataset, different marker/linestyle for pooling
                    color = hex_colors[d_idx % len(hex_colors)]
                    marker = pooling_styles[pooling]["marker"]
                    linestyle = pooling_styles[pooling]["linestyle"]

                    template_ax_plot(
                        ax,
                        data,
                        color=color,
                        marker=marker,
                        markersize=4,
                        linewidth=2,
                        linestyle=linestyle,
                    )

                except Exception as e:
                    print(
                        f"⚠️  Failed to plot {dataset}/{stream_type}_{pooling}/{measure}: {e}"
                    )

        # Configure subplot
        ax.set_title(f"{measure}", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Configure axis labels
        if m_idx == 0:
            ax.set_ylabel("Similarity", fontsize=12)
        else:
            # Remove Y-axis elements for non-first subplots
            ax.tick_params(left=False)
            ax.set_ylabel("")  # Set an empty string as ylabel

        ax.set_xlabel("Layer", fontsize=12)

    # Set Y limits: min = 0 or minimum data value, max = 1
    if min_all_data_values:
        y_min = min(0, min(min_all_data_values))
        y_max = 1.0
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    # === ADD CUSTOM LEGENDS ONLY AT THE END ===
    # Only add legends to the first subplot (0,0)
    first_ax = axes[0]

    # Create custom legend for datasets (colors)
    dataset_legend_elements = []
    for d_idx, dataset in enumerate(datasets):
        color = hex_colors[d_idx % len(hex_colors)]
        dataset_legend_elements.append(
            Line2D([0], [0], color=color, linewidth=3, label=dataset)
        )

    # Create custom legend for pooling methods (markers/linestyles)
    pooling_legend_elements = []
    for pooling in pooling_methods:
        marker = pooling_styles[pooling]["marker"]
        linestyle = pooling_styles[pooling]["linestyle"]
        pooling_legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=marker,
                linestyle=linestyle,
                markersize=8,  # Larger marker for better visibility
                linewidth=2.5,  # Thicker line for better visibility
                markerfacecolor="black",  # Fill marker with black
                markeredgecolor="black",  # Black marker edge
                markeredgewidth=1,  # Visible marker edge
                label=pooling,
            )
        )

    # Add dataset legend first
    dataset_legend = first_ax.legend(
        handles=dataset_legend_elements,
        # title="",
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.0, 1),
        ncol=len(datasets),
    )

    # Add dataset legend as artist to preserve it when adding second legend
    first_ax.add_artist(dataset_legend)

    # Add pooling methods legend second
    first_ax.legend(
        handles=pooling_legend_elements,
        # title="Pooling Methods",
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.0, 0.95),
        ncol=len(pooling_methods),
        handlelength=5.0,  # Increase line length in legend
        handletextpad=1.0,  # Add space between line and text
    )

    # Set overall title
    plt.suptitle(f"{stream_type}", fontsize=16, y=0.98)
    plt.tight_layout()

    filename = f"similarity_measures_{stream_type}_{'_'.join(pooling_methods)}_{'_'.join(measures)}.png"
    save_plot_if_requested(save_plots, filename)


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

    # Set style for better plots
    plt.style.use("default")
    hex_colors = list(TABLEAU_COLORS.values())

    # Define markers and linestyles for different pooling methods
    pooling_styles = {
        "last": {"marker": "o", "linestyle": "-"},
        "mean": {"marker": "s", "linestyle": "--"},
        "none": {"marker": "^", "linestyle": ":"},
    }

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

    # Create subplots: 1 row, num_models columns
    height = 8
    width = 6 * len(all_models)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(all_models), figsize=(width, height), sharey=True
    )

    # Handle single subplot case
    if len(all_models) == 1:
        axes = [axes]

    # Collect all data values for Y limits
    all_data_values = []

    # Plot for each model (subplot)
    for m_idx, model in enumerate(all_models):
        ax = axes[m_idx]

        # Plot each combination of dataset and pooling method
        for d_idx, dataset in enumerate(datasets):
            # Check if this model exists in this dataset
            if model not in model_data_measures[dataset]:
                continue

            model_data = model_data_measures[dataset][model]

            # Check if the measure type exists
            if dataset_measure not in model_data:
                continue

            measure_data = model_data[dataset_measure]

            for pooling in pooling_methods:
                # Handle different pooling methods for different measure types
                if dataset_measure == "prompt_entropy":
                    # Prompt entropy uses "none" as pooling method
                    key = f"{stream_type}_none"
                else:
                    # Construct the key: stream_type_pooling
                    key = f"{stream_type}_{pooling}"

                # For intrinsic dimension, we need to handle additional parameters
                if dataset_measure == "intrinsic_dimension":
                    # Look for keys that start with our prefix (e.g., "output_layer_last")
                    matching_keys = [
                        k for k in measure_data.keys() if k.startswith(key)
                    ]
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
                    all_data_values.extend(data.flatten())

                    # Set styling: same color for dataset, different marker/linestyle for pooling
                    color = hex_colors[d_idx % len(hex_colors)]
                    marker = pooling_styles[pooling]["marker"]
                    linestyle = pooling_styles[pooling]["linestyle"]

                    # Plot the data
                    ax.plot(
                        range(len(data)),
                        data,
                        color=color,
                        marker=marker,
                        markersize=4,
                        linewidth=2,
                        linestyle=linestyle,
                    )

                except Exception as e:
                    print(
                        f"⚠️  Failed to plot {dataset}/{model}/{dataset_measure}/{actual_key}: {e}"
                    )

        # Configure subplot
        ax.set_title(f"{model}", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Configure axis labels
        if m_idx == 0:
            ax.set_ylabel(f"{dataset_measure.replace('_', ' ').title()}", fontsize=12)
        else:
            # Remove Y-axis elements for non-first subplots
            ax.tick_params(left=False)
            ax.set_ylabel("")

        ax.set_xlabel("Layer", fontsize=12)

    # Set Y limits based on all data
    if all_data_values:
        y_min = min(all_data_values)
        y_max = max(all_data_values)
        # Add some padding
        y_range = y_max - y_min
        y_min_padded = y_min - 0.05 * y_range
        y_max_padded = y_max + 0.05 * y_range

        for ax in axes:
            ax.set_ylim(y_min_padded, y_max_padded)

    # === ADD CUSTOM LEGENDS ONLY AT THE END ===
    # Only add legends to the first subplot (0,0)
    if len(all_models) > 0:
        first_ax = axes[0]

        # Create custom legend for datasets (colors)
        dataset_legend_elements = []
        for d_idx, dataset in enumerate(datasets):
            color = hex_colors[d_idx % len(hex_colors)]
            dataset_legend_elements.append(
                Line2D([0], [0], color=color, linewidth=3, label=dataset)
            )

        # Create custom legend for pooling methods (markers/linestyles)
        pooling_legend_elements = []
        for pooling in pooling_methods:
            marker = pooling_styles[pooling]["marker"]
            linestyle = pooling_styles[pooling]["linestyle"]
            pooling_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=marker,
                    linestyle=linestyle,
                    markersize=8,
                    linewidth=2.5,
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label=pooling,
                )
            )

        data_measures_kwargs = {
            "datasets": {},
            "pooling_methods": {},
        }
        if "coco_captioning" in datasets and dataset_measure == "intrinsic_dimension":
            data_measures_kwargs["datasets"]["loc"] = "upper right"
            data_measures_kwargs["datasets"]["bbox_to_anchor"] = (1, 0.95)
            data_measures_kwargs["pooling_methods"]["loc"] = "upper right"
            data_measures_kwargs["pooling_methods"]["bbox_to_anchor"] = (1, 1.0)
        else:
            data_measures_kwargs["datasets"]["loc"] = "lower right"
            data_measures_kwargs["datasets"]["bbox_to_anchor"] = (1, 0.0)
            data_measures_kwargs["pooling_methods"]["loc"] = "lower right"
            data_measures_kwargs["pooling_methods"]["bbox_to_anchor"] = (1, 0.05)

        # Add dataset legend first
        dataset_legend = first_ax.legend(
            handles=dataset_legend_elements,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=len(datasets),
            **data_measures_kwargs["datasets"],
        )

        # Add dataset legend as artist to preserve it when adding second legend
        first_ax.add_artist(dataset_legend)

        # Add pooling methods legend second
        first_ax.legend(
            handles=pooling_legend_elements,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=len(pooling_methods),
            handlelength=5.0,
            handletextpad=1.0,
            **data_measures_kwargs["pooling_methods"],
        )

    # Set overall title
    plt.suptitle(
        f"{dataset_measure.replace('_', ' ').title()} - {stream_type}",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout()

    filename = f"model_data_measures_{dataset_measure}_{stream_type}_{'_'.join(pooling_methods)}.png"
    save_plot_if_requested(save_plots, filename)


def plot_all_model_data_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """Plot all model data measures for different stream types and dataset measures."""
    # Define configurations for different combinations
    configs = [
        ("dataset_entropy", "output_layer", DEFAULT_POOLING_METHODS),
        ("prompt_entropy", "output_layer", PROMPT_ENTROPY_POOLING),
        ("intrinsic_dimension", "output_layer", DEFAULT_POOLING_METHODS),
        ("dataset_entropy", "post_mlp", DEFAULT_POOLING_METHODS),
        ("prompt_entropy", "post_mlp", PROMPT_ENTROPY_POOLING),
        ("intrinsic_dimension", "post_mlp", DEFAULT_POOLING_METHODS),
    ]

    for dataset_measure, stream_type, pooling_methods in configs:
        template_plot_model_data_measures(
            results_loader=results_loader,
            dataset_measure=dataset_measure,
            stream_type=stream_type,
            pooling_methods=pooling_methods,
            save_plots=save_plots,
        )


def plot_transplanting_layers_benchmarking(
    results_loader: ResultsLoader,
    transplantation_method: str,
    save_plots: bool = False,
):
    """
    Plot benchmarking and transplanting layers.

    Args:
        results_loader: ResultsLoader instance with loaded data
        transplantation_method: Either 'sliding_window_ws2_s2' or 'two_parts_s2'
    """

    # Load transplanting data for both datasets
    datasets = ["cocoqa_txt", "cocoqa_img"]

    # Validate transplantation method
    if "transplanting_layers_benchmarking" not in results_loader.data:
        print("⚠️  No transplanting layers benchmarking data loaded")
        return

    print(f"\n📊 Plotting transplanting layers benchmarking: {transplantation_method}")

    # Create figure with 1x2 subplots (shared y-axis)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

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

            # Prepare data for plotting
            x_positions = list(range(len(layer_numbers) + 2))  # +2 for FM and PM
            all_accuracies = list(layer_accuracies) + [fm_accuracy, pm_accuracy]
            x_labels = [str(num) for num in layer_numbers] + ["FM", "PM"]

            # Create bar plot
            bars = ax.bar(x_positions, all_accuracies, alpha=0.7, edgecolor="black")

            # Color coding: layers in blue, FM in green, PM in red
            for i, bar in enumerate(bars):
                if i < len(layer_numbers):
                    bar.set_color("skyblue")
                elif x_labels[i] == "FM":
                    bar.set_color("lightgreen")
                else:  # PM
                    bar.set_color("lightcoral")

            # Customize subplot
            ax.set_xlabel("Layer Number / Model Type", fontsize=12)
            if idx == 0:  # Only first subplot gets y-label
                ax.set_ylabel("Accuracy", fontsize=12)

            ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 10 else 0)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on top of bars
            for i, (pos, acc) in enumerate(zip(x_positions, all_accuracies)):
                ax.text(
                    pos,
                    acc + 0.01,
                    f"{acc:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
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
            ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")

    # Add overall title
    method_display = transplantation_method.replace("_", " ").title()
    plt.suptitle(
        f"Transplanting Layers Benchmarking: {method_display}",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="skyblue", label="Layer Transplantation"),
        Patch(facecolor="lightgreen", label="Finetuned Model (FM)"),
        Patch(facecolor="lightcoral", label="Pretrained Model (PM)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        fontsize=11,
    )

    # Remove Y-axis elements for non-first subplots
    axes[1].tick_params(left=False)
    axes[1].set_ylabel("")  # Set an empty string as ylabel

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for title and legend

    filename = f"transplanting_layers_benchmarking_{transplantation_method}.png"
    save_plot_if_requested(save_plots, filename)


def plot_all_transplanting_layers_benchmarking(
    results_loader: ResultsLoader, save_plots: bool = False
):
    """
    Plot all available transplanting layers benchmarking experiments.
    """
    print("\n📊 Plotting all transplanting layers benchmarking experiments...")

    # Available transplantation methods
    methods = ["sliding_window_ws2_s2", "two_parts_s2"]

    for method in methods:
        plot_transplanting_layers_benchmarking(
            results_loader, method, save_plots=save_plots
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
):
    """
    Plot caption benchmarking results for a specific metric and directory.

    Args:
        results_loader: ResultsLoader instance with loaded data
        metric_name: Metric to plot ('CIDEr', 'CLIP-S', 'RefCLIP-S')
        directory_name: Directory name (e.g., 'transplanting_layers_caption_benchmarking')
        save_plots: Whether to save plots instead of showing them
    """
    if "transplanting_layers_caption_benchmarking" not in results_loader.data:
        print("⚠️  No transplanting layers caption benchmarking data loaded")
        return

    print(f"\n📊 Plotting caption benchmarking: {metric_name} from {directory_name}")

    # Available transplantation methods
    transplantation_methods = ["two_parts_s2", "sliding_window_ws2_s2"]
    dataset_name = "coco_captioning"

    # Create figure with 1x2 subplots (shared y-axis)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

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

            # Prepare data for plotting
            x_positions = list(range(len(layer_numbers) + 2))
            all_metrics = list(layer_metrics) + [fm_metric, pm_metric]
            x_labels = [str(num) for num in layer_numbers] + ["FM", "PM"]

            # Collect all values for Y limits
            all_metric_values.extend(all_metrics)

            # Create bar plot
            bars = ax.bar(x_positions, all_metrics, alpha=0.7, edgecolor="black")

            # Color coding: layers in blue, FM in green, PM in red
            for i, bar in enumerate(bars):
                if i < len(layer_numbers):
                    bar.set_color("skyblue")
                elif x_labels[i] == "FM":
                    bar.set_color("lightgreen")
                else:  # PM
                    bar.set_color("lightcoral")

            # Customize subplot
            ax.set_xlabel("Layer Number / Model Type", fontsize=12)
            if idx == 0:  # Only first subplot gets y-label
                ax.set_ylabel(metric_name, fontsize=12)

            ax.set_title(f"{method}", fontsize=14, fontweight="bold")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 10 else 0)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on top of bars
            for i, (pos, val) in enumerate(zip(x_positions, all_metrics)):
                # Format value: round to 0 if very small, otherwise 2 decimals
                if abs(val) < 0.01:
                    display_val = "0"
                else:
                    display_val = f"{val:.2f}"

                ax.text(
                    pos,
                    val + 0.01,
                    display_val,
                    ha="center",
                    va="bottom",
                    fontsize=9,
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
            ax.set_title(f"{method}", fontsize=14, fontweight="bold")

    # Set Y limits based on collected data
    if all_metric_values:
        y_max = max(all_metric_values)
        y_max_adjusted = y_max * 1.1  # Add 10% padding
        for ax in axes:
            ax.set_ylim(0, y_max_adjusted)

    # Extract max_new_tokens from directory name
    max_tokens = _extract_max_tokens_from_directory(directory_name)

    # Add overall title with token information
    plt.suptitle(
        f"Transplanting layers caption benchmarking: {metric_name} (max_new_tokens={max_tokens})",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="skyblue", label="Layer Transplantation"),
        Patch(facecolor="lightgreen", label="Finetuned Model (FM)"),
        Patch(facecolor="lightcoral", label="Pretrained Model (PM)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        fontsize=11,
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
):
    """
    Plot caption benchmarking results for a specific metric (default directory).

    Args:
        results_loader: ResultsLoader instance with loaded data
        metric_name: Metric to plot ('CIDEr' or 'SPICE')
        save_plots: Whether to save plots instead of showing them
    """
    # Use default directory for backward compatibility
    default_directory = "transplanting_layers_caption_benchmarking"
    plot_caption_benchmarking_for_directory(
        results_loader, metric_name, default_directory, save_plots
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


def plot_all_transplanting_layers_caption_benchmarking(
    results_loader: ResultsLoader, save_plots: bool = False
):
    """
    Plot all available caption benchmarking experiments for all metrics.
    """
    print("\n📊 Plotting all transplanting layers caption benchmarking experiments...")

    # Print summary of available data
    _print_caption_benchmarking_summary(results_loader)

    # Available metrics - plot each metric separately
    metrics = ["CIDEr", "CLIP-S", "RefCLIP-S"]

    # Available directories from config
    caption_directories = results_loader.config.get("caption_benchmarking", {}).get(
        "directories", []
    )

    for metric in metrics:
        for directory in caption_directories:
            plot_caption_benchmarking_for_directory(
                results_loader, metric, directory, save_plots=save_plots
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
        stream_types: List of stream types to plot (e.g., ['output_layer', 'post_mlp'])
        similarity_type: Type of similarity to plot ("cosine_similarity", "homogeneity_score",
                       "homogeneity_score_euclidean", or "homogeneity_score_cosine")
        save_plots: Whether to save plots instead of showing them
    """
    print(f"\n📊 Plotting modalities similarities: {similarity_type}")

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
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    # Define colors for models
    model_colors = {"finetuned": "blue", "pretrained": "red"}

    # Define linestyles and markers for stream types (using dashed lines and markers for discrete data)
    stream_styles = {"output_layer": "-", "post_mlp": "--"}
    stream_markers = {"output_layer": "o", "post_mlp": "s"}

    # Collect all data values for Y limits
    all_data_values = []

    # Plot data for each dataset (subplot)
    for d_idx, dataset in enumerate(all_datasets):
        ax = axes[d_idx]

        # Plot each model and stream type combination
        for model in all_models:
            for stream_type in stream_types:
                try:
                    similarity_data = results_loader.get_modalities_similarity(
                        model, dataset, stream_type, similarity_type
                    )

                    if isinstance(similarity_data, torch.Tensor):
                        similarity_data = safe_tensor_to_numpy(similarity_data)

                    # Collect data for Y limits
                    if similarity_data.ndim == 1:
                        all_data_values.extend(similarity_data)
                    elif similarity_data.ndim == 2:
                        all_data_values.extend(similarity_data[:, 0])

                    # Create line plot with dashed lines and markers for discrete data
                    layers = range(len(similarity_data))
                    color = model_colors.get(model, "gray")
                    linestyle = stream_styles.get(stream_type, "--")
                    marker = stream_markers.get(stream_type, "o")

                    ax.plot(
                        layers,
                        similarity_data if similarity_data.ndim == 1 else similarity_data[:, 0],
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markersize=5,
                        linewidth=2,
                        label=f"{model}_{stream_type}",
                    )

                    if similarity_data.ndim == 2:
                        ax.fill_between(
                            layers,
                            similarity_data[:, 0] - similarity_data[:, 1],
                            similarity_data[:, 0] + similarity_data[:, 1],
                            alpha=0.25,
                        )

                except KeyError:
                    # Data not available for this combination
                    continue

        # Configure subplot
        ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=12)
        if d_idx == 0:  # Only first subplot gets y-label
            if similarity_type == "cosine_similarity":
                y_label = "Cosine Similarity"
            elif similarity_type.startswith("homogeneity_score_"):
                metric = similarity_type.split("_")[-1]
                y_label = f"Homogeneity Score ({metric})"
            elif similarity_type == "homogeneity_score":
                y_label = "Homogeneity Score"
            else:
                y_label = "Similarity"
            ax.set_ylabel(y_label, fontsize=12)
        else:
            # Remove Y-axis elements for non-first subplots
            ax.tick_params(left=False)
            ax.set_ylabel("")

        ax.grid(True, alpha=0.3)

        # Set Y limits based on collected data
    if all_data_values:
        y_min = min(all_data_values)
        y_max = max(all_data_values)

        # Handle cases where data range is very small or zero
        y_range = y_max - y_min
        if y_range < 1e-10:  # Very small or zero range
            # Set a small range around the single value
            y_min_padded = y_min - 0.1
            y_max_padded = y_max + 0.1
        else:
            # Add some padding for normal cases
            y_min_padded = y_min - 0.05 * y_range
            y_max_padded = y_max + 0.05 * y_range

        for ax in axes:
            ax.set_ylim(y_min_padded, y_max_padded)

    # Add legends only to the first subplot
    if len(all_datasets) > 0:
        first_ax = axes[0]

        # Create custom legend for models (colors)
        from matplotlib.lines import Line2D

        model_legend_elements = []
        for model, color in model_colors.items():
            model_legend_elements.append(
                Line2D([0], [0], color=color, linewidth=3, label=model)
            )

        # Create custom legend for stream types (linestyles and markers)
        stream_legend_elements = []
        for stream_type, linestyle in stream_styles.items():
            marker = stream_markers.get(stream_type, "o")
            stream_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=linestyle,
                    marker=marker,
                    markersize=8,
                    linewidth=2.5,
                    label=stream_type,
                )
            )

        similarity_type_kwargs = {"models": {}, "stream_types": {}}
        if similarity_type == "cosine_similarity":
            similarity_type_kwargs["models"]["loc"] = "upper left"
            similarity_type_kwargs["models"]["bbox_to_anchor"] = (0.0, 1)
            similarity_type_kwargs["stream_types"]["loc"] = "upper left"
            similarity_type_kwargs["stream_types"]["bbox_to_anchor"] = (0.0, 0.95)
        elif similarity_type.startswith("homogeneity_score_"):
            similarity_type_kwargs["models"]["loc"] = "lower left"
            similarity_type_kwargs["models"]["bbox_to_anchor"] = (0.0, 0.05)
            similarity_type_kwargs["stream_types"]["loc"] = "lower left"
            similarity_type_kwargs["stream_types"]["bbox_to_anchor"] = (0.0, 0.0)
        elif similarity_type == "homogeneity_score":
            similarity_type_kwargs["models"]["loc"] = "lower left"
            similarity_type_kwargs["models"]["bbox_to_anchor"] = (0.0, 0.05)
            similarity_type_kwargs["stream_types"]["loc"] = "lower left"
            similarity_type_kwargs["stream_types"]["bbox_to_anchor"] = (0.0, 0.0)
        else:
            raise ValueError(f"Invalid similarity type: {similarity_type}")

        # Add model legend first
        model_legend = first_ax.legend(
            handles=model_legend_elements,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=len(model_colors),
            **similarity_type_kwargs["models"],
        )

        # Add model legend as artist to preserve it when adding second legend
        first_ax.add_artist(model_legend)

        # Add stream types legend second
        first_ax.legend(
            handles=stream_legend_elements,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            handlelength=5.0,
            handletextpad=1.0,
            ncol=len(stream_styles),
            **similarity_type_kwargs["stream_types"],
        )

    # Set overall title based on similarity type
    if similarity_type == "cosine_similarity":
        title = "Cosine similarity of trailing text and image embeddings"
    elif similarity_type.startswith("homogeneity_score_"):
        metric = similarity_type.split("_")[-1]
        title = f"Homogeneity score ({metric}) of trailing text and image embeddings"
    elif similarity_type == "homogeneity_score":
        title = "Homogeneity score of trailing text and image embeddings"
    else:
        title = f"Similarity ({similarity_type}) of trailing text and image embeddings"
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    filename = f"modalities_similarities_{similarity_type}_{'_'.join(stream_types)}.png"
    save_plot_if_requested(save_plots, filename)


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

    # plot_all_output_layer_similarity_measures(results, save_plots=save_plots)
    # plot_all_post_mlp_similarity_measures(results, save_plots=save_plots)
    # plot_all_mean_heads_projection_similarity_measures(results, save_plots=save_plots)
    # plot_all_similarity_measures_matrices(results, save_plots=save_plots)
    # plot_all_model_data_measures(results, save_plots=save_plots)
    # plot_modalities_similarities(
    #     results,
    #     ["output_layer", "post_mlp"],
    #     "cosine_similarity",
    #     save_plots=save_plots,
    # )
    # plot_modalities_similarities(
    #     results,
    #     ["output_layer", "post_mlp"],
    #     "homogeneity_score_euclidean",
    #     save_plots=save_plots,
    # )
    # plot_modalities_similarities(
    #     results,
    #     ["output_layer", "post_mlp"],
    #     "homogeneity_score_cosine",
    #     save_plots=save_plots,
    # )
    # plot_all_transplanting_layers_benchmarking(results, save_plots=save_plots)
    # plot_all_transplanting_layers_caption_benchmarking(results, save_plots=save_plots)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multimodal analysis plots")
    parser.add_argument(
        "--save", action="store_true", help="Save plots instead of showing them"
    )
    args = parser.parse_args()

    main(save_plots=args.save)
