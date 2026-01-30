import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Importing make_axes_locatable to strictly control colorbar sizing relative to the heatmap
import numpy as np
from safetensors.torch import load_file


MODELS = ["llava_7b", "llava_13b", "llava-v1.5-7b-pretrained-lbasile", "llava-v1.5-13b-pretrained-lbasile", "InternVL2-8B", "InternVL2-8B-Pretrain",]
METRICS = ["linear_cka", "neighborhood_overlap"]
DATASET_NAMES = [
    "l11p/COCOQA-restval-ICQA",
    "l11p/VQAv2-COCO-restval-ICQ",
]
RESULTS_DIR = "/orfeo/cephfs/home/dssc/lpalaciosflores/multimodal_finetuned_representations/results"

# Helper to convert dataset name to directory name
def get_dataset_dir_name(dataset_name):
    # Removes 'l11p/' prefix, converts to lowercase, replaces '-' with '_' and appends '_img_vs_txt'
    name = dataset_name.replace("l11p/", "").lower().replace("-", "_")
    return f"{name}_img_vs_txt"

def load_data():
    """
    Loads data into a dictionary with the following hierarchy:
    results_data = {
        "model_name": {
            "metric_name": {
                "dataset_name": numpy_array
            }
        }
    }
    """
    data = {}
    
    for model in MODELS:
        model_data = {}
        for metric in METRICS:
            metric_data = {}
            for dataset_name in DATASET_NAMES:
                dir_name = get_dataset_dir_name(dataset_name)
                similarities_path = os.path.join(RESULTS_DIR, dir_name, "similarities")
                
                # Filename pattern: {model}_last_{metric}.safetensors
                filename = f"{model}_last_{metric}.safetensors"
                file_path = os.path.join(similarities_path, filename)
                
                if os.path.exists(file_path):
                    try:
                        loaded = load_file(file_path)
                        if metric in loaded:
                            val = loaded[metric].cpu().numpy()
                            # Apply transformation for information imbalance if needed (referencing plot_internvl_heatmap.py)
                            if "information_imbalance" in metric:
                                val = 1 - val
                            metric_data[dataset_name] = val
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            
            if metric_data:
                # Calculate element-wise mean across datasets
                try:
                    arrays = list(metric_data.values())
                    if len(arrays) > 0:
                        # Ensure all arrays have the same shape before stacking
                        first_shape = arrays[0].shape
                        if all(arr.shape == first_shape for arr in arrays):
                            mean_val = np.mean(np.stack(arrays), axis=0)
                            metric_data["average"] = mean_val
                        else:
                            print(f"Warning: Shapes do not match for {model} - {metric}, skipping average.")
                except Exception as e:
                    print(f"Error calculating average for {model} - {metric}: {e}")

                model_data[metric] = metric_data
        
        if model_data:
            data[model] = model_data
            
    return data

results_data = load_data()

def plot_heatmap_comparison(data_dict, model1, model2, metric, dataset_key, base_output_dir="plots/heatmap_comparisons", 
                            model1_title=None, model2_title=None,
                            fontsize_title=26, fontsize_label=18, fontsize_ticks=14, fontsize_cbar=14,
                            trim_whitespace=False):
    """
    Generates a publication-quality side-by-side heatmap comparison.
    
    Args:
        data_dict (dict): The loaded data structure.
        model1 (str): Name of the first model.
        model2 (str): Name of the second model.
        metric (str): Metric to plot.
        dataset_key (str): Dataset key (e.g., 'average' or 'l11p/COCOQA...').
        base_output_dir (str): Base directory to save the plot.
        model1_title (str, optional): Custom title for the first model subplot. Defaults to model1 name.
        model2_title (str, optional): Custom title for the second model subplot. Defaults to model2 name.
        fontsize_title (int, optional): Font size for subplot titles. Defaults to 26.
        fontsize_label (int, optional): Font size for axis labels. Defaults to 18.
        fontsize_ticks (int, optional): Font size for axis ticks. Defaults to 14.
        fontsize_cbar (int, optional): Font size for colorbar ticks. Defaults to 14.
        trim_whitespace (bool, optional): If True, uses bbox_inches='tight' when saving. Defaults to False.
    """
    # 1. Retrieve Data
    try:
        # Transpose matrices to swap axes: 
        # Original: Rows=Text, Cols=Image -> New: Rows=Image, Cols=Text
        matrix1 = data_dict[model1][metric][dataset_key].T
        matrix2 = data_dict[model2][metric][dataset_key].T
    except KeyError as e:
        print(f"Error: Missing data for {e}. Check models, metric, or dataset availability.")
        return

    # 2. Setup Plotting Style (Scientific Publication Quality)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # 3. Determine Global Limits for Shared Colorbar
    vmin = min(matrix1.min(), matrix2.min())
    vmax = max(matrix1.max(), matrix2.max())
    
    # 4. Create Figure - TOTAL MANUAL CONTROL
    # We disable constrained_layout/tight_layout to manually place everything.
    fig = plt.figure(figsize=(16, 8))
    
    # 5. Define Grid for Heatmaps
    # --------------------------------------------------------------------------------
    # CONTROL: HEATMAP SEPARATION & MARGINS
    # 'wspace': Gap between heatmaps.
    # 'left', 'right', 'top', 'bottom': Global figure margins (0-1).
    # Reducing 'left' removes the "horrible white space" on the side.
    # --------------------------------------------------------------------------------
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1,
                           left=0.05, right=0.92, top=0.85, bottom=0.10)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # 6. Plot Heatmaps
    im1 = ax1.imshow(matrix1, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
    im2 = ax2.imshow(matrix2, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
    
    # 7. Styling Axes
    # Determine titles to display
    t1 = model1_title if model1_title is not None else model1
    t2 = model2_title if model2_title is not None else model2
    
    for ax, title in zip([ax1, ax2], [t1, t2]):
        # ----------------------------------------------------------------------------
        # CONTROL: SUBTITLE POSITION
        # 'pad' controls the vertical distance (in points) between the subplot title
        # and the top of the heatmap.
        # - Larger pad -> Title moves up.
        # - Smaller pad -> Title moves down.
        # ----------------------------------------------------------------------------
        ax.set_title(title,
                     pad=16,
                     color='#222222', 
                     fontsize=fontsize_title,
                    #  fontweight='bold',
        )
        ax.set_xlabel("text input", labelpad=12, color='#333333', fontsize=fontsize_label)
        
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, color='#444444', labelsize=fontsize_ticks)
        
        num_layers = matrix1.shape[0] 
        ticks = range(0, num_layers, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        ax.grid(False)

    ax1.set_ylabel("visual input", labelpad=12, color='#333333', fontsize=fontsize_label)
    ax2.set_ylabel("visual input", labelpad=12, color='#333333', fontsize=fontsize_label)
    
    # Remove y-axis labels and ticks for the second plot
    # ax2.set_yticklabels([])
    # ax2.set_ylabel("")
    # ax2.tick_params(axis='y', left=False) 

    # 8. Colorbar (Manual Positioning)
    # This allows independent control of the colorbar's distance from the second plot.
    # We must force a draw first so that get_position() returns the final calculated bbox.
    fig.canvas.draw() 
    
    pos = ax2.get_position()
    
    # --------------------------------------------------------------------------------
    # CONTROL: COLORBAR POSITION
    # 'cbar_gap' controls the horizontal distance (in figure coordinates, 0-1)
    # between the right edge of the second heatmap and the left edge of the colorbar.
    # - Larger cbar_gap (e.g., 0.05) -> Colorbar moves further right.
    # - Smaller cbar_gap (e.g., 0.01) -> Colorbar moves closer to the plot.
    # --------------------------------------------------------------------------------
    cbar_gap = 0.02 
    cbar_width = 0.02 # Width of the colorbar
    
    # Create the colorbar axis [left, bottom, width, height]
    cax = fig.add_axes([pos.x1 + cbar_gap, pos.y0, cbar_width, pos.height])
    
    cbar = fig.colorbar(im1, cax=cax)
    cbar.outline.set_visible(False) 
    cbar.ax.tick_params(size=0, labelsize=fontsize_cbar)

    # fig.tight_layout()

    # 9. Main Title
    display_dataset = "Average" if dataset_key == "average" else dataset_key.split("/")[-1]
    metric_display = metric.replace("_", " ").title()
    
    # --------------------------------------------------------------------------------
    # CONTROL: MAIN TITLE POSITION
    # 'y' controls the vertical position of the main figure title in figure coordinates (0-1).
    # - y > 1.0 (e.g., 1.05) -> Title moves higher, potentially outside the saved area unless constrained.
    # - y < 1.0 (e.g., 0.95) -> Title moves lower, closer to the subplots.
    # --------------------------------------------------------------------------------
    # fig.suptitle(f"{metric_display} Comparison: {display_dataset}", fontweight='bold', y=0.98)

    # 10. Save
    clean_metric = metric.replace("_", "-")
    output_dir = os.path.join(base_output_dir, clean_metric)
    os.makedirs(output_dir, exist_ok=True)
    
    clean_dataset = "average" if dataset_key == "average" else get_dataset_dir_name(dataset_key)
    filename = f"multimodality_comparison_{model1}_vs_{model2}_{clean_dataset}_{clean_metric}.png"
    save_path = os.path.join(output_dir, filename)
    
    # We avoid bbox_inches='tight' because it can shift our manually placed axes.
    # Instead, we ensure the figure size (16,8) is sufficient for the content.
    save_kwargs = {}
    if trim_whitespace:
        save_kwargs['bbox_inches'] = 'tight'
        save_kwargs['pad_inches'] = 0.1
        
    plt.savefig(save_path, dpi=300, facecolor='white', **save_kwargs)
    print(f"✅ Generated Scientific Plot: {save_path}")
    plt.close(fig)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Example Configuration
    # SELECTED_METRIC = "neighborhood_overlap"
    SELECTED_METRIC = "linear_cka"
    SELECTED_DATASET = "average" 
    # MODEL_A = "InternVL2-8B-Pretrain"
    # MODEL_B = "InternVL2-8B"
    MODEL_A = "llava-v1.5-13b-pretrained-lbasile"
    MODEL_B = "llava_13b"
    
    # Optional Custom Titles (set to None to use model keys)
    MODEL_A_TITLE = "pretrained"
    MODEL_B_TITLE = "finetuned"
    # MODEL_A_TITLE = "pretrained (InternVL2 8B)"
    # MODEL_B_TITLE = "finetuned (InternVL2 8B)"
    
    print(f"Running comparison for {SELECTED_METRIC} on {SELECTED_DATASET}...")
    plot_heatmap_comparison(
        results_data, 
        MODEL_A, 
        MODEL_B, 
        SELECTED_METRIC, 
        SELECTED_DATASET,
        model1_title=MODEL_A_TITLE,
        model2_title=MODEL_B_TITLE,
        fontsize_title=22,
        fontsize_label=20,
        fontsize_ticks=18,
        fontsize_cbar=18,
        trim_whitespace=True
    )