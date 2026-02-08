import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from safetensors.torch import load_file

# Add parent directory to path to import utils if needed
sys.path.append(str(Path.cwd().parent))

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_NAMES = [
    "l11p/COCOQA-restval-ICQA",
    "l11p/VQAv2-COCO-restval-ICQ",
    # "l11p/MME-test-captioned",
    # "l11p/MMBench_EN-dev-captioned",
    # "l11p/GQA-val-balanced-noyesno-captioned",
]
MODELS = {
    "Pretrained": "l11p/llava-v1.5-13b-pretrained-lbasile",
    # "Finetuned": "lbasile/llava_13b"
    "Finetuned": "OpenGVLab/InternVL2-8B"
}
FINETUNED_MODEL_ID = MODELS["Finetuned"].split('/')[-1].replace('/', '_')
MODEL_IDS = {k: v.split('/')[-1] for k, v in MODELS.items()}
POOLING_METHODS = ["last"]
MEASURES = {
    # "neighborhood_overlap": "Neighborhood Overlap",
    # "linear_cka": "Linear CKA",
    # "information_imbalance": "II Complement (Text->Image)",
    # "information_imbalance_reversed": "II Complement (Image->Text)",
    "information_imbalance_mean": "II Complement (Mean)"
}
ID_RESULTS_DIR = Path("../results/model_comparison/intrinsic_dimension")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_data(model_key, measure_key, pooling_method, results_dir):
    model_id = MODEL_IDS[model_key]
    filename = f"{model_id}_{pooling_method}_{measure_key}.safetensors"
    path = results_dir / filename
    try:
        data = load_file(str(path))
        tensor = data[measure_key]
        val = tensor.cpu().numpy()
        if "information_imbalance" in measure_key:
            val = 1 - val
        return val
    except Exception as e:
        print(f"⚠️  Warning: Could not load {filename}: {e}")
        return None

def load_id_data(dataset_suffix, model_key, pooling_method, dataset_prefix, id_results_dir):
    model_id = MODEL_IDS[model_key]
    dataset_name = f"{dataset_prefix}_{dataset_suffix}"
    filename = f"{model_id}_{dataset_name}_{pooling_method}_output_layer_intrinsic_dimension.safetensors"
    path = id_results_dir / filename
    try:
        data = load_file(str(path))
        tensor = data["intrinsic_dimension"]
        return tensor.cpu().numpy()
    except Exception as e:
        print(f"⚠️  Warning: Could not load ID data from {filename}: {e}")
        return None

def plot_measure_comparison(measure_key, measure_title, pooling_method, dataset_prefix, results_dir, id_results_dir, plots_dir):
    sim_pm = load_data("Pretrained", measure_key, pooling_method, results_dir)
    sim_fm = load_data("Finetuned", measure_key, pooling_method, results_dir)
    
    if sim_pm is None or sim_fm is None:
        print(f"⚠️  Skipping plot for {measure_title} ({pooling_method}): Missing similarity data.")
        return

    id_img_pm = load_id_data("img", "Pretrained", pooling_method, dataset_prefix, id_results_dir)
    id_img_fm = load_id_data("img", "Finetuned", pooling_method, dataset_prefix, id_results_dir)
    id_txt_pm = load_id_data("txt", "Pretrained", pooling_method, dataset_prefix, id_results_dir)
    id_txt_fm = load_id_data("txt", "Finetuned", pooling_method, dataset_prefix, id_results_dir)

    has_id_data = all(x is not None for x in [id_img_pm, id_txt_pm, id_img_fm, id_txt_fm])
    if not has_id_data:
        print(f"⚠️  Plotting {measure_title} ({pooling_method}) without intrinsic dimension data.")
    
    sim_diff = sim_pm - sim_fm
    id_img_diff = id_img_pm - id_img_fm if has_id_data else None
    id_txt_diff = id_txt_pm - id_txt_fm if has_id_data else None
    max_diff = max(abs(np.nanmin(sim_diff)), abs(np.nanmax(sim_diff))) if np.any(sim_diff) else 0

    fig = plt.figure(figsize=(26, 12 if has_id_data else 9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.08], width_ratios=[1, 1, 1], hspace=0.25, wspace=0.2)
    
    models_config = [
        ("Pretrained", sim_pm, id_img_pm, id_txt_pm, "viridis", 0, 1),
        ("Finetuned", sim_fm, id_img_fm, id_txt_fm, "viridis", 0, 1),
        ("Difference (PM - FM)", sim_diff, id_img_diff, id_txt_diff, "coolwarm", -max_diff, max_diff)
    ]
    
    main_heatmaps = []
    for idx, (model_label, sim_data, id_img, id_txt, cmap, vmin, vmax) in enumerate(models_config):
        if has_id_data:
            gs_inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, idx], width_ratios=[6, 1], height_ratios=[1, 6], wspace=0.03, hspace=0.03)
            ax_top, ax_main, ax_right = fig.add_subplot(gs_inner[0, 0]), fig.add_subplot(gs_inner[1, 0]), fig.add_subplot(gs_inner[1, 1])
            ax_top.sharex(ax_main)
            ax_right.sharey(ax_main)
            # disable ticks for the x axes on top and y axes on right
            ax_top.tick_params(bottom=False, labelbottom=False)
            ax_right.tick_params(left=False, labelleft=False)
        else:
            ax_main = fig.add_subplot(gs[0, idx])
        
        hm = sns.heatmap(sim_data, ax=ax_main, cmap=cmap, vmin=vmin, vmax=vmax, center=0 if idx == 2 else None, square=False, cbar=False)
        if idx == 0 or idx == 2:
            main_heatmaps.append(hm.collections[0])
        
        if has_id_data:
            x_centers, y_centers = np.arange(len(id_img)) + 0.5, np.arange(len(id_txt)) + 0.5
            ax_top.plot(x_centers, id_img, color='#377eb8', lw=2)
            ax_top.fill_between(x_centers, id_img, 0, color='#377eb8', alpha=0.3)
            sns.despine(ax=ax_top, bottom=True, left=False)
            ax_top.tick_params(bottom=False, labelbottom=False, axis='y', labelsize=8)
            ax_top.set_ylabel("ID", fontsize=10)
            ax_right.plot(id_txt, y_centers, color='#e41a1c', lw=2)
            ax_right.fill_betweenx(y_centers, id_txt, 0, color='#e41a1c', alpha=0.3)
            sns.despine(ax=ax_right, left=True, bottom=False)
            ax_right.tick_params(left=False, labelleft=False, axis='x', labelsize=8)
            ax_right.set_xlabel("ID", fontsize=10)
            ax_top.set_title(f"{model_label}", fontsize=18, pad=10, fontweight='medium')
        else:
            ax_main.set_title(f"{model_label}", fontsize=18, pad=10, fontweight='medium')
        
        ax_main.set_xlabel("Image Layers", fontsize=13, labelpad=8)
        ax_main.set_ylabel("Text Layers", fontsize=13, labelpad=8)
        ax_main.invert_yaxis()
        ticks = range(0, sim_data.shape[0], 5)
        ax_main.set_xticks([t + 0.5 for t in ticks]); ax_main.set_xticklabels(ticks, fontsize=11)
        ax_main.set_yticks([t + 0.5 for t in ticks]); ax_main.set_yticklabels(ticks, fontsize=11)

    cax_shared = fig.add_subplot(gs[1, 0:2]); plt.colorbar(main_heatmaps[0], cax=cax_shared, orientation='horizontal', label=f"{measure_title} (0 to 1)")
    cax_diff = fig.add_subplot(gs[1, 2]); plt.colorbar(main_heatmaps[1], cax=cax_diff, orientation='horizontal', label=f"Difference ({measure_title})")
    plt.suptitle(f"{measure_title} ({pooling_method}) on {dataset_prefix}: Text vs Image Similarity", fontsize=22, fontweight='bold', y=0.98)
    save_path = plots_dir / f"multimodality_{pooling_method}_{measure_key}_comparison_with_id_{FINETUNED_MODEL_ID}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"✅ Saved plot to {save_path}")
    plt.show()

def load_multiple_datasets_data(model_key, measure_key, pooling_method, dataset_names):
    all_data = []
    for dataset_name in dataset_names:
        dataset_prefix = dataset_name.split('/')[-1].replace('.', '_').replace('-', '_').lower()
        results_dir = Path(f"../results/{dataset_prefix}_img_vs_txt/similarities")
        model_id = MODEL_IDS[model_key]
        filename = f"{model_id}_{pooling_method}_{measure_key}.safetensors"
        path = results_dir / filename
        try:
            data = load_file(str(path))
            tensor = data[measure_key]
            val = tensor.cpu().numpy()
            if "information_imbalance" in measure_key:
                val = 1 - val
            all_data.append(val)
        except Exception as e:
            print(f"⚠️  Warning: Could not load {filename} for dataset {dataset_name}: {e}")
    if not all_data:
        return None
    return np.mean(all_data, axis=0)

def plot_averaged_measure_comparison(measure_key, measure_title, pooling_method, dataset_names, plots_dir):
    sim_pm = load_multiple_datasets_data("Pretrained", measure_key, pooling_method, dataset_names)
    sim_fm = load_multiple_datasets_data("Finetuned", measure_key, pooling_method, dataset_names)

    if sim_pm is None or sim_fm is None:
        print(f"⚠️  Skipping averaged plot for {measure_title} ({pooling_method}): Missing similarity data.")
        return

    sim_diff = sim_pm - sim_fm
    max_diff = max(abs(np.nanmin(sim_diff)), abs(np.nanmax(sim_diff))) if np.any(sim_diff) else 0
    
    fig = plt.figure(figsize=(26, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.08], width_ratios=[1, 1, 1], hspace=0.25, wspace=0.2)

    models_config = [
        ("Pretrained", sim_pm, "viridis", 0, 1),
        ("Finetuned", sim_fm, "viridis", 0, 1),
        ("Difference (PM - FM)", sim_diff, "coolwarm", -max_diff, max_diff)
    ]

    main_heatmaps = []
    for idx, (model_label, sim_data, cmap, vmin, vmax) in enumerate(models_config):
        ax_main = fig.add_subplot(gs[0, idx])
        hm = sns.heatmap(sim_data, ax=ax_main, cmap=cmap, vmin=vmin, vmax=vmax, center=0 if idx == 2 else None, square=False, cbar=False)
        if idx == 0 or idx == 2:
            main_heatmaps.append(hm.collections[0])

        ax_main.set_xlabel("Image Layers", fontsize=13, labelpad=8)
        if idx == 0:
            ax_main.set_ylabel("Text Layers", fontsize=13, labelpad=8)
        ax_main.invert_yaxis()
        
        num_layers = sim_data.shape[0]
        ticks = range(0, num_layers, 5)
        ax_main.set_xticks([t + 0.5 for t in ticks]); ax_main.set_xticklabels(ticks, fontsize=11)
        ax_main.set_yticks([t + 0.5 for t in ticks]); ax_main.set_yticklabels(ticks if idx == 0 else [], fontsize=11)
        ax_main.set_title(f"{model_label}", fontsize=18, pad=10, fontweight='medium')

    cax_shared = fig.add_subplot(gs[1, 0:2]); plt.colorbar(main_heatmaps[0], cax=cax_shared, orientation='horizontal', label=f"{measure_title} (0 to 1)")
    cax_diff = fig.add_subplot(gs[1, 2]); plt.colorbar(main_heatmaps[1], cax=cax_diff, orientation='horizontal', label=f"Difference ({measure_title})")
    plt.suptitle(f"Averaged {measure_title} ({pooling_method}): Text vs Image Modality Similarity", fontsize=22, fontweight='bold', y=0.98)
    save_path = plots_dir / f"multimodality_averaged_{pooling_method}_{measure_key}_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"✅ Saved averaged plot to {save_path}")
    plt.show()



def plot_averaged_diagonals(pooling_method, dataset_names, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Filter measures
    measures_to_plot = {}
    has_mean = "information_imbalance_mean" in MEASURES
    
    for key, title in MEASURES.items():
        if has_mean and key in ["information_imbalance", "information_imbalance_reversed"]:
            continue
        measures_to_plot[key] = title
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(measures_to_plot)))
    
    for (key, title), color in zip(measures_to_plot.items(), colors):
        sim_pm = load_multiple_datasets_data("Pretrained", key, pooling_method, dataset_names)
        sim_fm = load_multiple_datasets_data("Finetuned", key, pooling_method, dataset_names)
        
        if sim_pm is None or sim_fm is None:
            continue
            
        diag_pm = np.diag(sim_pm)
        diag_fm = np.diag(sim_fm)
        diag_diff = np.diag(sim_pm - sim_fm)
        
        x_axis = range(len(diag_pm))
        
        # Plot 1: Diagonals of Pretrained and Finetuned
        axes[0].plot(x_axis, diag_pm, linestyle="--", color=color, label=f"{title}")
        axes[0].plot(x_axis, diag_fm, linestyle="-", color=color)
        
        # Plot 2: Diagonal of Difference
        axes[1].plot(x_axis, diag_diff, linestyle="-", color=color, label=title)

    # Styling Plot 1
    axes[0].set_title(f"Diagonal of Averaged Similarity Measures ({pooling_method})")
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Similarity")
    axes[0].set_ylim(0,1)
    
    # Legend 1: Measures (Colors)
    legend1 = axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1), title="Measures")
    axes[0].add_artist(legend1)
    
    # Legend 2: Models (Line Styles)
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Pretrained"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="Finetuned")
    ]
    axes[0].legend(handles=custom_lines, loc="upper left", bbox_to_anchor=(0, 0.75), title="Models")
    
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Styling Plot 2
    axes[1].set_title(f"Diagonal of Difference (Pretrained - Finetuned) ({pooling_method})")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Difference")
    axes[1].legend(title="Measures")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    save_path = plots_dir / f"multimodality_averaged_{pooling_method}_diagonals_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved averaged diagonals plot to {save_path}")
    plt.show()

def main():
    # ============================================================================
    # EXECUTION: INDIVIDUAL PLOTS
    # ============================================================================
    for dataset_name in DATASET_NAMES:
        print(f"\n{'='*80}\n📊 Processing dataset: {dataset_name}\n{'='*80}")
        dataset_prefix = dataset_name.split('/')[-1].replace('.', '_').replace('-', '_').lower()
        results_dir = Path(f"../results/{dataset_prefix}_img_vs_txt/similarities")
        plots_dir = Path(f"../plots/multimodality_similarity_{dataset_prefix}")
        plots_dir.mkdir(parents=True, exist_ok=True)
        for pooling in POOLING_METHODS:
            print(f"\n  Generating plots for pooling method: {pooling}")
            for key, title in MEASURES.items():
                plot_measure_comparison(key, title, pooling, dataset_prefix, results_dir, ID_RESULTS_DIR, plots_dir)

    # ============================================================================
    # EXECUTION: AVERAGED PLOTS
    # ============================================================================
    averaged_plots_dir = Path(f"../plots/multimodality_similarity_averaged")
    averaged_plots_dir.mkdir(parents=True, exist_ok=True)
    for pooling in POOLING_METHODS:
        print(f"\n{'='*80}\n📊 Generating averaged plots for pooling method: {pooling}\n{'='*80}")
        for key, title in MEASURES.items():
            plot_averaged_measure_comparison(key, title, pooling, DATASET_NAMES, averaged_plots_dir)
        plot_averaged_diagonals(pooling, DATASET_NAMES, averaged_plots_dir)

if __name__ == '__main__':
    main()
