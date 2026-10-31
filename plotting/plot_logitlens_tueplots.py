import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from safetensors.numpy import load_file

from all_benchmarks_plots_tueplots import get_rc

PROJECT_ROOT = os.path.expanduser("~/scratch/plm")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "logitlens")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = [
    ("l11p/COCOQA-restval-ICQA", ["images_qa"]),
    ("l11p/VQAv2-COCO-restval-ICQ", ["images_qa"]),
    ("lmms-lab/ScienceQA", ["images_qa"]),
    ("l11p/MME-test-captioned", ["images_qa"]),
    ("l11p/MMBench_EN-dev-captioned", ["images_qa"]),
    ("l11p/GQA-val-balanced-noyesno-captioned", ["images_qa"]),
    ("lmms-lab/POPE", ["images_qa"]),
    ("Lin-Chen/MMStar", ["images_qa"]),
    ("vikhyatk/CountBenchQA", ["images_qa"]),
]

MODELS = [
    ("nyu-visionx/cambrian-8b", "Finetuned"),
    ("lbasile/llava_7b", "Finetuned"),
    ("lbasile/llava_13b", "Finetuned"),
    ("lmms-lab/LLaVA-OneVision-1.5-8B-Instruct", "Finetuned"),
    ("lmms-lab/LLaVA-OneVision-1.5-4B-Instruct", "Finetuned"),
    ("OpenGVLab/InternVL2-8B", "Finetuned"),
]

BASELINES = {
    "scienceqa": 0.2,
    "mmbench_en_dev_captioned": 0.25,
    "mme_test_captioned": 0.5,
    "cocoqa_restval_icqa": 0.0,
    "vqav2_coco_restval_icq": 0.0,
    "gqa_val_balanced_noyesno_captioned": 0.0,
    "ai2d": 0.25,
    "pope": 0.5,
    "textvqa": 0.0,
    "mmstar": 0.25,
    "countbenchqa": 0.0,
}

GROUP_MC = ["l11p/MME-test-captioned", "l11p/MMBench_EN-dev-captioned", "lmms-lab/POPE", "Lin-Chen/MMStar"]
GROUP_QA = ["l11p/GQA-val-balanced-noyesno-captioned", "l11p/COCOQA-restval-ICQA", "l11p/VQAv2-COCO-restval-ICQ"]

# Helper to load and normalize
def load_accuracies(results_dir, model_slug, dataset_slug, subset, mode="logit_lens"):
    filename = f"{model_slug}_{dataset_slug}_{subset}_{mode}.safetensors"
    path = os.path.join(results_dir, "logitlens_evaluation", filename)
    if os.path.exists(path):
        return load_file(path)["accuracies"]
    if mode == "logit_lens":
        filename_legacy = f"{model_slug}_{dataset_slug}_{subset}.safetensors"
        path_legacy = os.path.join(results_dir, "logitlens_evaluation", filename_legacy)
        if os.path.exists(path_legacy):
            return load_file(path_legacy)["accuracies"]
    return None

def normalize_profile(arr, baseline):
    if arr is None: return None
    max_val = np.max(arr)
    if max_val <= baseline:
        denom = 1.0
    else:
        denom = max_val - baseline
    return (arr - baseline) / denom

def average_arrays(arrays):
    if not arrays:
        return None, None
    min_len = min(len(arr) for arr in arrays)
    clipped = [arr[:min_len] for arr in arrays]
    return np.mean(clipped, axis=0), np.std(clipped, axis=0)

def extract_profiles_for_model(model_slug):
    mc_ll_profiles = []
    mc_lp_profiles = []
    qa_ll_profiles = []

    for dataset_name, subsets in DATASETS:
        dataset_slug = dataset_name.split('/')[-1].replace("-", "_").lower()
        baseline = BASELINES.get(dataset_slug, 0.0)
        
        for subset in subsets:
            raw_ll = load_accuracies(RESULTS_DIR, model_slug, dataset_slug, subset, mode="logit_lens")
            raw_lp = load_accuracies(RESULTS_DIR, model_slug, dataset_slug, subset, mode="linear_probe")
            
            norm_ll = normalize_profile(raw_ll, baseline)
            norm_lp = normalize_profile(raw_lp, baseline)

            if dataset_name in GROUP_MC:
                if norm_ll is not None: mc_ll_profiles.append(norm_ll)
                if norm_lp is not None: mc_lp_profiles.append(norm_lp)
            
            if dataset_name in GROUP_QA:
                if norm_ll is not None: qa_ll_profiles.append(norm_ll)
                
    mean_mc_ll, std_mc_ll = average_arrays(mc_ll_profiles)
    mean_mc_lp, std_mc_lp = average_arrays(mc_lp_profiles)
    mean_qa_ll, std_qa_ll = average_arrays(qa_ll_profiles)
    
    return {
        "mc_ll": (mean_mc_ll, std_mc_ll),
        "mc_lp": (mean_mc_lp, std_mc_lp),
        "qa_ll": (mean_qa_ll, std_qa_ll)
    }

def plot_aggregated_panel(ax, model_slug, model_label):
    data = extract_profiles_for_model(model_slug)
    mean_mc_ll, std_mc_ll = data["mc_ll"]
    mean_mc_lp, std_mc_lp = data["mc_lp"]
    mean_qa_ll, std_qa_ll = data["qa_ll"]
    
    # Determine abstraction phase correctly based on model
    if model_slug == "llava_7b":
        ax.axvspan(5.5, 16.5, alpha=0.12, color="#DAA520", zorder=0, label="Abstraction phase")
    elif "LLaVA-OneVision-1.5-4B-Instruct" in model_slug:
        ax.axvspan(7.5, 26.5, alpha=0.12, color="#DAA520", zorder=0, label="Abstraction phase")
    
    # Remove generation phase as requested
    
    # Plot Multiple Choice (LL)
    if mean_mc_ll is not None:
        ax.plot(range(len(mean_mc_ll))[1::2], mean_mc_ll[1::2], 
            label="Multiple Choice (LL)", color="#669bbc",
            marker="s", markersize=3, linewidth=0.8)
        if std_mc_ll is not None:
            ax.fill_between(range(len(mean_mc_ll))[1::2], 
                            mean_mc_ll[1::2] - std_mc_ll[1::2]/2, 
                            mean_mc_ll[1::2] + std_mc_ll[1::2]/2, 
                            color="#669bbc", alpha=0.15)
            
    # Plot Multiple Choice (LP)
    if mean_mc_lp is not None:
        ax.plot(range(len(mean_mc_lp))[1::2], mean_mc_lp[1::2],
            label="Multiple Choice (LP)", color="#669bbc", linestyle="--",
            marker="^", markersize=3, linewidth=0.8)
        if std_mc_lp is not None:
            ax.fill_between(range(len(mean_mc_lp))[1::2], 
                            mean_mc_lp[1::2] - std_mc_lp[1::2]/2, 
                            mean_mc_lp[1::2] + std_mc_lp[1::2]/2, 
                            color="#669bbc", alpha=0.1)
            
    # Plot QA Open-ended (LL)
    if mean_qa_ll is not None:
        ax.plot(range(len(mean_qa_ll))[1::2], mean_qa_ll[1::2],
            label="Open-Ended (LL)", color="#e76f51",
            marker="o", markersize=3, linewidth=0.8)
        if std_qa_ll is not None:
            ax.fill_between(range(len(mean_qa_ll))[1::2], 
                            mean_qa_ll[1::2] - std_qa_ll[1::2]/2, 
                            mean_qa_ll[1::2] + std_qa_ll[1::2]/2, 
                            color="#e76f51", alpha=0.15)

    max_len = 0
    if mean_mc_ll is not None: max_len = max(max_len, len(mean_mc_ll))
    if mean_qa_ll is not None: max_len = max(max_len, len(mean_qa_ll))
    
    if max_len > 0:
        xticks = np.arange(0, max_len + 1, 4)
        ax.set_xticks(xticks)
        
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Normalized Perf.", fontsize=10)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_title(model_label, fontsize=12)

def main():
    # 1) Save individual plots for all models
    rc = dict(get_rc(ncols=1))
    # Add seaborn style but keeping tueplots base
    sns.set_style("whitegrid", rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True})
    
    with plt.rc_context(rc):
        for model_path, model_label in MODELS:
            model_slug = model_path.split('/')[-1]
            fig, ax = plt.subplots(figsize=(4.5, 4))
            plot_aggregated_panel(ax, model_slug, model_label)
            ax.legend(fontsize=8, loc='best')
            
            out_name = f"aggregated_normalized_tueplots_{model_slug}.pdf"
            plt.savefig(os.path.join(OUTPUT_DIR, out_name), bbox_inches='tight')
            plt.close()
            print(f"Saved Individual Plot: {out_name}")

    # 2) Save unitary panel with LLaVA-7B and OneVision-4B
    rc = dict(get_rc(ncols=2))
    sns.set_style("whitegrid", rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
        
        plot_aggregated_panel(axes[0], "llava_7b", "LLaVA 1.5 7B")
        plot_aggregated_panel(axes[1], "LLaVA-OneVision-1.5-4B-Instruct", "LLaVA-OneVision 4B")
        
        axes[1].set_ylabel("")  # Share Y axis visual
        
        # Unified legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        # Remove duplicate labels if any
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=9)
        
        out_name = "aggregated_normalized_tueplots_joint_llava7b_ov4b.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, out_name), bbox_inches='tight')
        plt.close()
        print(f"Saved Joint Plot: {out_name}")

if __name__ == "__main__":
    main()
