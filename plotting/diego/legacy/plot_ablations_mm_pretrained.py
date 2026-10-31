# LEGACY: superseded by ablations_mm_tueplots.py / ablations_text_tueplots.py.
# Hardcoded author-local paths below (/home/diego/...) — not runnable as-is,
# kept for provenance only.

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# *************************************************************************
alpha = 0.9
sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

base_path_vl = "/home/diego/Documents/area_science/ricerca/open/llava_finetuning/results/layer_skip_eval_mm"


def get_data_vl_window(base_path, window_size):

    tasks_metric = {
        "cv_bench": ("cv_bench_acc,none", "cv_bench"),
        "mme": ("mme_perception_score,none", "mme"),
        "gqa": ("exact_match,none", "gqa"),
        "mmbench_en_dev": ("gpt_eval_score,none", "mmbench_en_dev"),
        "seedbench": ("seed_image,none", "seedbench"),
    }

    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    perf = {}
    layers_per_task = {}

    for task, (metric, result_key) in tasks_metric.items():

        perf[task] = []
        layers_per_task[task] = []

        for layer in layers:
            task_folder = task
            file_name = f"{task}_L{layer}_W{window_size}.json"
            file_path = f"{base_path}/llava-v1.5-7b-pretrained-lbasile_sw{window_size}/{task_folder}/{file_name}"

            try:
                with open(file_path, "r") as f:
                    file = json.load(f)
                    perf[task].append(file["results"][result_key][metric])
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}")
                continue

    full_baselines = {}
    for task, (metric, result_key) in tasks_metric.items():
        task_folder = task
        file_name = f"{task}_L0_W0.json"
        file_path = f"{base_path}/llava-v1.5-7b-pretrained-lbasile_sw0/{task_folder}/{file_name}"

        try:
            with open(file_path, "r") as f:
                file = json.load(f)
                full_baselines[task] = file["results"][result_key][metric]
        except FileNotFoundError:
            print(f"Warning: Baseline file not found {file_path}")
            continue

    random_baselines = {
        "cv_bench": None,
        "mme": None,
        "gqa": None,
        "mmbench_en_dev": None,
        "seedbench": None,
    }

    # Load random baselines from sw32 where available
    for task, (metric, result_key) in tasks_metric.items():
        try:
            with open(
                f"{base_path}/llava-v1.5-7b-pretrained-lbasile_sw32/{task}/{task}_L0_W32.json",
                "r",
            ) as f:
                file = json.load(f)
                random_baselines[task] = file["results"][result_key][metric]
        except FileNotFoundError:
            print(f"Warning: Random baseline not found for {task}, using 0")
            random_baselines[task] = 0
            continue

    return perf, random_baselines, full_baselines, layers


def get_data_vl_progressive(base_path):

    tasks_metric = {
        "cv_bench": ("cv_bench_acc,none", "cv_bench"),
        "mme": ("mme_perception_score,none", "mme"),
        "gqa": ("exact_match,none", "gqa"),
        "seedbench": ("seed_image,none", "seedbench"),
    }

    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    perf = {}
    layers_per_task = {}

    for task, (metric, result_key) in tasks_metric.items():

        perf[task] = []
        layers_per_task[task] = []

        for layer in layers:
            task_folder = task
            file_name = f"{task}_L0_W{layer}.json"
            file_path = f"{base_path}/llava-v1.5-7b-pretrained-lbasile/{task_folder}/{file_name}"

            try:
                with open(file_path, "r") as f:
                    file = json.load(f)
                    perf[task].append(file["results"][result_key][metric])
                    layers_per_task[task].append(layer)
            except FileNotFoundError:
                print(f"Warning: Progressive file not found {file_path}")
                continue

    # Find common layers across all tasks with data
    common_layers = None
    for task in perf:
        if len(perf[task]) > 0:
            if common_layers is None:
                common_layers = set(layers_per_task[task])
            else:
                common_layers = common_layers.intersection(set(layers_per_task[task]))

    # Filter to only keep common layers
    if common_layers is not None:
        common_layers = sorted(list(common_layers))
        for task in perf:
            if len(perf[task]) > 0:
                filtered_perf = []
                for i, layer in enumerate(layers_per_task[task]):
                    if layer in common_layers:
                        filtered_perf.append(perf[task][i])
                perf[task] = filtered_perf
        layers = common_layers

    full_baselines = {}
    for task, (metric, result_key) in tasks_metric.items():
        task_folder = task
        file_name = f"{task}_L0_W0.json"
        file_path = f"{base_path}/llava-v1.5-7b-pretrained-lbasile_sw0/{task_folder}/{file_name}"

        try:
            with open(file_path, "r") as f:
                file = json.load(f)
                full_baselines[task] = file["results"][result_key][metric]
        except FileNotFoundError:
            print(f"Warning: Baseline file not found {file_path}")
            # Try to get from progressive directory
            file_path_alt = f"{base_path}/llava-v1.5-7b-pretrained-lbasile/{task_folder}/{file_name}"
            try:
                with open(file_path_alt, "r") as f:
                    file = json.load(f)
                    full_baselines[task] = file["results"][result_key][metric]
            except FileNotFoundError:
                print(f"Warning: Alternative baseline file not found {file_path_alt}")
                continue

    random_baselines = {
        "cv_bench": None,
        "mme": None,
        "gqa": None,
        "seedbench": None,
    }

    # Load random baselines from sw32 where available
    for task, (metric, result_key) in tasks_metric.items():
        try:
            with open(
                f"{base_path}/llava-v1.5-7b-pretrained-lbasile_sw32/{task}/{task}_L0_W32.json",
                "r",
            ) as f:
                file = json.load(f)
                random_baselines[task] = file["results"][result_key][metric]
        except FileNotFoundError:
            print(f"Warning: Random baseline not found for {task}, using 0")
            random_baselines[task] = 0
            continue

    return perf, random_baselines, full_baselines, layers


# ***************************************************************************************************

vl_window = 4
vl_perf, vl_random, vl_full, vl_layers = get_data_vl_window(
    base_path_vl, window_size=vl_window
)

vl_perf_progr, vl_random_progr, vl_full_progr, vl_layers_progr = (
    get_data_vl_progressive(base_path_vl)
)


# *********************************************************************


#####################################

# Combined averaged profiles plot

#######################################


def get_avg_task(performances, random_base, full_perf, task_type=None, task_types=None):
    vl_norm_values = []

    for task in performances:
        # Skip tasks that don't have data
        if len(performances[task]) == 0:
            print(f"Skipping {task} - no data available")
            continue
        if task not in full_perf or task not in random_base:
            print(f"Skipping {task} - missing baseline data")
            continue

        if task_type is None or (
            task_type is not None and task_types[task] == task_type
        ):
            data = performances[task]
            norm_acc = (np.array(data) - random_base[task]) / (
                full_perf[task] - random_base[task]
            )
            vl_norm_values.append(norm_acc)

    # Average across VL tasks
    vl_avg = np.mean(vl_norm_values, axis=0)
    vl_std = np.std(vl_norm_values, axis=0)

    return vl_avg, vl_std


###########

# VL

###########

# 1. VL tasks window 4 averaged
vl_tasks_to_average = {
    task: vl_perf[task] for task in vl_perf.keys() if len(vl_perf[task]) > 0
}
vl_avg, vl_std = get_avg_task(
    vl_tasks_to_average,
    random_base=vl_random,
    full_perf=vl_full,
    task_type=None,
    task_types=None,
)
shifted_layers_vl = np.array(vl_layers) + vl_window / 2
num_vl_tasks = len(vl_tasks_to_average)
vl_sem = vl_std / np.sqrt(num_vl_tasks)  # Standard error of the mean


# ********************************

vl_tasks_to_average_progr = {
    task: vl_perf_progr[task]
    for task in vl_perf_progr.keys()
    if len(vl_perf_progr[task]) > 0
}
vl_avg_prgr, vl_std_progr = get_avg_task(
    vl_tasks_to_average_progr,
    random_base=vl_random_progr,
    full_perf=vl_full_progr,
    task_type=None,
    task_types=None,
)
num_vl_tasks_progr = len(vl_tasks_to_average_progr)
vl_sem_progr = vl_std_progr / np.sqrt(num_vl_tasks_progr)  # Standard error of the mean


##############################################################
##############################################################

fig_combined = plt.figure(figsize=(4.5, 4))
gs_combined = GridSpec(1, 1)


ax_combined = fig_combined.add_subplot(gs_combined[0])
ax_combined.axvspan(
    6, 17, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
)
sns.lineplot(
    x=shifted_layers_vl,
    y=vl_avg,
    marker="o",
    linewidth=1,
    label="Image-ablation\nVision-language tasks",
    color="purple",
    ax=ax_combined,
)

ax_combined.set_xticks(np.array(list(range(0, 33, 4))))
# ax_combined.set_ylim(-0.05, 1.05)
ax_combined.set_ylabel("performance ", fontsize=14)
ax_combined.set_xlabel("layers", fontsize=12)
ax_combined.legend(fontsize=10, loc="best")
ax_combined.set_title("sliding window ablation (pretrained)", fontsize=15)
ax_combined.grid(True, alpha=0.4)

gs_combined.tight_layout(fig_combined)
plt.savefig("./results/plots/averaged_profiles_ablations_w4_pretrained.png", dpi=200)


##########################################

# PROGRESSIVE

##########################################

fig_combined = plt.figure(figsize=(4.5, 4))
gs_combined = GridSpec(1, 1)


ax_combined = fig_combined.add_subplot(gs_combined[0])
ax_combined.axvspan(
    6, 17, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
)
sns.lineplot(
    x=vl_layers_progr,
    y=vl_avg_prgr,
    marker="o",
    linewidth=1,
    label="Image ablation\nvision-language tasks",
    color="purple",
    ax=ax_combined,
)

ax_combined.fill_between(
    vl_layers_progr,
    vl_avg_prgr - vl_std_progr,
    vl_avg_prgr + vl_std_progr,
    color="purple",
    alpha=0.2,
)

ax_combined.set_xticks(np.array(list(range(0, 33, 4))))
ax_combined.set_ylabel("performance ", fontsize=14)
ax_combined.set_xlabel("layers", fontsize=12)
ax_combined.legend(fontsize=9, loc="best")
ax_combined.set_title("n° ablated layers from input (pretrained)", fontsize=15)
ax_combined.grid(True, alpha=0.4)

# *************************************************************

gs_combined.tight_layout(fig_combined)
plt.savefig(
    "./results/plots/averaged_profiles_ablations_progressive_std_pretrained.png",
    dpi=200,
)


######################

# Plot all tasks VL sliding window

######################

fig_vl_all = plt.figure(figsize=(4.5, 4))
gs_vl_all = GridSpec(1, 1)

ax_vl_all = fig_vl_all.add_subplot(gs_vl_all[0])
ax_vl_all.axvspan(6, 17, alpha=0.15, color="goldenrod", zorder=0)

for task, data in vl_perf.items():
    if len(data) == 0 or task not in vl_full or task not in vl_random:
        continue
    # Normalize data
    norm_acc = (np.array(data) - vl_random[task]) / (vl_full[task] - vl_random[task])

    sns.lineplot(
        x=shifted_layers_vl,
        y=norm_acc,
        marker="o",
        linewidth=1,
        label=task,
        alpha=alpha,
        ax=ax_vl_all,
    )

ax_vl_all.set_xticks(np.array(list(range(0, 33, 4))))
# ax_vl_all.set_ylim(-0.05, 1.05)
ax_vl_all.set_ylabel("performance", fontsize=14)
ax_vl_all.set_xlabel("layers", fontsize=12)
ax_vl_all.legend(fontsize=10, loc="best")
ax_vl_all.set_title("VL tasks - sliding window (pretrained)", fontsize=15)
ax_vl_all.grid(True, alpha=0.4)
ax_vl_all.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

gs_vl_all.tight_layout(fig_vl_all)
plt.savefig("./results/plots/all_tasks_vl_sliding_window_w4_pretrained.png", dpi=200)


######################

# Plot all tasks VL progressive

######################

fig_vl_prog = plt.figure(figsize=(4.5, 4))
gs_vl_prog = GridSpec(1, 1)

ax_vl_prog = fig_vl_prog.add_subplot(gs_vl_prog[0])
ax_vl_prog.axvspan(6, 17, alpha=0.15, color="goldenrod", zorder=0)

for task, data in vl_perf_progr.items():
    if len(data) == 0 or task not in vl_full_progr or task not in vl_random_progr:
        continue
    # Normalize data
    norm_acc = (np.array(data) - vl_random_progr[task]) / (
        vl_full_progr[task] - vl_random_progr[task]
    )

    sns.lineplot(
        x=vl_layers_progr,
        y=norm_acc,
        marker="o",
        linewidth=1,
        label=task,
        alpha=alpha,
        ax=ax_vl_prog,
    )

ax_vl_prog.set_xticks(np.array(list(range(0, 33, 4))))
# ax_vl_prog.set_ylim(-0.05, 1.05)
ax_vl_prog.set_ylabel("performance", fontsize=14)
ax_vl_prog.set_xlabel("layers", fontsize=12)
ax_vl_prog.legend(fontsize=10, loc="best")
ax_vl_prog.set_title("VL tasks - progressive (pretrained)", fontsize=15)
ax_vl_prog.grid(True, alpha=0.4)
ax_vl_prog.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

gs_vl_prog.tight_layout(fig_vl_prog)
plt.savefig("./results/plots/all_tasks_vl_progressive_pretrained.png", dpi=200)

print("\nAll plots saved successfully!")
print(f"Number of VL tasks (sliding window): {num_vl_tasks}")
print(f"Tasks included (sliding window): {list(vl_tasks_to_average.keys())}")
print(f"Number of VL tasks (progressive): {num_vl_tasks_progr}")
print(f"Tasks included (progressive): {list(vl_tasks_to_average_progr.keys())}")
