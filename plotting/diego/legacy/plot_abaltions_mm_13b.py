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

base_path_language = "/home/diego/Documents/area_science/ricerca/open/llava_finetuning/results/layer_skip_eval"
base_path_vl = "/home/diego/Documents/area_science/ricerca/open/llava_finetuning/results/layer_skip_eval_mm"


def get_data_language_new(
    base_path,
    lang_window,
):

    tasks_fs = {
        "commonsense_qa": 0,
        "hellaswag": 0,
        "mmlu": 0,
        # "arc_challenge": 0,
        "arc_easy": 0,
        "piqa": 0,
        "winogrande": 0,
    }

    tasks_metric = {
        "commonsense_qa": "acc,none",
        "hellaswag": "acc,none",
        "mmlu": "acc,none",
        # "arc_challenge": "acc,none",
        "arc_easy": "acc,none",
        "piqa": "acc,none",
        "winogrande": "acc,none",
    }

    tasks_type = {
        "commonsense_qa": "lik",
        "hellaswag": "lik",
        "mmlu": "lik",
        # "arc_challenge": "lik",
        "arc_easy": "lik",
        "piqa": "lik",
        "winogrande": "lik",
    }

    # 13B model has 40 layers (0-39)
    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    perf = {}
    for task, few_shot in tasks_fs.items():

        perf[task] = []
        for layer in layers:
            file_path = f"{base_path}/llava_13b_sw{lang_window}/{task}/{task}_L{layer:02d}_W{lang_window}_{few_shot}shot.json"
            with open(file_path, "r") as f:
                file = json.load(f)
                perf[task].append(file["results"][task][tasks_metric[task]])

    baselines = {}
    for task, few_shot in tasks_fs.items():
        file_path = (
            f"{base_path}/llava_13b_sw0/{task}/{task}_L00_W0_{few_shot}shot.json"
        )
        with open(file_path, "r") as f:
            file = json.load(f)
            baselines[task] = file["results"][task][tasks_metric[task]]

    random_baselines = {
        "commonsense_qa": 0.2,
        "hellaswag": 0.25,
        "mmlu": 0.25,
        # "arc_challenge": 0.25,
        "arc_easy": 0.25,
        "piqa": 0.5,
        "winogrande": 0.5,
        "nq_open": 0,
        "triviaqa": 0,
    }

    return perf, random_baselines, baselines, tasks_type


def get_data_vl_window(base_path, window_size):

    tasks_metric = {
        # "cv_bench": ("cv_bench_acc,none", "cv_bench"),
        "mme": ("mme_perception_score,none", "mme"),
        "gqa": ("exact_match,none", "gqa"),
        "mmbench_en_dev": ("gpt_eval_score,none", "mmbench_en_dev"),
        "mmbench_cn_dev": ("gpt_eval_score,none", "mmbench_cn_dev"),
        "seedbench": ("seed_image,none", "seedbench"),
        "vqav2_val": ("exact_match,none", "vqav2_val"),
    }

    # 13B model: includes layer 40 for VL tasks
    layers = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
    ]
    perf = {}
    layers_per_task = {}

    for task, (metric, result_key) in tasks_metric.items():

        perf[task] = []
        layers_per_task[task] = []

        for layer in layers:
            task_folder = task
            file_name = f"{task}_L{layer}_W{window_size}.json"
            file_path = (
                f"{base_path}/llava_13b_sw{window_size}/{task_folder}/{file_name}"
            )

            with open(file_path, "r") as f:
                file = json.load(f)
                perf[task].append(file["results"][result_key][metric])

    full_baselines = {}
    for task, (metric, result_key) in tasks_metric.items():
        task_folder = task
        file_name = f"{task}_L0_W0.json"
        file_path = f"{base_path}/llava_13b_sw0/{task_folder}/{file_name}"

        with open(file_path, "r") as f:
            file = json.load(f)
            full_baselines[task] = file["results"][result_key][metric]

    random_baselines = {
        # "cv_bench": None,
        "mme": None,
        "gqa": None,
        "mmbench_en_dev": None,
        "mmbench_cn_dev": None,
        "seedbench": None,
        "vqav2_val": None,
    }

    # Load random baselines from sw40 (13B's max window)
    for task, (metric, result_key) in tasks_metric.items():
        with open(
            f"{base_path}/llava_13b_sw40/{task}/{task}_L0_W40.json",
            "r",
        ) as f:
            file = json.load(f)
            random_baselines[task] = file["results"][result_key][metric]

    return perf, random_baselines, full_baselines, layers


def get_data_vl_progressive(base_path):

    tasks_metric = {
        # "cv_bench": ("cv_bench_acc,none", "cv_bench"),
        "mme": ("mme_perception_score,none", "mme"),
        "gqa": ("exact_match,none", "gqa"),
        "seedbench": ("seed_image,none", "seedbench"),
        "mmbench_en_dev": ("gpt_eval_score,none", "mmbench_en_dev"),
        "mmbench_cn_dev": ("gpt_eval_score,none", "mmbench_cn_dev"),
        "vqav2_val": ("exact_match,none", "vqav2_val"),
    }

    # 13B model: progressive up to 40 layers
    layers = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
    ]
    perf = {}
    layers_per_task = {}

    for task, (metric, result_key) in tasks_metric.items():

        perf[task] = []
        layers_per_task[task] = []

        for layer in layers:
            task_folder = task
            file_name = f"{task}_L0_W{layer}.json"
            file_path = f"{base_path}/llava_13b/{task_folder}/{file_name}"

            with open(file_path, "r") as f:
                file = json.load(f)
                perf[task].append(file["results"][result_key][metric])
                layers_per_task[task].append(layer)

    full_baselines = {}
    for task, (metric, result_key) in tasks_metric.items():
        task_folder = task
        file_name = f"{task}_L0_W0.json"
        file_path = f"{base_path}/llava_13b_sw0/{task_folder}/{file_name}"

        with open(file_path, "r") as f:
            file = json.load(f)
            full_baselines[task] = file["results"][result_key][metric]

    random_baselines = {
        # "cv_bench": None,
        "mme": None,
        "gqa": None,
        "seedbench": None,
        "mmbench_en_dev": None,
        "mmbench_cn_dev": None,
        "vqav2_val": None,
    }

    # Load random baselines from sw40 where available
    for task, (metric, result_key) in tasks_metric.items():
        with open(
            f"{base_path}/llava_13b_sw40/{task}/{task}_L0_W40.json",
            "r",
        ) as f:
            file = json.load(f)
            random_baselines[task] = file["results"][result_key][metric]

    return perf, random_baselines, full_baselines, layers


def get_data_language_progressive(base_path):

    tasks_fs = {
        "commonsense_qa": 0,
        "hellaswag": 0,
        # "mmlu": 0,
        # "arc_challenge": 0,
        "arc_easy": 0,
        "piqa": 0,
        "winogrande": 0,
    }

    tasks_metric = {
        "commonsense_qa": "acc,none",
        "hellaswag": "acc,none",
        # "mmlu": "acc,none",
        # "arc_challenge": "acc,none",
        "arc_easy": "acc,none",
        "piqa": "acc,none",
        "winogrande": "acc,none",
    }

    tasks_type = {
        "commonsense_qa": "lik",
        "hellaswag": "lik",
        # "mmlu": "lik",
        # "arc_challenge": "lik",
        "arc_easy": "lik",
        "piqa": "lik",
        "winogrande": "lik",
    }

    # 13B model: progressive up to 40 layers
    layers = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
    ]
    perf = {}
    layers_per_task = {}

    for task, few_shot in tasks_fs.items():

        perf[task] = []
        layers_per_task[task] = []

        for layer in layers:
            file_path = (
                f"{base_path}/llava_13b/{task}/{task}_L00_W{layer}_{few_shot}shot.json"
            )
            with open(file_path, "r") as f:
                file = json.load(f)
                perf[task].append(file["results"][task][tasks_metric[task]])
                layers_per_task[task].append(layer)

    baselines = {}
    for task, few_shot in tasks_fs.items():
        file_path = (
            f"{base_path}/llava_13b_sw0/{task}/{task}_L00_W0_{few_shot}shot.json"
        )
        with open(file_path, "r") as f:
            file = json.load(f)
            baselines[task] = file["results"][task][tasks_metric[task]]

    random_baselines = {
        "commonsense_qa": 0.2,
        "hellaswag": 0.25,
        # "mmlu": 0.25,
        "arc_challenge": 0.25,
        "arc_easy": 0.25,
        "piqa": 0.5,
        "winogrande": 0.5,
    }

    return perf, random_baselines, baselines, tasks_type, layers


# ***************************************************************************************************

lang_window = 5
language_perf, language_random, language_full, language_task_type = (
    get_data_language_new(base_path_language, lang_window=lang_window)
)


(
    language_perf_progr,
    language_random_progr,
    language_full_progr,
    language_task_type_progr,
    language_layers_progr,
) = get_data_language_progressive(base_path_language)


vl_window = 5
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

plot_gen_data = False


def get_avg_task(performances, random_base, full_perf, task_type=None, task_types=None):
    vl_norm_values = []

    for task in performances:
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
vl_tasks_to_average = {task: vl_perf[task] for task in vl_perf.keys()}
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

# 2. Language likelihood-based tasks averaged
x_lang = np.array(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
)
x_shift_lang = lang_window / 2
shifted_x_lang = x_lang + x_shift_lang

lang_lik_avg, lang_lik_std = get_avg_task(
    language_perf,
    random_base=language_random,
    full_perf=language_full,
    task_type="lik",
    task_types=language_task_type,
)
# Count number of likelihood tasks for SEM calculation
num_lik_tasks = sum(1 for task in language_perf if language_task_type[task] == "lik")
lang_lik_sem = lang_lik_std / np.sqrt(num_lik_tasks)  # Standard error of the mean


# ********************************

vl_tasks_to_average = {task: vl_perf_progr[task] for task in vl_perf_progr.keys()}
vl_avg_prgr, vl_std_progr = get_avg_task(
    vl_tasks_to_average,
    random_base=vl_random_progr,
    full_perf=vl_full_progr,
    task_type=None,
    task_types=None,
)
num_vl_tasks = len(vl_tasks_to_average)
vl_sem_progr = vl_std_progr / np.sqrt(num_vl_tasks)  # Standard error of the mean


##############################################################à
##############################################################

fig_combined = plt.figure(figsize=(4.5, 4))
gs_combined = GridSpec(1, 1)


ax_combined = fig_combined.add_subplot(gs_combined[0])
# Adjusted abstraction phase for 40 layers (13B model)
ax_combined.axvspan(
    6, 20, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
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

sns.lineplot(
    x=shifted_x_lang,
    y=lang_lik_avg,
    marker="s",
    linewidth=1,
    label="Language tasks\nmultiple choice",
    color="green",
    ax=ax_combined,
)

ax_combined.set_xticks(np.array(list(range(0, 41, 4))))
ax_combined.set_ylim(-0.05, 1.05)
ax_combined.set_ylabel("performance ", fontsize=14)
ax_combined.set_xlabel("layers", fontsize=12)
ax_combined.legend(fontsize=10, loc="best")
ax_combined.set_title("sliding window ablation (13B)", fontsize=15)
ax_combined.grid(True, alpha=0.4)

gs_combined.tight_layout(fig_combined)
plt.savefig(
    f"./results/plots/averaged_profiles_ablations_w{vl_window}_lang_w{lang_window}_13b.png",
    dpi=200,
)


######################

# Language tasks stratified by difficulty - sliding window

######################

# Define easy and hard tasks
easy_tasks = ["hellaswag", "piqa", "arc_easy"]
hard_tasks = ["commonsense_qa", "winogrande"]

# Calculate averages for easy tasks
lang_easy_avg, lang_easy_std = get_avg_task(
    {task: language_perf[task] for task in easy_tasks if task in language_perf},
    random_base=language_random,
    full_perf=language_full,
    task_type=None,
    task_types=None,
)
num_easy_tasks = len([task for task in easy_tasks if task in language_perf])
lang_easy_sem = lang_easy_std / np.sqrt(num_easy_tasks)

# Calculate averages for hard tasks
lang_hard_avg, lang_hard_std = get_avg_task(
    {task: language_perf[task] for task in hard_tasks if task in language_perf},
    random_base=language_random,
    full_perf=language_full,
    task_type=None,
    task_types=None,
)
num_hard_tasks = len([task for task in hard_tasks if task in language_perf])
lang_hard_sem = lang_hard_std / np.sqrt(num_hard_tasks)

# Create figure
fig_stratified = plt.figure(figsize=(4.5, 4))
gs_stratified = GridSpec(1, 1)

ax_stratified = fig_stratified.add_subplot(gs_stratified[0])
ax_stratified.axvspan(
    6, 20, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
)

# Plot VL tasks (same as average plot)
sns.lineplot(
    x=shifted_layers_vl,
    y=vl_avg,
    marker="o",
    linewidth=1,
    label="Image-ablation\nVision-language tasks",
    color="purple",
    ax=ax_stratified,
)

# Add standard deviation for VL tasks
ax_stratified.fill_between(
    shifted_layers_vl,
    vl_avg - vl_std,
    vl_avg + vl_std,
    color="purple",
    alpha=0.15,
)


# Plot easy language tasks (dashed green line)
sns.lineplot(
    x=shifted_x_lang,
    y=lang_easy_avg,
    marker="s",
    linewidth=1,
    linestyle="--",
    label="Language tasks (easy)",
    color="green",
    ax=ax_stratified,
)

# Add standard deviation for easy tasks
ax_stratified.fill_between(
    shifted_x_lang,
    lang_easy_avg - lang_easy_std,
    lang_easy_avg + lang_easy_std,
    color="green",
    alpha=0.15,
)

# Plot hard language tasks (solid line)
sns.lineplot(
    x=shifted_x_lang,
    y=lang_hard_avg,
    marker="^",
    linewidth=1,
    linestyle="-",
    label="Language tasks (hard)",
    color="darkgreen",
    ax=ax_stratified,
)

# Add standard deviation for hard tasks
ax_stratified.fill_between(
    shifted_x_lang,
    lang_hard_avg - lang_hard_std,
    lang_hard_avg + lang_hard_std,
    color="darkgreen",
    alpha=0.15,
)

ax_stratified.set_xticks(np.array(list(range(0, 41, 4))))
# ax_stratified.set_ylim(-0.05, 1.05)
ax_stratified.set_ylabel("performance ", fontsize=14)
ax_stratified.set_xlabel("layers", fontsize=12)
ax_stratified.legend(fontsize=9, loc="best")
ax_stratified.set_title("sliding window ablation (13B)", fontsize=15)
ax_stratified.grid(True, alpha=0.4)

gs_stratified.tight_layout(fig_stratified)
plt.savefig(
    f"./results/plots/averaged_profiles_language_difficulty_stratified_w{vl_window}_lang_w{lang_window}_13b.png",
    dpi=200,
)


##########################################

# PROGRESSIVE

# @########################################à#

# Calculate averages for easy tasks (progressive)
lang_easy_avg_progr, lang_easy_std_progr = get_avg_task(
    {
        task: language_perf_progr[task]
        for task in easy_tasks
        if task in language_perf_progr
    },
    random_base=language_random_progr,
    full_perf=language_full_progr,
    task_type=None,
    task_types=None,
)
num_easy_tasks_progr = len([task for task in easy_tasks if task in language_perf_progr])
lang_easy_sem_progr = lang_easy_std_progr / np.sqrt(num_easy_tasks_progr)

# Calculate averages for hard tasks (progressive)
lang_hard_avg_progr, lang_hard_std_progr = get_avg_task(
    {
        task: language_perf_progr[task]
        for task in hard_tasks
        if task in language_perf_progr
    },
    random_base=language_random_progr,
    full_perf=language_full_progr,
    task_type=None,
    task_types=None,
)
num_hard_tasks_progr = len([task for task in hard_tasks if task in language_perf_progr])
lang_hard_sem_progr = lang_hard_std_progr / np.sqrt(num_hard_tasks_progr)


# fig_combined = plt.figure(figsize=(4.5, 4))
# gs_combined = GridSpec(1, 1)


# ax_combined = fig_combined.add_subplot(gs_combined[0])
# ax_combined.axvspan(
#     6, 20, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
# )
# sns.lineplot(
#     x=vl_layers_progr,
#     y=vl_avg_prgr,
#     marker="o",
#     linewidth=1,
#     label="Image ablation\nvision-language tasks",
#     color="purple",
#     ax=ax_combined,
# )

# ax_combined.fill_between(
#     vl_layers_progr,
#     vl_avg_prgr - vl_std_progr,
#     vl_avg_prgr + vl_std_progr,
#     color="purple",
#     alpha=0.2,
# )

# # Plot easy language tasks (progressive, dashed green line)
# sns.lineplot(
#     x=language_layers_progr,
#     y=lang_easy_avg_progr,
#     marker="s",
#     linewidth=1,
#     linestyle="--",
#     label="Language tasks (easy)\nhellaswag, piqa, arc_easy",
#     color="green",
#     ax=ax_combined,
# )

# # Add standard deviation for easy tasks
# ax_combined.fill_between(
#     language_layers_progr,
#     lang_easy_avg_progr - lang_easy_std_progr,
#     lang_easy_avg_progr + lang_easy_std_progr,
#     color="green",
#     alpha=0.2,
# )

# # Plot hard language tasks (progressive, solid line)
# sns.lineplot(
#     x=language_layers_progr,
#     y=lang_hard_avg_progr,
#     marker="^",
#     linewidth=1,
#     linestyle="-",
#     label="Language tasks (hard)\nall others",
#     color="darkgreen",
#     ax=ax_combined,
# )

# # Add standard deviation for hard tasks
# ax_combined.fill_between(
#     language_layers_progr,
#     lang_hard_avg_progr - lang_hard_std_progr,
#     lang_hard_avg_progr + lang_hard_std_progr,
#     color="darkgreen",
#     alpha=0.2,
# )

# ax_combined.set_xticks(np.array(list(range(0, 41, 4))))
# ax_combined.set_ylabel("performance ", fontsize=14)
# ax_combined.set_xlabel("layers", fontsize=12)
# ax_combined.legend(fontsize=9, loc="best")
# ax_combined.set_title("n° ablated layers from input (13B)", fontsize=15)
# ax_combined.grid(True, alpha=0.4)

# # *************************************************************

# gs_combined.tight_layout(fig_combined)
# plt.savefig(
#     "./results/plots/averaged_profiles_ablations_progressive_std_13b.png", dpi=200
# )


######################

# Progressive plot with combined language tasks

######################

# Calculate averages for all language tasks combined (progressive)
lang_all_avg_progr, lang_all_std_progr = get_avg_task(
    language_perf_progr,
    random_base=language_random_progr,
    full_perf=language_full_progr,
    task_type="lik",
    task_types=language_task_type_progr,
)
num_all_tasks_progr = len(
    [task for task in language_perf_progr if language_task_type_progr[task] == "lik"]
)
lang_all_sem_progr = lang_all_std_progr / np.sqrt(num_all_tasks_progr)

fig_combined_all = plt.figure(figsize=(4.5, 4))
gs_combined_all = GridSpec(1, 1)

ax_combined_all = fig_combined_all.add_subplot(gs_combined_all[0])
ax_combined_all.axvspan(
    6, 20, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
)

# Plot VL tasks progressive
sns.lineplot(
    x=vl_layers_progr,
    y=vl_avg_prgr,
    marker="o",
    linewidth=1,
    label="Image ablation\nvision-language tasks",
    color="purple",
    ax=ax_combined_all,
)

ax_combined_all.fill_between(
    vl_layers_progr,
    vl_avg_prgr - vl_std_progr,
    vl_avg_prgr + vl_std_progr,
    color="purple",
    alpha=0.2,
)

# Plot all language tasks combined (progressive, green line)
sns.lineplot(
    x=language_layers_progr,
    y=lang_all_avg_progr,
    marker="s",
    linewidth=1,
    linestyle="-",
    label="Language tasks",
    color="green",
    ax=ax_combined_all,
)

# Add standard deviation for all language tasks
ax_combined_all.fill_between(
    language_layers_progr,
    lang_all_avg_progr - lang_all_std_progr,
    lang_all_avg_progr + lang_all_std_progr,
    color="green",
    alpha=0.2,
)

ax_combined_all.set_xticks(np.array(list(range(0, 41, 4))))
ax_combined_all.set_ylabel("performance ", fontsize=14)
ax_combined_all.set_xlabel("layers", fontsize=12)
ax_combined_all.legend(fontsize=10, loc="best")
ax_combined_all.set_title("n° ablated layers from input (13B)", fontsize=15)
ax_combined_all.grid(True, alpha=0.4)

gs_combined_all.tight_layout(fig_combined_all)
plt.savefig(
    "./results/plots/averaged_profiles_ablations_progressive_std_13b.png", dpi=200
)


######################

# Plot all tasks VL sliding window

######################

fig_vl_all = plt.figure(figsize=(4.5, 4))
gs_vl_all = GridSpec(1, 1)

ax_vl_all = fig_vl_all.add_subplot(gs_vl_all[0])
ax_vl_all.axvspan(6, 20, alpha=0.15, color="goldenrod", zorder=0)

for task, data in vl_perf.items():
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

ax_vl_all.set_xticks(np.array(list(range(0, 41, 4))))
ax_vl_all.set_ylim(-0.05, 1.05)
ax_vl_all.set_ylabel("performance", fontsize=14)
ax_vl_all.set_xlabel("layers", fontsize=12)
ax_vl_all.legend(fontsize=10, loc="best")
ax_vl_all.set_title("VL tasks - sliding window (13B)", fontsize=15)
ax_vl_all.grid(True, alpha=0.4)
ax_vl_all.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

gs_vl_all.tight_layout(fig_vl_all)
plt.savefig(
    f"./results/plots/all_tasks_vl_sliding_window_w{vl_window}_13b.png", dpi=200
)


######################

# Plot all tasks Language sliding window

######################

fig_lang_all = plt.figure(figsize=(4.5, 4))
gs_lang_all = GridSpec(1, 1)

ax_lang_all = fig_lang_all.add_subplot(gs_lang_all[0])
ax_lang_all.axvspan(6, 20, alpha=0.15, color="goldenrod", zorder=0)

for task, data in language_perf.items():
    if language_task_type[task] == "lik":  # Only plot likelihood-based tasks
        # Normalize data
        norm_acc = (np.array(data) - language_random[task]) / (
            language_full[task] - language_random[task]
        )

        sns.lineplot(
            x=shifted_x_lang,
            y=norm_acc,
            marker="s",
            linewidth=1,
            label=task,
            alpha=alpha,
            ax=ax_lang_all,
        )

ax_lang_all.set_xticks(np.array(list(range(0, 41, 4))))
# ax_lang_all.set_ylim(-0.15, 1.05)
ax_lang_all.set_ylabel("performance", fontsize=14)
ax_lang_all.set_xlabel("layers", fontsize=12)
ax_lang_all.legend(fontsize=10, loc="best")
ax_lang_all.set_title("Language tasks - sliding window (13B)", fontsize=15)
ax_lang_all.grid(True, alpha=0.4)
ax_lang_all.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

gs_lang_all.tight_layout(fig_lang_all)
plt.savefig(
    f"./results/plots/all_tasks_language_sliding_window_w{lang_window}_13b.png", dpi=200
)


######################

# Plot all tasks VL progressive

######################

fig_vl_prog = plt.figure(figsize=(4.5, 4))
gs_vl_prog = GridSpec(1, 1)

ax_vl_prog = fig_vl_prog.add_subplot(gs_vl_prog[0])
ax_vl_prog.axvspan(6, 20, alpha=0.15, color="goldenrod", zorder=0)

for task, data in vl_perf_progr.items():
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

ax_vl_prog.set_xticks(np.array(list(range(0, 41, 4))))
# ax_vl_prog.set_ylim(-0.05, 1.05)
ax_vl_prog.set_ylabel("performance", fontsize=14)
ax_vl_prog.set_xlabel("layers", fontsize=12)
ax_vl_prog.legend(fontsize=10, loc="best")
ax_vl_prog.set_title("VL tasks - progressive (13B)", fontsize=15)
ax_vl_prog.grid(True, alpha=0.4)
ax_vl_prog.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

gs_vl_prog.tight_layout(fig_vl_prog)
plt.savefig("./results/plots/all_tasks_vl_progressive_13b.png", dpi=200)
