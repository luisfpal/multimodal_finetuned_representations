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


def get_data_language(base_path):

    tasks_fs = {
        "commonsense_qa": 0,
        "hellaswag": 0,
        "mmlu": 0,
        "nq_open": 5,
        "triviaqa": 1,
        # "truthfulqa_gen": 0,
        "squadv2": 0,
    }

    tasks_metric = {
        "commonsense_qa": "acc,none",
        "hellaswag": "acc,none",
        "mmlu": "acc,none",
        "nq_open": "exact_match,remove_whitespace",
        "triviaqa": "exact_match,remove_whitespace",
        "ifeval": "inst_level_strict_acc,none",
        "truthfulqa_gen": "bleu_acc,none",  # , "rouge1_acc,none",
        "squadv2": "exact,none",
    }

    tasks_type = {
        "commonsense_qa": "lik",
        "hellaswag": "lik",
        "mmlu": "lik",
        "nq_open": "gen",
        "triviaqa": "gen",
        "truthfulqa_gen": "gen",
        "squadv2": "gen",
    }

    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    perf = {}
    for task, few_shot in tasks_fs.items():

        perf[task] = []
        for layer in layers:
            with open(
                f"{base_path}/llava-1.5-7b-hf_sw4/{task}_skip_{layer:02d}_{few_shot}shot.json",
                "r",
            ) as f:
                file = json.load(f)
                perf[task].append(file["results"][task][tasks_metric[task]])

    baselines = {}
    for task, few_shot in tasks_fs.items():
        if task == "squadv2":
            continue
        for layer in layers:
            with open(
                f"{base_path}/llava-1.5-7b-hf/{task}_skip_00_{few_shot}shot.json",
                "r",
            ) as f:
                file = json.load(f)
                baselines[task] = file["results"][task][tasks_metric[task]]

    random_baselines = {
        "commonsense_qa": 0.2,
        "hellaswag": 0.25,
        "mmlu": 0.25,
        "nq_open": 0,
        "triviaqa": 0,
        "ifeval": 0,
        "truthfulqa_gen": 0,
        "squadv2": 0,
    }

    return perf, random_baselines, baselines, tasks_type


def get_data_language_new(base_path):

    tasks_fs = {
        "commonsense_qa": 0,
        "hellaswag": 0,
        "mmlu": 0,
        # "arc_challenge": 0,
        "arc_easy": 0,
        "piqa": 0,
        "winogrande": 0,
        # "nq_open": 5,
        # "triviaqa": 1,
    }

    tasks_metric = {
        "commonsense_qa": "acc,none",
        "hellaswag": "acc,none",
        "mmlu": "acc,none",
        # "arc_challenge": "acc,none",
        "arc_easy": "acc,none",
        "piqa": "acc,none",
        "winogrande": "acc,none",
        # "nq_open": "exact_match,remove_whitespace",
        # "triviaqa": "exact_match,remove_whitespace",
    }

    tasks_type = {
        "commonsense_qa": "lik",
        "hellaswag": "lik",
        "mmlu": "lik",
        # "arc_challenge": "lik",
        "arc_easy": "lik",
        "piqa": "lik",
        "winogrande": "lik",
        # "nq_open": "gen",
        # "triviaqa": "gen",
    }

    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    perf = {}
    for task, few_shot in tasks_fs.items():

        perf[task] = []
        for layer in layers:
            file_path = f"{base_path}/llava-7b-lbasile_sw4/{task}/{task}_L{layer:02d}_W4_{few_shot}shot.json"
            try:
                with open(file_path, "r") as f:
                    file = json.load(f)
                    perf[task].append(file["results"][task][tasks_metric[task]])
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}")

    baselines = {}
    for task, few_shot in tasks_fs.items():
        file_path = (
            f"{base_path}/llava-7b-lbasile_sw0/{task}/{task}_L00_W0_{few_shot}shot.json"
        )
        try:
            with open(file_path, "r") as f:
                file = json.load(f)
                baselines[task] = file["results"][task][tasks_metric[task]]
        except FileNotFoundError:
            print(f"Warning: Baseline file not found {file_path}")

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
        # "ai2d": ("exact_match,flexible-extract", "ai2d"),
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
            file_path = f"{base_path}/llava-7b-lbasile_sw{window_size}/{task_folder}/{file_name}"

            with open(file_path, "r") as f:
                file = json.load(f)
                perf[task].append(file["results"][result_key][metric])

    full_baselines = {}
    for task, (metric, result_key) in tasks_metric.items():
        task_folder = task
        file_name = f"{task}_L0_W0.json"
        file_path = f"{base_path}/llava-7b-lbasile_sw0/{task_folder}/{file_name}"

        with open(file_path, "r") as f:
            file = json.load(f)
            full_baselines[task] = file["results"][result_key][metric]

    random_baselines = {
        # "cv_bench": None,
        "mme": None,
        "gqa": None,
        "mmbench_en_dev": None,
        "mmbench_cn_dev": None,
        # "ai2d": None,
        "seedbench": None,
    }

    # Load random baselines from sw32 where available
    for task, (metric, result_key) in tasks_metric.items():
        with open(
            f"{base_path}/llava-7b-lbasile_sw32/{task}/{task}_L0_W32.json",
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
        # "ai2d": ("exact_match,flexible-extract", "ai2d"),
        "seedbench": ("seed_image,none", "seedbench"),
        "mmbench_en_dev": ("gpt_eval_score,none", "mmbench_en_dev"),
        "mmbench_cn_dev": ("gpt_eval_score,none", "mmbench_cn_dev"),
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
            file_path = f"{base_path}/llava-7b-lbasile/{task_folder}/{file_name}"

            with open(file_path, "r") as f:
                file = json.load(f)
                perf[task].append(file["results"][result_key][metric])
                layers_per_task[task].append(layer)

    full_baselines = {}
    for task, (metric, result_key) in tasks_metric.items():
        task_folder = task
        file_name = f"{task}_L0_W0.json"
        file_path = f"{base_path}/llava-7b-lbasile_sw0/{task_folder}/{file_name}"

        with open(file_path, "r") as f:
            file = json.load(f)
            full_baselines[task] = file["results"][result_key][metric]

    random_baselines = {
        # "cv_bench": None,
        "mme": None,
        "gqa": None,
        # "ai2d": None,
        "seedbench": None,
        "mmbench_en_dev": None,
        "mmbench_cn_dev": None,
    }

    # Load random baselines from sw32 where available
    for task, (metric, result_key) in tasks_metric.items():

        with open(
            f"{base_path}/llava-7b-lbasile_sw32/{task}/{task}_L0_W32.json",
            "r",
        ) as f:
            file = json.load(f)
            random_baselines[task] = file["results"][result_key][metric]

    return perf, random_baselines, full_baselines, layers


def get_data_language_progressive(base_path):

    tasks_fs = {
        "commonsense_qa": 0,
        "hellaswag": 0,
        "mmlu": 0,
        "arc_challenge": 0,
        "arc_easy": 0,
        "piqa": 0,
        "winogrande": 0,
    }

    tasks_metric = {
        "commonsense_qa": "acc,none",
        "hellaswag": "acc,none",
        "mmlu": "acc,none",
        "arc_challenge": "acc,none",
        "arc_easy": "acc,none",
        "piqa": "acc,none",
        "winogrande": "acc,none",
    }

    tasks_type = {
        "commonsense_qa": "lik",
        "hellaswag": "lik",
        "mmlu": "lik",
        "arc_challenge": "lik",
        "arc_easy": "lik",
        "piqa": "lik",
        "winogrande": "lik",
    }

    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    perf = {}
    layers_per_task = {}

    for task, few_shot in tasks_fs.items():

        perf[task] = []
        layers_per_task[task] = []

        for layer in layers:
            file_path = f"{base_path}/llava-7b-lbasile/{task}/{task}_L00_W{layer}_{few_shot}shot.json"
            try:
                with open(file_path, "r") as f:
                    file = json.load(f)
                    perf[task].append(file["results"][task][tasks_metric[task]])
                    layers_per_task[task].append(layer)
            except FileNotFoundError:
                print(f"Warning: Progressive file not found {file_path}")

    baselines = {}
    for task, few_shot in tasks_fs.items():
        file_path = (
            f"{base_path}/llava-7b-lbasile_sw0/{task}/{task}_L00_W0_{few_shot}shot.json"
        )
        try:
            with open(file_path, "r") as f:
                file = json.load(f)
                baselines[task] = file["results"][task][tasks_metric[task]]
        except FileNotFoundError:
            print(f"Warning: Baseline file not found {file_path}")

    random_baselines = {
        "commonsense_qa": 0.2,
        "hellaswag": 0.25,
        "mmlu": 0.25,
        "arc_challenge": 0.25,
        "arc_easy": 0.25,
        "piqa": 0.5,
        "winogrande": 0.5,
    }

    return perf, random_baselines, baselines, tasks_type, layers


# ***************************************************************************************************

language_perf, language_random, language_full, language_task_type = (
    get_data_language_new(base_path_language)
)


(
    language_perf_progr,
    language_random_progr,
    language_full_progr,
    language_task_type_progr,
    language_layers_progr,
) = get_data_language_progressive(base_path_language)


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


# ax_combined.axvspan(
#     6, 17, alpha=0.15, color="steelblue", zorder=0, label="abstraction phase"
# )
# ax_combined.axvspan(
#     17, 32, alpha=0.15, color="coral", zorder=0, label="generation phase"
# )


###########

# VL

###########

# 1. VL tasks window 5 averaged
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
x_lang = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
window_size_lang = 4
x_shift_lang = window_size_lang / 2
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


# 3. Language open-ended generation tasks averaged (calculate before plotting)
lang_gen_avg, lang_gen_std = get_avg_task(
    language_perf,
    random_base=language_random,
    full_perf=language_full,
    task_type="gen",
    task_types=language_task_type,
)
# Count number of generation tasks for SEM calculation
num_gen_tasks = sum(1 for task in language_perf if language_task_type[task] == "gen")
lang_gen_sem = lang_gen_std / np.sqrt(num_gen_tasks)  # Standard error of the mean


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

sns.lineplot(
    x=shifted_x_lang,
    y=lang_lik_avg,
    marker="s",
    linewidth=1,
    label="Language tasks\nmultiple choice",
    color="green",
    ax=ax_combined,
)

if plot_gen_data:
    sns.lineplot(
        x=shifted_x_lang,
        y=lang_gen_avg,
        marker="^",
        linewidth=1,
        label="Language open-ended gen",
        color="orange",
        ax=ax_combined,
    )
    ax_combined.fill_between(
        shifted_x_lang,
        lang_gen_avg - lang_gen_std,
        lang_gen_avg + lang_gen_std,
        color="orange",
        alpha=0.2,
    )

ax_combined.set_xticks(np.array(list(range(0, 33, 4))))
ax_combined.set_ylim(-0.05, 1.05)
ax_combined.set_ylabel("performance ", fontsize=14)
ax_combined.set_xlabel("layers", fontsize=12)
ax_combined.legend(fontsize=10, loc="best")
ax_combined.set_title("sliding window ablation", fontsize=15)
ax_combined.grid(True, alpha=0.4)

gs_combined.tight_layout(fig_combined)
plt.savefig("./results/plots/averaged_profiles_ablations_w4.png", dpi=200)


######################

# Language tasks stratified by difficulty - sliding window

######################

# Define easy and hard tasks
easy_tasks = ["hellaswag", "piqa", "arc_easy"]
hard_tasks = ["mmlu", "commonsense_qa", "winogrande"]
# hard_tasks = [
#     task
#     for task in language_perf.keys()
#     if task not in easy_tasks
#     and language_task_type[task] == "lik"
#     and task != "arc_challenge"
# ]

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
    6, 17, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
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

# Add standard deviation for hard tasks
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

ax_stratified.set_xticks(np.array(list(range(0, 33, 4))))
# ax_stratified.set_ylim(-0.05, 1.05)
ax_stratified.set_ylabel("performance ", fontsize=14)
ax_stratified.set_xlabel("layers", fontsize=12)
ax_stratified.legend(fontsize=9, loc="best")
ax_stratified.set_title("sliding window ablation", fontsize=15)
ax_stratified.grid(True, alpha=0.4)

gs_stratified.tight_layout(fig_stratified)
plt.savefig(
    "./results/plots/averaged_profiles_language_difficulty_stratified_w4.png", dpi=200
)


##########################################

# PR0GRESSIVE

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

# Plot easy language tasks (progressive, dashed green line)
sns.lineplot(
    x=language_layers_progr,
    y=lang_easy_avg_progr,
    marker="s",
    linewidth=1,
    linestyle="--",
    label="Language tasks (easy)\nhellaswag, piqa, arc_easy",
    color="green",
    ax=ax_combined,
)

# Add standard deviation for easy tasks
ax_combined.fill_between(
    language_layers_progr,
    lang_easy_avg_progr - lang_easy_std_progr,
    lang_easy_avg_progr + lang_easy_std_progr,
    color="green",
    alpha=0.2,
)

# Plot hard language tasks (progressive, solid line)
sns.lineplot(
    x=language_layers_progr,
    y=lang_hard_avg_progr,
    marker="^",
    linewidth=1,
    linestyle="-",
    label="Language tasks (hard)\nall others",
    color="darkgreen",
    ax=ax_combined,
)

# Add standard deviation for hard tasks
ax_combined.fill_between(
    language_layers_progr,
    lang_hard_avg_progr - lang_hard_std_progr,
    lang_hard_avg_progr + lang_hard_std_progr,
    color="darkgreen",
    alpha=0.2,
)

ax_combined.set_xticks(np.array(list(range(0, 33, 4))))
ax_combined.set_ylabel("performance ", fontsize=14)
ax_combined.set_xlabel("layers", fontsize=12)
ax_combined.legend(fontsize=9, loc="best")
ax_combined.set_title("n° ablated layers from input", fontsize=15)
ax_combined.grid(True, alpha=0.4)

# *************************************************************

gs_combined.tight_layout(fig_combined)
plt.savefig("./results/plots/averaged_profiles_ablations_progressive_std.png", dpi=200)


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
    6, 17, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
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
    label="Language tasks\n(all combined)",
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

ax_combined_all.set_xticks(np.array(list(range(0, 33, 4))))
ax_combined_all.set_ylabel("performance ", fontsize=14)
ax_combined_all.set_xlabel("layers", fontsize=12)
ax_combined_all.legend(fontsize=10, loc="best")
ax_combined_all.set_title("n° ablated layers from input", fontsize=15)
ax_combined_all.grid(True, alpha=0.4)

gs_combined_all.tight_layout(fig_combined_all)
plt.savefig("./results/plots/averaged_profiles_ablations_progressive_std.png", dpi=200)


######################

# Plot all tasks VL sliding window

######################

fig_vl_all = plt.figure(figsize=(4.5, 4))
gs_vl_all = GridSpec(1, 1)

ax_vl_all = fig_vl_all.add_subplot(gs_vl_all[0])
ax_vl_all.axvspan(6, 17, alpha=0.15, color="goldenrod", zorder=0)

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

ax_vl_all.set_xticks(np.array(list(range(0, 33, 4))))
ax_vl_all.set_ylim(-0.05, 1.05)
ax_vl_all.set_ylabel("performance", fontsize=14)
ax_vl_all.set_xlabel("layers", fontsize=12)
ax_vl_all.legend(fontsize=10, loc="best")
ax_vl_all.set_title("VL tasks - sliding window", fontsize=15)
ax_vl_all.grid(True, alpha=0.4)
ax_vl_all.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

gs_vl_all.tight_layout(fig_vl_all)
plt.savefig("./results/plots/all_tasks_vl_sliding_window_w4.png", dpi=200)


######################

# Plot all tasks Language sliding window

######################

fig_lang_all = plt.figure(figsize=(4.5, 4))
gs_lang_all = GridSpec(1, 1)

ax_lang_all = fig_lang_all.add_subplot(gs_lang_all[0])
ax_lang_all.axvspan(6, 17, alpha=0.15, color="goldenrod", zorder=0)

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

ax_lang_all.set_xticks(np.array(list(range(0, 33, 4))))
# ax_lang_all.set_ylim(-0.15, 1.05)
ax_lang_all.set_ylabel("performance", fontsize=14)
ax_lang_all.set_xlabel("layers", fontsize=12)
ax_lang_all.legend(fontsize=10, loc="best")
ax_lang_all.set_title("Language tasks - sliding window", fontsize=15)
ax_lang_all.grid(True, alpha=0.4)
ax_lang_all.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

gs_lang_all.tight_layout(fig_lang_all)
plt.savefig("./results/plots/all_tasks_language_sliding_window_w4.png", dpi=200)


######################

# Plot all tasks VL progressive

######################

fig_vl_prog = plt.figure(figsize=(4.5, 4))
gs_vl_prog = GridSpec(1, 1)

ax_vl_prog = fig_vl_prog.add_subplot(gs_vl_prog[0])
ax_vl_prog.axvspan(6, 17, alpha=0.15, color="goldenrod", zorder=0)

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

ax_vl_prog.set_xticks(np.array(list(range(0, 33, 4))))
# ax_vl_prog.set_ylim(-0.05, 1.05)
ax_vl_prog.set_ylabel("performance", fontsize=14)
ax_vl_prog.set_xlabel("layers", fontsize=12)
ax_vl_prog.legend(fontsize=10, loc="best")
ax_vl_prog.set_title("VL tasks - progressive", fontsize=15)
ax_vl_prog.grid(True, alpha=0.4)
ax_vl_prog.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

gs_vl_prog.tight_layout(fig_vl_prog)
plt.savefig("./results/plots/all_tasks_vl_progressive.png", dpi=200)


######################

# Plot all tasks Language generative

######################

# Get generative language tasks
gen_tasks = {
    task: data
    for task, data in language_perf.items()
    if language_task_type[task] == "gen"
}

language_perf.keys()

if len(gen_tasks) > 0:
    fig_lang_gen = plt.figure(figsize=(4.5, 4))
    gs_lang_gen = GridSpec(1, 1)

    ax_lang_gen = fig_lang_gen.add_subplot(gs_lang_gen[0])
    ax_lang_gen.axvspan(6, 17, alpha=0.15, color="goldenrod", zorder=0)

    for task, data in gen_tasks.items():
        # Normalize data
        norm_acc = (np.array(data) - language_random[task]) / (
            language_full[task] - language_random[task]
        )

        sns.lineplot(
            x=shifted_x_lang,
            y=norm_acc,
            marker="^",
            linewidth=1,
            label=task,
            alpha=alpha,
            ax=ax_lang_gen,
        )

    ax_lang_gen.set_xticks(np.array(list(range(0, 33, 4))))
    ax_lang_gen.set_ylim(-0.05, 1.05)
    ax_lang_gen.set_ylabel("performance", fontsize=14)
    ax_lang_gen.set_xlabel("layers", fontsize=12)
    ax_lang_gen.legend(fontsize=10, loc="best")
    ax_lang_gen.set_title("Language generative tasks - sliding window", fontsize=15)
    ax_lang_gen.grid(True, alpha=0.4)
    ax_lang_gen.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    gs_lang_gen.tight_layout(fig_lang_gen)
    plt.savefig("./results/plots/all_tasks_language_generative.png", dpi=200)


######################

# Open-ended tasks plot (Flickr + Language generation)

######################


def get_data_openended_vl(base_path, window_size):
    """Load Flickr30k and COCO caption data for sliding window."""
    tasks_metric = {
        "flickr30k": ("flickr_CIDEr,none", "flickr30k_test"),
        "coco2017_cap_val": ("coco_CIDEr,none", "coco2017_cap_val"),
    }

    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    perf = {}

    for task, (metric, result_key) in tasks_metric.items():
        perf[task] = []
        for layer in layers:
            file_path = f"{base_path}/layer_skip_eval_mm/llava-7b-lbasile_sw{window_size}/{task}/{task}_L{layer}_W{window_size}.json"
            try:
                with open(file_path, "r") as f:
                    file = json.load(f)
                    perf[task].append(file["results"][result_key][metric])
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}")
                continue

    full_baselines = {}
    for task, (metric, result_key) in tasks_metric.items():
        file_path = f"{base_path}/layer_skip_eval_mm/llava-7b-lbasile_sw0/{task}/{task}_L0_W0.json"
        try:
            with open(file_path, "r") as f:
                file = json.load(f)
                full_baselines[task] = file["results"][result_key][metric]
        except FileNotFoundError:
            print(f"Warning: Baseline file not found {file_path}")
            continue

    random_baselines = {
        "flickr30k": None,
        "coco2017_cap_val": None,
    }

    # Load random baselines from sw32 where available
    for task, (metric, result_key) in tasks_metric.items():
        try:
            with open(
                f"{base_path}/layer_skip_eval_mm/llava-7b-lbasile_sw32/{task}/{task}_L0_W32.json",
                "r",
            ) as f:
                file = json.load(f)
                random_baselines[task] = file["results"][result_key][metric]
        except FileNotFoundError:
            print(f"Warning: Random baseline not found for {task}, using 0")
            random_baselines[task] = 0
            continue

    return perf, random_baselines, full_baselines, layers


def get_data_openended_lang(base_path, window_size):
    """Load nq_open and triviaqa data for sliding window."""
    tasks_fs = {
        "nq_open": 5,
        "triviaqa": 1,
    }

    tasks_metric = {
        "nq_open": "exact_match,remove_whitespace",
        "triviaqa": "exact_match,remove_whitespace",
    }

    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    perf = {}

    for task, few_shot in tasks_fs.items():
        perf[task] = []
        for layer in layers:
            file_path = f"{base_path}/layer_skip_eval/llava-1.5-7b-hf_sw{window_size}/{task}/{task}_skip_{layer:02d}_{few_shot}shot.json"
            try:
                with open(file_path, "r") as f:
                    file = json.load(f)
                    perf[task].append(file["results"][task][tasks_metric[task]])
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}")
                continue

    baselines = {}
    for task, few_shot in tasks_fs.items():
        file_path = f"{base_path}/layer_skip_eval/llava-1.5-7b-hf/{task}/{task}_skip_00_{few_shot}shot.json"
        try:
            with open(file_path, "r") as f:
                file = json.load(f)
                baselines[task] = file["results"][task][tasks_metric[task]]
        except FileNotFoundError:
            print(f"Warning: Baseline file not found {file_path}")
            continue

    random_baselines = {
        "nq_open": 0,
        "triviaqa": 0,
    }

    return perf, random_baselines, baselines, layers


# Load open-ended task data
base_path_root = (
    "/home/diego/Documents/area_science/ricerca/open/llava_finetuning/results"
)
openended_vl_perf, openended_vl_random, openended_vl_full, openended_vl_layers = (
    get_data_openended_vl(base_path_root, window_size=4)
)

(
    openended_lang_perf,
    openended_lang_random,
    openended_lang_full,
    openended_lang_layers,
) = get_data_openended_lang(base_path_root, window_size=4)

# Create the open-ended tasks plot
fig_openended = plt.figure(figsize=(4.5, 4))
gs_openended = GridSpec(1, 1)

ax_openended = fig_openended.add_subplot(gs_openended[0])
ax_openended.axvspan(
    6, 17, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase"
)

# Normalize and average Flickr30k and COCO
vl_gen_tasks_norm = []
if "flickr30k" in openended_vl_perf and len(openended_vl_perf["flickr30k"]) > 0:
    flickr_data = np.array(openended_vl_perf["flickr30k"])
    flickr_norm = (flickr_data - openended_vl_random["flickr30k"]) / (
        openended_vl_full["flickr30k"] - openended_vl_random["flickr30k"]
    )
    vl_gen_tasks_norm.append(flickr_norm)

if (
    "coco2017_cap_val" in openended_vl_perf
    and len(openended_vl_perf["coco2017_cap_val"]) > 0
):
    coco_data = np.array(openended_vl_perf["coco2017_cap_val"])
    coco_norm = (coco_data - openended_vl_random["coco2017_cap_val"]) / (
        openended_vl_full["coco2017_cap_val"] - openended_vl_random["coco2017_cap_val"]
    )
    vl_gen_tasks_norm.append(coco_norm)


if len(vl_gen_tasks_norm) > 0:
    vl_gen_avg = np.mean(vl_gen_tasks_norm, axis=0)
    vl_gen_std = np.std(vl_gen_tasks_norm, axis=0)
    shifted_layers_vl = np.array(openended_vl_layers) + 4 / 2

    sns.lineplot(
        x=shifted_layers_vl,
        y=vl_gen_avg,
        marker="o",
        linewidth=1,
        label="Image captioning",
        color="purple",
        ax=ax_openended,
    )

    ax_openended.fill_between(
        shifted_layers_vl,
        vl_gen_avg - vl_gen_std,
        vl_gen_avg + vl_gen_std,
        color="purple",
        alpha=0.2,
    )

# Average language generation tasks
lang_gen_tasks_norm = []
for task in ["nq_open", "triviaqa"]:
    if task in openended_lang_perf and len(openended_lang_perf[task]) > 0:
        task_data = np.array(openended_lang_perf[task])
        task_norm = (task_data - openended_lang_random[task]) / (
            openended_lang_full[task] - openended_lang_random[task]
        )
        lang_gen_tasks_norm.append(task_norm)

if len(lang_gen_tasks_norm) > 0:
    lang_gen_avg = np.mean(lang_gen_tasks_norm, axis=0)
    lang_gen_std = np.std(lang_gen_tasks_norm, axis=0)
    shifted_layers_lang = np.array(openended_lang_layers) + 4 / 2

    sns.lineplot(
        x=shifted_layers_lang,
        y=lang_gen_avg,
        marker="s",
        linewidth=1,
        label="Language generation",
        color="green",
        ax=ax_openended,
    )

    ax_openended.fill_between(
        shifted_layers_lang,
        lang_gen_avg - lang_gen_std,
        lang_gen_avg + lang_gen_std,
        color="green",
        alpha=0.2,
    )

ax_openended.set_xticks(np.array(list(range(0, 33, 4))))
ax_openended.set_ylim(-0.05, 1.05)
ax_openended.set_ylabel("performance", fontsize=14)
ax_openended.set_xlabel("layers", fontsize=12)
ax_openended.legend(fontsize=10, loc="best")
ax_openended.set_title("Open-ended generation tasks", fontsize=15)
ax_openended.grid(True, alpha=0.4)

gs_openended.tight_layout(fig_openended)
plt.savefig("./results/plots/openended_tasks_sliding_window_w4.png", dpi=200)
print("Open-ended tasks plot saved!")
