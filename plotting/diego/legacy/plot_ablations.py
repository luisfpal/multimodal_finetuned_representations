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

base_path = "/home/diego/Documents/area_science/ricerca/open/llava_finetuning/results/layer_skip_eval"


tasks_fs = {
    "commonsense_qa": 0,
    "hellaswag": 0,
    "mmlu": 0,
    "nq_open": 5,
    "triviaqa": 1,
    # "ifeval": 0,
}

tasks_metric = {
    "commonsense_qa": "acc,none",
    "hellaswag": "acc,none",
    "mmlu": "acc,none",
    "nq_open": "exact_match,remove_whitespace",
    "triviaqa": "exact_match,remove_whitespace",
    "ifeval": "inst_level_strict_acc,none",
}

tasks_type = {
    "commonsense_qa": "lik",
    "hellaswag": "lik",
    "mmlu": "lik",
    "nq_open": "gen",
    "triviaqa": "gen",
    "ifeval": "gen",
}


gen_tasks = {"nq_open", "ifeval", "triviaqa"}
lik_tasks = {"commonsense_qa", "hellaswag", "mmlu"}

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
    if task == "nq_open":
        few_shot = 0
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
}


fig = plt.figure(figsize=(8.5, 3.8))
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[0])
ax.axvspan(6, 16, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase")
ax.axvspan(16, 25, alpha=0.15, color="coral", zorder=0, label="generation phase")
x = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
for task, data in perf.items():
    print(task)
    if tasks_type[task] == "lik":
        norm_acc = (np.array(data) - random_baselines[task]) / (
            baselines[task] - random_baselines[task]
        )
        ax.plot(
            x,
            norm_acc,
            marker=".",
            label=f"{task}",
        )
ax.set_xticks(x)
ax.set_ylabel("normalized performance", fontsize=10)
ax.set_xlabel("layers")
ax.legend(fontsize=8)
ax.set_title("Likelihood based")

ax = fig.add_subplot(gs[1])
ax.axvspan(6, 16, alpha=0.15, color="goldenrod", zorder=0, label="abstraction phase")
ax.axvspan(16, 25, alpha=0.15, color="coral", zorder=0, label="generation phase")
for task, data in perf.items():
    if tasks_type[task] == "gen":
        norm_acc = (np.array(data) - random_baselines[task]) / (
            baselines[task] - random_baselines[task]
        )
        ax.plot(
            x,
            norm_acc,
            marker=".",
            label=f"{task}",
        )
ax.set_xticks(x)
ax.set_ylabel("normalized performance", fontsize=10)
ax.set_xlabel("layers")
ax.legend(fontsize=8)
ax.set_title("Open-ended generation")
gs.tight_layout(fig)
plt.savefig("./results/plots/sliding_window.png", dpi=200)


# # Build a long-format DataFrame
# rows = []
# for task, data in perf.items():
#     for i, layer in enumerate(layers):
#         rows.append(
#             {
#                 "task": task,
#                 "task_type": tasks_type[task],
#                 "layer": layer,
#                 "normalized_perf": (data[i] - random_baselines[task])
#                 / (baselines[task] - random_baselines[task]),
#             }
#         )

# df = pd.DataFrame(rows)

# # Plot with seaborn
# fig, ax = plt.subplots(figsize=(4.5, 4.5))
# # sns.lineplot(
# #     data=df,
# #     x="layer",
# #     y="normalized_perf",
# #     hue="task",           # Different color for each task
# #     style="task_type",    # Different linestyle for "lik" vs "gen"
# #     markers=True,
# #     ax=ax
# # )


# sns.lineplot(
#     data=df,
#     x="layer",
#     y="normalized_perf",
#     hue="task",
#     style="task_type",
#     markers={"lik": "o", "gen": "s"},  # Different markers per task_type
#     dashes={"lik": "", "gen": (2, 2)},
#     markersize=8,  # Global marker size
#     ax=ax,
# )


# ax.set_xticks(layers)
# ax.set_ylabel("normalized performance", fontsize=14)
# ax.set_xlabel("layers")
# ax.legend(fontsize=8)
# ax.set_title("Sliding window")
# plt.tight_layout()


# type_to_line = []

# fig = plt.figure(figsize=(4.5, 4.5))
# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# x = [0, 4, 8, 12, 16, 20, 25]
# for task, data in perf.items():
#     ax.plot(
#         x,
#         np.array(data) / baselines[task],
#         marker=".",
#         label=f"{task}",
#         linestyle=type_to_line[tasks_type[task]],
#     )
# ax.set_xticks(x)

# ax.set_ylabel("normalized performance", fontsize=14)
# ax.set_xlabel("layers")
# ax.legend(fontsize=8)
# ax.set_title("Sliding window")
# gs.tight_layout(fig)
