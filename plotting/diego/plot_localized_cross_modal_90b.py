"""Plot Spearman correlation between image reliance and localized cross-modal advantage for Llama-3-90B across benchmark tasks."""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

DATA_DIR = Path(
    "./results/layer_skip_eval_mm/Meta-Llama-3-90B-instruct_cross_modal_win10_65"
)
OUTPUT_PATH = Path(
    "./results/plots/spearman_image_reliance/localized_cross_modal_90b.pdf"
)

# Metric key per dataset: (metric_key, result_key)
TASK_METRIC: Dict[str, Tuple[str, str]] = {
    "chartqa": ("relaxed_overall,none", "chartqa"),
    "countbench": ("acc,none", "countbench"),
    "cv_bench": ("cv_bench_acc,none", "cv_bench"),
    "gqa": ("exact_match,none", "gqa"),
    "mmbench_cn_dev": ("gpt_eval_score,none", "mmbench_cn_dev"),
    "mmbench_en_dev": ("gpt_eval_score,none", "mmbench_en_dev"),
    "mme": ("mme_perception_score,none", "mme"),
    "mmmu_val": ("mmmu_acc,none", "mmmu_val"),
    "naturalbench": ("naturalbench_G_ACC,none", "naturalbench"),
    "scienceqa_img": ("exact_match,none", "scienceqa_img"),
    "seedbench": ("seed_image,none", "seedbench"),
    "vqav2_val": ("exact_match,none", "vqav2_val"),
}

# Display name for each dataset (matches the reference figure).
DISPLAY_NAME = {
    "chartqa": "ChartQA",
    "countbench": "CountBench",
    "cv_bench": "CVBench",
    "gqa": "GQA",
    "mmbench_cn_dev": "MMB-CN",
    "mmbench_en_dev": "MMB-EN",
    "mme": "MME",
    "mmmu_val": "MMMU",
    "naturalbench": "Naturalbench",
    "scienceqa_img": "SQA",
    "seedbench": "SEED",
    "vqav2_val": "VQAv2",
}

# Image reliance values estimated from the LLaVA-OneVision-1.5-4B reference
# figure (x-axis position of each dataset's marker).
IMAGE_RELIANCE = {
    "mmmu_val": 0.16,
    "scienceqa_img": 0.20,
    "cv_bench": 0.32,
    "gqa": 0.41,
    "seedbench": 0.48,
    "vqav2_val": 0.50,
    "mme": 0.55,
    "mmbench_en_dev": 0.73,
    "mmbench_cn_dev": 0.85,
    "chartqa": 0.84,
    "countbench": 0.87,
    "naturalbench": 1.00,
}

# Marker style by category (mirrors reference figure: open-ended vs MCQ).
OPEN_ENDED = {"gqa", "vqav2_val", "chartqa", "countbench"}
MCQ = {
    "mmbench_en_dev",
    "mmbench_cn_dev",
    "mme",
    "mmmu_val",
    "scienceqa_img",
    "seedbench",
    "cv_bench",
    "naturalbench",
}


def read_metric(path: Path, result_key: str, metric_key: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["results"][result_key][metric_key])


def localized_advantage(task: str) -> float:
    metric_key, result_key = TASK_METRIC[task]
    keep = read_metric(DATA_DIR / task / f"{task}_keep.json", result_key, metric_key)
    remove = read_metric(
        DATA_DIR / task / f"{task}_remove.json", result_key, metric_key
    )
    return (keep - remove) / keep


sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

tasks = sorted(TASK_METRIC)
xs = np.array([IMAGE_RELIANCE[t] for t in tasks])
ys = np.array([localized_advantage(t) for t in tasks])

rho, pval = spearmanr(xs, ys)

fig, ax = plt.subplots(figsize=(7.0, 5.2))

color_open = "#e0b251"
color_mcq = "#6e94b8"
color_other = "#d8825a"

for t, x, y in zip(tasks, xs, ys):
    if t in OPEN_ENDED:
        color, marker, size = color_open, "o", 70
    elif t in MCQ:
        color, marker, size = color_mcq, "s", 55
    else:
        color, marker, size = color_other, "^", 60
    ax.scatter(x, y, s=size, color=color, marker=marker, zorder=3)

# Label offsets per dataset (in axis fraction of x-range / data y units).
# Tuned to avoid overlaps similar to the reference figure.
label_offsets = {
    "mmmu_val": (0.012, 0.0015, "left"),
    "scienceqa_img": (0.012, -0.0015, "left"),
    "cv_bench": (-0.012, -0.0025, "right"),
    "gqa": (0.012, -0.0015, "left"),
    "seedbench": (0.0, -0.004, "center"),
    "vqav2_val": (0.012, 0.0, "left"),
    "mme": (0.012, -0.001, "left"),
    "mmbench_en_dev": (0.012, 0.0, "left"),
    "mmbench_cn_dev": (0.012, -0.002, "left"),
    "chartqa": (0.012, 0.002, "left"),
    "countbench": (-0.012, -0.002, "right"),
    "naturalbench": (-0.012, 0.002, "right"),
}

for t, x, y in zip(tasks, xs, ys):
    dx, dy, ha = label_offsets.get(t, (0.012, 0.0, "left"))
    ax.annotate(
        DISPLAY_NAME[t],
        xy=(x, y),
        xytext=(x + dx, y + dy),
        ha=ha,
        va="center",
        fontsize=11,
    )

ax.axhline(0.0, color="gray", linewidth=0.7, alpha=0.6)

ax.set_xlabel("Image reliance", fontsize=14)
ax.set_ylabel("Localized ft advantage", fontsize=14)
ax.set_title("Llama-3.2-Vision-90B", fontsize=15)
ax.tick_params(axis="both", labelsize=11)

# Spearman annotation box (bottom-right).
ax.text(
    0.98,
    0.04,
    f"Spearman\n$\\varrho$ = {rho:.2f}\n$p$ = {pval:.3f}",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.5"),
)

fig.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=300, format="pdf")
print(f"Saved to {OUTPUT_PATH}")
