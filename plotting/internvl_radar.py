"""Radar plot: InternVL2-8B vs InternVL2-8B-Pretrain vs LLaVA-1.5-7B (reference)."""

import json
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVALS_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations/evals")
OUTPUT_PATH = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/plots/internvl_evals/internvl_radar.png"
)

# ---------------------------------------------------------------------------
# Benchmark registry  (ordered for visual layout on the radar)
#   key        → human-readable name used everywhere else
#   dir        → subfolder under <model>/lmms_eval/
#   key        → the metric key inside the last line of the results .jsonl
#   max_val    → denominator to normalise the raw score into [0, 1]
#   abbrev     → label on the radar axis
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "MME": {
        "dir": "mme",
        "key": "mme_perception_score,none",
        "max_val": 2000,
        "abbrev": "MME",
    },
    "GQA": {"dir": "gqa", "key": "exact_match,none", "max_val": 1, "abbrev": "GQA",},
    "VQAv2": {
        "dir": "vqav2_val",
        "key": "exact_match,none",
        "max_val": 1,
        "abbrev": "VQAv2",
    },
    "VizWiz": {
        "dir": "vizwiz_vqa_val",
        "key": "exact_match,none",
        "max_val": 1,
        "abbrev": "VizWiz",
    },
    "TextVQA": {
        "dir": "textvqa_val",
        "key": "exact_match,none",
        "max_val": 1,
        "abbrev": "TextVQA",
    },
    "ScienceQA": {
        "dir": "scienceqa_img",
        "key": "exact_match,none",
        "max_val": 1,
        "abbrev": "SQA",
    },
    "POPE": {
        "dir": "pope",
        "key": "pope_f1_score,none",
        "max_val": 1,
        "abbrev": "POPE",
    },
    "SEEDBench": {
        "dir": "seedbench",
        "key": "seed_all,none",
        "max_val": 1,
        "abbrev": "SEED",
    },
    "LLaVA-Wild": {
        "dir": "llava_in_the_wild",
        "key": "gpt_eval_llava_all,none",
        "max_val": 100,
        "abbrev": "LLaVA-W",
    },
    "MMBench-EN": {
        "dir": "mmbench_en_dev",
        "key": "gpt_eval_score,none",
        "max_val": 100,
        "abbrev": "MMB-EN",
    },
    "MMBench-CN": {
        "dir": "mmbench_cn_dev",
        "key": "gpt_eval_score,none",
        "max_val": 100,
        "abbrev": "MMB-CN",
    },
}

# InternVL model folder names inside EVALS_ROOT
INTERNVL_MODELS = {
    "InternVL2-8B": "InternVL2-8B",
    "InternVL2-8B-Pretrain": "InternVL2-8B-Pretrain",
}

# LLaVA-1.5-7B paper scores (MODEL_ZOO.md), normalised to [0, 1].
# MME perception / 2000; everything else is reported as a percentage → / 100.
LLAVA_REF = {
    "MME": 1510.7 / 2000,
    "GQA": 62.0 / 100,
    "VQAv2": 78.5 / 100,
    "VizWiz": 50.0 / 100,
    "TextVQA": 58.2 / 100,
    "ScienceQA": 66.8 / 100,
    "POPE": 85.9 / 100,
    "SEEDBench": 58.6 / 100,
    "LLaVA-Wild": 65.4 / 100,
    "MMBench-EN": 64.3 / 100,
    "MMBench-CN": 58.3 / 100,
}

# ---------------------------------------------------------------------------
# Visual style per model
# ---------------------------------------------------------------------------
MODEL_STYLES = {
    "InternVL2-8B": {"color": "#1f77b4", "ls": "-", "marker": "o", "fill_alpha": 0.10},
    "InternVL2-8B-Pretrain": {
        "color": "#ff7f0e",
        "ls": "-",
        "marker": "s",
        "fill_alpha": 0.08,
    },
    "LLaVA-1.5-7B": {"color": "#9467bd", "ls": "--", "marker": "^", "fill_alpha": 0.15},
}

# Plot order: reference first (background), then the two InternVL variants
PLOT_ORDER = ["LLaVA-1.5-7B", "InternVL2-8B-Pretrain", "InternVL2-8B"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _read_last_jsonl(path: Path) -> dict:
    """Return the last JSON object in a .jsonl file (the summary line)."""
    with open(path) as fh:
        for line in fh:
            last = line
    return json.loads(last)


def load_internvl_scores() -> dict[str, dict[str, float]]:
    """Load and normalise every benchmark score for each InternVL model."""
    scores: dict[str, dict[str, float]] = {}

    for label, folder in INTERNVL_MODELS.items():
        scores[label] = {}
        for bench_name, info in BENCHMARKS.items():
            bench_dir = EVALS_ROOT / folder / "lmms_eval" / info["dir"]
            jsonl = next(bench_dir.glob("*results*.jsonl"))
            summary = _read_last_jsonl(jsonl)
            scores[label][bench_name] = summary[info["key"]] / info["max_val"]

    return scores


def load_all_scores() -> dict[str, dict[str, float]]:
    """Combine InternVL scores with the hardcoded LLaVA reference."""
    scores = load_internvl_scores()
    scores["LLaVA-1.5-7B"] = dict(LLAVA_REF)
    return scores


# ---------------------------------------------------------------------------
# Radar plot
# ---------------------------------------------------------------------------
def make_radar(scores: dict[str, dict[str, float]], save_path: Path) -> None:
    categories = list(BENCHMARKS.keys())
    n = len(categories)
    angles = [i / n * 2 * pi for i in range(n)] + [0]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))

    for label in PLOT_ORDER:
        style = MODEL_STYLES[label]
        values = [scores[label][c] for c in categories] + [scores[label][categories[0]]]

        ax.plot(
            angles,
            values,
            linestyle=style["ls"],
            color=style["color"],
            marker=style["marker"],
            markersize=5,
            linewidth=2,
            label=label,
        )
        ax.fill(angles, values, alpha=style["fill_alpha"], color=style["color"])

    # Axis formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([BENCHMARKS[c]["abbrev"] for c in categories], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="grey")
    ax.tick_params(axis="x", pad=12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    # plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scores = load_all_scores()

    # Sanity-print normalised values so you can eyeball them before the plot
    print(f"{'Benchmark':<14}", end="")
    for label in PLOT_ORDER:
        print(f"{label:>26}", end="")
    print()
    print("-" * (14 + 26 * len(PLOT_ORDER)))
    for b in BENCHMARKS:
        print(f"{BENCHMARKS[b]['abbrev']:<14}", end="")
        for label in PLOT_ORDER:
            print(f"{scores[label][b]:>26.4f}", end="")
        print()

    make_radar(scores, OUTPUT_PATH)
