import json
import os
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from _tueplots_style import get_rc, out_path


EVALS_ROOT = Path("/u/dssc/lbasile/scratch/plm/evals")

BENCHMARKS_EXT = {
    "SugarCrepe": {"dir": "../compositionality/sugarcrepe", "key": "mean", "max_val": 1, "abbrev": "SugarCrepe"},
    'ARO': {'dir': '../compositionality/aro', 'key': 'mean', 'max_val': 1, 'abbrev': 'ARO'},
    "MME": {"dir": "mme", "key": "total", "max_val": 2800, "abbrev": "MME"},
    "GQA": {"dir": "gqa", "key": "exact_match,none", "max_val": 1, "abbrev": "GQA"},
    "ScienceQA": {"dir": "scienceqa_img", "key": "exact_match,none", "max_val": 1, "abbrev": "SQA"},
    "POPE": {"dir": "pope", "key": "pope_accuracy,none", "max_val": 1, "abbrev": "POPE"},
    "CVBENCH": {"dir": "cv_bench", "key": "cv_bench_acc,none", "max_val": 1, "abbrev": "CVBench"},
    "SEEDBench": {"dir": "seedbench", "key": "seed_all,none", "max_val": 1, "abbrev": "SEED"},
    "MMBench-EN": {"dir": "mmbench_en_dev", "key": "gpt_eval_score,none", "max_val": 100, "abbrev": "MMB-EN"},
    "MMBench-CN": {"dir": "mmbench_cn_dev", "key": "gpt_eval_score,none", "max_val": 100, "abbrev": "MMB-CN"},
    "VQAv2": {"dir": "vqav2_val", "key": "exact_match,none", "max_val": 1, "abbrev": "VQAv2"},
    "Naturalbench": {"dir": "naturalbench", "key": "naturalbench_G_ACC,none", "max_val": 1, "abbrev": "Naturalbench"},
    "MMMU": {"dir": "mmmu_val", "key": "mmmu_acc,none", "max_val": 1, "abbrev": "MMMU"},
    "ChartQA": {"dir": "chartqa", "key": "relaxed_overall,none", "max_val": 1, "abbrev": "ChartQA"},
    "MMStar": {"dir": "mmstar", "key": "average,none", "max_val": 1, "abbrev": "MMStar"},
    "MMMU-Pro-Std": {"dir": "mmmu_pro_standard", "key": "mmmu_acc,none", "max_val": 1, "abbrev": "MMMU-Pro-S"},
    "RealWorldQA": {"dir": "realworldqa", "key": "exact_match,flexible-extract", "max_val": 1, "abbrev": "RWQA"},
    "CountBench": {"dir": "countbench", "key": "acc,none", "max_val": 1, "abbrev": "CountBench"},
    "COCO": {"dir": "coco2017_cap_val", "key": "coco_CIDEr,none", "max_val": 1, "abbrev": "COCO"},
    "Flickr": {"dir": "flickr30k", "key": "flickr_CIDEr,none", "max_val": 1, "abbrev": "Flickr"},
}

# CORRELATION_REMOVE = {"COCO", "Flickr", "VizWiz", "ARO", "DocVQA", "InfoVQA", "MMMU", "MMVP", "Vlmsareblind", "BLINK", "MMMU-Pro-S", "BiVLC"}
CORRELATION_REMOVE = {"COCO", "Flickr", "VizWiz", "ARO-Order", "ARO-VGR", "ARO-VGA", "DocVQA", "InfoVQA", "AI2D", "MMVP", "Vlmsareblind", "BLINK", "BiVLC"}
IMAGE_RELIANCE_BASELINE_MODELS = ["LLaVA-OneVision-1.5-8B-Instruct", "InternVL2-8B", "pixtral-12b", "cambrian-8b"]

PAIRS = {
    "llava_7b": {
        "base": "llava_7b",
        "noimage": "llava_7b_noimage",
        "loc": "llava_7b_6_17",
        "comp": "llava_7b_6_17_complementary",
        "title": "LLaVA-7B",
    },
    "LLaVA-OneVision-1.5-4B": {
        "base": "LLaVA-OneVision-1.5-4B",
        "noimage": "LLaVA-OneVision-1.5-4B_noimage",
        "loc": "LLaVA-OneVision-1.5-4B-8-26",
        "comp": "LLaVA-OneVision-1.5-4B-8-26-comp",
        "title": "OneVision-4B",
    },
}


def load_model_scores(models):
    scores = {model: {} for model in models}

    for model in models:
        model_path = EVALS_ROOT / model / "lmms_eval"
        if not model_path.exists():
            print(f"Warning: lmms_eval not found for {model}")
            continue

        for name, info in BENCHMARKS_EXT.items():
            bench_dir = model_path / info["dir"]
            if not bench_dir.exists():
                scores[model][name] = 0
                continue

            results_files = [f for f in os.listdir(bench_dir) if f.endswith(".jsonl") and "results" in f]
            if not results_files:
                scores[model][name] = 0
                continue

            if "results_files" in info:
                split_scores = []
                for rf in info["results_files"]:
                    target = bench_dir / rf
                    if not target.exists():
                        continue
                    with open(target, "r") as f:
                        lines = f.readlines()
                    if not lines:
                        continue
                    result = json.loads(lines[-1])
                    split_scores.append(result.get(info["key"], 0) / info["max_val"])
                scores[model][name] = np.mean(split_scores) if split_scores else 0
                continue

            # ARO (aggregate) averages the 4 per-task accuracies (coco_order, flickr30k_order, vg_relation, vg_attribution).
            if name == 'ARO':
                aro_scores = []
                for rf in sorted(results_files):
                    with open(bench_dir / rf, 'r') as af:
                        aro_lines = af.readlines()
                    if not aro_lines:
                        continue
                    aro_result = json.loads(aro_lines[-1])
                    aro_scores.append(aro_result.get('accuracy', 0) / info['max_val'])
                scores[model][name] = np.mean(aro_scores) if aro_scores else 0
                continue

            if "results_file" in info:
                target = bench_dir / info["results_file"]
                if not target.exists():
                    scores[model][name] = 0
                    continue
                with open(target, "r") as f:
                    lines = f.readlines()
                if not lines:
                    scores[model][name] = 0
                    continue
                last_result = json.loads(lines[-1])
                scores[model][name] = last_result.get(info["key"], 0) / info["max_val"]
                continue

            with open(bench_dir / results_files[0], "r") as f:
                lines = f.readlines()
                if not lines:
                    scores[model][name] = 0
                    continue
                if name == "BLINK":
                    blink_scores = []
                    for line in lines:
                        result = json.loads(line)
                        blink_scores.append(result.get(info["key"], 0) / info["max_val"])
                    scores[model][name] = np.mean(blink_scores)
                    continue
                if info.get("aggregate_lines") == "mean":
                    vals = []
                    for line in lines:
                        result = json.loads(line)
                        if info["key"] in result:
                            vals.append(result[info["key"]] / info["max_val"])
                    scores[model][name] = np.mean(vals) if vals else 0
                    continue
                last_result = json.loads(lines[-1])

                if name == "MME":
                    val = last_result.get("mme_perception_score,none", 0) + last_result.get("mme_cognition_score,none", 0)
                else:
                    val = last_result.get(info["key"], 0)

                scores[model][name] = val / info["max_val"]

    return scores


def compute_image_reliance(scores):
    means = {}
    stds = {}
    for task in BENCHMARKS_EXT.keys():
        base = np.array([scores[m][task] for m in IMAGE_RELIANCE_BASELINE_MODELS])
        noimage = np.array([scores[m + "_noimage"][task] for m in IMAGE_RELIANCE_BASELINE_MODELS])
        if base.size == 0:
            continue
        remove = np.where(base < 0.5 * base.max())
        base = np.delete(base, remove)
        noimage = np.delete(noimage, remove)
        diff = (base - noimage) / base
        valid_diff = diff[np.isfinite(diff)]
        if valid_diff.size == 0:
            continue
        means[task] = valid_diff.mean()
        stds[task] = valid_diff.std()
    return means, stds


CATEGORY_COLORS = {
    "Compositional": "#e76f51",
    "Open-ended": "#e9c46a",
    "Multiple choice": "#669bbc",
}

CATEGORY_MARKERS = {
    "Compositional": "^",
    "Open-ended": "o",
    "Multiple choice": "s",
}

COMPOSITIONAL_KEYS = {"SugarCrepe", "ARO", "ARO-order", "ARO-vg_relation", "ARO-vg_attribution"}
OPEN_ENDED_KEYS = {"ChartQA", "GQA", "VQAv2", "CountBench"}


def _category_for(bench_key):
    if bench_key in COMPOSITIONAL_KEYS:
        return "Compositional"
    if bench_key in OPEN_ENDED_KEYS:
        return "Open-ended"
    return "Multiple choice"


def _correlation_panel(ax, scores, means, pair, color_by_category=False):
    x_vals, y_vals, labels, cats = [], [], [], []
    for bench_key, bench_info in BENCHMARKS_EXT.items():
        if bench_info["abbrev"] in CORRELATION_REMOVE or bench_key in CORRELATION_REMOVE:
            continue
        s_loc = scores.get(pair["loc"], {}).get(bench_key, 0)
        s_comp = scores.get(pair["comp"], {}).get(bench_key, 0)
        dx = means.get(bench_key, np.nan)
        if s_loc > 1e-6 and np.isfinite(dx):
            dy = (s_loc - s_comp) / s_loc
            x_vals.append(dx)
            y_vals.append(dy)
            labels.append(bench_info["abbrev"])
            cats.append(_category_for(bench_key))

    if not x_vals:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    spearman_corr, p_value = stats.spearmanr(x_vals, y_vals, alternative="greater")

    if color_by_category:
        for cat_name in CATEGORY_COLORS:
            xs = [x for x, c in zip(x_vals, cats) if c == cat_name]
            ys = [y for y, c in zip(y_vals, cats) if c == cat_name]
            if xs:
                ax.scatter(xs, ys, s=40, alpha=0.95,
                           color=CATEGORY_COLORS[cat_name],
                           marker=CATEGORY_MARKERS[cat_name],
                           edgecolor="white", linewidth=0.6, zorder=3)
    else:
        ax.scatter(x_vals, y_vals, s=28, alpha=0.9, color="#1f77b4", edgecolor="white", linewidth=0.6, zorder=3)

    texts = [ax.text(x, y, lab, fontsize=10) for x, y, lab in zip(x_vals, y_vals, labels)]
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            ax=ax,
            expand_points=(2.4, 2.6),
            expand_text=(2.0, 2.2),
            force_points=(1.0, 1.3),
            force_text=(0.9, 1.1),
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
        )
    except ImportError:
        pass

    xpad = (max(x_vals) - min(x_vals)) * 0.08
    ypad = (max(y_vals) - min(y_vals)) * 0.08
    ax.set_xlim(min(x_vals) - xpad, max(x_vals) + xpad)
    ax.set_ylim(min(y_vals) - ypad, max(y_vals) + ypad)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.3)

    stats_text = f"Spearman\n$\\varrho={spearman_corr:.2f}$\n$p={p_value:.3f}$"
    ax.text(0.96, 0.04, stats_text, transform=ax.transAxes,
            horizontalalignment="right", verticalalignment="bottom", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9))

    ax.set_xlabel("Image reliance")
    ax.set_ylabel("Localized ft advantage")
    ax.set_title(pair["title"])
    ax.grid(True, linestyle="-", alpha=0.15)


def plot_correlation_two_panel(scores, means, save_path=None, color_by_category=False):
    rc = dict(get_rc(ncols=2))
    rc["figure.figsize"] = (12, 4.5)
    rc["figure.constrained_layout.use"] = False
    rc["axes.labelsize"] = 13
    rc["axes.titlesize"] = 14
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10
    rc["legend.fontsize"] = 13

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2)
        _correlation_panel(axes[0], scores, means, PAIRS["llava_7b"], color_by_category=color_by_category)
        _correlation_panel(axes[1], scores, means, PAIRS["LLaVA-OneVision-1.5-4B"], color_by_category=color_by_category)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.18 if color_by_category else 0.13, wspace=0.22)

        if color_by_category:
            from matplotlib.lines import Line2D

            handles = [
                Line2D([0], [0], marker="s", linestyle="", color="none", label="Task type:"),
            ] + [
                Line2D([0], [0], marker=CATEGORY_MARKERS[name], linestyle="", color=col,
                       markeredgecolor="white", markeredgewidth=0.6, markersize=9, label=name)
                for name, col in CATEGORY_COLORS.items()
            ]
            fig.legend(handles=handles, loc="lower center", ncol=len(CATEGORY_COLORS) + 1,
                       bbox_to_anchor=(0.5, -0.04), frameon=False, handletextpad=0.4,
                       columnspacing=1.2)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved {save_path}")
        plt.show()
        plt.close(fig)


def _radar_panel(ax, scores, pair, categories):
    base_key = pair["base"]
    loc_key = pair["loc"]
    comp_key = pair["comp"]

    targets = [
        (base_key, "Base", "black", "--"),
        (loc_key, "Localized", "teal", "-"),
        (comp_key, "Complementary", "maroon", "-"),
    ]

    base_scores = scores.get(base_key, {})
    normalized = {}
    max_val = 0.0
    for key, _label, _color, _ls in targets:
        if key not in scores:
            continue
        normalized[key] = {}
        for cat in categories:
            b = base_scores.get(cat, 0)
            c = scores[key].get(cat, 0)
            val = (c / b) if b > 1e-6 else (1.0 if c > 0 else 0.0)
            normalized[key][cat] = val
            max_val = max(max_val, val)

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    for key, label_text, color, ls in targets:
        if key not in normalized:
            continue
        values = [normalized[key].get(cat, 0) for cat in categories]
        values += values[:1]
        if ls == "--":
            ax.plot(angles, values, linewidth=1.5, linestyle="--", label=label_text, color=color, alpha=0.85)
        else:
            ax.plot(angles, values, linewidth=1.5, linestyle="solid", label=label_text, color=color)
            ax.fill(angles, values, alpha=0.06, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([BENCHMARKS_EXT[c]["abbrev"] for c in categories])
    ax.tick_params(axis="x", pad=22)
    ax.set_yticks([0.8, 0.9, 1.0, 1.1])
    ax.set_ylim(0.75, max(1.1, max_val + 0.01))
    ax.plot(angles, [1.0] * (N + 1), color="grey", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_title(pair["title"], pad=25)
    ax.grid(True, alpha=0.3)


def plot_radar_two_panel(scores, save_path=None):
    # radar_drop = {"BLINK", "Vlmsareblind", "DocVQA", "MMMU", "InfoVQA", "MMMU-Pro-Std", "BiVLC", "VizWiz", "MMVP"}
    radar_drop = {"VizWiz", "ARO-order", "ARO-vg_relation", "ARO-vg_attribution", "DocVQA", "InfoVQA", "AI2D", "MMVP", "Vlmsareblind", "BLINK", "BiVLC"}
    radar_categories = [k for k in BENCHMARKS_EXT.keys() if k not in radar_drop]

    rc = dict(get_rc(ncols=2))
    rc["figure.figsize"] = (15, 8)
    rc["figure.constrained_layout.use"] = False
    rc["xtick.labelsize"] = 11
    rc["ytick.labelsize"] = 8
    rc["axes.titlesize"] = 16
    rc["legend.fontsize"] = 13

    with plt.rc_context(rc):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection="polar")
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")

        _radar_panel(ax1, scores, PAIRS["llava_7b"], radar_categories)
        _radar_panel(ax2, scores, PAIRS["LLaVA-OneVision-1.5-4B"], radar_categories)

        ax1.set_position([0.06, 0.18, 0.34, 0.66])
        ax2.set_position([0.56, 0.18, 0.34, 0.66])

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02))

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved {save_path}")
        plt.show()
        plt.close(fig)


LAYER_DEPTH_MODELS = [
    "llava_7b_frozen", "llava_7b",
    "llava_7b_0_2", "llava_7b_0_4", "llava_7b_0_6", "llava_7b_0_8",
    "llava_7b_0_12", "llava_7b_0_16", "llava_7b_0_24",
    "llava_7b_8_32", "llava_7b_16_32", "llava_7b_20_32",
    "llava_7b_24_32", "llava_7b_26_32", "llava_7b_28_32", "llava_7b_30_32",
]


def plot_layer_depth(scores, baseline_model="llava_7b", save_path=None):
    bottom_up = {}
    top_down = {}

    base_scores = scores.get(baseline_model, {})

    def normalized_avg(model_name):
        m_scores = scores.get(model_name, {})
        ratios = []
        for bench, b in base_scores.items():
            v = m_scores.get(bench, 0)
            if b > 1e-6 and v > 1e-6:
                ratios.append(v / b)
        return sum(ratios) / len(ratios) if ratios else None

    for model_name in LAYER_DEPTH_MODELS:
        if model_name not in scores:
            continue
        avg = normalized_avg(model_name)
        if avg is None:
            continue

        if model_name == "llava_7b_frozen":
            bottom_up[0] = avg
            top_down[32] = avg
        elif model_name == "llava_7b":
            bottom_up[32] = avg
            top_down[0] = avg
        elif model_name.startswith("llava_7b_"):
            parts = model_name.replace("llava_7b_", "").split("_")
            if len(parts) == 2:
                try:
                    start, end = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                if start == 0:
                    bottom_up[end] = avg
                elif end == 32:
                    top_down[start] = avg

    bu_x = sorted(bottom_up.keys())
    bu_y = [bottom_up[x] for x in bu_x]
    td_x = sorted(top_down.keys())
    td_y = [top_down[x] for x in td_x]

    rc = dict(get_rc(ncols=1))
    rc["figure.figsize"] = (6, 4)
    rc["figure.constrained_layout.use"] = True
    rc["axes.labelsize"] = 13
    rc["axes.titlesize"] = 14
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10
    rc["legend.fontsize"] = 11

    with plt.rc_context(rc):
        fig, ax = plt.subplots()
        ax.axvspan(5.5, 16.5, alpha=0.12, color="#DAA520", zorder=0, label="Abstraction phase")
        ax.plot(bu_x, bu_y, marker="o", linewidth=1.2, color="#2ca02c", label="Bottom-Up $[0, N-1]$")
        ax.plot(td_x, td_y, marker="s", linewidth=1.2, color="#ff7f0e", label="Top-Down $[N, 31]$")

        ax.set_xticks(list(range(0, 33, 4)))
        all_y = bu_y + td_y
        ymin, ymax = min(all_y), max(all_y)
        pad = (ymax - ymin) * 0.07
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Average performance")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved {save_path}")
        plt.show()
        plt.close(fig)


TABLE_ROWS = [
    ("LLaVA-7B",            "llava_7b",        "$[0, 16]$",                "llava_7b_0_16"),
    ("LLaVA-7B",            "llava_7b",        "$[16, 32]$",               "llava_7b_16_32"),
    ("LLaVA-7B",            "llava_7b",        "$[6, 16]$",                "llava_7b_6_17"),
    ("LLaVA-7B",            "llava_7b",        r"$\setminus[6, 16]$",      "llava_7b_6_17_complementary"),
    ("LLaVA-OneVision-4B",  "LLaVA-OneVision-1.5-4B", "$[0, 18]$",         "LLaVA-OneVision-1.5-4B-0-18"),
    ("LLaVA-OneVision-4B",  "LLaVA-OneVision-1.5-4B", "$[18, 36]$",        "LLaVA-OneVision-1.5-4B-18-36"),
    ("LLaVA-OneVision-4B",  "LLaVA-OneVision-1.5-4B", "$[8, 26]$",         "LLaVA-OneVision-1.5-4B-8-26"),
    ("LLaVA-OneVision-4B",  "LLaVA-OneVision-1.5-4B", r"$\setminus[8, 26]$","LLaVA-OneVision-1.5-4B-8-26-comp"),
]

CAPTIONING_KEYS = {"COCO", "Flickr"}


def _table_category(bench_key, bench_info):
    if bench_key in CAPTIONING_KEYS:
        return "Captioning"
    if bench_info["abbrev"] in CORRELATION_REMOVE or bench_key in CORRELATION_REMOVE:
        return None
    return _category_for(bench_key)


def print_latex_table(scores):
    rows_data = []
    for model_name, base_key, layer_label, model_key in TABLE_ROWS:
        m = scores.get(model_key, {})
        b = scores.get(base_key, {})
        if not m or not b:
            continue

        per_cat = {"Multiple choice": [], "Open-ended": [], "Compositional": [], "Captioning": []}
        all_ratios = []
        for bench_key, bench_info in BENCHMARKS_EXT.items():
            cat = _table_category(bench_key, bench_info)
            if cat is None:
                continue
            bv = b.get(bench_key, 0)
            mv = m.get(bench_key, 0)
            if bv > 1e-6 and mv > 1e-6:
                ratio = mv / bv * 100
                per_cat[cat].append(ratio)
                all_ratios.append(ratio)

        def avg(xs):
            return f"{sum(xs)/len(xs):.1f}" if xs else "-"

        rows_data.append((model_name, layer_label,
                          avg(per_cat["Multiple choice"]),
                          avg(per_cat["Open-ended"]),
                          avg(per_cat["Compositional"]),
                          avg(per_cat["Captioning"]),
                          avg(all_ratios)))

    header = (
        r"\begin{tabular}{llcccccc}"
        "\n\\toprule\n"
        r" Model & \makecell{Trained \\ layers} & \makecell{MC \\ VQA} $\uparrow$ & \makecell{OE \\ VQA}$\uparrow$& Compositional $\uparrow$ & Captioning $\uparrow$ & Overall $\uparrow$ & \makecell{Train \\ time} $\downarrow$\\"
        "\n\\midrule"
    )
    print(header)
    prev_model = None
    for model_name, layer, mc, oe, comp, cap, overall in rows_data:
        if prev_model is not None and prev_model != model_name:
            print(r"\midrule")
        m_cell = model_name if prev_model != model_name else ""
        print(f"{m_cell} & {layer} & {mc} & {oe} & {comp} & {cap} & {overall} & \\\\")
        prev_model = model_name
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    needed = set(IMAGE_RELIANCE_BASELINE_MODELS) | {m + "_noimage" for m in IMAGE_RELIANCE_BASELINE_MODELS}
    for pair in PAIRS.values():
        needed.update([pair["base"], pair["noimage"], pair["loc"], pair["comp"]])
    needed.update(LAYER_DEPTH_MODELS)
    for _, base_key, _, model_key in TABLE_ROWS:
        needed.update([base_key, model_key])

    scores = load_model_scores(sorted(needed))
    means, _stds = compute_image_reliance(scores)

    plot_correlation_two_panel(scores, means, save_path=out_path("correlation_two_panel.pdf"))
    plot_correlation_two_panel(scores, means, save_path=out_path("correlation_two_panel_colored.pdf"), color_by_category=True)
    plot_radar_two_panel(scores, save_path=out_path("radar_two_panel.pdf"))
    plot_layer_depth(scores, save_path=out_path("layer_depth.pdf"))
    print_latex_table(scores)


if __name__ == "__main__":
    main()
