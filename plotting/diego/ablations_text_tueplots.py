"""Text-only layer-skip ablations in the tueplots NeurIPS style.

Renders both the per-task (`all_tasks`) and the averaged (`average`) views,
for both the `progressive` and `sliding` schedules, for every model.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tueplots_style import OUTPUT_DIR, get_rc  # noqa: E402

DATA_ROOT = Path("/u/dssc/lbasile/scratch/plm/diego/layer_skip_eval")
OUT_DIR = OUTPUT_DIR / "ablations_text"

SW_WINDOW = 4

TEXT_TASKS_METRIC = {
    "arc_easy": ("acc,none", "arc_easy"),
    "commonsense_qa": ("acc,none", "commonsense_qa"),
    "hellaswag": ("acc,none", "hellaswag"),
    "mmlu": ("acc,none", "mmlu"),
    "piqa": ("acc,none", "piqa"),
    "winogrande": ("acc,none", "winogrande"),
}

VARIANT_STYLES = {
    "progressive": {"color": "#991499", "linestyle": "-", "marker": "o", "markerfacecolor": "#991499"},
    "sliding":     {"color": "#991499", "linestyle": (0, (4, 4)), "marker": "^", "markerfacecolor": "#991499"},
}

PANEL_CONFIGS = {
    "llava_7b":     {"title": "LLaVA-7B",     "num_layers": 32, "base_dir": "old_7b/llava-7b-lbasile"},
    "OneVision_4B": {"title": "OneVision-4B", "num_layers": 36, "base_dir": "LLaVA-OneVision-1.5-4B-Instruct"},
    "InternVL2_8B": {"title": "InternVL2-8B", "num_layers": 32, "base_dir": "InternVL2-8B"},
    "llava_13b":    {"title": "LLaVA-13B",    "num_layers": 40, "base_dir": "llava_13b"},
    "cambrian_8b":  {"title": "Cambrian-8B",  "num_layers": 32, "base_dir": "cambrian-8b"},
    "OneVision_8B": {"title": "OneVision-8B", "num_layers": 36, "base_dir": "LLaVA-OneVision-1.5-8B-Instruct"},
}

MODEL_ORDER = [
    "llava_7b", "llava_13b", "cambrian_8b",
    "OneVision_4B", "OneVision_8B","InternVL2_8B",
]

_SORTED_TASKS = sorted(TEXT_TASKS_METRIC)
TASK_PALETTE= ('#f94144', '#f8961e', '#f9c74f', '#90be6d', '#469583', '#557CAF', '#0077B6')
TASK_COLORS = dict(zip(_SORTED_TASKS, TASK_PALETTE))


def _build_variants(panel: dict) -> List[dict]:
    base = panel["base_dir"]
    return [
        {"label": "progressive", "dir": base, "type": "progressive"},
        {"label": "sliding", "dir": f"{base}_sw{SW_WINDOW}", "type": "sliding", "window": SW_WINDOW},
    ]


def _read_metric(path: Path, result_key: str, metric_key: str) -> Optional[float]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get("results", {}).get(result_key)
    if not isinstance(block, dict):
        return None
    v = block.get(metric_key)
    return float(v) if v is not None else None


def _scan_points(task_dir: Path, task: str, result_key: str, metric_key: str,
                 vtype: str, window: int) -> Dict[float, float]:
    if vtype == "sliding":
        pat = re.compile(rf"^{re.escape(task)}_L(\d+)_W{window}_0shot\.json$")
        x_fn = lambda m: int(m.group(1)) + window / 2
    else:
        pat = re.compile(rf"^{re.escape(task)}_L0+_W(\d+)_0shot\.json$")
        x_fn = lambda m: int(m.group(1))
    points: Dict[float, float] = {}
    for f in task_dir.iterdir():
        m = pat.match(f.name)
        if m:
            val = _read_metric(f, result_key, metric_key)
            if val is not None:
                points[x_fn(m)] = val
    return points


def load_normalized_tasks(panel: dict, variant: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    variant_dir = DATA_ROOT / variant["dir"]
    norm_dir = DATA_ROOT / panel["base_dir"]
    if not variant_dir.is_dir() or not norm_dir.is_dir():
        return {}
    vtype = variant["type"]
    window = variant.get("window", 0)
    num_layers = panel["num_layers"]

    raw = {}
    norm_params = {}
    for task, (metric_key, result_key) in TEXT_TASKS_METRIC.items():
        task_dir = variant_dir / task
        norm_td = norm_dir / task
        if not task_dir.is_dir() or not norm_td.is_dir():
            continue
        full = _read_metric(norm_td / f"{task}_L00_W0_0shot.json", result_key, metric_key)
        random_ = _read_metric(norm_td / f"{task}_L00_W{num_layers}_0shot.json", result_key, metric_key)
        if full is None or random_ is None or abs(full - random_) < 1e-12:
            continue
        points = _scan_points(task_dir, task, result_key, metric_key, vtype, window)
        if points:
            raw[task] = points
            norm_params[task] = (full, random_)

    if not raw:
        return {}
    x_counts: Dict[float, int] = {}
    for pts in raw.values():
        for x in pts:
            x_counts[x] = x_counts.get(x, 0) + 1
    threshold = len(raw) / 2
    expected_x = sorted(x for x, c in x_counts.items() if c > threshold)
    x_arr = np.array(expected_x)
    out = {}
    for task, points in raw.items():
        if not all(x in points for x in expected_x):
            continue
        full, random_ = norm_params[task]
        vals = np.array([points[x] for x in expected_x])
        out[task] = (x_arr, (vals - random_) / (full - random_))
    return out


def load_average(panel: dict, variant: dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tasks = load_normalized_tasks(panel, variant)
    if not tasks:
        return None
    x = next(iter(tasks.values()))[0]
    stacked = np.vstack([v for _, v in tasks.values()])
    return x, np.mean(stacked, axis=0), np.std(stacked, axis=0)


def _plot_panel_average(ax, panel: dict, show_std: bool = False) -> None:
    num_layers = panel["num_layers"]
    for variant in _build_variants(panel):
        result = load_average(panel, variant)
        if result is None:
            continue
        x, mean, std = result
        style = VARIANT_STYLES[variant["label"]]
        ax.plot(x, mean, marker=style["marker"], markersize=5, linewidth=2,
                color=style["color"], linestyle=style["linestyle"],
                markerfacecolor=style.get("markerfacecolor", style["color"]))
        if show_std:
            ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xticks(np.arange(0, num_layers + 1, 4))
    ax.set_xlabel("Layers")
    ax.set_ylabel("Performance")
    ax.set_title(panel["title"])
    ax.grid(True, alpha=0.3)


def _plot_panel_tasks(ax, panel: dict, variant: dict) -> None:
    num_layers = panel["num_layers"]
    task_curves = load_normalized_tasks(panel, variant)
    if not task_curves:
        ax.set_title(f"{panel['title']} (no data)")
        ax.grid(True, alpha=0.3)
        return
    for task, (x, norm) in sorted(task_curves.items()):
        ax.plot(x, norm, marker="o", markersize=5, linewidth=2, color=TASK_COLORS[task])
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xticks(np.arange(0, num_layers + 1, 4))
    ax.set_xlabel("Layers")
    ax.set_ylabel("Performance")
    ax.set_title(panel["title"])
    ax.grid(True, alpha=0.3)


def _schedule_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color=VARIANT_STYLES["progressive"]["color"],
               linestyle=VARIANT_STYLES["progressive"]["linestyle"],
               marker=VARIANT_STYLES["progressive"]["marker"],
               markerfacecolor=VARIANT_STYLES["progressive"].get("markerfacecolor", VARIANT_STYLES["progressive"]["color"]),
               markersize=5, linewidth=2, label="progressive"),
        Line2D([0], [0], color=VARIANT_STYLES["sliding"]["color"],
               linestyle=VARIANT_STYLES["sliding"]["linestyle"],
               marker=VARIANT_STYLES["sliding"]["marker"],
               markerfacecolor=VARIANT_STYLES["sliding"].get("markerfacecolor", VARIANT_STYLES["sliding"]["color"]),
               markersize=5, linewidth=2, label="sliding window"),
    ]


TASK_PRETTY = {
    "arc_easy":       "ARC-Easy",
    "commonsense_qa": "CommonsenseQA",
    "hellaswag":      "HellaSwag",
    "mmlu":           "MMLU",
    "piqa":           "PIQA",
    "winogrande":     "WinoGrande",
}


def _datasets_legend_handles() -> List:
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=TASK_COLORS[t], edgecolor=TASK_COLORS[t],
              label=TASK_PRETTY.get(t, t))
        for t in _SORTED_TASKS
    ]


def plot_average() -> None:
    rc = get_rc(ncols=3, nrows=2, **{"figure.figsize": (11, 6)})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3)
        for ax, key in zip(axes.flat, MODEL_ORDER):
            _plot_panel_average(ax, PANEL_CONFIGS[key])
        for ax in axes[:, 1:].flat:
            ax.set_ylabel("")
        fig.legend(handles=_schedule_legend_handles(),
                   title="Skip layers:",
                   loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "ablations_text_average.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def plot_all_tasks(schedule_label: str) -> None:
    rc = get_rc(ncols=3, nrows=2, **{"figure.figsize": (11, 6)})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3)
        for ax, key in zip(axes.flat, MODEL_ORDER):
            panel = PANEL_CONFIGS[key]
            variant = next(v for v in _build_variants(panel) if v["label"] == schedule_label)
            _plot_panel_tasks(ax, panel, variant)
        for ax in axes[:, 1:].flat:
            ax.set_ylabel("")
        fig.legend(handles=_datasets_legend_handles(),
                   title="Dataset:",
                   loc="lower center", bbox_to_anchor=(0.5, -0.09),
                   ncol=len(_SORTED_TASKS), frameon=False)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"ablations_text_tasks_{schedule_label}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def main() -> None:
    plot_average()
    plot_all_tasks("progressive")
    plot_all_tasks("sliding")


if __name__ == "__main__":
    main()
