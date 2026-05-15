from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tueplots_style import get_rc  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "diego" / "probing_tasks"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tueplots" / "diego"

EMBEDDING = "res2"
POOLING = "text_avg"
METRIC_PICK = "mean"

TASKS = ["CoordInv", "SOMO", "ObjNum", "SubjNum", "Tense"]

MODELS = [
    ("llava_7b_hf",       "LLaVA-7B"),
    ("llava_13b_hf",      "LLaVA-13B"),
    ("cambrian_8b",       "Cambrian-8B"),
    ("onevision_4b_hf",   "OneVision-4B"),
    ("onevision_8b_hf",   "OneVision-8B"),
    ("internvl_8b_hf",    "InternVL2-8B"),
]

TASK_PALETTE = ('#f94144', '#f8961e', '#f9c74f', '#90be6d', '#469583')

LAYER_RE = re.compile(r"\.layers\.(\d+)")


def _parse_layer(key):
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def _load_curve(model_slug, task):
    path = RESULTS_ROOT / model_slug / EMBEDDING / POOLING / f"{task}_accuracy_logistic.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text())
    points = []
    for key, values in payload.items():
        layer = _parse_layer(key)
        if layer is None or not values:
            continue
        points.append((layer, float(np.mean(values)) if METRIC_PICK == "mean" else float(values[int(METRIC_PICK)])))
    if not points:
        return None, None
    points.sort()
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def _panel(ax, model_slug, model_label, tasks, show_xlabel):
    max_layer = 0
    for idx, task in enumerate(tasks):
        layers, accs = _load_curve(model_slug, task)
        if layers is None:
            continue
        color = TASK_PALETTE[idx % len(TASK_PALETTE)]
        ax.plot(layers[1::2], accs[1::2], color=color, linewidth=2, marker="o", markersize=5)
        max_layer = max(max_layer, int(layers.max()))
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.9)
    ax.set_xticks(np.arange(0, max_layer + 1, 4))
    ax.set_xlim(-1, max_layer + 1)
    ax.set_ylim(0.45, 1.0)
    if show_xlabel:
        ax.set_xlabel("Layer")
    ax.set_title(model_label)
    ax.grid(True, alpha=0.3)


def plot_probing_grid(tasks=None, save_name=None):
    if tasks is None:
        tasks = TASKS
    if save_name is None:
        save_name = f"probing_{'_'.join(tasks)}.pdf"

    with plt.rc_context(get_rc(ncols=3, nrows=2, **{"figure.figsize": (12, 7)})):
        fig, axes = plt.subplots(2, 3)
        axes_flat = axes.flatten()

        for i, (ax, (slug, label)) in enumerate(zip(axes_flat, MODELS)):
            _panel(ax, slug, label, tasks, show_xlabel=(i >= 3))

        for ax in axes[:, 0]:
            ax.set_ylabel("Accuracy")

        handles = [
            mpatches.Patch(facecolor=TASK_PALETTE[i % len(TASK_PALETTE)], label=task)
            for i, task in enumerate(tasks)
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=len(tasks), frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / save_name
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


def main():
    plot_probing_grid(tasks=["CoordInv", "SOMO"], save_name="probing_CoordInv_SOMO.pdf")
    plot_probing_grid(tasks=TASKS, save_name="probing_all.pdf")


if __name__ == "__main__":
    main()
