import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from rich import print

PARENT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PARENT_DIR / "results"
PLOTS_DIR = PARENT_DIR / "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

START2END_CMAP = LinearSegmentedColormap.from_list(
    "start2end_gradient", ["#3b4cc0", "#b40426"]
)
END2START_CMAP = LinearSegmentedColormap.from_list(
    "end2start_gradient", ["#1b9e77", "#d95f02"]
)

# QA_RESULTS_DIR = RESULTS_DIR / "transplanting_layers_qa_benchmarking"
# CAPTION_RESULTS_DIR = RESULTS_DIR / "transplanting_layers_caption_benchmarking"
QA_RESULTS_DIR = RESULTS_DIR / "transplanting_layers_qa_benchmarking_v2"
CAPTION_RESULTS_DIR = RESULTS_DIR / "transplanting_layers_caption_benchmarking_v2"

# v1
cocoqa_txt_start2end_path = (
    QA_RESULTS_DIR
    / "cocoqa_txt"
    / (
        # "llava-1.5-7b-hf_vicuna-7b-v1.5_two_parts_s2.csv"
        "llava-1.5-7b-hf_llava-1.5-7b-hf_vicuna-7b-v1.5_pretrained_connector_two_parts_s2.csv"
    )
)

cocoqa_txt_end2start_path = (
    QA_RESULTS_DIR
    / "cocoqa_txt"
    / (
        # "llava-1.5-7b-hf_llava-1.5-7b-hf_vicuna-7b-v1.5_pretrained_connector_two_parts_s2_reverse_cumulative.csv"
        "llava-1.5-7b-hf_llava-1.5-7b-hf_vicuna-7b-v1.5_pretrained_connector_two_parts_s2_reverse_cumulative.csv"
    )
)

cocoqa_img_start2end_path = (
    QA_RESULTS_DIR
    / "cocoqa_img"
    / (
        # "llava-1.5-7b-hf_vicuna-7b-v1.5_two_parts_s2.csv"
        "llava-1.5-7b-hf_llava-1.5-7b-hf_vicuna-7b-v1.5_pretrained_connector_two_parts_s2.csv"
    )
)
cocoqa_img_end2start_path = (
    QA_RESULTS_DIR
    / "cocoqa_img"
    / (
        "llava-1.5-7b-hf_llava-1.5-7b-hf_vicuna-7b-v1.5_"
        "pretrained_connector_two_parts_s2_reverse_cumulative.csv"
    )
)

coco_captioning_start2end_path = (
    CAPTION_RESULTS_DIR
    / "coco_captioning"
    / (
        "llava-1.5-7b-hf_llava-1.5-7b-hf_vicuna-7b-v1.5_pretrained_connector_two_parts_s2.csv"
    )
)
coco_captioning_end2start_path = (
    CAPTION_RESULTS_DIR
    / "coco_captioning"
    / (
        "llava-1.5-7b-hf_llava-1.5-7b-hf_vicuna-7b-v1.5_pretrained_connector_two_parts_s2_reverse_cumulative.csv"
    )
)

DATA_SPECS: Dict[str, Dict] = {
    "cocoqa_txt": {
        "metrics": ["accuracy"],
        "start2end_path": cocoqa_txt_start2end_path,
        "end2start_path": cocoqa_txt_end2start_path,
        "pm_label": "mm_pretrained_connector",
        "fm_label": "mm_model",
    },
    "cocoqa_img": {
        "metrics": ["accuracy"],
        "start2end_path": cocoqa_img_start2end_path,
        "end2start_path": cocoqa_img_end2start_path,
        "pm_label": "mm_pretrained_connector",
        "fm_label": "mm_model",
    },
    "coco_captioning": {
        "metrics": ["CIDEr", "CLIP-S"],
        "start2end_path": coco_captioning_start2end_path,
        "end2start_path": coco_captioning_end2start_path,
        "pm_label": "mm_pretrained_connector",
        "fm_label": "mm_model",
    },
}


def _extract_split_series(
    df: pd.DataFrame, value_column: str
) -> Tuple[List[int], List[float]]:
    rows = [
        (int(idx.split("_")[-1]), df.loc[idx, value_column])
        for idx in df.index
        if idx.startswith("split_layer_")
    ]
    rows.sort(key=lambda item: item[0])
    layers = [layer for layer, _ in rows]
    values = [float(val) for _, val in rows]
    return layers, values


def _load_csv_series(
    csv_path: Path, metrics: List[str], include_references: bool
) -> Dict[str, Dict]:
    df = pd.read_csv(csv_path, index_col=0)
    layers = None
    series: Dict[str, Dict] = {"metrics": {}}

    for metric in metrics:
        layers, values = _extract_split_series(df, metric)
        series["metrics"][metric] = values

    series["layers"] = layers if layers is not None else []

    if include_references:
        series["pm"] = {
            metric: float(df.loc["mm_pretrained_connector", metric])
            for metric in metrics
        }
        series["fm"] = {metric: float(df.loc["mm_model", metric]) for metric in metrics}

    return series


def load_cumulative_data() -> Dict[str, Dict]:
    cumulative_data = {"start2end": {}, "end2start": {}}

    for task, spec in DATA_SPECS.items():
        start_series = _load_csv_series(
            spec["start2end_path"], spec["metrics"], include_references=True
        )
        end_series = _load_csv_series(
            spec["end2start_path"], spec["metrics"], include_references=False
        )

        # Reuse the x-axis labels from start2end data and
        # ! flip the reverse-cumulative metrics so that
        # ! they progress in the same start→end direction on the plot
        for metric in spec["metrics"]:
            end_series["metrics"][metric] = list(
                reversed(end_series["metrics"][metric])
            )

        end_series["layers"] = start_series["layers"]

        cumulative_data["start2end"][task] = start_series
        cumulative_data["end2start"][task] = end_series

    return cumulative_data


def _generate_colors(cmap: LinearSegmentedColormap, count: int) -> List:
    if count == 1:
        return [cmap(0.5)]
    return [cmap(i / (count - 1)) for i in range(count)]


def _plot_metric_series(
    ax,
    x_positions: List[int],
    values: List[float],
    colors: List,
    linestyle: str,
    label: str,
):
    ax.plot(
        x_positions,
        values,
        linestyle=linestyle,
        color="#a0a0a0",
        linewidth=1,
        alpha=0.6,
    )
    ax.scatter(
        x_positions,
        values,
        c=colors,
        s=50,
        edgecolor="none",
        label=label,
    )


def _plot_pm_fm(ax, pm_value: float, fm_value: float, first_pos: int, last_pos: int):
    ax.scatter(
        first_pos,
        pm_value,
        marker="s",
        s=70,
        color="#555555",
        label="PM",
        zorder=3,
    )
    ax.scatter(
        last_pos,
        fm_value,
        marker="D",
        s=70,
        color="#b40426",
        label="FM",
        zorder=3,
    )


def plot_qa_figures(cumulative_data: Dict[str, Dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    tasks = [("cocoqa_txt", "Text QA"), ("cocoqa_img", "Visual QA")]

    for ax, (task_key, title) in zip(axes, tasks):
        start_series = cumulative_data["start2end"][task_key]
        end_series = cumulative_data["end2start"][task_key]

        layers = start_series["layers"]
        x_positions = list(range(len(layers) + 2))
        split_positions = x_positions[1:-1]

        start_colors = _generate_colors(START2END_CMAP, len(layers))
        end_colors = _generate_colors(END2START_CMAP, len(layers))

        _plot_metric_series(
            ax,
            split_positions,
            start_series["metrics"]["accuracy"],
            start_colors,
            linestyle="-",
            label="start→end",
        )
        _plot_metric_series(
            ax,
            split_positions,
            end_series["metrics"]["accuracy"],
            end_colors,
            linestyle="--",
            label="end→start",
        )

        _plot_pm_fm(
            ax,
            start_series["pm"]["accuracy"],
            start_series["fm"]["accuracy"],
            x_positions[0],
            x_positions[-1],
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            ["PM"] + [str(layer) for layer in layers] + ["FM"],
            rotation=45,
        )
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Number of patched layers")

    axes[0].set_ylabel("Accuracy after patching")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=START2END_CMAP(0.8),
            label="start→end",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=END2START_CMAP(0.8),
            label="end→start",
        ),
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="#555555", label="PM"
        ),
        Line2D(
            [0], [0], marker="D", color="none", markerfacecolor="#b40426", label="FM"
        ),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()

    output_path = PLOTS_DIR / "qa_transplantation_start_end_v2.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_captioning_figures(cumulative_data: Dict[str, Dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics = [("CIDEr", "CIDEr"), ("CLIP-S", "CLIP-S")]
    task_key = "coco_captioning"

    start_series = cumulative_data["start2end"][task_key]
    end_series = cumulative_data["end2start"][task_key]
    layers = start_series["layers"]
    x_positions = list(range(len(layers) + 2))
    split_positions = x_positions[1:-1]

    for ax, (metric_key, ylabel) in zip(axes, metrics):
        start_colors = _generate_colors(START2END_CMAP, len(layers))
        end_colors = _generate_colors(END2START_CMAP, len(layers))

        _plot_metric_series(
            ax,
            split_positions,
            start_series["metrics"][metric_key],
            start_colors,
            linestyle="-",
            label="start→end",
        )
        _plot_metric_series(
            ax,
            split_positions,
            end_series["metrics"][metric_key],
            end_colors,
            linestyle="--",
            label="end→start",
        )

        _plot_pm_fm(
            ax,
            start_series["pm"][metric_key],
            start_series["fm"][metric_key],
            x_positions[0],
            x_positions[-1],
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            ["PM"] + [str(layer) for layer in layers] + ["FM"],
            rotation=45,
        )
        ax.set_xlabel("Number of patched layers")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=12)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=START2END_CMAP(0.8),
            label="start→end",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=END2START_CMAP(0.8),
            label="end→start",
        ),
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="#555555", label="PM"
        ),
        Line2D(
            [0], [0], marker="D", color="none", markerfacecolor="#b40426", label="FM"
        ),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()

    output_path = PLOTS_DIR / "captioning_transplantation_start_end_v2.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    cumulative_data = load_cumulative_data()
    qa_plot = plot_qa_figures(cumulative_data)
    caption_plot = plot_captioning_figures(cumulative_data)

    print(f"[green]Saved QA plot to:[/green] {qa_plot}")
    print(f"[green]Saved captioning plot to:[/green] {caption_plot}")


if __name__ == "__main__":
    main()
