"""
Unified intrinsic-dimension plots for pretrained vs finetuned models.

Mirrors the output-folder and run logic of plot_imbalance2.py:
- per-task plots in results/plots/intrinsic_dim/<mode>/single_tasks/
- average-over-tasks plot in results/plots/intrinsic_dim/<mode>/
- all-data plot in results/plots/intrinsic_dim/<mode>/
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _model_dir(cfg: dict, mode: str) -> str:
    return cfg["pre_path"] if mode == "pre" else cfg["ft_path"]


def _load_id_array(path: str, gride_scale: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        if gride_scale >= arr.shape[1]:
            raise ValueError(
                f"GRIDE scale index {gride_scale} out of bounds for {path} with shape {arr.shape}"
            )
        return arr[:, gride_scale]
    return arr


def _candidate_paths(
    model_dir: str,
    task: str,
    token_type: str,
    mode: str,
    data_type: str,
    task_suffix: str | None,
) -> list[str]:
    base = f"{model_dir}/{task}/{token_type}"
    candidates = []

    if task_suffix:
        candidates.append(f"{base}/{mode}_id_{task_suffix}_{data_type}.npy")

    candidates.append(f"{base}/{mode}_id_{data_type}_open_ended.npy")
    return candidates


def load_id_series(
    cfg: dict,
    mode: str,
    task: str,
    token_type: str,
    data_type: str,
    gride_scale: int,
    task_suffix_map: dict[str, str],
) -> np.ndarray:
    model_dir = _model_dir(cfg, mode)
    suffix = task_suffix_map.get(task)
    candidates = _candidate_paths(
        model_dir=model_dir,
        task=task,
        token_type=token_type,
        mode=mode,
        data_type=data_type,
        task_suffix=suffix,
    )

    for path in candidates:
        if os.path.exists(path):
            return _load_id_array(path, gride_scale)

    searched = "\n  - ".join(candidates)
    raise FileNotFoundError(
        f"Missing intrinsic-dimension file for task='{task}', mode='{mode}', data_type='{data_type}'.\n"
        f"Searched:\n  - {searched}"
    )


def load_average_id(
    cfg: dict,
    mode: str,
    tasks: list[str],
    token_type: str,
    data_type: str,
    gride_scale: int,
    task_suffix_map: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [
        load_id_series(
            cfg=cfg,
            mode=mode,
            task=task,
            token_type=token_type,
            data_type=data_type,
            gride_scale=gride_scale,
            task_suffix_map=task_suffix_map,
        )
        for task in tasks
    ]

    stacked = np.stack(arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    layers = np.arange(stacked.shape[1])
    return layers, mean, std


def _plot_curves_figure(
    cfg: dict,
    title: str,
    filename: str,
    img_payload_by_mode: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    txt_payload_by_mode: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    show_std: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    highlight_region = cfg.get("highlight_region", {})
    vspan = highlight_region.get("ft") or highlight_region.get("pre")
    panel_map = {
        "img": (axes[0], "Image representations", img_payload_by_mode),
        "txt": (axes[1], "Text representations", txt_payload_by_mode),
    }

    for ax, panel_title, payload_by_mode in panel_map.values():

        if vspan:
            x0, x1 = vspan
            ax.axvspan(x0, x1, alpha=0.15, color=BACKGROUND_COLOR, zorder=0)

        for mode in MODES:
            layers, mean, std = payload_by_mode[mode]
            ax.plot(
                layers,
                mean,
                color=MODE_COLORS[mode],
                linestyle=MODE_LINESTYLES[mode],
                marker="o",
                markersize=4,
                alpha=0.9,
                label=MODE_LABELS[mode],
            )

            if show_std:
                ax.fill_between(
                    layers,
                    mean - std,
                    mean + std,
                    color=MODE_COLORS[mode],
                    alpha=0.2,
                    linewidth=0,
                )

        ax.set_title(panel_title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Intrinsic dimension")
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved → {filename}")


def _plot_tasks_profiles_figure(
    cfg: dict,
    title: str,
    filename: str,
    tasks_payload_img: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    tasks_payload_txt: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    highlight_region = cfg.get("highlight_region", {})
    vspan = highlight_region.get("ft") or highlight_region.get("pre")
    task_colors = dict(
        zip(
            tasks_payload_img.keys(),
            sns.color_palette("tab10", n_colors=len(tasks_payload_img)),
        )
    )

    panel_map = {
        "img": (axes[0], "Image representations", tasks_payload_img),
        "txt": (axes[1], "Text representations", tasks_payload_txt),
    }

    for ax, panel_title, payload_by_task in panel_map.values():
        if vspan:
            x0, x1 = vspan
            ax.axvspan(x0, x1, alpha=0.15, color=BACKGROUND_COLOR, zorder=0)

        for task, payload_by_mode in payload_by_task.items():
            for mode in MODES:
                layers, values = payload_by_mode[mode]
                ax.plot(
                    layers,
                    values,
                    color=task_colors[task],
                    linestyle=MODE_LINESTYLES[mode],
                    marker="o",
                    markersize=3.5,
                    alpha=0.9,
                    label=f"{task} ({MODE_LABELS[mode]})",
                )

        ax.set_title(panel_title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Intrinsic dimension")
        ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved → {filename}")


def plot_id(
    cfg: dict,
    tasks: list[str],
    token_type: str,
    gride_scale: int,
    plot_mode: str = "average",
) -> None:
    assert plot_mode in ["average", "single", "all_data"]

    label = cfg["label"]

    if plot_mode == "single":
        print(f"  [{label}] pre+ft | {token_type} | all single-task profiles")

        tasks_payload_img: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        tasks_payload_txt: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for task in tasks:
            tasks_payload_img[task] = {}
            tasks_payload_txt[task] = {}
            for mode in MODES:
                img = load_id_series(
                    cfg, mode, task, token_type, "img", gride_scale, TASK_SUFFIX_MAP
                )
                txt = load_id_series(
                    cfg, mode, task, token_type, "txt", gride_scale, TASK_SUFFIX_MAP
                )
                tasks_payload_img[task][mode] = (np.arange(len(img)), img)
                tasks_payload_txt[task][mode] = (np.arange(len(txt)), txt)

        out_dir = f"{OUTPUT_DIR}/pre_ft/single_tasks"
        filename = (
            f"{out_dir}/{cfg['filename_prefix']}_all_tasks_{token_type}_pre_ft_id.png"
        )

        _plot_tasks_profiles_figure(
            cfg=cfg,
            title=f"{label} — single-task profiles (pre+ft, {token_type})",
            filename=filename,
            tasks_payload_img=tasks_payload_img,
            tasks_payload_txt=tasks_payload_txt,
        )
        return

    if plot_mode == "all_data":
        print(f"  [{label}] pre+ft | {token_type} | all_data")

        img_payload_by_mode = {}
        txt_payload_by_mode = {}
        for mode in MODES:
            img = load_id_series(
                cfg,
                mode,
                "all_data",
                token_type,
                "img",
                gride_scale,
                TASK_SUFFIX_MAP,
            )
            txt = load_id_series(
                cfg,
                mode,
                "all_data",
                token_type,
                "txt",
                gride_scale,
                TASK_SUFFIX_MAP,
            )
            img_payload_by_mode[mode] = (np.arange(len(img)), img, np.zeros_like(img))
            txt_payload_by_mode[mode] = (np.arange(len(txt)), txt, np.zeros_like(txt))

        out_dir = f"{OUTPUT_DIR}/pre_ft"
        filename = (
            f"{out_dir}/{cfg['filename_prefix']}"
            f"_all_data_{token_type}_pre_ft_id.png"
        )

        _plot_curves_figure(
            cfg=cfg,
            title=f"{label} — all data (pre+ft, {token_type})",
            filename=filename,
            img_payload_by_mode=img_payload_by_mode,
            txt_payload_by_mode=txt_payload_by_mode,
            show_std=False,
        )
        return

    print(f"  [{label}] pre+ft | {token_type} | avg({tasks})")
    img_payload_by_mode = {}
    txt_payload_by_mode = {}
    for mode in MODES:
        layers_img, mean_img, std_img = load_average_id(
            cfg=cfg,
            mode=mode,
            tasks=tasks,
            token_type=token_type,
            data_type="img",
            gride_scale=gride_scale,
            task_suffix_map=TASK_SUFFIX_MAP,
        )
        layers_txt, mean_txt, std_txt = load_average_id(
            cfg=cfg,
            mode=mode,
            tasks=tasks,
            token_type=token_type,
            data_type="txt",
            gride_scale=gride_scale,
            task_suffix_map=TASK_SUFFIX_MAP,
        )
        img_payload_by_mode[mode] = (layers_img, mean_img, std_img)
        txt_payload_by_mode[mode] = (layers_txt, mean_txt, std_txt)

    out_dir = f"{OUTPUT_DIR}/pre_ft"
    filename = f"{out_dir}/{cfg['filename_prefix']}_avg_{token_type}_pre_ft_id.png"

    _plot_curves_figure(
        cfg=cfg,
        title=f"{label} — avg over {len(tasks)} tasks (pre+ft, {token_type})",
        filename=filename,
        img_payload_by_mode=img_payload_by_mode,
        txt_payload_by_mode=txt_payload_by_mode,
        show_std=True,
    )


MODELS: dict[str, dict] = {
    "llava_7b": {
        "pre_path": "./results/intrinsic_dim/llava-v1.5-7b-pretrained-lbasile",
        "ft_path": "./results/intrinsic_dim/llava_7b",
        "tasks": ["vqav2", "cocoqa", "gqa", "mmbench", "seed"],
        "highlight_region": {
            "pre": [5, 16],
            "ft": [5, 16],
        },
        "label": "LLaVA 7B",
        "filename_prefix": "id_llava_7b",
    },
    "internvl2_8b": {
        "pre_path": "./results/intrinsic_dim/InternVL2-8B-Pretrain",
        "ft_path": "./results/intrinsic_dim/InternVL2-8B",
        "tasks": ["vqav2", "cocoqa", "gqa", "mmbench", "seed"],
        "highlight_region": {
            "pre": [5, 16],
            "ft": [5, 16],
        },
        "label": "InternVL2-8B",
        "filename_prefix": "id_internvl2_8b",
    },
    "llava_13b": {
        "pre_path": "./results/intrinsic_dim/llava-v1.5-13b-pretrained-lbasile",
        "ft_path": "./results/intrinsic_dim/llava_13b",
        "tasks": ["vqav2", "cocoqa", "gqa", "mmbench", "seed"],
        "highlight_region": {
            "pre": [5, 20],
            "ft": [5, 20],
        },
        "label": "LLaVA 13B",
        "filename_prefix": "id_llava_13b",
    },
    "llava_onevision_8b": {
        "pre_path": "./results/intrinsic_dim/LLaVA-OneVision-1.5-8B-stage1",
        "ft_path": "./results/intrinsic_dim/LLaVA-OneVision-1.5-8B-stage2-2600",
        "tasks": ["vqav2", "cocoqa", "gqa", "mmbench", "seed"],
        "highlight_region": {
            "pre": [5, 16],
            "ft": [5, 16],
        },
        "label": "LLaVA OneVision",
        "filename_prefix": "id_llava_one_vision_8b",
    },
}


TASK_SUFFIX_MAP = {
    "vqav2": "split01_human",
    "cocoqa": "split01_human",
    "gqa": "synthetic",
    "mmbench": "synthetic",
    "seed": "synthetic",
    "mme": "synthetic",
}

MODE_COLORS = {"pre": "#4878CF", "ft": "#D65F5F"}
MODE_LINESTYLES = {"pre": "--", "ft": "-"}
MODE_LABELS = {"pre": "Pretrained", "ft": "Finetuned"}
BACKGROUND_COLOR = "goldenrod"

OUTPUT_DIR = "./results/plots/intrinsic_dim"

GRIDE_SCALE = 2
TOKEN_POSITION = "last_token"
MODEL_NAME = "llava_7b"
TASKS = ["vqav2", "cocoqa", "gqa", "mmbench", "seed"]
MODES = ["pre", "ft"]


cfg = MODELS[MODEL_NAME]

sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

plot_id(
    cfg=cfg,
    tasks=TASKS,
    token_type=TOKEN_POSITION,
    gride_scale=GRIDE_SCALE,
    plot_mode="single",
)

plot_id(
    cfg=cfg,
    tasks=TASKS,
    token_type=TOKEN_POSITION,
    gride_scale=GRIDE_SCALE,
    plot_mode="average",
)

plot_id(
    cfg=cfg,
    tasks=["all_data"],
    token_type=TOKEN_POSITION,
    gride_scale=GRIDE_SCALE,
    plot_mode="all_data",
)

plt.show()
