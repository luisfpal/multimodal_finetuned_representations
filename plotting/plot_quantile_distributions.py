"""
plot_quantile_distributions.py
──────────────────────────────
Generates one figure per training-stage (Open-Ended VQA / VQAv2 / Benchmarks).
Each figure is a grid: rows = datasets (+ avg row), cols = models.
Two columns per model: combined (gray histogram) and independent (green/purple histograms).
"""

import argparse
import resource
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
from safetensors import safe_open

matplotlib.rcParams["font.family"] = "sans-serif"

RESULTS_DIR = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"
    "results/multimodality_comparison"
)
PLOTS_DIR = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/plots/quantile_distributions"
)

# ══════════════════════════════════════════════════════════════════════════════
# VISUAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

POOLING_METHODS = ["last", "mean_context"]

DPI = 150
CELL_W = 4.0  # wider cells for histograms
CELL_H = 2.5

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODELS = OrderedDict(
    [
        ("llava-v1.5-7b-pretrained-lbasile", "pretrained\n(LLaVA 7B)"),
        ("llava_7b", "finetuned\n(LLaVA 7B)"),
        ("llava-v1.5-13b-pretrained-lbasile", "pretrained\n(LLaVA 13B)"),
        ("llava_13b", "finetuned\n(LLaVA 13B)"),
        ("InternVL2-8B-Pretrain", "pretrained\n(InternVL2 8B)"),
        ("InternVL2-8B", "finetuned\n(InternVL2 8B)"),
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE / DATASET REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

STAGES = [
    (
        "cocoqa_gqa",
        "Open-Ended VQA",
        [
            (
                "cocoqa_restval_icqa_split01_human",
                "unique_images",
                "COCOQA Split01 (Human)",
            ),
            (
                "cocoqa_restval_icqa_split01_synthetic",
                "unique_images",
                "COCOQA Split01 (Synth)",
            ),
            (
                "cocoqa_restval_icqa_split02_human",
                "unique_images",
                "COCOQA Split02 (Human)",
            ),
            (
                "cocoqa_restval_icqa_split02_synthetic",
                "unique_images",
                "COCOQA Split02 (Synth)",
            ),
            ("gqa_val_balanced_noyesno_captioned_synthetic", "unique_images", "GQA"),
        ],
    ),
    (
        "vqav2",
        "VQAv2",
        [
            (
                "vqav2_coco_restval_icq_split01_human",
                "unique_images",
                "VQAv2 S01 (Human)",
            ),
            (
                "vqav2_coco_restval_icq_split01_synthetic",
                "unique_images",
                "VQAv2 S01 (Synth)",
            ),
            (
                "vqav2_coco_restval_icq_split02_human",
                "unique_images",
                "VQAv2 S02 (Human)",
            ),
            (
                "vqav2_coco_restval_icq_split02_synthetic",
                "unique_images",
                "VQAv2 S02 (Synth)",
            ),
            (
                "vqav2_coco_restval_icq_split03_human",
                "unique_images",
                "VQAv2 S03 (Human)",
            ),
            (
                "vqav2_coco_restval_icq_split03_synthetic",
                "unique_images",
                "VQAv2 S03 (Synth)",
            ),
        ],
    ),
    (
        "single_base",
        "Benchmarks (Repeated Images)",
        [
            ("mmbench_en_dev_captioned_synthetic", "repeated_images", "MMBench"),
            ("mme_test_captioned_synthetic", "repeated_images", "MME"),
            ("pope_test_captioned_synthetic", "repeated_images", "POPE"),
            ("seed_bench_images_captioned_synthetic", "repeated_images", "SEED"),
        ],
    ),
    (
        "single_unique",
        "Benchmarks (Unique Images)",
        [
            ("mmbench_en_dev_captioned_synthetic", "unique_images", "MMBench (Unique)"),
            ("mme_test_captioned_synthetic", "unique_images", "MME (Unique)"),
            ("pope_test_captioned_synthetic", "unique_images", "POPE (Unique)"),
            ("seed_bench_images_captioned_synthetic", "unique_images", "SEED (Unique)"),
        ],
    ),
    (
        "open_ended",
        "Benchmarks (Open-Ended)",
        [
            (
                "mmbench_en_dev_captioned_open_ended_synthetic",
                "unique_images",
                "MMBench (Open-Ended)",
            ),
            (
                "seed_bench_images_captioned_open_ended_synthetic",
                "unique_images",
                "SEED (Open-Ended)",
            ),
        ],
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════


def _load_tensor(
    results_dir: Path,
    model_id: str,
    dataset: str,
    variant: str,
    metric: str,
    pooling: str,
):
    p = (
        results_dir
        / dataset
        / pooling
        / variant
        / "similarities"
        / f"{model_id}_{metric}.safetensors"
    )
    if not p.exists():
        return None
    try:
        with safe_open(p, framework="np", device="cpu") as f:
            if metric in f.keys():
                return f.get_tensor(metric).copy()
    except Exception:
        pass
    return None


def load_all(results_dir: Path, pooling: str) -> dict:
    data = OrderedDict()
    all_datasets_variants = {
        (ds, var) for _, _, ds_list in STAGES for ds, var, _ in ds_list
    }

    for model_id, label in MODELS.items():
        model_data = {}
        for dataset, variant in all_datasets_variants:
            arrs = {}
            for metric in ["information_imbalance_reversed", "information_imbalance"]:
                arr = _load_tensor(
                    results_dir, model_id, dataset, variant, metric, pooling
                )
                if arr is not None:
                    arrs[metric] = arr
            if arrs:
                model_data[f"{dataset}_{variant}"] = arrs
        data[model_id] = model_data
        n = len(model_data)
        status = "✓" if n else "✗"
        print(f"  {status}  {label:<30} – {n} dataset(s)")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-CELL RENDERER  (matches reference exactly)
# ══════════════════════════════════════════════════════════════════════════════


def render_cell_combined(ax, t_img, i_txt, thresholds_c, x_limit=None):
    """
    Renders the COMBINED (pooled) histogram and its threshold lines.
    Gray histogram, blue/orange/red dashed quantile lines.
    """
    vals = []
    if t_img is not None:
        vals.append(t_img.flatten())
    if i_txt is not None:
        vals.append(i_txt.flatten())
    if not vals:
        return
    cell_data = np.concatenate(vals)
    if x_limit is None:
        q3_c = thresholds_c[2][1]
        xmax = float(np.percentile(cell_data, 98))
        x_limit = max(xmax, q3_c * 1.15)

    ax.hist(
        cell_data,
        bins=100,
        range=(0, x_limit),
        color="gray",
        alpha=0.45,
        density=True,
        edgecolor="none",
    )
    q1_c, q2_c, q3_c = thresholds_c[0][1], thresholds_c[1][1], thresholds_c[2][1]
    ax.axvline(q1_c, color="blue", linestyle="--", lw=1.5, alpha=0.85, zorder=5)
    ax.axvline(q2_c, color="orange", linestyle="--", lw=1.5, alpha=0.85, zorder=5)
    ax.axvline(q3_c, color="red", linestyle="--", lw=1.5, alpha=0.85, zorder=5)
    ax.axvspan(0, q1_c, color="blue", alpha=0.08, zorder=0)
    ax.axvspan(q1_c, q2_c, color="orange", alpha=0.08, zorder=0)
    ax.axvspan(q2_c, q3_c, color="red", alpha=0.08, zorder=0)
    ax.set_xlim(left=0, right=x_limit)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.4, color="gray", alpha=0.3, axis="x")
    ax.grid(True, lw=0.4, color="gray", alpha=0.15, axis="y")


def render_cell_independent(ax, t_img, i_txt, thresholds_t, thresholds_i, x_limit=None):
    """
    Renders the per-MODALITY histograms: green for T→I, purple for I→T.
    Each modality's quantile lines use dotted/dash-dot style.
    """
    vals = []
    if t_img is not None:
        vals.append(t_img.flatten())
    if i_txt is not None:
        vals.append(i_txt.flatten())
    if not vals:
        return
    cell_data = np.concatenate(vals)
    q3_t = thresholds_t[2][1] if thresholds_t else 0
    q3_i = thresholds_i[2][1] if thresholds_i else 0
    if x_limit is None:
        xmax = float(np.percentile(cell_data, 98))
        x_limit = max(xmax, max(q3_t, q3_i) * 1.15)

    if t_img is not None:
        ax.hist(
            t_img.flatten(),
            bins=100,
            range=(0, x_limit),
            color="#2ca02c",
            alpha=0.45,
            density=True,
            edgecolor="none",
        )
        if thresholds_t:
            q1_t, q2_t, q3_t = (
                thresholds_t[0][1],
                thresholds_t[1][1],
                thresholds_t[2][1],
            )
            ax.axvline(
                q1_t, color="#2ca02c", linestyle=":", lw=2.2, alpha=0.85, zorder=6
            )
            ax.axvline(
                q2_t, color="#2ca02c", linestyle=":", lw=2.2, alpha=0.85, zorder=6
            )
            ax.axvline(
                q3_t, color="#2ca02c", linestyle=":", lw=2.2, alpha=0.85, zorder=6
            )

    if i_txt is not None:
        ax.hist(
            i_txt.flatten(),
            bins=100,
            range=(0, x_limit),
            color="#9467bd",
            alpha=0.45,
            density=True,
            edgecolor="none",
        )
        if thresholds_i:
            q1_i, q2_i, q3_i = (
                thresholds_i[0][1],
                thresholds_i[1][1],
                thresholds_i[2][1],
            )
            ax.axvline(
                q1_i, color="#9467bd", linestyle="-.", lw=1.8, alpha=0.85, zorder=7
            )
            ax.axvline(
                q2_i, color="#9467bd", linestyle="-.", lw=1.8, alpha=0.85, zorder=7
            )
            ax.axvline(
                q3_i, color="#9467bd", linestyle="-.", lw=1.8, alpha=0.85, zorder=7
            )

    ax.set_xlim(left=0, right=x_limit)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.4, color="gray", alpha=0.3, axis="x")
    ax.grid(True, lw=0.4, color="gray", alpha=0.15, axis="y")


def render_missing(ax):
    ax.set_facecolor("#ececec")
    ax.text(
        0.5,
        0.5,
        "N/A",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#999",
        style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#ccc")


# ══════════════════════════════════════════════════════════════════════════════
# LEGEND (matches reference: inside upper-left)
# ══════════════════════════════════════════════════════════════════════════════


def _make_combined_legend_handles(thresholds_c):
    """Legend handles for the combined column."""
    lo1, hi1, s1, p1 = thresholds_c[0]
    lo2, hi2, s2, p2 = thresholds_c[1]
    lo3, hi3, s3, p3 = thresholds_c[2]
    return [
        mlines.Line2D(
            [],
            [],
            color="gray",
            marker="s",
            markersize=8,
            markeredgecolor="none",
            alpha=0.5,
            label="T→I + I→T (pooled)",
            linestyle="None",
        ),
        mlines.Line2D(
            [],
            [],
            color="blue",
            marker="None",
            linestyle="--",
            lw=1.5,
            label=f"$q_{{{p1}}}$ = {hi1:.3f}",
        ),
        mlines.Line2D(
            [],
            [],
            color="orange",
            marker="None",
            linestyle="--",
            lw=1.5,
            label=f"$q_{{{p2}}}$ = {hi2:.3f}",
        ),
        mlines.Line2D(
            [],
            [],
            color="red",
            marker="None",
            linestyle="--",
            lw=1.5,
            label=f"$q_{{{p3}}}$ = {hi3:.3f}",
        ),
    ]


def _make_independent_legend_handles(thresholds_t, thresholds_i):
    """Legend handles for the independent column."""
    ret = [
        mlines.Line2D(
            [],
            [],
            color="#2ca02c",
            marker="s",
            markersize=8,
            markeredgecolor="none",
            alpha=0.5,
            label="T→I distribution",
            linestyle="None",
        ),
        mlines.Line2D(
            [],
            [],
            color="#9467bd",
            marker="o",
            markersize=8,
            markeredgecolor="none",
            alpha=0.5,
            label="I→T distribution",
            linestyle="None",
        ),
    ]
    if thresholds_t and thresholds_t[0][1] != 0.03:
        p1t, p2t, p3t = thresholds_t[0][3], thresholds_t[1][3], thresholds_t[2][3]
        q1t, q2t, q3t = thresholds_t[0][1], thresholds_t[1][1], thresholds_t[2][1]
        ret += [
            mlines.Line2D(
                [],
                [],
                color="#2ca02c",
                marker="None",
                linestyle=":",
                lw=2.2,
                label=f"T→I  $q_{{{p1t}}}$={q1t:.3f}  $q_{{{p2t}}}$={q2t:.3f}  $q_{{{p3t}}}$={q3t:.3f}",
            ),
        ]
    if thresholds_i and thresholds_i[0][1] != 0.03:
        p1i, p2i, p3i = thresholds_i[0][3], thresholds_i[1][3], thresholds_i[2][3]
        q1i, q2i, q3i = thresholds_i[0][1], thresholds_i[1][1], thresholds_i[2][1]
        ret += [
            mlines.Line2D(
                [],
                [],
                color="#9467bd",
                marker="None",
                linestyle="-.",
                lw=1.8,
                label=f"I→T  $q_{{{p1i}}}$={q1i:.3f}  $q_{{{p2i}}}$={q2i:.3f}  $q_{{{p3i}}}$={q3i:.3f}",
            ),
        ]
    return ret


# ══════════════════════════════════════════════════════════════════════════════
# STAGE AVERAGE
# ══════════════════════════════════════════════════════════════════════════════


def stage_avg(model_id: str, stage_datasets: list, data: dict):
    t_imgs, i_txts = [], []
    model_data = data.get(model_id, {})
    for ds, var, _ in stage_datasets:
        entry = model_data.get(f"{ds}_{var}", {})
        if "information_imbalance_reversed" in entry:
            t_imgs.append(entry["information_imbalance_reversed"])
        if "information_imbalance" in entry:
            i_txts.append(entry["information_imbalance"])

    shapes_t = {a.shape for a in t_imgs}
    shapes_i = {a.shape for a in i_txts}
    t_img = (
        np.mean(np.stack(t_imgs), 0)
        if len(t_imgs) >= 2 and len(shapes_t) == 1
        else None
    )
    i_txt = (
        np.mean(np.stack(i_txts), 0)
        if len(i_txts) >= 2 and len(shapes_i) == 1
        else None
    )
    return t_img, i_txt


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════


def get_row_thresholds(row_data_items, stage_key, uniform_logic, n_points):
    # Calculate strict non-overlapping marker size s_max based on exact grid geometry in points
    dx_pt = (CELL_W * 72) / n_points
    dy_pt = (CELL_H * 72) / n_points
    s_max = min(dx_pt, dy_pt) ** 2

    if not row_data_items:
        return [
            (0, 0.03, s_max, 20),
            (0.03, 0.05, s_max * 0.25, 35),
            (0.05, 0.07, s_max * 0.05, 50),
        ]

    combo = np.concatenate(row_data_items)

    # Benchmarks usually have higher overall imbalance (lower information)
    # so we need to grab lower quantiles to keep the plot dense.
    if not uniform_logic and stage_key in ("single_base", "single_unique"):
        # Experimental benchmark quantiles (tuning targets)
        p1, p2, p3 = 30, 50, 70
        q1 = float(np.percentile(combo, p1))
        q2 = float(np.percentile(combo, p2))
        q3 = float(np.percentile(combo, p3))
    else:
        # Standard open-ended quantiles
        p1, p2, p3 = 20, 35, 50
        q1 = float(np.percentile(combo, p1))
        q2 = float(np.percentile(combo, p2))
        q3 = float(np.percentile(combo, p3))

    return [
        (0.000, q1, s_max, p1),
        (q1, q2, s_max * 0.25, p2),
        (q2, q3, s_max * 0.05, p3),
    ]


def build_stage_figure(
    stage_key: str,
    stage_label: str,
    stage_datasets: list,
    model_keys: list,
    data: dict,
    uniform_logic: bool,
) -> plt.Figure:
    row_labels = [lbl for _, _, lbl in stage_datasets] + ["Average"]
    nR = len(row_labels)
    nMod = len(model_keys)
    nC = nMod * 2  # two columns per model: combined + independent

    # Layout
    wspace = 0.28
    hspace = 0.35
    top_margin = 0.55  # inches
    bottom_margin = 0.55
    left_margin = 1.4  # room for row labels
    right_margin = 0.25

    fig_w = left_margin + nC * CELL_W + (nC - 1) * CELL_W * wspace + right_margin
    fig_h = top_margin + nR * CELL_H + (nR - 1) * CELL_H * hspace + bottom_margin

    fig, axes = plt.subplots(nR, nC, figsize=(fig_w, fig_h), squeeze=False)

    fig.subplots_adjust(
        left=left_margin / fig_w,
        right=1 - right_margin / fig_w,
        top=1 - top_margin / fig_h,
        bottom=bottom_margin / fig_h,
        wspace=wspace,
        hspace=hspace,
    )


    legend_combined_placed = False
    legend_indep_placed = False

    for r, row_lbl in enumerate(row_labels):
        is_avg = r == nR - 1

        # Gather data and compute ALL three threshold sets once per row
        row_t = []
        row_i = []
        cell_inputs = []
        for mod_id in model_keys:
            model_data = data.get(mod_id, {})
            if is_avg:
                t_img, i_txt = stage_avg(mod_id, stage_datasets, data)
            else:
                ds_name, var_name, _ = stage_datasets[r]
                entry = model_data.get(f"{ds_name}_{var_name}", {})
                t_img = entry.get("information_imbalance_reversed")
                i_txt = entry.get("information_imbalance")
            cell_inputs.append((t_img, i_txt))
            if t_img is not None:
                row_t.append(t_img.flatten())
            if i_txt is not None:
                row_i.append(i_txt.flatten())

        thresholds_c = get_row_thresholds(
            row_t + row_i, stage_key, uniform_logic, n_points=32
        )
        thresholds_t = (
            get_row_thresholds(row_t, stage_key, uniform_logic, n_points=32)
            if row_t
            else None
        )
        thresholds_i = (
            get_row_thresholds(row_i, stage_key, uniform_logic, n_points=32)
            if row_i
            else None
        )

        # Shared x-limit across all 4 columns so plots are directly comparable
        all_vals = row_t + row_i
        if all_vals:
            all_data = np.concatenate(all_vals)
            q3_max = max(
                thresholds_c[2][1],
                thresholds_t[2][1] if thresholds_t else 0,
                thresholds_i[2][1] if thresholds_i else 0,
            )
            x_limit = max(float(np.percentile(all_data, 98)), q3_max * 1.15)
        else:
            x_limit = 1.0

        for c, (mod_id, (t_img, i_txt)) in enumerate(zip(model_keys, cell_inputs)):
            col_comb = c * 2  # combined column for this model
            col_indep = c * 2 + 1  # independent column for this model

            ax_comb = axes[r, col_comb]
            ax_indep = axes[r, col_indep]

            has_data = t_img is not None or i_txt is not None

            # ── Combined column ──────────────────────────────────────────────
            if has_data:
                render_cell_combined(
                    ax_comb, t_img, i_txt, thresholds_c, x_limit=x_limit
                )

                # Annotation: combined quantiles only
                p1_c, p2_c, p3_c = (
                    thresholds_c[0][3],
                    thresholds_c[1][3],
                    thresholds_c[2][3],
                )
                q1_c, q2_c, q3_c = (
                    thresholds_c[0][1],
                    thresholds_c[1][1],
                    thresholds_c[2][1],
                )
                ann = (
                    f"$q_{{{p1_c}}}$={q1_c:.3f}\n"
                    f"$q_{{{p2_c}}}$={q2_c:.3f}\n"
                    f"$q_{{{p3_c}}}$={q3_c:.3f}"
                )
                ax_comb.text(
                    0.97,
                    0.97,
                    ann,
                    transform=ax_comb.transAxes,
                    fontsize=7,
                    color="black",
                    va="top",
                    ha="right",
                    zorder=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="#aaa",
                        alpha=1.0,
                    ),
                )

                if not legend_combined_placed:
                    leg = ax_comb.legend(
                        handles=_make_combined_legend_handles(thresholds_c),
                        loc="upper left",
                        fontsize=7,
                        framealpha=0.92,
                        edgecolor="#aaa",
                        handlelength=1.2,
                        borderpad=0.5,
                    )
                    leg.set_zorder(10)
                    legend_combined_placed = True
            else:
                render_missing(ax_comb)

            # ── Independent column ───────────────────────────────────────────
            if has_data:
                render_cell_independent(
                    ax_indep, t_img, i_txt, thresholds_t, thresholds_i, x_limit=x_limit
                )

                # Annotation: T→I and I→T quantiles
                ann_lines = []
                if thresholds_t:
                    p1t, p2t, p3t = (
                        thresholds_t[0][3],
                        thresholds_t[1][3],
                        thresholds_t[2][3],
                    )
                    q1t, q2t, q3t = (
                        thresholds_t[0][1],
                        thresholds_t[1][1],
                        thresholds_t[2][1],
                    )
                    ann_lines.append(
                        f"T→I  $q_{{{p1t}}}$={q1t:.3f}  $q_{{{p2t}}}$={q2t:.3f}  $q_{{{p3t}}}$={q3t:.3f}"
                    )
                if thresholds_i:
                    p1i, p2i, p3i = (
                        thresholds_i[0][3],
                        thresholds_i[1][3],
                        thresholds_i[2][3],
                    )
                    q1i, q2i, q3i = (
                        thresholds_i[0][1],
                        thresholds_i[1][1],
                        thresholds_i[2][1],
                    )
                    ann_lines.append(
                        f"I→T  $q_{{{p1i}}}$={q1i:.3f}  $q_{{{p2i}}}$={q2i:.3f}  $q_{{{p3i}}}$={q3i:.3f}"
                    )
                if ann_lines:
                    ax_indep.text(
                        0.97,
                        0.97,
                        "\n".join(ann_lines),
                        transform=ax_indep.transAxes,
                        fontsize=7,
                        color="black",
                        va="top",
                        ha="right",
                        zorder=10,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="#aaa",
                            alpha=1.0,
                        ),
                    )

                if not legend_indep_placed:
                    leg = ax_indep.legend(
                        handles=_make_independent_legend_handles(
                            thresholds_t, thresholds_i
                        ),
                        loc="upper left",
                        fontsize=7,
                        framealpha=0.92,
                        edgecolor="#aaa",
                        handlelength=1.2,
                        borderpad=0.5,
                    )
                    leg.set_zorder(10)
                    legend_indep_placed = True
            else:
                render_missing(ax_indep)

            # ── Row label (left of row) ──────────────────────────────────────
            if c == 0:
                ax_comb.annotate(
                    row_lbl,
                    xy=(0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-52, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    rotation=90,
                    annotation_clip=False,
                )

            # ── Column headers (top row only) ────────────────────────────────
            if r == 0:
                model_label = MODELS[mod_id]
                ax_comb.set_title(
                    f"{model_label}\n(Combined)", fontsize=8, fontweight="bold", pad=4
                )
                ax_indep.set_title(
                    f"{model_label}\n(Independent)",
                    fontsize=8,
                    fontweight="bold",
                    pad=4,
                )

            # ── X-axis label (bottom row only) ───────────────────────────────
            lbl = "Imbalance Distance" if r == nR - 1 else ""
            ax_comb.set_xlabel(lbl, fontsize=8)
            ax_indep.set_xlabel(lbl, fontsize=8)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Generate Information Imbalance Grid")
    parser.add_argument(
        "--uniform-logic",
        action="store_true",
        help="Force the open-ended quantiles (20/35/45) across all datasets, including benchmarks.",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  Information Imbalance Grid")
    if args.uniform_logic:
        print("  [WARNING: Running with --uniform-logic !]")
    print("=" * 64)
    print(f"\n  results : {RESULTS_DIR}")
    print(f"  plots   : {PLOTS_DIR}")

    for pooling in POOLING_METHODS:
        print("\n" + "─" * 64)
        print(f"  [ POOLING: {pooling} ]")
        print("─" * 64)

        plots_dir = PLOTS_DIR / pooling
        plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Loading data for {pooling}...")
        data = load_all(RESULTS_DIR, pooling)

        if not any(data.values()):
            print(f"  No data found for {pooling}. Skipping.")
            continue

        print("\n  Generating figures...")

        CROSS_MODEL_PAIRS = [
            ("llava_7b", "llava-v1.5-7b-pretrained-lbasile"),
            ("llava_13b", "llava-v1.5-13b-pretrained-lbasile"),
            ("InternVL2-8B", "InternVL2-8B-Pretrain"),
        ]

        jobs = []
        paired_set = set()
        for m_fine, m_pre in CROSS_MODEL_PAIRS:
            group = [m for m in (m_pre, m_fine) if m in data]
            if group:
                jobs.append((m_fine, group))
                paired_set.update(group)

        for model_id in MODELS:
            if model_id in data and model_id not in paired_set:
                jobs.append((model_id, [model_id]))

        for target_model_id, model_keys in jobs:
            for stage_key, stage_label, stage_datasets in STAGES:
                if pooling == "mean_context" and stage_key == "single_base":
                    print(
                        f"\n  ── Skipping {stage_label} ({stage_key}) for {pooling} [No Data]"
                    )
                    continue

                stage_has_data = any(
                    f"{ds}_{var}" in data.get(mod_id, {})
                    for mod_id in model_keys
                    for ds, var, _ in stage_datasets
                )
                if not stage_has_data:
                    print(
                        f"\n  ── Skipping {stage_label} ({stage_key}) for {target_model_id} [No Data]"
                    )
                    continue

                print(f"\n  ── {stage_label} ({stage_key}) [{target_model_id}]")
                try:
                    fig = build_stage_figure(
                        stage_key,
                        f"{stage_label} - {MODELS[target_model_id]}",
                        stage_datasets,
                        model_keys,
                        data,
                        args.uniform_logic,
                    )
                    out = plots_dir / f"{target_model_id}_{stage_key}.png"
                    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    fig.savefig(str(out), dpi=DPI, facecolor="white")
                    plt.close("all")
                    print(f"    saving {out.name}  (RSS {mem_mb:.0f} MB)")
                    sys.stdout.flush()
                except Exception:
                    traceback.print_exc()
                    sys.stdout.flush()

    print("\n  Done.")


if __name__ == "__main__":
    main()
