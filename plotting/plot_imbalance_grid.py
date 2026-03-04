"""
Multimodal Information Imbalance - Layer-Pair Scatter Grids
============================================================

Generates one figure per dataset variant (Open-Ended VQA, VQAv2, Benchmarks).
Each figure is a grid: rows = datasets, cols = models (pretrained + finetuned pairs).

Every cell plots (text_layer, image_layer) coordinates with scatter points.
Marker size is determined by quantile-based bucketing of imbalance values:
  - Open-ended VQA datasets: 20th, 35th, 50th percentiles
  - Benchmark datasets: 30th, 50th, 70th percentiles (empirically tuned for visibility)

Green squares = text->image imbalance (information_imbalance_reversed)
Purple circles = image->text imbalance (information_imbalance)
Goldenrod band (layers 5.5-17.5) marks embedding/attention -> MLP transition
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib

matplotlib.use("Agg")
from safetensors import safe_open

matplotlib.rcParams["font.family"] = "sans-serif"

RESULTS_DIR = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"
    "results/multimodality_comparison"
)
PLOTS_DIR = Path(
    "/u/dssc/lpalaciosflores/multimodal_finetuned_representations/plots/imbalance_grids"
)

# ══════════════════════════════════════════════════════════════════════════════
# VISUAL CONSTANTS  (tuned to match reference single-plot)
# ══════════════════════════════════════════════════════════════════════════════

POOLING_METHODS = ["last", "mean_context"]
VARIANT = "unique_images"

ABSTRACTION = (5.5, 17.5)  # goldenrod band on both axes

STYLE_T_IMG = dict(marker="s", edgecolors="green", facecolors="none", linewidths=1.5)
STYLE_I_TXT = dict(marker="o", edgecolors="purple", facecolors="none", linewidths=1.5)

DPI = 150
CELL_W = 3.2  # inches per cell
CELL_H = 3.0

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
    except Exception as e:
        pass
    return None


def load_all(results_dir: Path, pooling: str) -> dict:
    """Load information_imbalance data for all models and dataset-variant pairs."""
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


def render_cell(ax, t_img, i_txt, thresholds):
    """
    Renders the custom scatter logic.
    t_img: shape (N, N), information_imbalance_reversed
    i_txt: shape (N, N), information_imbalance
    """
    n = 32
    if t_img is not None:
        try:
            n = t_img.shape[0]
        except Exception:
            pass

    # Grid lines and limits
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1, 3))
    ax.set_yticks(range(1, n + 1, 3))
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.4, color="gray", alpha=0.5)

    # Goldenrod abstraction band
    ax.axvspan(ABSTRACTION[0], ABSTRACTION[1], color="goldenrod", alpha=0.18, zorder=0)
    ax.axhspan(ABSTRACTION[0], ABSTRACTION[1], color="goldenrod", alpha=0.18, zorder=0)

    # Index grids: X=text layer, Y=visual layer
    vis_idx, txt_idx = np.indices((n, n))
    X = (txt_idx + 1).astype(float)
    Y = (vis_idx + 1).astype(float)

    for lo, hi, sz, p in thresholds:
        sz_pad = sz * 0.85  # pad to prevent adjacent marker touch-overlap
        try:
            if t_img is not None:
                mask = (t_img >= lo) & (t_img < hi)
                if mask.any():
                    ax.scatter(X[mask], Y[mask], s=sz_pad, zorder=3, **STYLE_T_IMG)
            if i_txt is not None:
                mask = (i_txt >= lo) & (i_txt < hi)
                if mask.any():
                    ax.scatter(X[mask], Y[mask], s=sz_pad, zorder=4, **STYLE_I_TXT)
        except Exception:
            pass


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


def _make_dynamic_legend_handles(thresholds):
    kw = dict(linestyle="None", markerfacecolor="none")
    lo1, hi1, s1, p1 = thresholds[0]
    lo2, hi2, s2, p2 = thresholds[1]
    lo3, hi3, s3, p3 = thresholds[2]

    # Convert area (s) to diameter (markersize in pt)
    d1 = np.sqrt(s1)
    d2 = np.sqrt(s2)
    d3 = np.sqrt(s3)

    return [
        mlines.Line2D(
            [], [], color="green", marker="s", markersize=8, label="text → img", **kw
        ),
        mlines.Line2D(
            [], [], color="purple", marker="o", markersize=8, label="img → text", **kw
        ),
        mlines.Line2D([], [], color="none", marker="None", label=" ", **kw),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            markersize=d1,
            label=f"imb. < {hi1:.3f}",
            **kw,
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            markersize=d2,
            label=f"{lo2:.3f} ≤ imb. < {hi2:.3f}",
            **kw,
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            markersize=d3,
            label=f"{lo3:.3f} ≤ imb. < {hi3:.3f}",
            **kw,
        ),
    ]


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
    nC = len(model_keys)

    # Layout
    wspace = 0.28
    hspace = 0.35
    top_margin = 0.55  # inches
    bottom_margin = 0.55
    left_margin = 1.4  # room for row labels
    right_margin = 0.25

    fig_w = left_margin + nC * CELL_W + (nC - 1) * CELL_W * wspace + right_margin
    fig_h = top_margin + nR * CELL_H + (nR - 1) * CELL_H * hspace + bottom_margin

    print(
        f"      [DEBUG] Calling subplots with nR={nR} nC={nC} fig_w={fig_w} fig_h={fig_h}"
    )
    try:
        import sys

        sys.stdout.flush()
        fig, axes = plt.subplots(nR, nC, figsize=(fig_w, fig_h), squeeze=False)
        print("      [DEBUG] Subplots success")
        sys.stdout.flush()
    except Exception as e:
        print(f"      [DEBUG] EXCEPTION IN SUBPLOTS: {e}")
        import sys

        sys.exit(2)

    fig.subplots_adjust(
        left=left_margin / fig_w,
        right=1 - right_margin / fig_w,
        top=1 - top_margin / fig_h,
        bottom=bottom_margin / fig_h,
        wspace=wspace,
        hspace=hspace,
    )

    legend_placed = False

    for r, row_lbl in enumerate(row_labels):
        is_avg = r == nR - 1
        print(f"      [DEBUG] Row {r}: {row_lbl} (is_avg={is_avg})")

        # 1. Gather all data for this row to compute row-specific thresholds
        row_data = []
        cell_inputs = []  # tuple of (t_img, i_txt) for later rendering
        for c, mod_id in enumerate(model_keys):
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
                row_data.append(t_img.flatten())
            if i_txt is not None:
                row_data.append(i_txt.flatten())

        # Compute dynamic thresholds specifically for this row's data
        thresholds = get_row_thresholds(row_data, stage_key, uniform_logic, n_points=32)

        # 2. Render each cell in the row using the row-specific thresholds
        for c, mod_id in enumerate(model_keys):
            print(f"        [DEBUG] Col {c}: {mod_id}")
            ax = axes[r, c]
            t_img, i_txt = cell_inputs[c]

            if t_img is not None or i_txt is not None:
                print(f"          [DEBUG] calling render_cell")
                render_cell(ax, t_img, i_txt, thresholds)

                # Annotate exact threshold limits on each valid plot
                q1, q2, q3 = thresholds[0][1], thresholds[1][1], thresholds[2][1]
                p1, p2, p3 = thresholds[0][3], thresholds[1][3], thresholds[2][3]
                threshold_txt = f"$q_{{{p1}}}$={q1:.3f}\n$q_{{{p2}}}$={q2:.3f}\n$q_{{{p3}}}$={q3:.3f}"
                ax.text(
                    0.97,
                    0.03,
                    threshold_txt,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="black",
                    va="bottom",
                    ha="right",
                    zorder=10,
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        edgecolor="#aaa",
                        alpha=1.0,
                    ),
                )

                # Legend in top-left cell only
                if not legend_placed:
                    print(f"          [DEBUG] rendering legend")
                    leg = ax.legend(
                        handles=_make_dynamic_legend_handles(thresholds),
                        loc="upper left",
                        fontsize=7,
                        framealpha=0.92,
                        edgecolor="#aaa",
                        handlelength=1.2,
                        borderpad=0.5,
                    )
                    leg.set_zorder(10)
                    legend_placed = True
            else:
                print(f"          [DEBUG] calling render_missing")
                render_missing(ax)

            # Column headers (model names) – top row only
            if r == 0:
                ax.set_title(MODELS[mod_id], fontsize=9, fontweight="bold", pad=5)

            # Axis labels: x only on bottom row, no per-row y-labels
            ax.set_xlabel("text input" if r == nR - 1 else "", fontsize=8)
            ax.set_ylabel("", fontsize=8)

        # Row label to the left of the left-most axis — large offset, no clip
        ax_left = axes[r, 0]
        ax_left.annotate(
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

    print("      [DEBUG] build_stage_figure: returning fig")
    import sys

    sys.stdout.flush()
    return fig

    # Shared axis labels
    l_frac = left_margin / fig_w
    fig.text(
        l_frac * 0.28,
        0.5,
        "visual input",
        ha="center",
        va="center",
        fontsize=9,
        rotation=90,
    )
    fig.text(
        0.5,
        bottom_margin / fig_h * 0.45,
        "text input",
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.suptitle(
        f"Information Imbalance  —  {stage_label}",
        fontsize=12,
        fontweight="bold",
        y=1.002,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Generate Information Imbalance Grid"
        )
        parser.add_argument(
            "--uniform-logic",
            action="store_true",
            help="Force the open-ended quantiles (20/35/50) across all datasets, including benchmarks.",
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
            print("\n" + "-" * 64)
            print(f"  [ POOLING: {pooling} ]")
            print("-" * 64)

            plots_dir = PLOTS_DIR / pooling
            plots_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Loading data for {pooling}...")
            data = load_all(RESULTS_DIR, pooling)

            # Check if any data loaded at all
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

                    print(f"\n  ── {stage_label} ({stage_key}) [{target_model_id}]")
                    import sys

                    sys.stdout.flush()

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
                        print(f"      [DEBUG] Calling savefig for {out}")
                        import sys
                        import resource

                        mem_mb = (
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                        )
                        print(f"      [DEBUG] MEMORY USAGE (Max RSS): {mem_mb:.2f} MB")
                        sys.stdout.flush()

                        try:
                            fig.savefig(str(out), dpi=DPI, facecolor="white")
                        except BaseException as be:
                            print("      [DEBUG] CAUGHT BASE EXCEPTION:", type(be), be)
                            import traceback

                            traceback.print_exc()
                            sys.exit(1)

                        print("      [DEBUG] Savefig complete")
                        plt.close("all")

                        print(f"     ✓  {out.name}")
                        sys.stdout.flush()

                    except Exception:
                        print("      [DEBUG] EXCEPTION IN BUILD/SAVE")
                        import traceback

                        traceback.print_exc()
                        sys.stdout.flush()

        print("\n  Done.")
    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
