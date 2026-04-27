"""Cambrian encoder ablation: heatmap and imbalance scatter figures.

Plots how multimodality similarity changes when the Cambrian-8B pretrained
model uses fewer vision encoders (1, 2, 3 vs the default 4).

Output:
    plots/similarity_grids/last/cambrian_encoder_ablation/
        linear_cka/           1_encoder.png  2_encoders.png  3_encoders.png  4_encoders.png
        neighborhood_overlap/ (same)
        information_imbalance/(same, scatter style)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open


# ==============================================================================
# SECTION 1 — CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path("/u/dssc/lpalaciosflores/multimodal_finetuned_representations")
RESULTS_DIR = PROJECT_ROOT / "results" / "multimodality_comparison"
PLOTS_DIR = (
    PROJECT_ROOT / "plots" / "similarity_grids" / "last" / "cambrian_encoder_ablation"
)

POOLING_METHOD = "last"
VARIANT_TAG = "unique_images"
DPI_PNG = 200

HEATMAP_METRICS: tuple[str, ...] = ("linear_cka", "neighborhood_overlap")

# Used for 1–3 encoder ablation variants
DATASETS_TO_AVERAGE: tuple[str, ...] = (
    "cocoqa_restval_icqa_human",
    "gqa_val_balanced_noyesno_captioned_synthetic",
    "vqav2_coco_restval_icq_synthetic",
    "mmbench_en_dev_captioned_open_ended_synthetic",
    "seed_bench_images_captioned_open_ended_synthetic",
)

# Used for the 4-encoder default model (different dataset splits)
DATASETS_TO_AVERAGE_4ENC: tuple[str, ...] = (
    "cocoqa_restval_icqa_split01_human",
    "gqa_val_balanced_noyesno_captioned_synthetic",
    "vqav2_coco_restval_icq_split01_human",
    "mmbench_en_dev_captioned_open_ended_synthetic",
    "seed_bench_images_captioned_open_ended_synthetic",
)

# (n_encoders, subtitle_label, model_id)
ENCODER_CONFIGS: tuple[tuple[int, str, str], ...] = (
    (1, "siglip", "cambrian-8b-pretrained__vision-siglip"),
    (1, "clip", "cambrian-8b-pretrained__vision-clip"),
    (1, "dinov2", "cambrian-8b-pretrained__vision-dinov2"),
    (1, "convnext", "cambrian-8b-pretrained__vision-convnext"),
    (2, "siglip+clip", "cambrian-8b-pretrained__vision-siglip+clip"),
    (2, "siglip+dinov2", "cambrian-8b-pretrained__vision-siglip+dinov2"),
    (2, "siglip+convnext", "cambrian-8b-pretrained__vision-siglip+convnext"),
    (2, "clip+dinov2", "cambrian-8b-pretrained__vision-clip+dinov2"),
    (2, "clip+convnext", "cambrian-8b-pretrained__vision-clip+convnext"),
    (2, "dinov2+convnext", "cambrian-8b-pretrained__vision-dinov2+convnext"),
    (3, "siglip+clip+dinov2", "cambrian-8b-pretrained__vision-siglip+clip+dinov2"),
    (3, "siglip+clip+convnext", "cambrian-8b-pretrained__vision-siglip+clip+convnext"),
    (3, "siglip+dinov2+convnext", "cambrian-8b-pretrained__vision-siglip+dinov2+convnext"),
    (3, "clip+dinov2+convnext", "cambrian-8b-pretrained__vision-clip+dinov2+convnext"),
    (4, "siglip+clip+dinov2+convnext", "cambrian-8b-pretrained"),
)

EXPECTED_SOURCE_ORIENTATION = ("text", "image")
DISPLAY_ORIENTATION = ("image", "text")

# Scatter / imbalance visual constants (verbatim from plot_imbalance_grid.py)
ABSTRACTION = (5.5, 17.5)
STYLE_T_IMG = dict(marker="o", edgecolors="purple", facecolors="none", linewidths=0.85)
STYLE_I_TXT = dict(marker="s", edgecolors="green", facecolors="none", linewidths=0.85)
MARKER_SIZE_SCALE = 0.63
SCATTER_CELL_W = 3.2
SCATTER_CELL_H = 3.0

HEATMAP_CELL = 3.5  # inches, square cells


# ==============================================================================
# SECTION 2 — DATA LOADING
# ==============================================================================


def _similarities_path(dataset_dir: str, model_id: str, metric: str) -> Path:
    return (
        RESULTS_DIR
        / dataset_dir
        / POOLING_METHOD
        / VARIANT_TAG
        / "similarities"
        / f"{model_id}_{metric}.safetensors"
    )


def _normalize_modality_label(value: str | None, fallback: str) -> str:
    label = str(value or "").strip().lower()
    return label if label in {"text", "image"} else fallback


def _resolve_matrix_orientation(metadata: dict[str, str] | None) -> tuple[str, str]:
    if not metadata:
        return EXPECTED_SOURCE_ORIENTATION
    row_modality = _normalize_modality_label(
        metadata.get("row_modality"), EXPECTED_SOURCE_ORIENTATION[0]
    )
    col_modality = _normalize_modality_label(
        metadata.get("col_modality"), EXPECTED_SOURCE_ORIENTATION[1]
    )
    if row_modality == col_modality:
        return EXPECTED_SOURCE_ORIENTATION
    return row_modality, col_modality


def _orient_for_display(
    matrix: np.ndarray,
    source_orientation: tuple[str, str],
) -> np.ndarray:
    if source_orientation == DISPLAY_ORIENTATION:
        return matrix
    return matrix.T


def load_matrix(
    dataset_dir: str,
    model_id: str,
    metric: str,
    *,
    transform_imbalance: bool = True,
) -> np.ndarray | None:
    path = _similarities_path(dataset_dir, model_id, metric)
    if not path.exists():
        return None
    try:
        with safe_open(path, framework="np", device="cpu") as f:
            if metric not in f.keys():
                return None
            matrix = f.get_tensor(metric).copy()
            source_orientation = _resolve_matrix_orientation(f.metadata())
    except Exception as exc:
        print(f"  ⚠ failed to load {path}: {exc}")
        return None

    matrix = _orient_for_display(matrix, source_orientation)
    if transform_imbalance and "information_imbalance" in metric:
        matrix = 1.0 - matrix
    return matrix


def average_over_datasets(
    model_id: str,
    metric: str,
    *,
    datasets: tuple[str, ...] = DATASETS_TO_AVERAGE,
    transform_imbalance: bool = True,
) -> np.ndarray | None:
    """Average over available datasets; warn on missing ones, None only if all fail."""
    matrices: list[np.ndarray] = []
    missing: list[str] = []

    for dataset_dir in datasets:
        m = load_matrix(dataset_dir, model_id, metric, transform_imbalance=transform_imbalance)
        if m is None:
            missing.append(dataset_dir)
        else:
            matrices.append(m)

    if missing:
        print(
            f"  ⚠ {model_id}/{metric}: {len(missing)}/{len(datasets)} datasets missing"
        )
    if not matrices:
        return None
    shapes = {m.shape for m in matrices}
    if len(shapes) != 1:
        print(f"  ⚠ {model_id}/{metric}: shape mismatch {sorted(shapes)}, skipping")
        return None
    return np.mean(np.stack(matrices), axis=0)


def load_imbalance_pair(
    model_id: str,
    *,
    datasets: tuple[str, ...] = DATASETS_TO_AVERAGE,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load raw (t_img, i_txt) averages for scatter rendering (no 1-x transform)."""
    t_img = average_over_datasets(
        model_id, "information_imbalance_reversed", datasets=datasets, transform_imbalance=False
    )
    i_txt = average_over_datasets(
        model_id, "information_imbalance", datasets=datasets, transform_imbalance=False
    )
    return t_img, i_txt


# ==============================================================================
# SECTION 3 — HEATMAP FIGURES (linear_cka, neighborhood_overlap)
# ==============================================================================


def _apply_publication_style(
    fontsize_title: int,
    fontsize_label: int,
    fontsize_ticks: int,
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.titlesize"] = fontsize_title
    plt.rcParams["axes.labelsize"] = fontsize_label
    plt.rcParams["xtick.labelsize"] = fontsize_ticks
    plt.rcParams["ytick.labelsize"] = fontsize_ticks
    plt.rcParams["figure.titlesize"] = fontsize_title + 2
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _layer_ticks(num_layers: int) -> range:
    return range(0, num_layers, 5)


def _render_heatmap_cell(
    ax: plt.Axes,
    matrix: np.ndarray | None,
    vmin: float,
    vmax: float,
    fontsize_ticks: int,
) -> matplotlib.image.AxesImage | None:
    if matrix is None:
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
        return None

    im = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks(_layer_ticks(matrix.shape[1]))
    ax.set_yticks(_layer_ticks(matrix.shape[0]))
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=1.2,
        color="#444444",
        labelsize=fontsize_ticks,
    )
    ax.grid(False)
    return im


def _compute_vmin_vmax(
    matrices: list[np.ndarray | None],
) -> tuple[float, float]:
    valid = [m for m in matrices if m is not None]
    if not valid:
        return 0.0, 1.0
    return float(min(m.min() for m in valid)), float(max(m.max() for m in valid))


def build_heatmap_figure(
    n_encoders: int,
    combos: list[tuple[str, str]],  # [(label, model_id), ...]
    metric: str,
    matrices: list[np.ndarray | None],
) -> plt.Figure:
    _apply_publication_style(fontsize_title=11, fontsize_label=9, fontsize_ticks=7)

    n_cols = len(combos)
    vmin, vmax = _compute_vmin_vmax(matrices)

    left_m = 0.80
    right_m = 0.15
    top_m = 1.10
    bottom_m = 0.70
    cbar_gap = 0.20
    cbar_w_in = 0.18

    cell_total = HEATMAP_CELL
    wspace_frac = 0.10
    wspace_in = cell_total * wspace_frac

    fig_w = (
        left_m
        + n_cols * cell_total
        + (n_cols - 1) * wspace_in
        + cbar_gap
        + cbar_w_in
        + right_m
    )
    fig_h = top_m + cell_total + bottom_m

    data_right = left_m + n_cols * cell_total + (n_cols - 1) * wspace_in
    gs_right = (data_right + cbar_gap / 2) / fig_w
    gs_left = left_m / fig_w
    gs_top = 1.0 - top_m / fig_h
    gs_bottom = bottom_m / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        1,
        n_cols + 1,
        width_ratios=[1] * n_cols + [cbar_w_in / cell_total],
        wspace=wspace_frac,
        left=gs_left,
        right=gs_right,
        top=gs_top,
        bottom=gs_bottom,
    )

    axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    last_im = None

    for c, (label, _model_id) in enumerate(combos):
        ax = axes[c]
        im = _render_heatmap_cell(ax, matrices[c], vmin, vmax, fontsize_ticks=7)
        if im is not None:
            last_im = im

        ax.set_title(label, fontsize=8, pad=5, color="#222222")
        ax.set_xlabel("text input" if c == 0 else "", labelpad=6, color="#333333", fontsize=8)
        ax.set_ylabel("visual input" if c == 0 else "", labelpad=6, color="#333333", fontsize=8)

    encoder_word = "encoder" if n_encoders == 1 else "encoders"
    metric_label = metric.replace("_", " ")
    fig.suptitle(
        f"Pretrained Cambrian 8B — {n_encoders} {encoder_word} — {metric_label}",
        fontsize=12,
        fontweight="bold",
        y=1.0 - (top_m * 0.12) / fig_h,
    )

    if last_im is not None:
        fig.canvas.draw()
        pos = axes[-1].get_position()
        cax = fig.add_axes(
            [pos.x1 + cbar_gap / fig_w, pos.y0, cbar_w_in / fig_w, pos.height]
        )
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0, labelsize=7)

    return fig


def save_png(fig: plt.Figure, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    return path


# ==============================================================================
# SECTION 4 — IMBALANCE SCATTER FIGURES (information_imbalance)
# ==============================================================================


def get_row_thresholds(
    row_data_items: list[np.ndarray],
    n_points: int = 32,
) -> list[tuple[float, float, float, int]]:
    dx_pt = (SCATTER_CELL_W * 72) / n_points
    dy_pt = (SCATTER_CELL_H * 72) / n_points
    s_max = min(dx_pt, dy_pt) ** 2

    if not row_data_items:
        return [
            (0.0, 0.03, s_max, 20),
            (0.03, 0.05, s_max * 0.25, 35),
            (0.05, 0.07, s_max * 0.05, 50),
        ]

    combo = np.concatenate(row_data_items)
    p1, p2, p3 = 30, 50, 70
    q1 = float(np.percentile(combo, p1))
    q2 = float(np.percentile(combo, p2))
    q3 = float(np.percentile(combo, p3))

    return [
        (0.0, q1, s_max, p1),
        (q1, q2, s_max * 0.25, p2),
        (q2, q3, s_max * 0.05, p3),
    ]


def render_cell(
    ax: plt.Axes,
    t_img: np.ndarray | None,
    i_txt: np.ndarray | None,
    thresholds: list[tuple[float, float, float, int]],
) -> None:
    n = 32
    if t_img is not None:
        try:
            n = t_img.shape[0]
        except Exception:
            pass

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1, 3))
    ax.set_yticks(range(1, n + 1, 3))
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.4, color="gray", alpha=0.5)

    ax.axvspan(ABSTRACTION[0], ABSTRACTION[1], color="goldenrod", alpha=0.18, zorder=0)
    ax.axhspan(ABSTRACTION[0], ABSTRACTION[1], color="goldenrod", alpha=0.18, zorder=0)

    vis_idx, txt_idx = np.indices((n, n))
    X = (txt_idx + 1).astype(float)
    Y = (vis_idx + 1).astype(float)

    for lo, hi, sz, _p in thresholds:
        sz_pad = sz * MARKER_SIZE_SCALE
        try:
            if i_txt is not None:
                mask = (i_txt >= lo) & (i_txt < hi)
                if mask.any():
                    ax.scatter(X[mask], Y[mask], s=sz_pad, zorder=3, **STYLE_I_TXT)
            if t_img is not None:
                mask = (t_img >= lo) & (t_img < hi)
                if mask.any():
                    ax.scatter(X[mask], Y[mask], s=sz_pad, zorder=4, **STYLE_T_IMG)
        except Exception:
            pass


def render_missing(ax: plt.Axes) -> None:
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


def _make_direction_legend_handles() -> list:
    kw = dict(linestyle="None", markerfacecolor="none")
    return [
        mlines.Line2D([], [], color="green", marker="s", markersize=8, label="text → img", **kw),
        mlines.Line2D([], [], color="purple", marker="o", markersize=8, label="img → text", **kw),
    ]


def _make_size_legend_handles(thresholds: list[tuple]) -> list:
    kw = dict(linestyle="None", markerfacecolor="none")
    lo1, hi1, s1, _p1 = thresholds[0]
    lo2, hi2, s2, _p2 = thresholds[1]
    lo3, hi3, s3, _p3 = thresholds[2]

    return [
        mlines.Line2D(
            [], [], color="black", marker="o", markersize=np.sqrt(s1),
            label=f"imb. < {hi1:.3f}", **kw,
        ),
        mlines.Line2D(
            [], [], color="black", marker="o", markersize=np.sqrt(s2),
            label=f"{lo2:.3f} ≤ imb. < {hi2:.3f}", **kw,
        ),
        mlines.Line2D(
            [], [], color="black", marker="o", markersize=np.sqrt(s3),
            label=f"{lo3:.3f} ≤ imb. < {hi3:.3f}", **kw,
        ),
    ]


def build_scatter_figure(
    n_encoders: int,
    combos: list[tuple[str, str]],
    cell_data: list[tuple[np.ndarray | None, np.ndarray | None]],
) -> plt.Figure:
    n_cols = len(combos)

    wspace = 0.28
    left_m = 0.80
    right_m = 0.25
    top_m = 0.80
    bottom_m = 1.25

    wspace_in = SCATTER_CELL_W * wspace
    fig_w = left_m + n_cols * SCATTER_CELL_W + (n_cols - 1) * wspace_in + right_m
    fig_h = top_m + SCATTER_CELL_H + bottom_m

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left=left_m / fig_w,
        right=1.0 - right_m / fig_w,
        top=1.0 - top_m / fig_h,
        bottom=bottom_m / fig_h,
        wspace=wspace,
    )

    row_data = [
        arr.flatten()
        for t_img, i_txt in cell_data
        for arr in (t_img, i_txt)
        if arr is not None
    ]
    thresholds = get_row_thresholds(row_data, n_points=32)

    for c, (label, _model_id) in enumerate(combos):
        ax = axes[0, c]
        t_img, i_txt = cell_data[c]
        if t_img is not None or i_txt is not None:
            render_cell(ax, t_img, i_txt, thresholds)
        else:
            render_missing(ax)

        ax.set_title(label, fontsize=8, fontweight="bold", pad=5)
        ax.set_xlabel("text input", fontsize=8)
        ax.set_ylabel("visual input" if c == 0 else "", fontsize=8)

    encoder_word = "encoder" if n_encoders == 1 else "encoders"
    fig.suptitle(
        f"Pretrained Cambrian 8B — {n_encoders} {encoder_word} — information imbalance",
        fontsize=12,
        fontweight="bold",
        y=1.0 - (top_m * 0.10) / fig_h,
    )

    if thresholds:
        dir_leg = fig.legend(
            handles=_make_direction_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.32, 0.01),
            ncol=2,
            fontsize=9,
            framealpha=0.95,
            edgecolor="#aaa",
            handlelength=1.2,
            borderpad=0.5,
            columnspacing=1.1,
            handletextpad=0.5,
        )
        fig.add_artist(dir_leg)
        size_leg = fig.legend(
            handles=_make_size_legend_handles(thresholds),
            loc="lower center",
            bbox_to_anchor=(0.72, 0.01),
            ncol=1,
            fontsize=9,
            framealpha=0.95,
            edgecolor="#aaa",
            handlelength=1.2,
            borderpad=0.5,
            columnspacing=1.1,
            handletextpad=0.5,
        )
        size_leg.set_zorder(10)

    return fig


# ==============================================================================
# SECTION 5 — MAIN
# ==============================================================================


def _encoder_count_label(n: int) -> str:
    return f"{n}_encoder" if n == 1 else f"{n}_encoders"


def main() -> None:
    print("=" * 72)
    print("  cambrian_encoder_ablation_plot.py")
    print(f"  pooling={POOLING_METHOD} | variant={VARIANT_TAG}")
    print(f"  datasets={len(DATASETS_TO_AVERAGE)} | output={PLOTS_DIR}")
    print("=" * 72)

    groups: dict[int, list[tuple[str, str]]] = {1: [], 2: [], 3: [], 4: []}
    for n_enc, label, model_id in ENCODER_CONFIGS:
        groups[n_enc].append((label, model_id))

    written: list[Path] = []
    skipped: list[str] = []

    # ── Heatmap loop ────────────────────────────────────────────────────────
    for metric in HEATMAP_METRICS:
        print(f"\n{'─' * 64}")
        print(f"  metric: {metric}")
        output_dir = PLOTS_DIR / metric

        for n_enc in (1, 2, 3, 4):
            combos = groups[n_enc]
            stem = _encoder_count_label(n_enc)
            datasets = DATASETS_TO_AVERAGE_4ENC if n_enc == 4 else DATASETS_TO_AVERAGE
            print(f"\n  {n_enc} encoder(s)  [{len(combos)} combos]")

            matrices: list[np.ndarray | None] = []
            for _label, model_id in combos:
                m = average_over_datasets(model_id, metric, datasets=datasets)
                matrices.append(m)
                status = "✓" if m is not None else "✗"
                print(f"    {status}  {model_id}")

            if all(m is None for m in matrices):
                skipped.append(f"{metric}/{stem}")
                print("    skipped: no data for any combo")
                continue

            fig = build_heatmap_figure(n_enc, combos, metric, matrices)
            path = save_png(fig, output_dir, stem)
            plt.close(fig)
            written.append(path)
            print(f"    ✓ {path}")

    # ── Scatter loop ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print("  metric: information_imbalance (scatter)")
    output_dir = PLOTS_DIR / "information_imbalance"

    for n_enc in (1, 2, 3, 4):
        combos = groups[n_enc]
        stem = _encoder_count_label(n_enc)
        datasets = DATASETS_TO_AVERAGE_4ENC if n_enc == 4 else DATASETS_TO_AVERAGE
        print(f"\n  {n_enc} encoder(s)  [{len(combos)} combos]")

        cell_data: list[tuple[np.ndarray | None, np.ndarray | None]] = []
        for _label, model_id in combos:
            pair = load_imbalance_pair(model_id, datasets=datasets)
            cell_data.append(pair)
            status = "✓" if any(p is not None for p in pair) else "✗"
            print(f"    {status}  {model_id}")

        if all(t is None and i is None for t, i in cell_data):
            skipped.append(f"information_imbalance/{stem}")
            print("    skipped: no data for any combo")
            continue

        fig = build_scatter_figure(n_enc, combos, cell_data)
        path = save_png(fig, output_dir, stem)
        plt.close(fig)
        written.append(path)
        print(f"    ✓ {path}")

    print("\n" + "=" * 72)
    print(f"  wrote {len(written)} file(s)")
    print(f"  skipped {len(skipped)} figure(s)")
    for item in skipped:
        print(f"    - {item}")
    print("=" * 72)


if __name__ == "__main__":
    main()
