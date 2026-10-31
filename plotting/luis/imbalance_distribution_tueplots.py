"""Information-imbalance distributions in the tueplots NeurIPS style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, kurtosis, skew

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tueplots_style import OUTPUT_DIR, get_rc  # noqa: E402

LUIS_ROOT = Path("/u/dssc/lbasile/scratch/plm/luis")
DATA_DIR = LUIS_ROOT / "data" / "multimodality"
OUT_DIR = OUTPUT_DIR / "imbalance_distribution"

POOLING_METHOD = "last"
VARIANT_TAG = "unique_images"

DATASETS_TO_AVERAGE: tuple[str, ...] = (
    "cocoqa_restval_icqa_split01_human",
    "gqa_val_balanced_noyesno_captioned_synthetic",
    "vqav2_coco_restval_icq_split01_human",
    "mmbench_en_dev_captioned_open_ended_synthetic",
    "seed_bench_images_captioned_open_ended_synthetic",
)

MODEL_PAIRS = (
    ("llava_7b", "LLaVA 7B", "llava-v1.5-7b-pretrained-lbasile", "pretrained", "llava_7b"),
    ("llava_13b", "LLaVA 13B", "llava-v1.5-13b-pretrained-lbasile", "pretrained", "llava_13b"),
    ("LLaVA-OneVision-1.5-4B-Instruct", "OneVision 4B", "LLaVA-OneVision-1.5-4B-Base", "pretrained", "llava_onevision_4b"),
    ("LLaVA-OneVision-1.5-8B-Instruct", "OneVision 8B", "LLaVA-OneVision-1.5-8B-Base", "pretrained", "llava_onevision_8b"),
    ("InternVL2-8B", "InternVL2 8B", "InternVL2-8B-Pretrain", "pretrained", "internvl2_8b"),
    ("cambrian-8b", "Cambrian 8B", "cambrian-8b-pretrained", "pretrained", "cambrian_8b"),
)

CURRENT_PERCENTILES = [17, 34, 50]
REFERENCE_PERCENTILES = [10, 25, 75, 90]

C_HIST = "#a8c8f0"
C_KDE = "#1a5fa8"
C_CDF = "#1a5fa8"
C_SPLIT = ["#2ca02c", "#98df8a", "#d62728", "#ffaaa5"]

MULTIPLIERS: list[float] = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
THRESHOLD_COLORS: list[str] = [
    "#1a6faf", "#5ba4cf", "#8bc34a", "#7b2d8b", "#f4a261", "#e07b00", "#c0392b",
]


def _path(dataset: str, model_id: str, metric: str) -> Path:
    return (
        DATA_DIR / dataset / POOLING_METHOD / VARIANT_TAG / "similarities"
        / f"{model_id}_{metric}.json"
    )


def _load_matrix(dataset: str, model_id: str, metric: str) -> np.ndarray | None:
    p = _path(dataset, model_id, metric)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"WARN: invalid JSON {p}: {exc}")
        return None
    if "matrix" not in payload:
        return None
    return np.asarray(payload["matrix"], dtype=float)


def _average_over_datasets(model_id: str, metric: str) -> np.ndarray | None:
    mats = [_load_matrix(d, model_id, metric) for d in DATASETS_TO_AVERAGE]
    mats = [m for m in mats if m is not None]
    if not mats:
        return None
    if len({m.shape for m in mats}) > 1:
        return None
    return np.mean(np.stack(mats, 0), axis=0)


def load_pool(fine_id: str, pre_id: str) -> dict[str, np.ndarray | None]:
    parts = {
        "fine_fwd": _average_over_datasets(fine_id, "information_imbalance"),
        "fine_rev": _average_over_datasets(fine_id, "information_imbalance_reversed"),
        "pre_fwd": _average_over_datasets(pre_id, "information_imbalance"),
        "pre_rev": _average_over_datasets(pre_id, "information_imbalance_reversed"),
    }
    flat = [v.flatten() for v in parts.values() if v is not None]
    parts["pool"] = np.concatenate(flat) if flat else np.array([])
    return parts


def compute_stats(pool: np.ndarray) -> dict:
    if pool.size == 0:
        return {}
    pcts = sorted(set(CURRENT_PERCENTILES + REFERENCE_PERCENTILES))
    return {
        "n": pool.size, "mean": float(np.mean(pool)), "std": float(np.std(pool)),
        "median": float(np.median(pool)), "skew": float(skew(pool)),
        "kurt": float(kurtosis(pool)), "min": float(pool.min()),
        "max": float(pool.max()), "pct": {p: float(np.percentile(pool, p)) for p in pcts},
    }


def _val_to_pct(pool: np.ndarray, val: float) -> float:
    return float(np.mean(pool <= val)) * 100.0


def find_mode_relative_thresholds(pool: np.ndarray, n_pts: int = 2048) -> dict:
    lo = float(np.percentile(pool, 0.1))
    hi = float(np.percentile(pool, 99.9))
    kde = gaussian_kde(pool, bw_method="scott")
    x = np.linspace(lo, hi, n_pts)
    y = kde(x)
    peak_idx = int(np.argmax(y))
    mode_val = float(x[peak_idx])
    p4_rank = round(_val_to_pct(pool, mode_val))

    thresholds = []
    for i, mult in enumerate(MULTIPLIERS, start=1):
        delta_i = mult * mode_val
        rank_i = max(1, min(99, int(round(_val_to_pct(pool, delta_i)))))
        thresholds.append({"name": f"P{i}", "rank": rank_i, "val": delta_i, "mult": mult})

    return {
        "mode_val": mode_val, "mode_pct": p4_rank, "peak_height": float(y[peak_idx]),
        "x_grid": x, "kde_vals": y, "thresholds": thresholds,
    }


def _threshold_lines(ax, pt: dict) -> None:
    for thresh, color in zip(pt["thresholds"], THRESHOLD_COLORS):
        ax.axvline(thresh["val"], color=color, linewidth=1.1, linestyle="-", zorder=6, alpha=0.9)


def _threshold_legend(ax, pt: dict, *, fontsize: float = 6.5, loc: str = "upper right") -> None:
    title = (
        f"P4 = KDE mode  (delta = {pt['mode_val']:.4f})\n"
        f"Pi = percentile( mult x delta_P4 )"
    )
    handles = []
    for thresh, color in zip(pt["thresholds"], THRESHOLD_COLORS):
        mult_str = f"{thresh['mult']:.2f}".rstrip("0").rstrip(".")
        label = f"{thresh['name']} [{mult_str}x]   P{thresh['rank']}   delta={thresh['val']:.4f}"
        handles.append(mlines.Line2D([0], [0], color=color, linewidth=2, label=label))
    ax.legend(
        handles=handles, title=title, title_fontsize=fontsize - 0.5, loc=loc,
        fontsize=fontsize, framealpha=0.95, edgecolor="#bbb",
        handlelength=1.5, borderpad=0.6, labelspacing=0.25,
    )


def plot_individual(fine_id: str, pre_id: str, title: str, slug: str) -> None:
    data = load_pool(fine_id, pre_id)
    pool = data["pool"]
    if pool.size == 0:
        print(f"WARN: no data for {title} - skipping")
        return
    pt = find_mode_relative_thresholds(pool)
    x_kde, y_kde = pt["x_grid"], pt["kde_vals"]

    rc = get_rc(ncols=4)
    with plt.rc_context(rc):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.34)
        ax_hist = fig.add_subplot(gs[0])
        ax_zoom = fig.add_subplot(gs[1])
        ax_cdf = fig.add_subplot(gs[2])
        ax_split = fig.add_subplot(gs[3])
        fig.suptitle(title)

        ax_hist.hist(pool, bins=60, density=True, color=C_HIST, edgecolor="none", alpha=0.7)
        ax_hist.plot(x_kde, y_kde, color=C_KDE, linewidth=1)
        ax_hist.set_xlabel("information imbalance delta")
        ax_hist.set_ylabel("density")
        ax_hist.set_title("Full distribution")
        ax_hist.grid(True, alpha=0.3)
        _threshold_lines(ax_hist, pt)
        _threshold_legend(ax_hist, pt)

        x_lo = max(0.0, pt["thresholds"][0]["val"] - 0.015)
        x_hi = pt["thresholds"][-1]["val"] * 1.12
        ax_zoom.hist(pool, bins=80, density=True, color=C_HIST, edgecolor="none", alpha=0.7)
        ax_zoom.plot(x_kde, y_kde, color=C_KDE, linewidth=1)
        ax_zoom.set_xlim(x_lo, x_hi)
        ax_zoom.set_xlabel("information imbalance delta")
        ax_zoom.set_ylabel("density")
        ax_zoom.set_title("Peak region (P1 -> P7 zoom)")
        ax_zoom.grid(True, alpha=0.3)
        _threshold_lines(ax_zoom, pt)
        _threshold_legend(ax_zoom, pt)

        sorted_pool = np.sort(pool)
        cdf = np.arange(1, len(sorted_pool) + 1) / len(sorted_pool)
        ax_cdf.plot(sorted_pool, cdf, color=C_CDF, linewidth=1)
        ax_cdf.set_xlabel("information imbalance delta")
        ax_cdf.set_ylabel("cumulative fraction")
        ax_cdf.set_title("CDF")
        ax_cdf.set_ylim(0, 1.02)
        ax_cdf.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax_cdf.grid(True, alpha=0.3)
        _threshold_lines(ax_cdf, pt)
        for thresh, color in zip(pt["thresholds"], THRESHOLD_COLORS):
            ax_cdf.axhline(thresh["rank"] / 100, color=color, linewidth=0.5, linestyle=":", alpha=0.45)

        for (mat, lbl), color in zip(
            [
                (data["fine_fwd"], "finetuned  txt->img"),
                (data["fine_rev"], "finetuned  img->txt"),
                (data["pre_fwd"], "pretrained txt->img"),
                (data["pre_rev"], "pretrained img->txt"),
            ],
            C_SPLIT,
        ):
            if mat is None:
                continue
            flat = mat.flatten()
            kde4 = gaussian_kde(flat, bw_method="scott")
            x4 = np.linspace(float(np.percentile(flat, 0.5)), float(np.percentile(flat, 99.5)), 256)
            ax_split.plot(x4, kde4(x4), color=color, linewidth=1, label=lbl)

        ax_split.set_xlabel("information imbalance delta")
        ax_split.set_ylabel("density")
        ax_split.set_title("Per-direction decomposition")
        ax_split.grid(True, alpha=0.3)
        ax_split.legend()
        _threshold_lines(ax_split, pt)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{slug}.pdf"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)
        print(f"Saved {out}")


def plot_overview(pairs_data: list[tuple]) -> None:
    rc = get_rc(ncols=3, nrows=2)
    with plt.rc_context(rc):
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.33, hspace=0.38)
        for i, (pool, title, _slug) in enumerate(pairs_data):
            ax = fig.add_subplot(gs[i])
            if pool.size == 0:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            pt = find_mode_relative_thresholds(pool)
            ax.hist(pool, bins=45, density=True, color=C_HIST, edgecolor="none", alpha=0.7)
            ax.plot(pt["x_grid"], pt["kde_vals"], color=C_KDE, linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("delta")
            ax.set_ylabel("density")
            ax.grid(True, alpha=0.3)
            _threshold_lines(ax, pt)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "overview_2x3.pdf"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)
        print(f"Saved {out}")


def main() -> None:
    pairs_data = []
    for fine_id, fine_label, pre_id, _pre_label, slug in MODEL_PAIRS:
        data = load_pool(fine_id, pre_id)
        pairs_data.append((data["pool"], fine_label, slug))
        plot_individual(fine_id, pre_id, fine_label, slug)
    plot_overview(pairs_data)


if __name__ == "__main__":
    main()
