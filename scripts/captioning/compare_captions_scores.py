"""Compare caption-image similarity scores across datasets.

Loads per-dataset JSON files produced by evaluate_captions_semantically.py,
computes descriptive statistics and quartile-bin counts, generates a
per-dataset histogram with annotated statistics, and runs three hypothesis
tests between the caption groups:

  * Welch t-test          – equality of means
  * Two-sample KS test    – full distribution equality (max-CDF divergence)
  * Anderson-Darling test – full distribution equality (tail-sensitive)

All outputs land in  PROJECT_ROOT / results / caption_evaluation /
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration  — paths / filenames must stay in sync with the eval script
# ---------------------------------------------------------------------------
HOME_DIR = Path.home()
PROJECT_ROOT = HOME_DIR / "multimodal_finetuned_representations"
RESULTS_BASE_DIR = PROJECT_ROOT / "results" / "caption_evaluation"

# Models to evaluate (must match keys in evaluate_captions_semantically.py)
MODELS = ["clip", "siglip", "siglip2", "siglip2-giant", "jina"]

_RESULTS_FILENAME = "semantic_similarity.json"

# Dataset groups
REAL_DATASETS = [
    "l11p/COCOQA-restval-ICQA",
    "l11p/VQAv2-COCO-restval-ICQ",
]
SYNTHETIC_DATASETS = [
    "l11p/COCOQA-restval-ICQA__synthetic",
    "l11p/VQAv2-COCO-restval-ICQ__synthetic",
    "l11p/MMBench_EN-dev-captioned",
    "l11p/MME-test-captioned",
    "l11p/GQA-val-balanced-noyesno-captioned",
]
ALL_DATASETS = REAL_DATASETS + SYNTHETIC_DATASETS

console = Console()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _dataset_model_dir(dataset_name: str, model_label: str) -> Path:
    """Per-dataset, per-model subdirectory."""
    return RESULTS_BASE_DIR / dataset_name.replace("/", "_") / model_label


def get_results_filepath(dataset_name: str, model_label: str) -> Path:
    """Path to the per-dataset JSON produced by the evaluation script."""
    return _dataset_model_dir(dataset_name, model_label) / _RESULTS_FILENAME


def load_all_scores(model_label: str) -> Dict[str, List[float]]:
    """Return {dataset_name: [similarity_scores]} for every available result file."""
    loaded: Dict[str, List[float]] = {}
    for ds in ALL_DATASETS:
        path = get_results_filepath(ds, model_label)
        if not path.exists():
            console.print(f"  [yellow]{ds}: file not found — skipping[/yellow]")
            continue
        with open(path) as fh:
            data = json.load(fh)
        scores = [entry["similarity"] for entry in data.values()]
        loaded[ds] = scores
        console.print(f"  [green]{ds}: {len(scores)} scores[/green]")
    return loaded


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_statistics(scores: List[float]) -> dict:
    """Descriptive stats + count of samples in each quartile bin.

    Bins (via np.digitize, right=True)
    ───────────────────────────────────
      [min, p25]   ← values ≤ p25
      (p25, p50]
      (p50, p75]
      (p75, max]   ← values > p75
    """
    arr = np.array(scores, dtype=np.float64)
    p25, p50, p75 = (float(np.percentile(arr, p)) for p in (25, 50, 75))

    # digitize: bin 0 → x ≤ p25, bin 1 → p25 < x ≤ p50, …
    bin_indices = np.digitize(arr, bins=[p25, p50, p75], right=True)
    bin_labels = ["[min, p25]", "(p25, p50]", "(p50, p75]", "(p75, max]"]
    bin_counts = {
        label: int(np.sum(bin_indices == i)) for i, label in enumerate(bin_labels)
    }

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "count": len(arr),
        "percentile_bins": bin_counts,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Consistent colour palette across all histograms
_COLOURS = dict(
    hist="steelblue",
    mean="crimson",
    p25="darkorange",
    p50="seagreen",
    p75="darkorchid",
    std_band="crimson",
)


def _short_label(dataset_name: str) -> str:
    """Human-readable short title with [Real] / [Syn] tag."""
    cat = "Real" if dataset_name in REAL_DATASETS else "Syn"
    short = dataset_name.split('/')[-1]
    # Prettify __synthetic suffix
    if "__synthetic" in short:
        short = short.replace("__synthetic", " (Synthetic)")
    return f"[{cat}] {short}"


def plot_dataset_histogram(
    scores: List[float],
    dataset_name: str,
    ds_stats: dict,
    save_dir: Path,
) -> Path:
    """Per-dataset histogram with mean/quartile lines, ±1-std band, and
    a bin-count inset box.  Returns the path of the saved figure."""

    fig, ax = plt.subplots(figsize=(10, 5))
    arr = np.array(scores)

    # ── histogram ──
    ax.hist(arr, bins=50, color=_COLOURS["hist"], edgecolor="white", alpha=0.85)

    # ── statistics vertical lines ──
    ax.axvline(
        ds_stats["mean"],
        color=_COLOURS["mean"],
        ls="--",
        lw=2,
        label=f"mean  = {ds_stats['mean']:.4f}",
    )
    ax.axvline(
        ds_stats["p25"],
        color=_COLOURS["p25"],
        ls=":",
        lw=1.5,
        label=f"p25   = {ds_stats['p25']:.4f}",
    )
    ax.axvline(
        ds_stats["p50"],
        color=_COLOURS["p50"],
        ls="-",
        lw=1.5,
        label=f"p50   = {ds_stats['p50']:.4f}",
    )
    ax.axvline(
        ds_stats["p75"],
        color=_COLOURS["p75"],
        ls=":",
        lw=1.5,
        label=f"p75   = {ds_stats['p75']:.4f}",
    )

    # ── ±1 std shading ──
    ax.axvspan(
        ds_stats["mean"] - ds_stats["std"],
        ds_stats["mean"] + ds_stats["std"],
        alpha=0.12,
        color=_COLOURS["std_band"],
        label="± 1 std",
    )

    # ── bin-count inset (top-left) ──
    inset_lines = "\n".join(
        f"{lbl}:  {cnt}" for lbl, cnt in ds_stats["percentile_bins"].items()
    )
    ax.text(
        0.02,
        0.97,
        inset_lines,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9),
    )

    ax.set_title(_short_label(dataset_name))
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    out = save_dir / "similarity_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Welch t-tests  (2 real × 3 synthetic)
# ---------------------------------------------------------------------------
def run_welch_tests(all_scores: Dict[str, List[float]]) -> dict:
    """Run one Welch t-test for every (real, synthetic) pair.

    Returns nested dict  {real_ds: {syn_ds: {t_statistic, p_value, …}}}
    suitable for direct JSON serialisation.
    """
    results: dict = {}
    for real_ds in REAL_DATASETS:
        if real_ds not in all_scores:
            continue
        results[real_ds] = {}
        real_mean = float(np.mean(all_scores[real_ds]))

        for syn_ds in SYNTHETIC_DATASETS:
            if syn_ds not in all_scores:
                continue
            t_stat, p_val = stats.ttest_ind(
                all_scores[real_ds], all_scores[syn_ds], equal_var=False
            )
            syn_mean = float(np.mean(all_scores[syn_ds]))
            results[real_ds][syn_ds] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant_at_0.05": bool(p_val < 0.05),
                "mean_real": real_mean,
                "mean_synthetic": syn_mean,
                "mean_difference": real_mean - syn_mean,
            }
    return results


def run_real_vs_real_test(all_scores: Dict[str, List[float]]) -> dict | None:
    """Welch t-test between the two Real datasets, if both are available.

    Returns a single flat dict (JSON-ready) or None when data is missing.
    """
    if not all(ds in all_scores for ds in REAL_DATASETS):
        return None

    a, b = REAL_DATASETS
    t_stat, p_val = stats.ttest_ind(all_scores[a], all_scores[b], equal_var=False)
    mean_a = float(np.mean(all_scores[a]))
    mean_b = float(np.mean(all_scores[b]))

    return {
        "dataset_a": a,
        "dataset_b": b,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_at_0.05": bool(p_val < 0.05),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_difference": mean_a - mean_b,
        "n_a": len(all_scores[a]),
        "n_b": len(all_scores[b]),
    }


# ---------------------------------------------------------------------------
# KS + Anderson-Darling tests  (full distribution equality)
# ---------------------------------------------------------------------------
def _ks_pair(a: List[float], b: List[float]) -> dict:
    """Two-sample KS test for one pair → JSON-ready dict."""
    stat, p_val = stats.ks_2samp(a, b)
    return {
        "ks_statistic":        float(stat),
        "p_value":             float(p_val),
        "significant_at_0.05": bool(p_val < 0.05),
    }


def _ad_pair(a: List[float], b: List[float]) -> dict:
    """Two-sample Anderson-Darling test (anderson_ksamp, k=2) → JSON-ready dict.

    Significance is derived by comparing the test statistic against the
    critical values returned by scipy (at 15 %, 10 %, 5 %, 2.5 %, 1 %).
    We do *not* use ``res.significance_level`` because it misbehaves for
    negative statistics and at asymptotic boundaries (see scipy warnings).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # expected "p-value floored/capped" warnings
        res = stats.anderson_ksamp([a, b])

    stat = float(res.statistic)
    cvs  = res.critical_values          # [15 %, 10 %, 5 %, 2.5 %, 1 %]

    # Walk critical values from most to least stringent.
    # Do not use res.significance_level — it misbehaves at boundaries.
    sig_pct: float | None
    if   stat > cvs[4]:
        sig_pct = 1.0
    elif stat > cvs[3]:
        sig_pct = 2.5
    elif stat > cvs[2]:
        sig_pct = 5.0
    elif stat > cvs[1]:
        sig_pct = 10.0
    elif stat > cvs[0]:
        sig_pct = 15.0
    else:
        sig_pct = None          # below every standard critical value

    return {
        "ad_statistic": stat,
        "critical_values": {
            "15%":  float(cvs[0]),
            "10%":  float(cvs[1]),
            "5%":   float(cvs[2]),
            "2.5%": float(cvs[3]),
            "1%":   float(cvs[4]),
        },
        "significance_level_pct": sig_pct,
        "significant_at_0.05":    bool(sig_pct is not None and sig_pct <= 5.0),
    }


def run_ks_tests(all_scores: Dict[str, List[float]]) -> dict:
    """KS test for every (real, synthetic) pair.

    Returns the same nested  {real_ds: {syn_ds: …}}  structure as
    ``run_welch_tests`` so the JSON layout stays consistent.
    """
    results: dict = {}
    for real_ds in REAL_DATASETS:
        if real_ds not in all_scores:
            continue
        results[real_ds] = {}
        for syn_ds in SYNTHETIC_DATASETS:
            if syn_ds not in all_scores:
                continue
            results[real_ds][syn_ds] = _ks_pair(
                all_scores[real_ds], all_scores[syn_ds]
            )
    return results


def run_ad_tests(all_scores: Dict[str, List[float]]) -> dict:
    """Anderson-Darling test for every (real, synthetic) pair."""
    results: dict = {}
    for real_ds in REAL_DATASETS:
        if real_ds not in all_scores:
            continue
        results[real_ds] = {}
        for syn_ds in SYNTHETIC_DATASETS:
            if syn_ds not in all_scores:
                continue
            results[real_ds][syn_ds] = _ad_pair(
                all_scores[real_ds], all_scores[syn_ds]
            )
    return results


# ---------------------------------------------------------------------------
# Rich table display for Welch results
# ---------------------------------------------------------------------------
def _stars(p: float) -> str:
    """Standard significance-star notation."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def print_welch_table(welch: dict, model_label: str):
    """Print a 2 × 3 Rich table.  Rows = real datasets, columns = synthetic."""
    table = Table(title=f"Welch t-test ({model_label}) — Real (rows) × Synthetic (cols)")
    table.add_column("", style="cyan", no_wrap=True)

    # only include synthetic columns that actually have data
    syn_present = [s for s in SYNTHETIC_DATASETS if any(s in v for v in welch.values())]
    for s in syn_present:
        table.add_column(s.split("/")[-1], justify="center")

    for real_ds, row in welch.items():
        cells: List[str] = [real_ds.split("/")[-1]]
        for syn_ds in syn_present:
            if syn_ds in row:
                r = row[syn_ds]
                stars = _stars(r["p_value"])
                color = "green" if r["significant_at_0.05"] else "red"
                cells.append(
                    f"[{color}]t = {r['t_statistic']:+.2f}   "
                    f"p = {r['p_value']:.2e}  {stars}[/{color}]\n"
                    f"Δμ = {r['mean_difference']:+.4f}"
                )
            else:
                cells.append("N/A")
        table.add_row(*cells)

    console.print(table)


# ---------------------------------------------------------------------------
# Rich table display for KS / AD results
# ---------------------------------------------------------------------------
def _ad_sig_str(r: dict) -> str:
    """Human-readable significance string for an AD result dict."""
    return f"{r['significance_level_pct']}%" if r["significance_level_pct"] is not None else ">15%"


def print_ks_table(ks: dict, model_label: str):
    """Rich table for KS results – same 2 × 3 layout as print_welch_table."""
    table = Table(title=f"KS two-sample test ({model_label}) — Real (rows) × Synthetic (cols)")
    table.add_column("", style="cyan", no_wrap=True)

    syn_present = [s for s in SYNTHETIC_DATASETS if any(s in v for v in ks.values())]
    for s in syn_present:
        table.add_column(s.split("/")[-1], justify="center")

    for real_ds, row in ks.items():
        cells: List[str] = [real_ds.split("/")[-1]]
        for syn_ds in syn_present:
            if syn_ds in row:
                r      = row[syn_ds]
                stars  = _stars(r["p_value"])
                color  = "green" if r["significant_at_0.05"] else "red"
                cells.append(
                    f"[{color}]D = {r['ks_statistic']:.4f}\n"
                    f"p = {r['p_value']:.2e}  {stars}[/{color}]"
                )
            else:
                cells.append("N/A")
        table.add_row(*cells)

    console.print(table)


def print_ad_table(ad: dict, model_label: str):
    """Rich table for Anderson-Darling results."""
    table = Table(title=f"Anderson-Darling test ({model_label}) — Real (rows) × Synthetic (cols)")
    table.add_column("", style="cyan", no_wrap=True)

    syn_present = [s for s in SYNTHETIC_DATASETS if any(s in v for v in ad.values())]
    for s in syn_present:
        table.add_column(s.split("/")[-1], justify="center")

    for real_ds, row in ad.items():
        cells: List[str] = [real_ds.split("/")[-1]]
        for syn_ds in syn_present:
            if syn_ds in row:
                r     = row[syn_ds]
                color = "green" if r["significant_at_0.05"] else "red"
                cells.append(
                    f"[{color}]A² = {r['ad_statistic']:.2f}\n"
                    f"sig ≈ {_ad_sig_str(r)}[/{color}]"
                )
            else:
                cells.append("N/A")
        table.add_row(*cells)

    console.print(table)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    console.print("[bold]Caption Score Comparison[/bold]\n")

    for model_label in MODELS:
        console.print(f"\n[bold magenta]=== Evaluating Model: {model_label} ===[/bold magenta]")

        # ── 1 · load ─────────────────────────────────────────────────────────
        console.print(f"[bold]1 · Loading scores for {model_label} …[/bold]")
        all_scores = load_all_scores(model_label)
        if not all_scores:
            console.print(
                f"[yellow]No results found for {model_label} — skipping.[/yellow]"
            )
            continue

        # ── 2 · per-dataset statistics ───────────────────────────────────────
        console.print(f"\n[bold]2 · Per-dataset statistics ({model_label})[/bold]")
        all_stats: Dict[str, dict] = {}
        for ds, scores in all_scores.items():
            s = compute_statistics(scores)
            all_stats[ds] = s
            console.print(
                f"  {ds.split('/')[-1]}:  "
                f"mean={s['mean']:.4f}  std={s['std']:.4f}  "
                f"bins={s['percentile_bins']}"
            )
            # save individual stats into the dataset/model directory
            ds_dir = _dataset_model_dir(ds, model_label)
            ds_dir.mkdir(parents=True, exist_ok=True)
            ds_stats_path = ds_dir / "statistics.json"
            with open(ds_stats_path, "w") as fh:
                json.dump(s, fh, indent=2)
            console.print(f"  [blue]saved → {ds_stats_path}[/blue]")

        # combined summary goes into combined/<model>/
        combined_dir = RESULTS_BASE_DIR / "combined" / model_label
        combined_dir.mkdir(parents=True, exist_ok=True)
        stats_path = combined_dir / "statistics_summary.json"
        with open(stats_path, "w") as fh:
            json.dump(all_stats, fh, indent=2)
        console.print(f"  [blue]saved → {stats_path}[/blue]")

        # ── 3 · histograms ───────────────────────────────────────────────────
        console.print(f"\n[bold]3 · Histograms ({model_label})[/bold]")
        for ds, scores in all_scores.items():
            out = plot_dataset_histogram(scores, ds, all_stats[ds], _dataset_model_dir(ds, model_label))
            console.print(f"  saved → {out}")

        # ── 4 · Welch tests ──────────────────────────────────────────────────
        console.print(f"\n[bold]4 · Welch t-tests ({model_label})[/bold]")
        welch = run_welch_tests(all_scores)
        if welch:
            print_welch_table(welch, model_label)
            welch_path = combined_dir / "welch_tests_real_vs_synthetic.json"
            with open(welch_path, "w") as fh:
                json.dump(welch, fh, indent=2)
            console.print(f"  [blue]saved → {welch_path}[/blue]")
        else:
            console.print("  [yellow]Not enough data for Welch tests.[/yellow]")

        # ── 5 · Real vs Real test ─────────────────────────────────────────────
        console.print(f"\n[bold]5 · Real vs Real t-test ({model_label})[/bold]")
        real_vs_real = run_real_vs_real_test(all_scores)
        if real_vs_real:
            stars = _stars(real_vs_real["p_value"])
            color = "green" if real_vs_real["significant_at_0.05"] else "red"
            console.print(
                f"  [{color}]{REAL_DATASETS[0].split('/')[-1]}  vs  "
                f"{REAL_DATASETS[1].split('/')[-1]}:  "
                f"t = {real_vs_real['t_statistic']:+.2f}   "
                f"p = {real_vs_real['p_value']:.2e}  {stars}  "
                f"Δμ = {real_vs_real['mean_difference']:+.4f}[/{color}]"
            )
            rvr_path = combined_dir / "welch_test_real_vs_real.json"
            with open(rvr_path, "w") as fh:
                json.dump(real_vs_real, fh, indent=2)
            console.print(f"  [blue]saved → {rvr_path}[/blue]")
        else:
            console.print("  [yellow]Both real datasets required — skipping.[/yellow]")

        # ── 6 · Distribution tests (KS + Anderson-Darling) ───────────────────
        console.print(f"\n[bold]6 · Distribution tests — KS & Anderson-Darling ({model_label})[/bold]")
        console.print(
            "  [dim]With n ≥ 10 000 the KS p-value is hyper-sensitive; "
            "the D statistic (max CDF gap) is the primary effect-size metric.[/dim]"
        )

        # ── Real × Synthetic grids ──
        ks_grid = run_ks_tests(all_scores)
        ad_grid = run_ad_tests(all_scores)

        if ks_grid:
            print_ks_table(ks_grid, model_label)
            ks_path = combined_dir / "ks_tests_real_vs_synthetic.json"
            with open(ks_path, "w") as fh:
                json.dump(ks_grid, fh, indent=2)
            console.print(f"  [blue]saved → {ks_path}[/blue]")

        if ad_grid:
            print_ad_table(ad_grid, model_label)
            ad_path = combined_dir / "ad_tests_real_vs_synthetic.json"
            with open(ad_path, "w") as fh:
                json.dump(ad_grid, fh, indent=2)
            console.print(f"  [blue]saved → {ad_path}[/blue]")

        # ── Real × Real ──
        if all(ds in all_scores for ds in REAL_DATASETS):
            a, b = REAL_DATASETS
            ks_rvr = _ks_pair(all_scores[a], all_scores[b])
            ad_rvr = _ad_pair(all_scores[a], all_scores[b])

            short_a, short_b = a.split("/")[-1], b.split("/")[-1]
            console.print(f"\n  [bold]Real vs Real:[/bold]  {short_a}  vs  {short_b}")

            # KS line
            ks_stars = _stars(ks_rvr["p_value"])
            ks_color = "green" if ks_rvr["significant_at_0.05"] else "red"
            console.print(
                f"    [dim]KS:[/dim]  [{ks_color}]"
                f"D = {ks_rvr['ks_statistic']:.4f}   "
                f"p = {ks_rvr['p_value']:.2e}  {ks_stars}[/{ks_color}]"
            )

            # AD line
            ad_color = "green" if ad_rvr["significant_at_0.05"] else "red"
            console.print(
                f"    [dim]AD:[/dim]  [{ad_color}]"
                f"A² = {ad_rvr['ad_statistic']:.2f}   "
                f"sig ≈ {_ad_sig_str(ad_rvr)}[/{ad_color}]"
            )

            # persist (attach dataset labels for traceability)
            for result in (ks_rvr, ad_rvr):
                result["dataset_a"] = a
                result["dataset_b"] = b

            ks_rvr_path = combined_dir / "ks_test_real_vs_real.json"
            with open(ks_rvr_path, "w") as fh:
                json.dump(ks_rvr, fh, indent=2)
            console.print(f"  [blue]saved → {ks_rvr_path}[/blue]")

            ad_rvr_path = combined_dir / "ad_test_real_vs_real.json"
            with open(ad_rvr_path, "w") as fh:
                json.dump(ad_rvr, fh, indent=2)
            console.print(f"  [blue]saved → {ad_rvr_path}[/blue]")


if __name__ == "__main__":
    main()
