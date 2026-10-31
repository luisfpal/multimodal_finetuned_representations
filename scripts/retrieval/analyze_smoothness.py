"""
Analyze smoothness of retrieval metrics across layers.

For each combination of:
  - direction file: imgavg_vs_textavg
  - pseudolabels: yes / no
  - part_type: res2 / norm1
  - metric: euclidean / cosine
  - retrieval score: recall@k, mrr, mean_percentile_rank, mean_rank

we compute, per model and averaged across the two retrieval directions:
  - peak position (normalized 0..1 along layer axis)
  - smoothness = mean(|2nd diff|) / range  (lower = smoother)
  - unimodality = fraction of step pairs respecting "rise then fall" (1.0 = perfectly unimodal)

Then we average across models (skipping cambrian for res2 since it has a different
layer schedule).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("results/embedding_similarity/retrieval_metrics/coco_retrieval_23k_detailed")
MODELS = ["cambrian_8b", "internvl_8b_hf", "llava_13b", "llava_7b",
          "onevision_4b_hf", "onevision_8b_hf"]
SCORE_KEYS = ["10", "30", "100", "300", "1000",
              "mean_rank", "mean_percentile_rank", "mrr"]


def load_curve(d, metric, part_type, sub, key):
    layers = d["results"][metric][part_type][sub]
    items = sorted(((int(k), v[key]) for k, v in layers.items()), key=lambda x: x[0])
    xs = np.array([i for i, _ in items], dtype=float)
    ys = np.array([v for _, v in items], dtype=float)
    return xs, ys


def smoothness(ys):
    """mean |2nd diff| / range. Lower = smoother."""
    if len(ys) < 3:
        return np.nan
    rng = ys.max() - ys.min()
    if rng <= 0:
        return np.nan
    d2 = np.diff(ys, n=2)
    return float(np.mean(np.abs(d2)) / rng)


def unimodality(ys, want="rise_fall"):
    """
    Fraction of consecutive steps that respect a single-peak shape.
    For 'rise_fall' (higher-is-better): all diffs before peak >= 0, all after <= 0.
    Returns value in [0, 1]; 1.0 = perfectly unimodal.
    """
    if len(ys) < 3:
        return np.nan
    if want == "rise_fall":
        peak = int(np.argmax(ys))
    else:
        peak = int(np.argmin(ys))
    diffs = np.diff(ys)
    n = len(diffs)
    if n == 0:
        return np.nan
    good = 0
    for i, dv in enumerate(diffs):
        before_peak = i < peak
        if want == "rise_fall":
            ok = (dv >= 0) if before_peak else (dv <= 0)
        else:
            ok = (dv <= 0) if before_peak else (dv >= 0)
        good += int(ok)
    return good / n


def peak_pos(ys, want="rise_fall"):
    if len(ys) == 0:
        return np.nan
    idx = int(np.argmax(ys)) if want == "rise_fall" else int(np.argmin(ys))
    return idx / max(len(ys) - 1, 1)


def analyze():
    rows = []
    for model in MODELS:
        for pseudo in [False, True]:
            fname = "retrieval_imgavg_vs_textavg"
            if pseudo:
                fname += "_pseudolabels"
            fname += ".json"
            path = ROOT / model / fname
            if not path.exists():
                continue
            with open(path) as f:
                d = json.load(f)

            for metric in d["results"]:
                for part_type in d["results"][metric]:
                    for key in SCORE_KEYS:
                        per_dir = []
                        layer_count = None
                        for sub in d["results"][metric][part_type]:
                            xs, ys = load_curve(d, metric, part_type, sub, key)
                            per_dir.append(ys)
                            layer_count = len(ys)
                        ys_avg = np.mean(np.stack(per_dir), axis=0)

                        if key == "mean_rank":
                            want = "fall_rise"
                        else:
                            want = "rise_fall"

                        rows.append({
                            "model": model,
                            "pseudo": pseudo,
                            "metric": metric,
                            "part_type": part_type,
                            "score": key,
                            "n_layers": layer_count,
                            "smoothness": smoothness(ys_avg),
                            "unimodality": unimodality(ys_avg, want),
                            "peak_pos": peak_pos(ys_avg, want),
                            "range": float(ys_avg.max() - ys_avg.min()),
                        })
    return rows


def aggregate(rows):
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        if r["model"] == "cambrian_8b" and r["part_type"] == "res2":
            continue
        k = (r["pseudo"], r["metric"], r["part_type"], r["score"])
        agg[k].append(r)

    out = []
    for k, lst in agg.items():
        s = np.array([r["smoothness"] for r in lst], dtype=float)
        u = np.array([r["unimodality"] for r in lst], dtype=float)
        p = np.array([r["peak_pos"] for r in lst], dtype=float)
        out.append({
            "pseudo": k[0],
            "metric": k[1],
            "part_type": k[2],
            "score": k[3],
            "n_models": len(lst),
            "mean_smoothness": float(np.nanmean(s)),
            "mean_unimodality": float(np.nanmean(u)),
            "mean_peak_pos": float(np.nanmean(p)),
        })
    return out


def print_table(agg, score_filter=None):
    rows = [r for r in agg if (score_filter is None or r["score"] in score_filter)]
    rows.sort(key=lambda r: (r["score"], r["pseudo"], r["part_type"], r["metric"]))
    header = f"{'score':>8} {'pseudo':>7} {'part':>5} {'metric':>10} | {'smooth':>8} {'unimod':>7} {'peak':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['score']:>8} {str(r['pseudo']):>7} {r['part_type']:>5} {r['metric']:>10}"
              f" | {r['mean_smoothness']:>8.4f} {r['mean_unimodality']:>7.3f} {r['mean_peak_pos']:>5.2f}")


def best_combos(agg):
    from collections import defaultdict
    by_score = defaultdict(list)
    for r in agg:
        by_score[r["score"]].append(r)

    print("\n=== Best combos per score (rank by unimodality DESC, then smoothness ASC) ===")
    print(f"{'score':>22} -> {'pseudo':>7} {'part':>5} {'metric':>10} | unimod  smooth  peak")
    for score, lst in by_score.items():
        lst.sort(key=lambda r: (-r["mean_unimodality"], r["mean_smoothness"]))
        b = lst[0]
        print(f"{score:>22} -> {str(b['pseudo']):>7} {b['part_type']:>5} {b['metric']:>10}"
              f" | {b['mean_unimodality']:.3f}  {b['mean_smoothness']:.4f}  {b['mean_peak_pos']:.2f}")


def overall_ranking(agg):
    from collections import defaultdict
    by_combo = defaultdict(list)
    for r in agg:
        by_combo[(r["pseudo"], r["part_type"], r["metric"])].append(r)
    rank_rows = []
    for (pseudo, pt, m), lst in by_combo.items():
        unim = np.mean([r["mean_unimodality"] for r in lst])
        smo = np.mean([r["mean_smoothness"] for r in lst])
        rank_rows.append((pseudo, pt, m, unim, smo))
    rank_rows.sort(key=lambda x: (-x[3], x[4]))
    print("\n=== Overall ranking (averaged across all retrieval scores) ===")
    print(f"{'pseudo':>7} {'part':>5} {'metric':>10} | unimod  smooth")
    for pseudo, pt, m, u, s in rank_rows:
        print(f"{str(pseudo):>7} {pt:>5} {m:>10} | {u:.3f}  {s:.4f}")


def per_model_table(rows, score):
    print(f"\n=== per-model unimodality / smoothness for score = {score} ===")
    sel = [r for r in rows if r["score"] == score]
    sel.sort(key=lambda r: (r["model"], r["pseudo"], r["part_type"], r["metric"]))
    header = f"{'model':>16} {'pseudo':>7} {'part':>5} {'metric':>10} | {'smooth':>8} {'unimod':>7} {'peak':>5} {'n':>3}"
    print(header)
    for r in sel:
        print(f"{r['model']:>16} {str(r['pseudo']):>7} {r['part_type']:>5} {r['metric']:>10}"
              f" | {r['smoothness']:>8.4f} {r['unimodality']:>7.3f} {r['peak_pos']:>5.2f}"
              f" {r['n_layers']:>3}")


if __name__ == "__main__":
    rows = analyze()
    agg = aggregate(rows)

    print("=== Aggregate (averaged across 5 models for res2 [cambrian dropped], 6 for norm1) ===")
    print("smooth = mean|2nd diff| / range  (LOWER = smoother)")
    print("unimod = fraction of step pairs respecting single-peak shape (HIGHER = better)")
    print("peak   = peak position normalized to [0, 1] along the layer axis\n")

    print("\n--- recall@30 (representative) ---")
    print_table(agg, score_filter={"30"})
    print("\n--- mrr ---")
    print_table(agg, score_filter={"mrr"})
    print("\n--- mean_percentile_rank ---")
    print_table(agg, score_filter={"mean_percentile_rank"})
    print("\n--- mean_rank ---")
    print_table(agg, score_filter={"mean_rank"})
    print("\n--- recall@1000 ---")
    print_table(agg, score_filter={"1000"})

    best_combos(agg)
    overall_ranking(agg)

    per_model_table(rows, "mean_percentile_rank")
