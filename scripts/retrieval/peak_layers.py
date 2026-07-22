"""Print, for each model, the layer at which retrieval performance peaks."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("results/embedding_similarity/retrieval_metrics/coco_retrieval_23k_detailed")
MODELS = ["cambrian_8b", "internvl_8b_hf", "llava_13b", "llava_7b",
          "onevision_4b_hf", "onevision_8b_hf"]
SCORES = ["10", "30", "100", "1000", "mrr", "mean_percentile_rank"]


def load_curve(d, metric, part_type, sub, key):
    layers = d["results"][metric][part_type][sub]
    items = sorted(((int(k), v[key]) for k, v in layers.items()), key=lambda x: x[0])
    xs = np.array([i for i, _ in items], dtype=int)
    ys = np.array([v for _, v in items], dtype=float)
    return xs, ys


def peak_for(d, metric, part_type, key):
    """Average both retrieval directions, return (peak_layer_idx, peak_value, n_layers)."""
    per_dir = []
    xs_ref = None
    for sub in d["results"][metric][part_type]:
        xs, ys = load_curve(d, metric, part_type, sub, key)
        per_dir.append(ys)
        xs_ref = xs
    ys_avg = np.mean(np.stack(per_dir), axis=0)
    if key == "mean_rank":
        idx = int(np.argmin(ys_avg))
    else:
        idx = int(np.argmax(ys_avg))
    return int(xs_ref[idx]), float(ys_avg[idx]), len(xs_ref)


def main(pseudo=False):
    suffix = "_pseudolabels" if pseudo else ""
    fname = f"retrieval_imgavg_vs_textavg{suffix}.json"
    print(f"\n{'='*80}")
    print(f"Peak layer for each model — pseudolabels = {pseudo}")
    print(f"{'='*80}\n")

    for part_type in ["norm1", "res2"]:
        for metric in ["euclidean", "cosine"]:
            print(f"\n--- part_type={part_type}, metric={metric} ---")
            header = f"{'model':>16} {'n_lay':>6} | " + "  ".join(f"{s:>10}" for s in SCORES)
            print(header)
            print("-" * len(header))
            for model in MODELS:
                path = ROOT / model / fname
                if not path.exists():
                    continue
                with open(path) as f:
                    d = json.load(f)
                if part_type not in d["results"][metric]:
                    continue
                cells = []
                n_layers = None
                for s in SCORES:
                    layer, val, n = peak_for(d, metric, part_type, s)
                    cells.append(f"L{layer:>2} ({val:.3f})")
                    n_layers = n
                print(f"{model:>16} {n_layers:>6} | " + "  ".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main(pseudo=False)
    main(pseudo=True)
