"""Cross-modal retrieval metrics between paired representations stored in two folders.

Row i of folder 1 is the positive for row i of folder 2. For each shared (layer, part_type in
{res2, norm1}) metrics are computed in both directions with Euclidean and cosine ranking:
NDCG@k, mean rank, mean percentile rank and MRR.

Outputs: JSON files at --out_path (and --out_path_pseudolabels).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RANKING_METRICS = ("euclidean", "cosine")

# Number of decoder layers per model. Used as the upper bound of the per-layer
# loop: for `res2`, files exist at l0..l{n-1}; for `norm1`/`norm2`, an extra
# post-stack norm tensor is saved at l{n} (see get_target_layers in
# embeddings_vlms.py).
MODEL_NUM_LAYERS = {
    "cambrian_8b": 32,
    "internvl_8b_hf": 32,
    "llava_7b": 32,
    "llava_13b": 40,
    "onevision_4b_hf": 36,
    "onevision_8b_hf": 36,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute retrieval metrics for paired-row representations across two "
            "folders. Each row i in folder1 is a positive pair with row i in "
            "folder2. For each shared (layer, part_type) where part_type in "
            "{res2, norm1}, metrics are computed in both directions and under "
            "both ranking criteria (raw Euclidean distance and cosine similarity, "
            "the latter equivalent to Euclidean on L2-normalized vectors). "
            "Per-layer entries report NDCG@k plus the smooth alternatives "
            "mean_rank, mean_percentile_rank (= AUC for single-relevance), and "
            "mrr without cutoff."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model folder under --results_root, e.g. 'internvl_8b_hf'.",
    )
    parser.add_argument(
        "--suffix1",
        type=str,
        required=True,
        help="Suffix used in folder1 file names, e.g. 'imgavg' for files like 'l0res2_imgavg.pt'.",
    )
    parser.add_argument(
        "--suffix2",
        type=str,
        required=True,
        help="Suffix used in folder2 file names, e.g. 'textavg' for files like 'l0res2_textavg.pt'.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results/embedding_similarity",
        help="Root containing per-model representation folders.",
    )
    parser.add_argument(
        "--folder1_name",
        type=str,
        default="coco_image_only",
        help="Dataset folder name under <results_root>/<model>/ for suffix1.",
    )
    parser.add_argument(
        "--folder2_name",
        type=str,
        default="coco_text_only",
        help="Dataset folder name under <results_root>/<model>/ for suffix2.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[10, 30, 100, 300],
        help="Cutoffs k for NDCG@k.",
    )
    parser.add_argument(
        "--part_types",
        type=str,
        nargs="+",
        default=["res2", "norm1"],
        help="Part types to evaluate. Each is matched independently across both folders.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--retrieval_dataset_name",
        type=str,
        default="coco_retrieval_23k_detailed",
        help="Subfolder name under <results_root>/retrieval_metrics/ where outputs land.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output JSON path. Defaults to "
        "<results_root>/retrieval_metrics/<retrieval_dataset_name>/<model>/"
        "retrieval_<suffix1>_vs_<suffix2>.json (contains both euclidean and cosine).",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Optional .npy of integer pseudolabels (one per row, -1 for unlabeled). "
        "If provided together with --mask_file, also writes a second JSON with "
        "multi-relevance retrieval metrics (positives = same-label items).",
    )
    parser.add_argument(
        "--mask_file",
        type=str,
        default=None,
        help="Optional .npy boolean mask selecting rows to keep for the "
        "pseudolabel-based multi-relevance evaluation.",
    )
    parser.add_argument(
        "--out_path_pseudolabels",
        type=str,
        default=None,
        help="Output JSON path for pseudolabel multi-relevance metrics. "
        "Defaults to the same as --out_path with '_pseudolabels' before the .json suffix.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="paired",
        choices=["paired", "winoground"],
        help="paired: row i in folder1 is the positive for row i in folder2 "
        "(default; used for coco / coco_karpathy / imagenet). "
        "winoground: side-major flattened [2N,D] tensors reshape to [N,2,D]; "
        "reports per-layer text/image/group scores. --ks / --label_file are ignored.",
    )
    return parser.parse_args()


def layer_file(folder: str, layer_id: int, part: str, suffix: str) -> Path:
    """Path to the saved tensor 'l{layer_id}{part}_{suffix}.pt' under <folder>/layers/."""
    return Path(folder) / "layers" / f"l{layer_id}{part}_{suffix}.pt"


@torch.inference_mode()
def retrieval_metrics_paired(
    emb_q: torch.Tensor,
    emb_d: torch.Tensor,
    ks,
    metric: str,
    device: str,
):
    """Retrieval metrics for binary single-relevance retrieval where the positive
    for query i is doc i.

    Returns a dict with:
      - 'ndcg':                 {k: NDCG@k}. With one relevant item, IDCG@k = 1,
                                so NDCG@k = 1/log2(rank+1) if rank <= k else 0.
      - 'mean_rank':            mean rank of the gold item (1 = best, N = worst).
      - 'mean_percentile_rank': 1 - (rank-1)/(N-1) averaged over queries. Equals
                                ROC-AUC for single-relevance ranking; in [0, 1],
                                higher is better. Smooth in similarity space.
      - 'mrr':                  mean reciprocal rank with no cutoff, mean(1/rank).

    metric='euclidean': rank by ascending Euclidean distance on raw vectors.
    metric='cosine':    rank by descending cosine similarity (vectors L2-normalized first).
    The full N x N score matrix is computed in one shot in float32.
    """
    assert emb_q.shape == emb_d.shape, f"shape mismatch: {emb_q.shape} vs {emb_d.shape}"
    n = emb_q.shape[0]
    emb_q = emb_q.float().to(device)
    emb_d = emb_d.float().to(device)

    diag = torch.arange(n, device=device)
    if metric == "euclidean":
        scores = torch.cdist(emb_q, emb_d)  # [n, n], lower is better
        pos_score = scores[diag, diag].unsqueeze(1)
        rank = 1 + (scores < pos_score).sum(dim=1).long()
    else:  # cosine
        emb_q = F.normalize(emb_q, p=2, dim=1)
        emb_d = F.normalize(emb_d, p=2, dim=1)
        scores = emb_q @ emb_d.T  # [n, n], higher is better
        pos_score = scores[diag, diag].unsqueeze(1)
        rank = 1 + (scores > pos_score).sum(dim=1).long()

    rank_f = rank.float()
    log_term = 1.0 / torch.log2(rank_f + 1.0)

    ndcg = {}
    for k in ks:
        contrib = torch.where(rank <= k, log_term, torch.zeros_like(log_term))
        ndcg[k] = contrib.mean().item()

    return {
        "ndcg": ndcg,
        "mean_rank": rank_f.mean().item(),
        "mean_percentile_rank": (1.0 - (rank_f - 1.0) / (n - 1)).mean().item(),
        "mrr": (1.0 / rank_f).mean().item(),
    }


@torch.inference_mode()
def retrieval_metrics_multilabel(
    emb_q: torch.Tensor,
    emb_d: torch.Tensor,
    labels: torch.Tensor,
    ks,
    metric: str,
    device: str,
):
    """Multi-relevance retrieval metrics: for query i, all docs j with the
    same label are positives. Binary relevance (0/1).

    Conventions:
      - 'mean_rank' and 'mrr' use the rank of the FIRST positive (1=best).
      - 'mean_percentile_rank' is per-query Mann-Whitney AUC averaged across
        queries (positives ranked above negatives), in [0, 1], higher better.
        Equals (1 - (rank-1)/(N-1)) when there is exactly one positive.
      - 'ndcg@k' uses binary DCG@k normalized by IDCG@k =
        sum_{r=1..min(k, n_pos_i)} 1/log2(r+1).

    Queries with no positives or no negatives are skipped from each metric.
    """
    assert emb_q.shape == emb_d.shape, f"shape mismatch: {emb_q.shape} vs {emb_d.shape}"
    n = emb_q.shape[0]
    emb_q = emb_q.float().to(device)
    emb_d = emb_d.float().to(device)
    labels_t = labels.to(device).long()

    if metric == "euclidean":
        scores = torch.cdist(emb_q, emb_d)  # lower is better
        order = scores.argsort(dim=1, descending=False)
    else:  # cosine
        emb_q = F.normalize(emb_q, p=2, dim=1)
        emb_d = F.normalize(emb_d, p=2, dim=1)
        scores = emb_q @ emb_d.T  # higher is better
        order = scores.argsort(dim=1, descending=True)

    # rel[i, p] = 1 iff doc at rank p+1 (for query i) shares query i's label.
    rel = labels_t[order] == labels_t.unsqueeze(1)  # [n, n] bool
    n_pos = rel.sum(dim=1).float()  # [n]

    rank_pos = torch.arange(1, n + 1, device=device, dtype=torch.float32)  # [n]
    discount = 1.0 / torch.log2(rank_pos + 1.0)
    cum_disc = torch.cumsum(discount, dim=0)  # IDCG up to rank r

    ndcg = {}
    for k in ks:
        dcg_k = (rel[:, :k].float() * discount[:k]).sum(dim=1)
        n_ideal = torch.clamp(n_pos, max=float(k)).long()  # [n]
        idcg_k = torch.zeros(n, device=device)
        valid = n_ideal > 0
        idcg_k[valid] = cum_disc[n_ideal[valid] - 1]
        ndcg_k = torch.where(idcg_k > 0, dcg_k / idcg_k, torch.zeros_like(dcg_k))
        # Average only over queries that have at least one positive.
        has_pos_mask = n_pos > 0
        ndcg[k] = (
            ndcg_k[has_pos_mask].mean().item() if has_pos_mask.any() else float("nan")
        )

    has_pos = n_pos > 0
    first_pos_idx = rel.float().argmax(dim=1)  # 0-indexed position of first True
    first_rank = (first_pos_idx + 1).float()
    mean_rank = first_rank[has_pos].mean().item() if has_pos.any() else float("nan")
    mrr = (1.0 / first_rank[has_pos]).mean().item() if has_pos.any() else float("nan")

    # Per-query AUC via the rank-sum identity:
    #   AUC = (sum_{i in pos} rank_from_bottom_i - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    # where rank_from_bottom = N - rank_from_top + 1, and rank_from_top = position+1.
    positions = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(
        0
    )  # [1, n]
    rank_bottom = n - positions  # [1, n], values N..1
    sum_rb_pos = (rel.float() * rank_bottom).sum(dim=1)  # [n]
    n_neg = n - n_pos
    valid_auc = (n_pos > 0) & (n_neg > 0)
    auc = torch.zeros(n, device=device)
    auc[valid_auc] = (
        sum_rb_pos[valid_auc] - n_pos[valid_auc] * (n_pos[valid_auc] + 1) / 2
    ) / (n_pos[valid_auc] * n_neg[valid_auc])
    mean_pr = auc[valid_auc].mean().item() if valid_auc.any() else float("nan")

    return {
        "ndcg": ndcg,
        "mean_rank": mean_rank,
        "mean_percentile_rank": mean_pr,
        "mrr": mrr,
        "n_queries_with_positives": int(has_pos.sum().item()),
        "mean_n_positives": (
            float(n_pos[has_pos].mean().item()) if has_pos.any() else 0.0
        ),
    }


@torch.inference_mode()
def winoground_scores(
    emb_img: torch.Tensor,
    emb_txt: torch.Tensor,
    metric: str,
    device: str,
):
    """Winoground per-entry text/image/group scores.

    Inputs are side-major flattened [2N, D] tensors as written by the
    extractor: row 2i = side 0, row 2i+1 = side 1 for entry i. Reshapes
    to [N, 2, D] and builds sim[n, image_side, text_side] (cosine if
    requested, else negative Euclidean — both higher-is-better). Then,
    per the Winoground paper:

      text_score  = mean over entries of
                    [sim(i0,c0) > sim(i0,c1) AND sim(i1,c1) > sim(i1,c0)]
      image_score = mean over entries of
                    [sim(i0,c0) > sim(i1,c0) AND sim(i1,c1) > sim(i0,c1)]
      group_score = mean over entries of [text_correct AND image_correct]
    """
    assert emb_img.shape == emb_txt.shape, (
        f"shape mismatch: {emb_img.shape} vs {emb_txt.shape}"
    )
    n2 = emb_img.shape[0]
    assert n2 % 2 == 0, (
        f"winoground task requires even row count (side-major), got {n2}"
    )
    n = n2 // 2

    img = emb_img.float().to(device).view(n, 2, -1)
    txt = emb_txt.float().to(device).view(n, 2, -1)

    if metric == "cosine":
        img = F.normalize(img, p=2, dim=-1)
        txt = F.normalize(txt, p=2, dim=-1)
        sim = torch.einsum("nid,njd->nij", img, txt)  # [N, 2, 2]
    else:  # euclidean — invert sign so "higher=better" stays consistent
        diff = img.unsqueeze(2) - txt.unsqueeze(1)  # [N, 2, 2, D]
        sim = -diff.norm(dim=-1)  # [N, 2, 2]

    text_correct = (sim[:, 0, 0] > sim[:, 0, 1]) & (sim[:, 1, 1] > sim[:, 1, 0])
    image_correct = (sim[:, 0, 0] > sim[:, 1, 0]) & (sim[:, 1, 1] > sim[:, 0, 1])
    group_correct = text_correct & image_correct

    return {
        "text_score": text_correct.float().mean().item(),
        "image_score": image_correct.float().mean().item(),
        "group_score": group_correct.float().mean().item(),
        "n_entries": int(n),
    }


def main():
    args = parse_args()

    folder1 = os.path.join(args.results_root, args.model, args.folder1_name)
    folder2 = os.path.join(args.results_root, args.model, args.folder2_name)
    print(f"folder1: {folder1}")
    print(f"folder2: {folder2}")

    if args.model not in MODEL_NUM_LAYERS:
        raise SystemExit(
            f"Unknown model {args.model!r}. Add it to MODEL_NUM_LAYERS "
            f"(known: {sorted(MODEL_NUM_LAYERS)})."
        )
    n_layers = MODEL_NUM_LAYERS[args.model]

    # ----- winoground task: separate metrics + early return ----------------
    if args.task == "winoground":
        wino: dict = {metric: {} for metric in RANKING_METRICS}
        parts_evaluated = set()
        for part in args.part_types:
            last_layer = n_layers if part in ("norm1", "norm2") else n_layers - 1
            for layer_id in range(last_layer + 1):
                f1 = layer_file(folder1, layer_id, part, args.suffix1)
                f2 = layer_file(folder2, layer_id, part, args.suffix2)
                if not (f1.is_file() and f2.is_file()):
                    continue
                parts_evaluated.add(part)
                e1 = torch.load(f1, map_location="cpu")
                e2 = torch.load(f2, map_location="cpu")
                line = f"[layer {layer_id}, part {part}]"
                for metric in RANKING_METRICS:
                    m = winoground_scores(e1, e2, metric, args.device)
                    wino[metric].setdefault(part, {})[str(layer_id)] = m
                    line += (
                        f"  [{metric}] text={m['text_score']:.4f} "
                        f"image={m['image_score']:.4f} group={m['group_score']:.4f}"
                    )
                print(line)

        if not parts_evaluated:
            raise SystemExit(
                f"No layer files found under {folder1} / {folder2} "
                f"for suffixes {args.suffix1!r}/{args.suffix2!r} and parts {args.part_types}."
            )

        out_path = args.out_path or os.path.join(
            args.results_root,
            "retrieval_metrics",
            args.retrieval_dataset_name,
            args.model,
            f"winoground_{args.suffix1}_vs_{args.suffix2}.json",
        )
        payload = {
            "task": "winoground",
            "model": args.model,
            "folder1": folder1,
            "folder2": folder2,
            "suffix1": args.suffix1,
            "suffix2": args.suffix2,
            "metrics": list(RANKING_METRICS),
            "part_types_evaluated": sorted(parts_evaluated),
            "description": (
                "Winoground per-entry text/image/group scores keyed "
                "[metric][part][layer]. Inputs are side-major flattened "
                "[2N,D] tensors (rows: (id=0,side=0),(id=0,side=1),"
                "(id=1,side=0),...) reshaped to [N,2,D]. With "
                "sim[n, image_side, text_side]: "
                "text_score = mean of [sim(i0,c0)>sim(i0,c1) AND sim(i1,c1)>sim(i1,c0)]; "
                "image_score = mean of [sim(i0,c0)>sim(i1,c0) AND sim(i1,c1)>sim(i0,c1)]; "
                "group_score = mean of [text AND image]. "
                "Cosine = dot on L2-normalized vectors; Euclidean = -L2-distance."
            ),
            "results": wino,
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved winoground metrics to {out_path}")
        return
    # ----------------------------------------------------------------------

    dir_1to2 = f"{args.suffix1}_to_{args.suffix2}"
    dir_2to1 = f"{args.suffix2}_to_{args.suffix1}"

    # results[metric][part][direction][layer_id] = {k: ndcg_k, mean_rank, ...}
    results: dict = {metric: {} for metric in RANKING_METRICS}

    do_pseudo = args.label_file is not None and args.mask_file is not None
    if do_pseudo:
        labels_np = np.load(args.label_file)
        mask_np = np.load(args.mask_file).astype(bool)
        if labels_np.shape[0] != mask_np.shape[0]:
            raise SystemExit(
                f"label/mask length mismatch: {labels_np.shape[0]} vs {mask_np.shape[0]}"
            )
        mask_t = torch.from_numpy(mask_np)
        labels_kept = torch.from_numpy(labels_np[mask_np]).long()
        print(
            f"pseudolabel eval: {int(mask_np.sum())} rows kept, "
            f"{len(np.unique(labels_kept.numpy()))} clusters"
        )
        results_pseudo: dict = {metric: {} for metric in RANKING_METRICS}

    def layer_entry(m):
        entry = {str(k): m["ndcg"][k] for k in args.ks}
        entry["mean_rank"] = m["mean_rank"]
        entry["mean_percentile_rank"] = m["mean_percentile_rank"]
        entry["mrr"] = m["mrr"]
        return entry

    def fmt(m):
        ndcg_str = ", ".join(f"ndcg@{k}={m['ndcg'][k]:.4f}" for k in args.ks)
        return (
            f"{ndcg_str}, mean_rank={m['mean_rank']:.2f}, "
            f"mpr={m['mean_percentile_rank']:.4f}, mrr={m['mrr']:.4f}"
        )

    parts_evaluated = set()
    for part in args.part_types:
        # norm1/norm2 add a post-stack norm tensor at index n_layers; res2 stops
        # at n_layers - 1 (see get_target_layers in embeddings_vlms.py).
        last_layer = n_layers if part in ("norm1", "norm2") else n_layers - 1
        for layer_id in range(last_layer + 1):
            f1 = layer_file(folder1, layer_id, part, args.suffix1)
            f2 = layer_file(folder2, layer_id, part, args.suffix2)
            if not (f1.is_file() and f2.is_file()):
                continue
            parts_evaluated.add(part)
            print(f"[layer {layer_id}, part {part}]")
            e1 = torch.load(f1, map_location="cpu")
            e2 = torch.load(f2, map_location="cpu")

            if do_pseudo:
                if e1.shape[0] != mask_t.shape[0] or e2.shape[0] != mask_t.shape[0]:
                    raise SystemExit(
                        f"mask length {mask_t.shape[0]} does not match embedding rows "
                        f"{e1.shape[0]} / {e2.shape[0]} at layer {layer_id} part {part}"
                    )
                e1_sub = e1[mask_t]
                e2_sub = e2[mask_t]

            for metric in RANKING_METRICS:
                m_12 = retrieval_metrics_paired(e1, e2, args.ks, metric, args.device)
                m_21 = retrieval_metrics_paired(e2, e1, args.ks, metric, args.device)

                # print(f"  [{metric}] {dir_1to2}: {fmt(m_12)}")
                # print(f"  [{metric}] {dir_2to1}: {fmt(m_21)}")

                part_block = results[metric].setdefault(part, {})
                part_block.setdefault(dir_1to2, {})[str(layer_id)] = layer_entry(m_12)
                part_block.setdefault(dir_2to1, {})[str(layer_id)] = layer_entry(m_21)

                if do_pseudo:
                    mp_12 = retrieval_metrics_multilabel(
                        e1_sub, e2_sub, labels_kept, args.ks, metric, args.device
                    )
                    mp_21 = retrieval_metrics_multilabel(
                        e2_sub, e1_sub, labels_kept, args.ks, metric, args.device
                    )
                    # print(f"  [{metric}/pseudo] {dir_1to2}: {fmt(mp_12)}")
                    # print(f"  [{metric}/pseudo] {dir_2to1}: {fmt(mp_21)}")
                    pblock = results_pseudo[metric].setdefault(part, {})
                    pblock.setdefault(dir_1to2, {})[str(layer_id)] = layer_entry(mp_12)
                    pblock.setdefault(dir_2to1, {})[str(layer_id)] = layer_entry(mp_21)

    if not parts_evaluated:
        raise SystemExit(
            f"No layer files found under {folder1} / {folder2} "
            f"for suffixes {args.suffix1!r}/{args.suffix2!r} and parts {args.part_types}."
        )

    out_path = args.out_path or os.path.join(
        args.results_root,
        "retrieval_metrics",
        args.retrieval_dataset_name,
        args.model,
        f"retrieval_{args.suffix1}_vs_{args.suffix2}.json",
    )
    payload = {
        "model": args.model,
        "folder1": folder1,
        "folder2": folder2,
        "suffix1": args.suffix1,
        "suffix2": args.suffix2,
        "ks": args.ks,
        "metrics": list(RANKING_METRICS),
        "part_types_evaluated": sorted(parts_evaluated),
        "description": (
            "results[metric][part][direction][layer] holds per-layer entries "
            "with ndcg@k (numeric string keys from --ks) plus mean_rank, "
            "mean_percentile_rank (=AUC, in [0,1], higher better), and mrr "
            "without cutoff. metric in {euclidean, cosine}; binary "
            "single-relevance, paired rows."
        ),
        "results": results,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved retrieval metrics to {out_path}")

    if do_pseudo:
        out_path_pseudo = args.out_path_pseudolabels or (
            os.path.splitext(out_path)[0] + "_pseudolabels.json"
        )
        payload_pseudo = {
            "model": args.model,
            "folder1": folder1,
            "folder2": folder2,
            "suffix1": args.suffix1,
            "suffix2": args.suffix2,
            "ks": args.ks,
            "metrics": list(RANKING_METRICS),
            "part_types_evaluated": sorted(parts_evaluated),
            "label_file": args.label_file,
            "mask_file": args.mask_file,
            "n_rows_kept": int(mask_t.sum().item()),
            "n_clusters": int(len(np.unique(labels_kept.numpy()))),
            "description": (
                "Multi-relevance retrieval metrics using cluster pseudolabels. "
                "Positives for query i = all docs j with the same label. "
                "mean_rank/mrr use the rank of the FIRST positive (1=best). "
                "mean_percentile_rank is per-query Mann-Whitney AUC averaged "
                "across queries (in [0, 1], higher better; reduces to "
                "(1 - (rank-1)/(N-1)) for single-positive). NDCG@k uses "
                "binary IDCG@k = sum_{r=1..min(k, n_pos_i)} 1/log2(r+1). "
                "Evaluated on the rows selected by mask_file."
            ),
            "results": results_pseudo,
        }
        os.makedirs(os.path.dirname(out_path_pseudo) or ".", exist_ok=True)
        with open(out_path_pseudo, "w", encoding="utf-8") as f:
            json.dump(payload_pseudo, f, indent=2)
        print(f"Saved pseudolabel retrieval metrics to {out_path_pseudo}")


if __name__ == "__main__":
    main()
