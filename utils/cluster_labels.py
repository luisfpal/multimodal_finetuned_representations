"""Replace instance labels with CLIP-ADP cluster IDs and cap the dominant cluster."""

import csv
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CLUSTER_CSVS: dict[str, Path] = {
    "mini-imagenet": PROJECT_ROOT / "data/mini-imagenet/mini-imagenet_val_clip_adp_clusters.csv",
    "imagenet_short_text": PROJECT_ROOT / "data/imagenet_animals/imagenet_animals_train_clip_adp_clusters.csv",
}


def apply_cluster_labels(
    representations: torch.Tensor,
    labels: np.ndarray,
    dataset_name: str,
    seed: int = 42,
    return_mask: bool = False,
) -> tuple[torch.Tensor, np.ndarray] | tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Replace instance labels with CLIP-ADP cluster IDs, then cap the dominant cluster.

    Positional mapping: CSV row i corresponds to representations[:, i, :].
    Capping: largest cluster is truncated to the size of the second-largest.
    """
    is_mini_imagenet = "mini-imagenet" in dataset_name.lower()
    is_imagenet_animals = "imagenet_short_text" in dataset_name.lower()
    if not (is_mini_imagenet or is_imagenet_animals):
        mask = np.ones(len(labels), dtype=bool)
        return (representations, labels, mask) if return_mask else (representations, labels)

    key = "mini-imagenet" if is_mini_imagenet else "imagenet_short_text"
    with _CLUSTER_CSVS[key].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(labels)
    if len(rows) < n:
        raise ValueError(f"Cluster CSV has {len(rows)} rows but need {n}")

    cluster_ids = np.array([str(int(rows[i]["cluster_id"])) for i in range(n)])
    mask = np.ones(n, dtype=bool)

    unique, counts = np.unique(cluster_ids, return_counts=True)
    order = np.argsort(-counts)
    if len(order) >= 2:
        top_cluster = unique[order[0]]
        second_count = int(counts[order[1]])
        top_indices = np.where(cluster_ids == top_cluster)[0].tolist()
        if len(top_indices) > second_count:
            rng = random.Random(seed)
            rng.shuffle(top_indices)
            keep = set(top_indices[:second_count])
            mask = np.array([cluster_ids[i] != top_cluster or i in keep for i in range(n)])
            representations = representations[:, mask, :]
            cluster_ids = cluster_ids[mask]

    return (representations, cluster_ids, mask) if return_mask else (representations, cluster_ids)
