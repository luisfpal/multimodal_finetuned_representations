"""Cluster CLIP image embeddings with dadapy (ADP) and report per-cluster populations."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dadapy import Data
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from utils.operations_utils import seed_all

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
MAXK = 128
RANGE_MAX = 128
K = 16
Z = 1.65
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots" / "clustering"

DATASET_CONFIGS: dict[str, dict] = {
    "mini-imagenet": {
        "hf_name": "timm/mini-imagenet",
        "split": "validation",
        "out_path": DATA_DIR / "mini-imagenet" / "mini-imagenet_val_clip_adp_clusters.csv",
        "plot_path": PLOTS_DIR / "mini-imagenet_val_cluster_population_overview.png",
    },
    "cocoqa": {
        "hf_name": "l11p/COCOQA-restval-ICQA",
        "split": "train",
        "out_path": DATA_DIR / "coco" / "cocoqa_restval_clip_adp_clusters.csv",
        "plot_path": PLOTS_DIR / "cocoqa_restval_cluster_population_overview.png",
    },
    "imagenet-animals": {
        "hf_name": "RitAreaSciencePark/imagenet_short_text_100_classes_x_100_samples",
        "split": "train",
        "out_path": DATA_DIR / "imagenet_animals" / "imagenet_animals_train_clip_adp_clusters.csv",
        "plot_path": PLOTS_DIR / "imagenet_animals_train_cluster_population_overview.png",
    },
}


def load_mini_imagenet() -> tuple[list, list[int], list[str], dict[str, str]]:
    dataset = load_dataset("timm/mini-imagenet", split="validation")
    class_names: list[str] = dataset.features["label"].names  # wnids, indexed by label_id
    images = [sample["image"] for sample in dataset]
    label_ids = [sample["label"] for sample in dataset]

    cat_csv = DATA_DIR / "mini-imagenet" / "mini-imagenet_mcq_categorization.csv"
    wnid_to_class: dict[str, str] = {}
    with cat_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wnid_to_class[row["wnid"]] = row["class_name"].lower()

    return images, label_ids, class_names, wnid_to_class


def load_imagenet_animals() -> tuple[list, list[int], list[str], dict[str, str]]:
    dataset = load_dataset(
        "RitAreaSciencePark/imagenet_short_text_100_classes_x_100_samples", split="train"
    )
    images: list = []
    label_ids: list[int] = []
    offsets: list[str] = []
    offset_to_class: dict[str, str] = {}
    for sample in dataset:
        images.append(sample["image"])
        label_ids.append(sample["label"])
        offsets.append(sample["offset"])
        if sample["offset"] not in offset_to_class:
            offset_to_class[sample["offset"]] = sample["synset"].split(".")[0].lower()
    return images, label_ids, offsets, offset_to_class


def load_cocoqa() -> tuple[list, list[str]]:
    dataset = load_dataset("l11p/COCOQA-restval-ICQA", split="train")
    images = [sample["image"] for sample in dataset]
    image_ids = [sample["image_id"] for sample in dataset]
    return images, image_ids


@torch.no_grad()
def embed_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    all_embeddings: list[torch.Tensor] = []
    for start in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        emb = model.get_image_features(**inputs)
        all_embeddings.append(emb.cpu().float())
    return torch.cat(all_embeddings, dim=0).numpy()


def cluster_with_adp(embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    emb = torch.tensor(embeddings, device=DEVICE)
    distances = torch.cdist(emb, emb).cpu().numpy()

    maxk = min(MAXK, len(embeddings) - 1)
    data = Data(distances=distances, maxk=maxk)
    id_list, _, _ = data.return_id_scaling_gride(range_max=RANGE_MAX)
    data.set_id(id_list[int(math.log2(K))])
    data.compute_density_kNN(k=K)
    labels: np.ndarray = data.compute_clustering_ADP(Z=Z)

    n_clusters = len(set(labels[labels >= 0]))
    return labels, n_clusters


def _gini(sizes: np.ndarray) -> float:
    s = np.sort(sizes.astype(float))
    n = len(s)
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * s).sum() / (n * s.sum()))


def print_cluster_stats(cluster_ids: np.ndarray, n_clusters: int, name: str) -> None:
    counts = Counter(int(c) for c in cluster_ids if c >= 0)
    sizes = np.array(sorted(counts.values(), reverse=True))
    n_ge_10 = int((sizes >= 10).sum())
    print(f"\n=== {name} cluster stats ===")
    print(f"  N clusters:                  {n_clusters}")
    print(f"  Total assigned:              {int(sizes.sum())}")
    print(f"  Max instances/cluster:       {int(sizes[0])}")
    print(f"  Min instances/cluster:       {int(sizes[-1])}")
    print(f"  Mean instances/cluster:      {sizes.mean():.1f}")
    print(f"  Median:                      {np.median(sizes):.1f}")
    print(f"  Clusters with >= 10 images:  {n_ge_10} / {n_clusters}")


def plot_cluster_population_overview(
    cluster_ids: np.ndarray,
    dataset_title: str,
    out_path: Path,
    label_top_n: int = 5,
) -> None:
    counts = Counter(int(c) for c in cluster_ids if c >= 0)
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cid_labels = [f"cl-{cid}" for cid, _ in ranked]
    sizes = np.array([sz for _, sz in ranked])
    n = len(sizes)
    total = int(sizes.sum())

    cumulative_pct = np.cumsum(sizes) / total * 100
    max_min_ratio = sizes[0] / sizes[-1]
    gini_val = _gini(sizes)
    top10_pct = float(cumulative_pct[min(9, n - 1)])
    top20_pct = float(cumulative_pct[min(19, n - 1)])
    n_ge_10 = int((sizes >= 10).sum())

    subtitle = (
        f"clusters={n} | samples={total} | max/min={max_min_ratio:.1f}x | "
        f"gini={gini_val:.3f} | top-10={top10_pct:.1f}% | top-20={top20_pct:.1f}%"
    )
    print(f"  Clusters with >= 10 images: {n_ge_10} / {n}")

    cmap = plt.colormaps["viridis"]
    # rank 0 (largest) → dark blue (0.15), last rank → yellow-green (0.92)
    colors = [cmap(0.15 + 0.77 * i / max(n - 1, 1)) for i in range(n)]
    ranks = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(max(10, n * 0.28), 5))
    ax2 = ax1.twinx()

    ax1.bar(ranks, sizes, color=colors, width=0.85, zorder=2)

    for i in range(min(label_top_n, n)):
        ax1.text(
            ranks[i],
            sizes[i] + total * 0.004,
            cid_labels[i],
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

    ax2.plot(ranks, cumulative_pct, color="crimson", linewidth=1.8, zorder=3)
    ax2.axhline(80, color="crimson", linewidth=1.0, linestyle="--", alpha=0.65)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Cumulative population (%)", color="crimson", fontsize=10)
    ax2.tick_params(axis="y", colors="crimson", labelsize=9)

    ax1.set_xlabel("Cluster rank (sorted by population)", fontsize=10)
    ax1.set_ylabel("Population", fontsize=10)
    ax1.set_xlim(-0.6, n - 0.4)
    tick_step = max(1, n // 10)
    ax1.set_xticks(ranks[::tick_step])
    ax1.tick_params(axis="both", which="major", labelsize=9)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    fig.suptitle(dataset_title, fontsize=12, fontweight="bold", y=1.01)
    ax1.set_title(subtitle, fontsize=8.5, pad=3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def save_mini_imagenet_clusters(
    label_ids: list[int],
    class_names: list[str],         # wnids indexed by label_id
    wnid_to_class: dict[str, str],  # wnid → human class_name (lowercase)
    cluster_ids: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label_id", "label_name", "class_name", "cluster_id"])
        for idx, (label_id, cluster_id) in enumerate(zip(label_ids, cluster_ids)):
            wnid = class_names[label_id]
            writer.writerow([idx, label_id, wnid, wnid_to_class[wnid], int(cluster_id)])
    print(f"Saved cluster mapping: {out_path}")


def save_imagenet_animals_clusters(
    label_ids: list[int],
    offsets: list[str],
    offset_to_class: dict[str, str],  # offset → human class_name (lowercase)
    cluster_ids: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "offset", "label", "class_name", "cluster_id"])
        for idx, (label_id, offset, cluster_id) in enumerate(
            zip(label_ids, offsets, cluster_ids)
        ):
            writer.writerow([idx, offset, label_id, offset_to_class[offset], int(cluster_id)])
    print(f"Saved cluster mapping: {out_path}")


def save_coco_clusters(
    image_ids: list[str],
    cluster_ids: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "image_id", "cluster_id"])
        for idx, (image_id, cluster_id) in enumerate(zip(image_ids, cluster_ids)):
            writer.writerow([idx, image_id, int(cluster_id)])
    print(f"Saved cluster mapping: {out_path}")


def run_mini_imagenet(model: CLIPModel, processor: CLIPProcessor) -> None:
    cfg = DATASET_CONFIGS["mini-imagenet"]
    print("\nLoading mini-imagenet validation split...")
    images, label_ids, class_names, wnid_to_class = load_mini_imagenet()
    n_gt_classes = len(set(label_ids))
    print(f"Loaded {len(images)} images | {n_gt_classes} unique classes")

    embeddings = embed_images(model, processor, images)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Clustering with Advanced Density Peak...")
    cluster_ids, n_clusters = cluster_with_adp(embeddings)
    ratio = n_clusters / n_gt_classes
    print(f"  N clusters: {n_clusters} | GT classes: {n_gt_classes} | ratio: {ratio:.3f}")
    print_cluster_stats(cluster_ids, n_clusters, "mini-imagenet")

    save_mini_imagenet_clusters(label_ids, class_names, wnid_to_class, cluster_ids, cfg["out_path"])
    plot_cluster_population_overview(
        cluster_ids,
        "Mini-ImageNet Validation Clusters — Population Overview",
        cfg["plot_path"],
    )


def run_imagenet_animals(model: CLIPModel, processor: CLIPProcessor) -> None:
    cfg = DATASET_CONFIGS["imagenet-animals"]
    print("\nLoading imagenet_short_text_100_classes_x_100_samples train split...")
    images, label_ids, offsets, offset_to_class = load_imagenet_animals()
    n_gt_classes = len(set(label_ids))
    print(f"Loaded {len(images)} images | {n_gt_classes} unique classes")

    embeddings = embed_images(model, processor, images)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Clustering with Advanced Density Peak...")
    cluster_ids, n_clusters = cluster_with_adp(embeddings)
    ratio = n_clusters / n_gt_classes
    print(f"  N clusters: {n_clusters} | GT classes: {n_gt_classes} | ratio: {ratio:.3f}")
    print_cluster_stats(cluster_ids, n_clusters, "imagenet-animals")

    save_imagenet_animals_clusters(label_ids, offsets, offset_to_class, cluster_ids, cfg["out_path"])
    plot_cluster_population_overview(
        cluster_ids,
        "ImageNet Animals Train Clusters — Population Overview",
        cfg["plot_path"],
    )


def run_cocoqa(model: CLIPModel, processor: CLIPProcessor) -> None:
    cfg = DATASET_CONFIGS["cocoqa"]
    print("\nLoading COCOQA-restval-ICQA train split...")
    images, image_ids = load_cocoqa()
    print(f"Loaded {len(images)} images")

    embeddings = embed_images(model, processor, images)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Clustering with Advanced Density Peak...")
    cluster_ids, n_clusters = cluster_with_adp(embeddings)
    print_cluster_stats(cluster_ids, n_clusters, "cocoqa")

    save_coco_clusters(image_ids, cluster_ids, cfg["out_path"])
    plot_cluster_population_overview(
        cluster_ids,
        "COCOQA Restval Train Clusters — Population Overview",
        cfg["plot_path"],
    )


def plot_from_csv(dataset: str) -> None:
    """Re-generate the population overview plot from an existing cluster CSV."""
    cfg = DATASET_CONFIGS[dataset]
    cluster_ids: list[int] = []
    with cfg["out_path"].open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_ids.append(int(row["cluster_id"]))
    arr = np.array(cluster_ids)
    n_clusters = len(set(arr[arr >= 0]))
    print_cluster_stats(arr, n_clusters, dataset)
    title_map = {
        "mini-imagenet": "Mini-ImageNet Validation Clusters — Population Overview",
        "cocoqa": "COCOQA Restval Train Clusters — Population Overview",
        "imagenet-animals": "ImageNet Animals Train Clusters — Population Overview",
    }
    plot_cluster_population_overview(arr, title_map[dataset], cfg["plot_path"])


def main() -> None:
    seed_all(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["mini-imagenet", "cocoqa", "imagenet-animals", "all"],
        default="cocoqa",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Re-generate plot from existing CSV without re-clustering.",
    )
    args = parser.parse_args()

    if args.plot_only:
        datasets = (
            ["mini-imagenet", "cocoqa", "imagenet-animals"]
            if args.dataset == "all"
            else [args.dataset]
        )
        for ds in datasets:
            plot_from_csv(ds)
        return

    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=MODEL_DTYPE).to(DEVICE).eval()

    if args.dataset in ("mini-imagenet", "all"):
        run_mini_imagenet(model, processor)
    if args.dataset in ("cocoqa", "all"):
        run_cocoqa(model, processor)
    if args.dataset in ("imagenet-animals", "all"):
        run_imagenet_animals(model, processor)


if __name__ == "__main__":
    main()
