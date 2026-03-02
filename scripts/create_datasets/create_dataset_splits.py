"""Generate deterministic, disjoint index splits for OpenVQA datasets.

Each split is a JSON file at ``data/splits/{dataset_slug}/{split_name}.json``
containing ``{"indices": [...], "metadata": {...}}``.

Splits are created by shuffling all dataset indices with a fixed seed and
partitioning sequentially.  This guarantees disjointness by construction.

Usage:
    python scripts/create_datasets/create_dataset_splits.py
"""

import json
import os

import numpy as np

# ── Split configurations ─────────────────────────────────────────────────────
# Each entry: (dataset_slug, total_size, [(split_name, split_size), ...])
# The last split gets "the rest" (total - sum of earlier sizes).
SPLIT_CONFIGS: list[tuple[str, int, list[tuple[str, int]]]] = [
    (
        "cocoqa_restval_icqa",
        17244,
        [("split01", 10000), ("split02", 7244)],
    ),
    (
        "vqav2_coco_restval_icq",
        30504,
        [("split01", 10000), ("split02", 10000), ("split03", 10504)],
    ),
]

SEED = 42


def create_splits(
    dataset_slug: str,
    total_size: int,
    splits: list[tuple[str, int]],
    output_dir: str,
    seed: int = SEED,
) -> None:
    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(total_size).tolist()

    offset = 0
    for split_name, split_size in splits:
        indices = sorted(all_indices[offset : offset + split_size])
        offset += split_size

        split_data = {
            "indices": indices,
            "metadata": {
                "dataset": dataset_slug,
                "split_name": split_name,
                "total_size": total_size,
                "split_size": len(indices),
                "seed": seed,
            },
        }

        split_dir = os.path.join(output_dir, dataset_slug)
        os.makedirs(split_dir, exist_ok=True)
        path = os.path.join(split_dir, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  wrote {path} ({len(indices)} indices)")

    assert offset == total_size, (
        f"Split sizes don't sum to total: {offset} != {total_size}"
    )


def main() -> None:
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_dir = os.path.join(project_root, "data", "splits")
    os.makedirs(output_dir, exist_ok=True)

    for dataset_slug, total_size, splits in SPLIT_CONFIGS:
        print(f"\n{dataset_slug} (N={total_size}):")
        create_splits(dataset_slug, total_size, splits, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
