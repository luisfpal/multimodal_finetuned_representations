"""
Patch VQAv2 (validation) with human captions from l11p/VQAv2-COCO-restval-ICQ.

Pipeline:
1. Load l11p/VQAv2-COCO-restval-ICQ to extract captions (list of strings per image)
2. Load lmms-lab/VQAv2 (validation split)
3. Map COCO filename image_id -> integer image_id
4. Filter VQAv2 rows with matching captions
5. Add 'captions' column (list of strings)
6. Shuffle and push to hub as VQAv2-COCO-restval-human-caps
"""

import os
import argparse
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def _coco_filename_to_int_id(filename: str) -> int:
    """Convert COCO filename like 'COCO_val2014_000000522418.jpg' to integer 522418."""
    return int(filename.split("_")[-1].split(".")[0])


def main():
    parser = argparse.ArgumentParser(
        description="Patch VQAv2 with human captions from l11p/VQAv2-COCO-restval-ICQ"
    )
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to HuggingFace Hub")
    parser.add_argument("--num-proc", type=int, default=16,
                        help="Number of parallel processes for filter/map (default: 16)")
    parser.add_argument("--hub-username", type=str, default="l11p",
                        help="HuggingFace username")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PATCHING VQAv2 DATASET WITH HUMAN CAPTIONS")
    print("="*60)

    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        cache_dir = os.path.expanduser(os.path.expandvars(cache_dir))

    # Step 1: Load the ICQ dataset to extract human captions
    icq_dataset_id = f"{args.hub_username}/VQAv2-COCO-restval-ICQ"
    print(f"\n[1/5] Loading {icq_dataset_id} for captions...")
    icq_dataset = load_dataset(icq_dataset_id, split="train", cache_dir=cache_dir)
    print(f"Loaded {len(icq_dataset)} samples from ICQ dataset")

    # Build mapping: int_image_id -> list of caption strings
    # Columns: image_id (str, e.g. 'COCO_val2014_000000522418.jpg'), captions (list[str])
    print("Building captions lookup...")
    icq_image_ids = icq_dataset['image_id']
    icq_captions = icq_dataset['captions']

    captions_map = {}
    for img_id_str, caps in zip(icq_image_ids, icq_captions):
        if caps and len(caps) > 0:
            int_id = _coco_filename_to_int_id(img_id_str)
            captions_map[int_id] = caps

    print(f"Built captions map: {len(captions_map)} images with captions")

    # Free ICQ dataset memory — we only need the map
    del icq_dataset, icq_image_ids, icq_captions

    # Step 2: Load VQAv2 dataset
    print("\n[2/5] Loading VQAv2 dataset (validation split)...")
    dataset = load_dataset("lmms-lab/VQAv2", split="validation", cache_dir=cache_dir)
    original_len = len(dataset)
    print(f"Loaded {original_len} samples")

    # Step 3: Filter rows with matching captions
    valid_ids = set(captions_map.keys())

    def filter_fn(example):
        return example['image_id'] in valid_ids

    print("\n[3/5] Filtering dataset...")
    dataset_filtered = dataset.filter(filter_fn, num_proc=args.num_proc)
    print(f"Kept {len(dataset_filtered)}/{original_len} samples")

    # Step 4: Add captions column
    def add_captions(example):
        example['captions'] = captions_map[example['image_id']]
        return example

    print("\n[4/5] Adding captions column...")
    dataset_with_captions = dataset_filtered.map(add_captions, num_proc=args.num_proc)

    # Shuffle
    dataset_with_captions = dataset_with_captions.shuffle(seed=42)

    unique_images = len(set(dataset_with_captions['image_id']))
    print(f"\nFinal: {len(dataset_with_captions)} samples from {unique_images} unique images")

    if len(dataset_with_captions) > 0:
        sample = dataset_with_captions[0]
        print(f"\nSample:")
        print(f"  image_id: {sample['image_id']}")
        print(f"  num captions: {len(sample['captions'])}")
        print(f"  first caption: {sample['captions'][0][:100]}...")

    # Step 5: Push to hub
    if args.push_to_hub:
        repo_id = f"{args.hub_username}/VQAv2-COCO-restval-human-caps"
        print(f"\n[5/5] Pushing to Hub: {repo_id}")
        token = os.getenv("HF_TOKEN")
        if token:
            dataset_with_captions.push_to_hub(repo_id, token=token)
        else:
            dataset_with_captions.push_to_hub(repo_id)
        print("✅ Pushed successfully!")
    else:
        print("\n[5/5] Skipping push to hub (use --push-to-hub to push)")

    print("\n" + "="*60)
    print("✅ MISSION ACCOMPLISHED!")
    print("="*60)


if __name__ == "__main__":
    main()
