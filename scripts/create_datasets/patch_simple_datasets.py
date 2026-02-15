"""
Patch datasets with generated captions.

Supported datasets: MME, MMBench_EN, VQAv2 (synthetic), COCOQA (synthetic).

Optimized pipeline:
1. Load dataset (Arrow/Memory-mapped)
2. Load captions
3. Filter dataset (keep only samples with valid captions)
4. Map dataset (add caption column)
5. Push to hub
"""

import os
import json
import argparse
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def get_num_proc():
    # Use half of available CPUs
    try:
        count = os.cpu_count()
        if count:
            return max(1, count // 2)
    except:
        pass
    return 1

def patch_mme_dataset(captions_file: str, push_to_hub: bool, hub_username: str):
    print("\n" + "="*60)
    print("PATCHING MME DATASET (OPTIMIZED)")
    print("="*60)

    # Load captions
    print(f"Loading captions from {captions_file}...")
    with open(captions_file, 'r') as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")

    # Load dataset
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        cache_dir = os.path.expanduser(os.path.expandvars(cache_dir))

    print("Loading MME dataset (test split)...")
    dataset = load_dataset("lmms-lab/MME", split="test", cache_dir=cache_dir)
    original_len = len(dataset)
    print(f"Loaded {original_len} samples")

    # Filter
    # We close over 'captions' here.
    # Note: JSON keys are strings. MME question_id might be int or string.
    # The original code used captions.get(qid).
    # We will assume this behavior is correct.
    
    def filter_fn(example):
        qid = example['question_id']
        cap = captions.get(qid, "FAILED")
        return cap != "FAILED" and cap is not None

    print("Filtering dataset (removing missing/failed captions)...")
    dataset_filtered = dataset.filter(filter_fn, num_proc=get_num_proc())
    
    print(f"Kept {len(dataset_filtered)}/{original_len} samples")

    # Add captions
    def add_caption(example):
        qid = example['question_id']
        example['caption'] = captions[qid]
        return example

    print("Adding captions column...")
    dataset_with_captions = dataset_filtered.map(add_caption, num_proc=get_num_proc())

    # Shuffle
    dataset_with_captions = dataset_with_captions.shuffle(seed=42)

    unique_images_final = len(set(dataset_with_captions['question_id']))
    print(f"Final: {len(dataset_with_captions)} samples from {unique_images_final} unique images")

    if len(dataset_with_captions) > 0:
        sample = dataset_with_captions[0]
        print(f"\nSample: question_id={sample['question_id']}, caption={sample['caption'][:100]}...")

    if push_to_hub:
        repo_id = f"{hub_username}/MME-test-captioned"
        print(f"\nPushing to Hub: {repo_id}")
        token = os.getenv("HF_TOKEN")
        if token:
            dataset_with_captions.push_to_hub(repo_id, token=token)
        else:
            dataset_with_captions.push_to_hub(repo_id)
        print("✅ Pushed successfully!")

    return dataset_with_captions


def _coco_filename_to_int_id(filename: str) -> int:
    """Convert COCO filename like 'COCO_val2014_000000522418.jpg' to integer 522418."""
    # Extract the numeric part after the last underscore, before the extension
    return int(filename.split("_")[-1].split(".")[0])


def patch_vqav2_synthetic_dataset(captions_file: str, push_to_hub: bool, hub_username: str, num_proc: int = 8):
    """Patch VQAv2 (validation) with synthetic captions.
    
    Captions JSON keys are COCO filenames (e.g. 'COCO_val2014_000000522418.jpg').
    VQAv2 uses integer image_id (e.g. 522418). We map between the two formats.
    Multiple VQAv2 rows share the same image_id (multiple questions per image).
    """
    print("\n" + "="*60)
    print("PATCHING VQAv2 DATASET WITH SYNTHETIC CAPTIONS")
    print("="*60)

    # Load captions and build int-keyed lookup
    print(f"Loading captions from {captions_file}...")
    with open(captions_file, 'r') as f:
        raw_captions = json.load(f)
    print(f"Loaded {len(raw_captions)} captions")

    # Pre-compute {int_image_id: caption_string}
    captions = {}
    skipped = 0
    for filename, cap in raw_captions.items():
        if cap is None or cap == "FAILED":
            skipped += 1
            continue
        int_id = _coco_filename_to_int_id(filename)
        captions[int_id] = cap
    print(f"Valid captions: {len(captions)} (skipped {skipped} failed/null)")

    # Load dataset
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        cache_dir = os.path.expanduser(os.path.expandvars(cache_dir))

    print("Loading VQAv2 dataset (validation split)...")
    dataset = load_dataset("lmms-lab/VQAv2", split="validation", cache_dir=cache_dir)
    original_len = len(dataset)
    print(f"Loaded {original_len} samples")

    # Filter: keep rows whose image_id has a valid caption
    valid_ids = set(captions.keys())  # freeze for closure
    
    def filter_fn(example):
        return example['image_id'] in valid_ids

    print("Filtering dataset...")
    dataset_filtered = dataset.filter(filter_fn, num_proc=num_proc)
    print(f"Kept {len(dataset_filtered)}/{original_len} samples")

    # Add captions
    def add_caption(example):
        example['caption'] = captions[example['image_id']]
        return example

    print("Adding caption column...")
    dataset_with_captions = dataset_filtered.map(add_caption, num_proc=num_proc)

    # Shuffle
    dataset_with_captions = dataset_with_captions.shuffle(seed=42)

    unique_images = len(set(dataset_with_captions['image_id']))
    print(f"Final: {len(dataset_with_captions)} samples from {unique_images} unique images")

    if len(dataset_with_captions) > 0:
        sample = dataset_with_captions[0]
        print(f"\nSample: image_id={sample['image_id']}, caption={sample['caption'][:100]}...")

    if push_to_hub:
        repo_id = f"{hub_username}/VQAv2-COCO-restval-synthetic-caps"
        print(f"\nPushing to Hub: {repo_id}")
        token = os.getenv("HF_TOKEN")
        if token:
            dataset_with_captions.push_to_hub(repo_id, token=token)
        else:
            dataset_with_captions.push_to_hub(repo_id)
        print("✅ Pushed successfully!")

    return dataset_with_captions


def patch_cocoqa_synthetic_dataset(captions_file: str, push_to_hub: bool, hub_username: str, num_proc: int = 4):
    """Patch COCOQA-restval-ICQA with synthetic captions.
    
    Captions JSON keys match the dataset's image_id directly (both are COCO filenames).
    Adds a 'synthetic-caption' column. Pushes to the same COCOQA-restval-ICQA dataset.
    """
    print("\n" + "="*60)
    print("PATCHING COCOQA DATASET WITH SYNTHETIC CAPTIONS")
    print("="*60)

    # Load captions
    print(f"Loading captions from {captions_file}...")
    with open(captions_file, 'r') as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")

    # Load dataset
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        cache_dir = os.path.expanduser(os.path.expandvars(cache_dir))

    print("Loading COCOQA-restval-ICQA dataset...")
    dataset = load_dataset(f"{hub_username}/COCOQA-restval-ICQA", split="train", cache_dir=cache_dir)
    original_len = len(dataset)
    print(f"Loaded {original_len} samples")

    # Filter
    def filter_fn(example):
        cap = captions.get(example['image_id'], "FAILED")
        return cap != "FAILED" and cap is not None

    print("Filtering dataset (removing missing/failed captions)...")
    dataset_filtered = dataset.filter(filter_fn, num_proc=num_proc)
    print(f"Kept {len(dataset_filtered)}/{original_len} samples")

    # Add synthetic caption
    def add_caption(example):
        example['synthetic-caption'] = captions[example['image_id']]
        return example

    print("Adding synthetic-caption column...")
    dataset_with_captions = dataset_filtered.map(add_caption, num_proc=num_proc)

    # Shuffle
    dataset_with_captions = dataset_with_captions.shuffle(seed=42)
    print(f"Final dataset: {len(dataset_with_captions)} samples")

    if len(dataset_with_captions) > 0:
        sample = dataset_with_captions[0]
        print(f"\nSample: image_id={sample['image_id']}, synthetic-caption={sample['synthetic-caption'][:100]}...")

    if push_to_hub:
        repo_id = f"{hub_username}/COCOQA-restval-ICQA"
        print(f"\nPushing to Hub: {repo_id} (overwrite)")
        token = os.getenv("HF_TOKEN")
        if token:
            dataset_with_captions.push_to_hub(repo_id, token=token)
        else:
            dataset_with_captions.push_to_hub(repo_id)
        print("✅ Pushed successfully!")

    return dataset_with_captions


def patch_vqav2_icq_synthetic_dataset(captions_file: str, push_to_hub: bool, hub_username: str, num_proc: int = 8):
    """Patch VQAv2-COCO-restval-ICQ with synthetic captions.
    
    Captions JSON keys match the dataset's image_id directly (both are COCO filenames).
    Adds a 'synthetic-caption' column. Pushes to the same VQAv2-COCO-restval-ICQ dataset.
    """
    print("\n" + "="*60)
    print("PATCHING VQAv2-COCO-restval-ICQ WITH SYNTHETIC CAPTIONS")
    print("="*60)

    # Load captions
    print(f"Loading captions from {captions_file}...")
    with open(captions_file, 'r') as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")

    # Load dataset
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        cache_dir = os.path.expanduser(os.path.expandvars(cache_dir))

    print("Loading VQAv2-COCO-restval-ICQ dataset...")
    dataset = load_dataset(f"{hub_username}/VQAv2-COCO-restval-ICQ", split="train", cache_dir=cache_dir)
    original_len = len(dataset)
    print(f"Loaded {original_len} samples")

    # Filter
    def filter_fn(example):
        cap = captions.get(example['image_id'], "FAILED")
        return cap != "FAILED" and cap is not None

    print("Filtering dataset (removing missing/failed captions)...")
    dataset_filtered = dataset.filter(filter_fn, num_proc=num_proc)
    print(f"Kept {len(dataset_filtered)}/{original_len} samples")

    # Add synthetic caption
    def add_caption(example):
        example['synthetic-caption'] = captions[example['image_id']]
        return example

    print("Adding synthetic-caption column...")
    dataset_with_captions = dataset_filtered.map(add_caption, num_proc=num_proc)

    # Shuffle
    dataset_with_captions = dataset_with_captions.shuffle(seed=42)
    print(f"Final dataset: {len(dataset_with_captions)} samples")

    if len(dataset_with_captions) > 0:
        sample = dataset_with_captions[0]
        print(f"\nSample: image_id={sample['image_id']}, synthetic-caption={sample['synthetic-caption'][:100]}...")

    if push_to_hub:
        repo_id = f"{hub_username}/VQAv2-COCO-restval-ICQ"
        print(f"\nPushing to Hub: {repo_id} (overwrite)")
        token = os.getenv("HF_TOKEN")
        if token:
            dataset_with_captions.push_to_hub(repo_id, token=token)
        else:
            dataset_with_captions.push_to_hub(repo_id)
        print("✅ Pushed successfully!")

    return dataset_with_captions

def patch_mmbench_dataset(captions_file: str, split: str, push_to_hub: bool, hub_username: str):
    print("\n" + "="*60)
    print(f"PATCHING MMBench_EN DATASET ({split} split) (OPTIMIZED)")
    print("="*60)

    # Load captions
    print(f"Loading captions from {captions_file}...")
    with open(captions_file, 'r') as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")

    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        cache_dir = os.path.expanduser(os.path.expandvars(cache_dir))

    print(f"Loading MMBench_EN dataset ({split} split)...")
    dataset = load_dataset("lmms-lab/MMBench_EN", split=split, cache_dir=cache_dir)
    original_len = len(dataset)
    print(f"Loaded {original_len} samples")

    # Filter
    def filter_fn(example):
        idx = str(example['index'])
        cap = captions.get(idx, "FAILED")
        return cap != "FAILED" and cap is not None

    print("Filtering dataset...")
    dataset_filtered = dataset.filter(filter_fn, num_proc=get_num_proc())
    print(f"Kept {len(dataset_filtered)}/{original_len} samples")

    # Add captions
    def add_caption(example):
        idx = str(example['index'])
        example['caption'] = captions[idx]
        return example

    print("Adding captions column...")
    dataset_with_captions = dataset_filtered.map(add_caption, num_proc=get_num_proc())

    # Shuffle
    dataset_with_captions = dataset_with_captions.shuffle(seed=42)
    print(f"Final dataset: {len(dataset_with_captions)} samples")

    if len(dataset_with_captions) > 0:
        sample = dataset_with_captions[0]
        print(f"Sample: index={sample['index']}, caption={sample['caption'][:100]}...")

    if push_to_hub:
        repo_id = f"{hub_username}/MMBench_EN-{split}-captioned"
        print(f"\nPushing to Hub: {repo_id}")
        token = os.getenv("HF_TOKEN")
        if token:
            dataset_with_captions.push_to_hub(repo_id, token=token)
        else:
            dataset_with_captions.push_to_hub(repo_id)
        print("✅ Pushed successfully!")

    return dataset_with_captions

def main():
    parser = argparse.ArgumentParser(description="Patch datasets with generated captions")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mme", "mmbench", "vqav2-synthetic", "cocoqa-synthetic", "vqav2-icq-synthetic"],
                        help="Dataset to patch")
    parser.add_argument("--captions-file", type=str, required=True,
                        help="Path to captions JSON file")
    parser.add_argument("--split", type=str, default="dev",
                        help="Split for MMBench (dev or test)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to HuggingFace Hub")
    parser.add_argument("--hub-username", type=str, default="l11p",
                        help="HuggingFace username")
    parser.add_argument("--num-proc", type=int, default = 12,
                        help="Number of parallel processes for filter/map (default: 12)")
    args = parser.parse_args()

    if args.dataset == "mme":
        patch_mme_dataset(args.captions_file, args.push_to_hub, args.hub_username)
    elif args.dataset == "mmbench":
        patch_mmbench_dataset(args.captions_file, args.split, args.push_to_hub, args.hub_username)
    elif args.dataset == "vqav2-synthetic":
        patch_vqav2_synthetic_dataset(args.captions_file, args.push_to_hub, args.hub_username, args.num_proc)
    elif args.dataset == "cocoqa-synthetic":
        patch_cocoqa_synthetic_dataset(args.captions_file, args.push_to_hub, args.hub_username, args.num_proc)
    elif args.dataset == "vqav2-icq-synthetic":
        patch_vqav2_icq_synthetic_dataset(args.captions_file, args.push_to_hub, args.hub_username, args.num_proc)

    print("\n✅ MISSION ACCOMPLISHED!")

if __name__ == "__main__":
    main()