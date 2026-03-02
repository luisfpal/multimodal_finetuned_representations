"""
Patch GQA dataset with generated captions.

Complex pipeline (Optimized):
1. Load images (val_balanced_images) - Lazy Load
2. Load instructions (val_balanced_instructions)
3. Filter out yes/no answers (Metadata only)
4. Select longest question per unique image ID (Metadata only)
5. Merge images with instructions (Optimized: Using Index Mapping + Map)
6. Add captions
7. Push to hub
"""

import os
import json
import argparse
from collections import defaultdict
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def get_num_proc():
    # Use 1/4 of available CPUs
    try:
        count = os.cpu_count()
        if count:
            return max(1, count // 4)
    except:
        pass
    return 1

def main():
    parser = argparse.ArgumentParser(description="Patch GQA dataset with captions")
    parser.add_argument("--captions-file", type=str, required=True,
                        help="Path to captions JSON file")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to HuggingFace Hub")
    parser.add_argument("--hub-username", type=str, default="l11p",
                        help="HuggingFace username")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PATCHING GQA DATASET - OPTIMIZED PIPELINE")
    print("="*60)

    # Configuration
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        cache_dir = os.path.expanduser(os.path.expandvars(cache_dir))

    # Step 1: Load datasets
    print("\n[1/6] Loading GQA datasets...")
    print("  Loading val_balanced_images (Lazy)...")
    images_dataset = load_dataset(
        "lmms-lab/GQA",
        "val_balanced_images",
        split="val",
        cache_dir=cache_dir
    )
    print(f"  Loaded {len(images_dataset)} images (lazy access)")

    print("  Loading val_balanced_instructions...")
    instructions_dataset = load_dataset(
        "lmms-lab/GQA",
        "val_balanced_instructions",
        split="val",
        cache_dir=cache_dir
    )
    print(f"  Loaded {len(instructions_dataset)} instruction samples")

    # Step 2: Filter yes/no answers
    print("\n[2/6] Filtering out yes/no answers...")
    # We perform this in memory as instructions are small text data
    
    image_to_questions = defaultdict(list)
    original_count = 0
    filtered_count = 0
    
    for sample in tqdm(instructions_dataset, desc="  Filtering & Grouping"):
        original_count += 1
        if sample['answer'] not in ['yes', 'no']:
            image_to_questions[sample['imageId']].append(sample)
            filtered_count += 1

    print(f"  Kept {filtered_count}/{original_count} samples "
          f"(removed {original_count - filtered_count} yes/no)")
    print(f"  Found {len(image_to_questions)} unique images after filtering")

    # Step 3: Select longest question per image
    print("\n[3/6] Selecting longest question per unique imageId...")
    selected_samples = []
    
    for image_id, samples in tqdm(image_to_questions.items(), desc="  Selecting longest"):
        longest = max(samples, key=lambda x: len(x['question']))
        # Remove 'id' from instructions as per requirement
        if 'id' in longest:
            del longest['id']
        selected_samples.append(longest)

    print(f"  Selected {len(selected_samples)} samples (one per image)")

    # Step 4: Prepare Index for Efficient Merge
    print("\n[4/6] Indexing images and captions...")
    
    # Load captions
    print(f"  Loading captions from {args.captions_file}...")
    with open(args.captions_file, 'r') as f:
        captions = json.load(f)
    print(f"  Loaded {len(captions)} captions")
    
    # Build Image Index (id -> dataset index)
    # This avoids loading images into RAM. We access the 'id' column directly.
    print("  Building image index (scanning IDs)...")
    image_ids = images_dataset['id']
    image_id_to_index = {iid: idx for idx, iid in enumerate(image_ids)}
    
    # Step 5: Filter and Prepare Metadata
    print("\n[5/6] Filtering missing images/captions...")
    
    final_metadata = []
    missing_images = 0
    missing_captions = 0
    failed_captions = 0
    
    for sample in tqdm(selected_samples, desc="  Checking"):
        img_id = sample['imageId']
        
        # Check image exists in dataset
        if img_id not in image_id_to_index:
            missing_images += 1
            continue
            
        # Check caption exists and is valid
        cap = captions.get(img_id)
        if cap is None:
            missing_captions += 1
            continue
        if cap == "FAILED":
            failed_captions += 1
            continue
            
        sample['caption'] = cap
        # Store index for O(1) retrieval in map
        sample['__image_index__'] = image_id_to_index[img_id]
        final_metadata.append(sample)

    print(f"  Ready to create dataset with {len(final_metadata)} samples")
    print(f"  Dropped: {missing_images} missing images, {missing_captions} missing caps, {failed_captions} failed caps")

    # Step 6: Create Dataset and Hydrate Images
    print("\n[6/6] Creating final dataset and hydrating images...")
    
    # Create intermediate dataset (metadata only) - Fast
    meta_dataset = Dataset.from_list(final_metadata)
    
    # Map to add images using parallel processing
    def add_image(example):
        idx = example['__image_index__']
        # Fetch image from the memory-mapped dataset
        example['image'] = images_dataset[idx]['image']
        return example

    print(f"  Hydrating images using {get_num_proc()} processes...")
    final_dataset = meta_dataset.map(
        add_image, 
        num_proc=get_num_proc(),
        remove_columns=['__image_index__'], # Clean up helper column
        desc="  Hydrating images"
    )

    # Shuffle
    final_dataset = final_dataset.shuffle(seed=42)

    print(f"\n✅ Final dataset ready: {len(final_dataset)} samples")

    # Show sample
    if len(final_dataset) > 0:
        sample = final_dataset[0]
        print(f"\nSample:")
        print(f"  imageId: {sample['imageId']}")
        print(f"  question: {sample['question']}")
        print(f"  answer: {sample['answer']}")
        print(f"  caption: {sample['caption'][:100]}...")
        # print(f"  image: {type(sample['image'])}")

    # Push to hub
    if args.push_to_hub:
        repo_id = f"{args.hub_username}/GQA-val-balanced-noyesno-captioned"
        print(f"\n🚀 Pushing to Hub: {repo_id}")
        token = os.getenv("HF_TOKEN")
        if token:
            final_dataset.push_to_hub(repo_id, token=token)
        else:
            final_dataset.push_to_hub(repo_id)
        print("✅ Pushed successfully!")
    else:
        print("\nSkipping push to hub (use --push-to-hub to push)")

    print("\n" + "="*60)
    print("✅ MISSION ACCOMPLISHED!")
    print("="*60)


if __name__ == "__main__":
    main()