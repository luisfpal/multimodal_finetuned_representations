"""
Create a unified dataset from COCO-QA Vietnamese test split and COCO-Karpathy restval split.
The final dataset has the captions and question-answer pairs in English.
The COCO-QA Vietnamese dataset is used for convenience and easy access on huggingface.

The resulting dataset combines images with both captions and question-answer pairs.
"""

import os
import argparse
from collections import defaultdict
from datasets import load_dataset, Features, Value, Sequence, Image, Dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Create COCOQA-restval-ICQA dataset")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset to Hugging Face Hub")
    parser.add_argument("--hub-username", type=str, default="l11p", help="Hugging Face username")
    args = parser.parse_args()

    # Configuration
    CACHE_DIR = os.getenv("HF_DATASETS_CACHE")
    # Fix for unexpanded variables like ${HOME}
    if CACHE_DIR:
        CACHE_DIR = os.path.expanduser(os.path.expandvars(CACHE_DIR))

    COCOQA_DATASET_ID = "ThucPD/coco-qa-vi"
    COCO_KARPATHY_ID = "yerevann/coco-karpathy"
    NEW_DATASET_NAME = "COCOQA-restval-ICQA"
    
    print(f"Using cache directory: {CACHE_DIR}")

    # 1. Load Datasets
    print(f"Loading {COCOQA_DATASET_ID} (test)...")
    cocoqa_dataset = load_dataset(COCOQA_DATASET_ID, cache_dir=CACHE_DIR, split='test')
    
    print(f"Loading {COCO_KARPATHY_ID} (restval)...")
    karpathy_dataset = load_dataset(COCO_KARPATHY_ID, cache_dir=CACHE_DIR, split='restval')

    # 2. Identify Common Images
    # Note from original script: filename(coco_karpathy) == image_id(coco_qa_vi)
    print("Identifying common image IDs...")
    cocoqa_unique_ids = set(cocoqa_dataset['image_id'])
    karpathy_unique_ids = set(karpathy_dataset['filename'])
    common_ids = cocoqa_unique_ids.intersection(karpathy_unique_ids)
    
    print(f"COCO-QA unique images: {len(cocoqa_unique_ids)}")
    print(f"COCO Karpathy unique images: {len(karpathy_unique_ids)}")
    print(f"Common images: {len(common_ids)}")

    if len(common_ids) == 0:
        print("Error: No common images found!")
        return

    # 3. Process Data
    # Group Q&A pairs by image_id (which corresponds to filename in Karpathy)
    
    print("Grouping COCO-QA questions and answers...")
    
    # Optimization 1: Process COCO-QA metadata by loading columns into memory
    qa_image_ids = cocoqa_dataset['image_id']
    qa_questions = cocoqa_dataset['question']
    qa_answers = cocoqa_dataset['answer']
    # If the dataset has 'image' column, we might need it if we were using it as source,
    # but here we use Karpathy as source for images/captions and just attach Q&A.
    # The original script prioritized Karpathy images.

    # Map image_id -> list of {question, answer}
    qa_map = defaultdict(list)
    # Map image_id -> index in cocoqa_dataset (to retrieve image later)
    image_id_to_cocoqa_idx = {}

    # Iterate with progress bar
    for idx, (img_id, q, a) in tqdm(enumerate(zip(qa_image_ids, qa_questions, qa_answers)), total=len(qa_image_ids), desc="Indexing Q&A"):
        if img_id in common_ids:
            qa_map[img_id].append({"question": q, "answer": a})
            if img_id not in image_id_to_cocoqa_idx:
                image_id_to_cocoqa_idx[img_id] = idx
    
    print(f"Processed COCO-QA. Found Q&A for {len(qa_map)} intersection images.")

    # 4. Create dataset using map (Optimized for Multi-CPU)
    print("Building new dataset with multiprocessing...")
    
    # We operate on the Karpathy dataset as the base since it has the images and captions
    
    # Optimization 2: Filter Karpathy first using multiprocessing
    # Filter for shared IDs and non-empty captions (as per original script)
    karpathy_filtered = karpathy_dataset.filter(
        lambda x: x['filename'] in common_ids and len(x['sentences']) > 0,
        num_proc=max(1, os.cpu_count() // 4)
    )
    
    print(f"Karpathy filtered dataset size: {len(karpathy_filtered)}")

    # Define features for the new dataset
    # matching the requested format:
    # {
    #     "image_id": image_id,
    #     "questions_and_answers": [
    #         {"question": question1, "answer": answer1},
    #         ...
    #     ],
    #     "captions": [caption1, caption2, ...],
    #     "image": image,
    # }
    
    features = Features({
        "image_id": Value("string"), # filename is string in karpathy
        "questions_and_answers": [{
            "question": Value("string"),
            "answer": Value("string")
        }],
        "captions": Sequence(Value("string")),
        "image": Image()
    })

    # function to merge data
    # qa_map will be pickled and sent to workers
    def merge_data(example):
        img_id = example['filename']
        
        # Get Q&A from pre-computed map
        qa_pairs = qa_map.get(img_id, [])
        
        # Get image from COCO-QA dataset
        cocoqa_idx = image_id_to_cocoqa_idx[img_id]
        image = cocoqa_dataset[cocoqa_idx]['image']

        return {
            "image_id": img_id,
            "questions_and_answers": qa_pairs,
            "captions": example['sentences'],
            "image": image 
        }

    # Optimization 3: Use map with multiple processes to construct the dataset
    new_dataset = karpathy_filtered.map(
        merge_data,
        features=features,
        remove_columns=karpathy_filtered.column_names, # Remove original columns
        num_proc=max(1, os.cpu_count() // 4)
    )
    
    # Filter out any entries that might have ended up with empty Q&A (though intersection check should prevent this)
    new_dataset = new_dataset.filter(
        lambda x: len(x['questions_and_answers']) > 0,
        num_proc=max(1, os.cpu_count() // 4)
    )
    
    print(f"New dataset created with {len(new_dataset)} samples.")
    
    # Verify sample
    if len(new_dataset) > 0:
        sample = new_dataset[0]
        print(f"Sample entry has {len(sample['captions'])} captions and {len(sample['questions_and_answers'])} Q&A pairs")

    # Shuffle
    new_dataset = new_dataset.shuffle(seed=42)

    # 6. Push to Hub
    if args.push_to_hub:
        # Check if user is logged in
        
        token = os.getenv("HF_TOKEN")
        if not token:
             try:
                 with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                     token = f.read().strip()
             except Exception:
                 pass
        
        repo_id = f"{args.hub_username}/{NEW_DATASET_NAME}"
        print(f"Pushing to Hub: {repo_id}")
        
        if token:
             print("Using found token for authentication.")
             new_dataset.push_to_hub(repo_id, token=token)
        else:
             print("No explicit token found. Relying on huggingface-cli login state...")
             new_dataset.push_to_hub(repo_id)

        print("Done!")
    else:
        print("Skipping push to hub (use --push-to-hub to push)")

if __name__ == "__main__":
    main()
