import os
import argparse
from collections import defaultdict
from datasets import load_dataset, Features, Value, Sequence, Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Create VQAv2-COCO-restval-ICQ dataset")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset to Hugging Face Hub")
    parser.add_argument("--hub-username", type=str, default="l11p", help="Hugging Face username")
    args = parser.parse_args()

    # Configuration
    CACHE_DIR = os.getenv("HF_DATASETS_CACHE")
    VQA_DATASET_ID = "lmms-lab/VQAv2"
    COCO_DATASET_ID = "yerevann/coco-karpathy"
    NEW_DATASET_NAME = "VQAv2-COCO-restval-ICQ"
    
    print(f"Using cache directory: {CACHE_DIR}")

    # 1. Load Datasets
    print(f"Loading {VQA_DATASET_ID} (validation)...")
    vqa_dataset = load_dataset(VQA_DATASET_ID, cache_dir=CACHE_DIR, split='validation')
    
    print(f"Loading {COCO_DATASET_ID} (restval)...")
    coco_dataset = load_dataset(COCO_DATASET_ID, cache_dir=CACHE_DIR, split='restval')

    # 2. Identify Common Images
    print("Identifying common image IDs...")
    vqa_unique_image_ids = set(vqa_dataset['image_id'])
    coco_unique_image_ids = set(coco_dataset['cocoid'])
    coco_vqa_common_ids = vqa_unique_image_ids.intersection(coco_unique_image_ids)
    
    print(f"VQA unique images: {len(vqa_unique_image_ids)}")
    print(f"COCO Karpathy unique images: {len(coco_unique_image_ids)}")
    print(f"Common images: {len(coco_vqa_common_ids)}")

    if len(coco_vqa_common_ids) == 0:
        print("Error: No common images found!")
        return

    # 3. Process Data
    # We want to group questions by image_id for VQA
    # And we need one image reference per image_id from VQA
    
    print("Grouping VQA questions and indexing images (Metadata only)...")
    from tqdm import tqdm
    
    # Optimization 1: Process VQA metadata by loading columns into memory
    # Accessing columns directly as lists is much faster than row-by-row iteration on the Dataset object
    all_image_ids = vqa_dataset['image_id']
    all_questions = vqa_dataset['question']
    all_answers = vqa_dataset['multiple_choice_answer']
    
    # Map image_id -> list of {question, answer}
    vqa_qa_map = defaultdict(list)
    # Map image_id -> index in vqa_dataset (to retrieve image later)
    # We only need one index per image_id.
    vqa_image_index_map = {}

    # Iterate with progress bar
    # Filtering on the fly to only keep those in intersection
    for idx, (img_id, question, answer) in tqdm(enumerate(zip(all_image_ids, all_questions, all_answers)), total=len(all_image_ids), desc="Indexing VQA"):
        if img_id in coco_vqa_common_ids:
            vqa_qa_map[img_id].append({"question": question, "answer": answer})
            if img_id not in vqa_image_index_map:
                vqa_image_index_map[img_id] = idx
    
    print(f"Processed VQA. Found Q&A pairs for {len(vqa_qa_map)} intersection images.")

    # 4. create dataset using map (Optimized for Multi-CPU)
    print("Building new dataset with multiprocessing...")
    
    # We operate on the COCO dataset as the base since it has unique images (mostly)
    # matching our target structure (one row per image).
    
    # Optimization 2: Filter COCO first using multiprocessing
    coco_filtered = coco_dataset.filter(
        lambda x: x['cocoid'] in coco_vqa_common_ids,
        num_proc=os.cpu_count()
    )
    
    # Define features for the new dataset
    features = Features({
        "image_id": Value("string"),
        "image": Image(),
        "captions": Sequence(Value("string")),
        "questions_and_answers": [{
            "question": Value("string"),
            "answer": Value("string")
        }]
    })

    # function to merge data
    # This will run in parallel workers. 
    # vqa_dataset, vqa_qa_map, vqa_image_index_map will be pickled and sent to workers.
    def merge_data(example):
        img_id = example['cocoid']
        filename = example['filename']
        
        # Get Q&A from pre-computed map
        qa_pairs = vqa_qa_map.get(img_id, [])
        
        # Get image from VQA dataset efficiently
        # We use the pre-computed index to access the specific row in VQA
        # Accessing ['image'] here triggers the decoding, distributed across workers
        vqa_idx = vqa_image_index_map[img_id]
        image = vqa_dataset[vqa_idx]['image']
        
        return {
            "image_id": filename,
            "image": image,
            "captions": example['sentences'],
            "questions_and_answers": qa_pairs
        }

    # Optimization 3: Use map with multiple processes to construct the dataset
    new_dataset = coco_filtered.map(
        merge_data,
        features=features,
        remove_columns=coco_filtered.column_names, # Remove original COCO columns
        num_proc=os.cpu_count()
    )
    
    print(f"New dataset created with {len(new_dataset)} samples.")
    
    # 6. Push to Hub
    if args.push_to_hub:
        # Check if user is logged in
        
        # Try to get token from many places to be robust
        token = os.getenv("HF_TOKEN")
        if not token:
             # Try reading from standard cache location if env var not set
             try:
                 with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                     token = f.read().strip()
             except Exception:
                 pass
        
        repo_id = f"{args.hub_username}/{NEW_DATASET_NAME}"
        print(f"Pushing to Hub: {repo_id}")
        
        if token:
             print("Using found token for authentication.")
             # Explicitly pass token if we found it
             new_dataset.push_to_hub(repo_id, token=token)
        else:
             print("No explicit token found. Relying on huggingface-cli login state...")
             new_dataset.push_to_hub(repo_id)

        print("Done!")
    else:
        print("Skipping push to hub (use --push-to-hub to push)")

if __name__ == "__main__":
    main()
