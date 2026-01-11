"""
Create a unified dataset from COCO-QA Vietnamese test split and COCO-Karpathy restval split.

The resulting dataset combines images with both captions and question-answer pairs.

The dataset is saved to disk in the following format:
{
    "image_id": image_id,
    "questions_and_answers": [
        {"question": question1, "answer": answer1},
        {"question": question2, "answer": answer2},
        ...
    ],
    "captions": [caption1, caption2, ...],
    "image": image,  # first image occurrence
}
"""

import os
from collections import defaultdict
from datasets import Dataset
from utils.dataset import get_hf_dataset_split

# Setup
HF_CACHE_DIR = os.path.expanduser("~/scratch/huggingface/datasets")

# Load datasets
coco_karpathy_dataset = get_hf_dataset_split(
    dataset_path_or_name="yerevann/coco-karpathy",
    cache_dir=HF_CACHE_DIR,
    split="restval",
)
# Dataset with image and captions
# Structure: (img, [caption1, caption2, ...])
# Full column names:
# (filepath, sentids, filename, imgid, split, sentences, cocoid, url)

cocoqavi_dataset = get_hf_dataset_split(
    dataset_path_or_name="ThucPD/coco-qa-vi",
    cache_dir=HF_CACHE_DIR,
    split="test",
)
# COCO-QA Vietnamese: using only the English part of the dataset
# Dataset with image and one question-answer pair per image
# Structure: (img, [question, answer])
# Full column names:
# (image, question, answer, translated_question, translated_answer, image_path, image_id, question_id, type, split)

# Note: filename(coco_karpathy) == image_id(coco_qa_vi)

# ----------------------------------------------------------------------------

print(f"\nCOCO Karpathy dataset size: {len(coco_karpathy_dataset)}")
print(f"COCO QA VI dataset size: {len(cocoqavi_dataset)}")

# Extract shared image IDs
coco_karpathy_ids = set(coco_karpathy_dataset["filename"])
cocoqa_ids = set(cocoqavi_dataset["image_id"])
intersection_image_ids = coco_karpathy_ids.intersection(cocoqa_ids)
print(f"\nUnique filenames in COCO Karpathy dataset: {len(coco_karpathy_ids)}")
print(f"Unique image IDs in COCO QA VI dataset: {len(cocoqa_ids)}")
print(f"\nIntersection image IDs: {len(intersection_image_ids)}")

print("\nFiltering datasets with shared image IDs...")
# Filter Karpathy dataset for shared IDs and non-empty captions
coco_karpathy_filtered = coco_karpathy_dataset.filter(
    lambda x: x["filename"] in intersection_image_ids and len(x["sentences"]) > 0,
    num_proc=8,
)
print(f"\nCOCO Karpathy filtered dataset size: {len(coco_karpathy_filtered)}")

# Filter COCOQA dataset for shared IDs
cocoqavi_filtered = cocoqavi_dataset.filter(
    lambda x: x["image_id"] in intersection_image_ids,
    num_proc=8,
)
print(f"\nCOCO QA VI filtered dataset size: {len(cocoqavi_filtered)}")

# ----------------------------------------------------------------------------

# Build image_id → captions mapping from Karpathy dataset
imageid_to_captions = {
    item["filename"]: item["sentences"] for item in coco_karpathy_filtered
}

# Group question-answer pairs by image_id from COCOQA dataset
qa_grouped = defaultdict(lambda: {"questions_and_answers": [], "image": None})

# For each imageid group the question-answer pairs with just one image
# {
#     "image_id": image_id,
#     "questions_and_answers": [
#         {"question": question1, "answer": answer1},
#         {"question": question2, "answer": answer2},
#         ...
#     ],
#     "image": image, # first image occurrence
# }

for item in cocoqavi_filtered:
    img_id = item["image_id"]
    qa_grouped[img_id]["questions_and_answers"].append(
        {"question": item["question"], "answer": item["answer"]}
    )
    # Store first image occurrence for each image_id
    if qa_grouped[img_id]["image"] is None:
        qa_grouped[img_id]["image"] = item["image"]

# ----------------------------------------------------------------------------

print("Building combined dataset...")
# Merge into final dataset
# {
#     "image_id": image_id,
#     "questions_and_answers": [
#         {"question": question1, "answer": answer1},
#         {"question": question2, "answer": answer2},
#         ...
#     ],
#     "captions": [caption1, caption2, ...],
#     "image": image,  # first image occurrence
# }
# !Note: multiple images can have the shared (question, answer) pairs

# Merge captions and question-answer pairs into final dataset
final_data = []
for image_id, qa_data in qa_grouped.items():
    captions = imageid_to_captions.get(image_id)
    if captions and qa_data["questions_and_answers"]:
        final_data.append(
            {
                "image_id": image_id,
                "questions_and_answers": qa_data["questions_and_answers"],
                "captions": captions,
                "image": qa_data["image"],
            }
        )

print(f"\nFinal combined dataset size: {len(final_data)}")

# Validate that we have both captions and Q&A pairs for all entries
if final_data:
    print(
        f"Sample entry has {len(final_data[0]['captions'])} captions and {len(final_data[0]['questions_and_answers'])} Q&A pairs"
    )

# Convert to HuggingFace Dataset and shuffle
final_dataset = Dataset.from_list(final_data).shuffle(seed=42)

# Save to disk
output_dir = os.path.expanduser("~/scratch/datasets/cocoqa_unified")
final_dataset.save_to_disk(output_dir)
