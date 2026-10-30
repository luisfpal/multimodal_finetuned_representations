"""Dataset and dataloader utilities for VQA, captioning, multiple-choice, and classification benchmarks."""

import ast
import base64
import csv
import hashlib
import json
import os
import re
from collections import Counter
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

import datasets
import numpy as np
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel

from .cambrian_utils import (
    is_cambrian_model,
    prepare_cambrian_batch_inputs,
)
from .constants import (
    MMBENCH_ANSWER_CHOICES,
    MME_ANSWER_CHOICES,
    MMLU_ANSWER_CHOICES,
    MMSTAR_ANSWER_CHOICES,
    SHORT_CAPTION_PROMPT,
    SQA_ANSWER_CHOICES,
    SYSTEM_ROLE,
    POPE_ANSWER_CHOICES,
    AI2D_ANSWER_CHOICES,
)
from .internvl_utils import apply_chat_template as apply_internvl_chat_template
from .internvl_utils import get_pixel_values
from .model_utils import ProcessorType
from .prompt_formatting import format_user_turn_text

SEED: int = 42

MULTIPLE_CHOICE_BENCHMARKS = [
    "mmlu",
    "scienceqa",
    "mmbench",
    "mme",
    "ai2d",
    "pope",
    "seed",
    "mmstar",
]

LLAVA_INSTRUCT_150K_NAME = "liuhaotian/LLaVA-Instruct-150K"
LLAVA_INSTRUCT_DETAIL_FILE = "detail_23k.json"
DEFAULT_COCO_TRAIN2014_DIR = Path(
    os.getenv(
        "COCO_TRAIN2014_DIR",
        "/orfeo/cephfs/scratch/area/ddoimo/data/mscoco/train2014",
    )
)
DEFAULT_HF_HUB_CACHE = Path(
    os.getenv("HF_HUB_CACHE", str(Path.home() / "scratch" / "huggingface" / "hub"))
)
CAPTIONING_PROMPT = "Provide a one-sentence caption about the image."
TEXTUAL_CAPTION_PREAMBLE = "This is a detailed description of an image:"


def is_llava_instruct_150k(dataset_path_or_name: str) -> bool:
    normalized = dataset_path_or_name.lower()
    return (
        normalized == LLAVA_INSTRUCT_150K_NAME.lower()
        or "llava-instruct-150k" in normalized
    )


def _resolve_llava_instruct_detail_path() -> Path:
    snapshots_dir = (
        DEFAULT_HF_HUB_CACHE
        / "datasets--liuhaotian--LLaVA-Instruct-150K"
        / "snapshots"
    )
    matches = sorted(snapshots_dir.glob(f"*/{LLAVA_INSTRUCT_DETAIL_FILE}"))
    if not matches:
        raise FileNotFoundError(
            f"Cached {LLAVA_INSTRUCT_DETAIL_FILE} not found under {snapshots_dir}. "
            f"Download {LLAVA_INSTRUCT_150K_NAME} first or set HF_HUB_CACHE."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_llava_instruct_detail_records() -> list[dict[str, Any]]:
    with _resolve_llava_instruct_detail_path().open(encoding="utf-8") as handle:
        return json.load(handle)


def _first_gpt_response(example: dict[str, Any]) -> Optional[str]:
    conversations = example.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if not isinstance(turn, dict) or turn.get("from") != "gpt":
            continue
        value = turn.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_coco_train2014_image_path(image_name: str, image_dir: Path) -> Path:
    if image_name.startswith("COCO_train2014_"):
        return image_dir / image_name
    image_id = Path(image_name).stem
    return image_dir / f"COCO_train2014_{image_id}.jpg"


def _select_llava_caption_records(
    dataset: list[dict[str, Any]],
    image_dir: Path,
    downsample_size: Optional[int],
) -> list[dict[str, Any]]:
    if downsample_size is not None and downsample_size <= 0:
        raise ValueError(f"downsample_size must be positive, got {downsample_size}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records: list[dict[str, Any]] = []

    for idx, example in enumerate(dataset):
        caption = _first_gpt_response(example)
        image_name = example.get("image")
        if not caption or not isinstance(image_name, str):
            continue
        image_path = _resolve_coco_train2014_image_path(image_name, image_dir)
        if not image_path.exists():
            continue

        record = {
            "idx": idx,
            "id": example.get("id"),
            "image_name": image_name,
            "image_path": str(image_path),
            "caption": caption,
            "gpt_response_length": len(caption),
        }

        records.append(record)

    records.sort(key=lambda record: record["gpt_response_length"], reverse=True)
    if downsample_size is not None:
        records = records[:downsample_size]

    if not records:
        raise ValueError(
            f"No usable {LLAVA_INSTRUCT_150K_NAME} caption records found. "
            "Check the split, GPT responses, and image directory."
        )
    if downsample_size is not None and len(records) < downsample_size:
        print(
            f"Requested top {downsample_size} GPT responses but only found "
            f"{len(records)} usable image-caption records."
        )
    return records


def get_hf_dataset_split(
    dataset_path_or_name: str,
    cache_dir: str,
    split: str = "test",
) -> datasets.Dataset:
    """
    Loads a specific split from a dataset, potentially from a local path or Hugging Face Hub.

    Args:
        dataset_path: Path to the dataset directory or Hugging Face dataset identifier.
        split: The name of the dataset split to load (e.g., 'train', 'validation', 'test').

    Returns:
        The loaded dataset split.

    Raises:
        Exception: If the dataset fails to load (e.g., path not found, invalid identifier).
    """
    try:
        dataset_kwargs = {
            "path": dataset_path_or_name,
            "split": split,
            "cache_dir": cache_dir,
        }
        if "mmlu" in dataset_path_or_name.lower():
            # MMLU dataset requires a specific name argument
            dataset_kwargs["name"] = "all"
        elif "scienceqa" in dataset_path_or_name.lower():
            # ScienceQA has FULL and IMG configs; we use the image-only subset
            dataset_kwargs["name"] = "ScienceQA-IMG"

        dataset = load_dataset(**dataset_kwargs)

        if isinstance(dataset, datasets.DatasetDict):
            # If split is explicitly None, or provided but not found
            if split and split in dataset:
                return dataset[split]

            # Fallback logic
            available_splits = list(dataset.keys())
            if len(available_splits) == 1:
                print(
                    f"  ⚠️ Requested split '{split}' not found. Using the only available split: '{available_splits[0]}'"
                )
                return dataset[available_splits[0]]

            if "train" in dataset:
                print(f"  ⚠️ Requested split '{split}' not found. Fallback to 'train'.")
                return dataset["train"]

            raise ValueError(
                f"Split '{split}' not found in dataset. Available splits: {available_splits}"
            )

        return dataset
    except Exception as e:
        # Catching a broad exception, but could be more specific depending on expected errors
        # from load_dataset (e.g., FileNotFoundError, datasets.exceptions.DatasetNotFoundError)
        raise RuntimeError(
            f"Failed to load dataset split '{split}' from '{dataset_path_or_name}'. Error: {e}"
        )


class MultipleChoiceDatasetBenchmark(Dataset):
    def __init__(
        self,
        dataset_path_or_name: str,
        cache_dir: str,
        split: str = "test",
        question_instruction_type: Optional[str] = "test",
        texts_qa: bool = False,
        images_qa: bool = False,
        downsample_size: Optional[int] = None,
        seed: Optional[int] = None,
        remove_yes_no: bool = False,
        filter_mmbench_4_choices: bool = False,
        allowed_indices: Optional[set[int]] = None,
        unique_images: bool = False,
        open_ended: bool = False,
    ):
        supported_datasets = [
            "scienceqa",
            "mmlu",
            "mmbench",
            "mme",
            "ai2d",
            "pope",
            "seed",
            "mmstar",
        ]

        self.dataset_path_or_name = dataset_path_or_name.lower()

        if not any(ds in self.dataset_path_or_name for ds in supported_datasets):
            raise ValueError(
                f"Dataset {dataset_path_or_name} is not supported. "
                f"Only {', '.join(supported_datasets)} are supported."
            )

        if texts_qa and images_qa:
            raise ValueError("texts_qa and images_qa cannot be both True.")

        self.question_instruction_type = question_instruction_type
        self.texts_qa = texts_qa
        self.images_qa = images_qa
        self.open_ended = open_ended

        # Load the initial dataset
        dataset = get_hf_dataset_split(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split=split,
        )

        # Optional semantic filtering by row index — applied first so downstream
        # filters and downsampling operate on the already-filtered set.
        if allowed_indices is not None:
            valid = sorted(allowed_indices & set(range(len(dataset))))
            before = len(dataset)
            dataset = dataset.select(valid)
            print(
                f"Applied allowed_indices filter "
                f"(from {before} to {len(dataset)} samples)"
            )

        # Deduplicate by imageHash
        if unique_images and "imageHash" in dataset.features:
            before = len(dataset)

            # groups[h] = all original indices that share this hash (None skipped)
            groups: dict[str, list[int]] = {}
            for idx, h in enumerate(dataset["imageHash"]):
                if h is not None:
                    groups.setdefault(h, []).append(idx)

            # seeded random choice per group → fully reproducible
            rng = random.Random(seed) if seed is not None else random.Random()
            chosen = [rng.choice(indices) for indices in groups.values()]

            chosen.sort()  # final dataset always in original index order
            dataset = dataset.select(chosen)

            print(
                f"Random unique-image filter: {before} → {len(dataset)} "
                f"({len(groups)} unique images)"
            )

        # Filter yes/no questions if requested
        if remove_yes_no:
            print("Filtering out Yes/No questions...")
            initial_len = len(dataset)
            # Assuming 'answer' field contains the answer (e.g., "Yes", "No", "A", "B")
            # Standardizing check to be case-insensitive against "yes" and "no"
            dataset = dataset.filter(
                lambda x: str(x["answer"]).strip().lower() not in ["yes", "no"]
            )
            print(
                f"Filtered {initial_len - len(dataset)} samples (from {initial_len} to {len(dataset)})"
            )

        if filter_mmbench_4_choices and "mmbench" in self.dataset_path_or_name:
            print("Filtering MMBench to samples with 4 choices...")
            initial_len = len(dataset)

            def has_4_choices(example):
                count = 0
                for choice in MMBENCH_ANSWER_CHOICES:
                    val = example.get(choice, None)
                    if val is not None and str(val).lower() != "nan":
                        count += 1
                return count == 4

            dataset = dataset.filter(has_4_choices)
            print(
                f"Filtered {initial_len - len(dataset)} samples (from {initial_len} to {len(dataset)})"
            )
            
        # Apply downsampling if requested and if dataset is larger than requested size
        if downsample_size and downsample_size < len(dataset):
            print(
                f"Downsampling dataset from {len(dataset)} to {downsample_size} samples"
            )
            dataset = dataset.shuffle(seed=seed).select(range(downsample_size))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question_data = self.dataset[idx]
        question_text = question_data["question"]

        # get the choices list based on the dataset
        if (
            "mmlu" in self.dataset_path_or_name
            or "scienceqa" in self.dataset_path_or_name
        ):
            choices_list = question_data["choices"]
        elif "mmbench" in self.dataset_path_or_name:
            choices_list = []
            for choice in MMBENCH_ANSWER_CHOICES:
                choice_data = question_data.get(choice, None)
                if choice_data is not None:
                    choices_list.append(choice_data)
        elif "ai2d" in self.dataset_path_or_name:
            choices_list = question_data["options"]
        elif "pope" in self.dataset_path_or_name:
            choices_list = POPE_ANSWER_CHOICES
        elif "mme" in self.dataset_path_or_name:
            choices_list = MME_ANSWER_CHOICES
        elif "seed" in self.dataset_path_or_name:
            choices_list = []
            for choice in ["choice_a", "choice_b", "choice_c", "choice_d"]:
                choice_data = question_data.get(choice, None)
                if choice_data is not None:
                    choices_list.append(choice_data)
        elif "mmstar" in self.dataset_path_or_name:
            choices_list = MMSTAR_ANSWER_CHOICES

        # hint or subject for mmlu, scienceqa and mmbench
        context_text = question_data.get("hint", None)
        context_text = "" if context_text == "nan" else context_text
        subject = question_data.get("subject", None)

        def format_subject(subject):
            return subject.replace("_", " ")

        # answer for each specific dataset
        if "scienceqa" in self.dataset_path_or_name:
            answer_letter = SQA_ANSWER_CHOICES[question_data["answer"]]
        elif "mmlu" in self.dataset_path_or_name:
            answer_letter = MMLU_ANSWER_CHOICES[question_data["answer"]]
        elif "ai2d" in self.dataset_path_or_name:
            # AI2D might differ in answer format (int index vs letter)
            answer_raw = question_data["answer"]
            if isinstance(answer_raw, int):
                answer_letter = AI2D_ANSWER_CHOICES[answer_raw]
            elif isinstance(answer_raw, str) and answer_raw.isdigit():
                answer_letter = AI2D_ANSWER_CHOICES[int(answer_raw)]
            else:
                answer_letter = answer_raw
        elif "pope" in self.dataset_path_or_name:
            # Map "yes"/"no" to "Yes"/"No"
            answer_raw = question_data["answer"]
            if str(answer_raw).lower() == "yes":
                answer_letter = "Yes"
            elif str(answer_raw).lower() == "no":
                answer_letter = "No"
            else:
                answer_letter = answer_raw
        else:
            # Yes/No for MME
            # A/B/C/D for MMBench and SEED-bench
            answer_letter = question_data["answer"]

        # get the image for visual question answering or the caption for textual question answering
        full_question_prompt = ""

        # Retrieve the image. It might be None if the sample doesn't have one or if texts_qa is True.
        image = None
        if not self.texts_qa:
            image = question_data.get("image", None)

            # Handle MMBench base64-encoded images
            if image is not None and "mmbench" in self.dataset_path_or_name:
                if isinstance(image, str):
                    # Decode base64 image
                    img_bytes = base64.b64decode(image)
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")

            # Ensure RGB
            if isinstance(image, Image.Image):
                image = image.convert("RGB")

        # Add caption context if texts_qa is True in place of the image
        if self.texts_qa:
            # by construction the datasets used here have synthetic-captions
            caption = question_data.get("synthetic-caption") or question_data.get(
                "caption"
            )
            if caption:
                full_question_prompt += f"Below is a description of an image. Use it to answer the question.\n\n{caption}\n\n"
            else:
                raise ValueError("No caption found")

        # add context (mmbench, scienceqa)
        if context_text is not None and context_text != "":
            if "scienceqa" in self.dataset_path_or_name:
                full_question_prompt += "Context: "
            full_question_prompt += f"{context_text} "

        # add subject-specific instruction for mmlu
        if subject is not None and "mmlu" in self.dataset_path_or_name:
            if self.question_instruction_type == "singular":
                full_question_prompt += f"The following is a multiple choice question (with answers) about {format_subject(subject)}.\n\n"
            elif self.question_instruction_type == "plural":
                full_question_prompt += f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"

        # quesstion pre-processing for MME to remove redundant instructions and ensure consistent formatting
        if "mme" in self.dataset_path_or_name:
            question_text = question_text.replace(" Please answer yes or no.", "")
            question_text = question_text.replace(
                "\nAnswer the question using a single word or phrase.", ""
            )
            question_text = question_text.replace(
                "\nAnswer the question with Yes or No.", ""
            )
            question_text = question_text.replace(" Answer:", "")
            question_text = question_text.replace(
                " Answer this question briefly[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n",
                "",
            )
            question_text = question_text.replace("[UNUSED_TOKEN_146]user\n", "")

        full_question_prompt += f"{question_text}"

        if not self.open_ended:
            # additional instruction for mmbench
            if "mmbench" in self.dataset_path_or_name:
                full_question_prompt += "\nThere are several options:\n"
            elif (
                "mme" not in self.dataset_path_or_name
                and "pope" not in self.dataset_path_or_name
            ):
                full_question_prompt += "\n"

            # add choices for multiple choice benchmarks except MME and POPE (which are Yes/No)
            if (
                "mme" not in self.dataset_path_or_name
                and "pope" not in self.dataset_path_or_name
            ):
                if choices_list is not None:
                    last_choice = choices_list[-1]
                    for letter_ord, choice_text in zip(
                        range(ord("A"), ord("Z") + 1), choices_list
                    ):
                        separator = (
                            ". " if "scienceqa" not in self.dataset_path_or_name else ": "
                        )
                        full_question_prompt += f"{chr(letter_ord)}{separator}{choice_text}"
                        full_question_prompt += "\n" if choice_text != last_choice else ""

        # these datasets have an imageHash field that can be used to uniquely identify the image
        imageHash = question_data.get("imageHash", None)

        sample = {
            "question": full_question_prompt,
            "answer_letter": answer_letter,
            "image_id": imageHash,
        }
        if self.images_qa:
            sample["image"] = image

        return sample


def extract_image_id(image_name: str) -> Optional[int]:
    """
    Extract the image id from the image name.

    Args:
        image_name: Image filename (e.g., "COCO_val2014_000000522418.jpg")

    Returns:
        Extracted image ID as integer, or None if extraction fails

    Example:
        extract_image_id("COCO_val2014_000000522418.jpg") => 522418
    """
    if not image_name:
        return None

    try:
        # Split by "2014_" and get the part after it
        if "2014_" not in image_name:
            return None

        s = image_name.split("2014_")[1]
        match = re.search(r"\d+", s)  # Find the first sequence of digits
        if match:
            return int(match.group())  # Automatically removes leading zeros
        return None  # No digits found
    except (IndexError, ValueError, AttributeError):
        return None


def _resolve_split_path(dataset_path_or_name: str, split_name: str) -> str:
    """Locate the JSON file for a precomputed dataset split.

    Searches ``data/splits/{dataset_slug}/{split_name}.json`` relative to the
    project root (two levels up from ``utils/``).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slug = dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    path = os.path.join(project_root, "data", "splits", slug, f"{split_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split file not found: {path}. "
            f"Run scripts/create_datasets/create_dataset_splits.py first."
        )
    return path


class OpenVQADataset(Dataset):
    """
    Dataset for Open-ended VQA datasets.

    Specifically, it is used for a version of the COCOQA dataset that has been modified to
    remove the images or the captions of the original dataset.

    Args:
        dataset_path_or_name: Path to the dataset directory or Hugging Face dataset identifier.
        cache_dir: Directory to cache the dataset.
        split: Dataset split to load.
        downsample_size: Maximum number of samples to use
        seed: Random seed for reproducibility
        images_qa: Whether to remove images from the dataset
        texts_qa: Whether to remove captions from the dataset
        remove_yes_no: Whether to remove Yes/No questions/answers from the dataset
        caption_source: For datasets with both human captions and synthetic caption
            (COCOQA-restval-ICQA, VQAv2-COCO-restval-ICQ). Use 'human' for the
            concatenated captions list, 'synthetic' for the single synthetic-caption.
            None (default) uses the existing behavior.
        dataset_split: Name of a precomputed split (e.g. "split01") whose indices
            are stored in ``data/splits/{dataset_slug}/{split_name}.json``.  When
            provided, seed-based downsampling is skipped and the split's indices
            are used to select rows instead.
    """

    def __init__(
        self,
        dataset_path_or_name: str,
        cache_dir: Optional[str] = None,
        split: str = "train",
        downsample_size: Optional[int] = None,
        seed: int = SEED,
        images_qa: bool = False,
        texts_qa: bool = False,
        remove_yes_no: bool = False,
        caption_source: Optional[str] = None,
        allowed_indices: Optional[set[int]] = None,
        dataset_split: Optional[str] = None,
    ):
        if images_qa and texts_qa:
            raise ValueError("images_qa and texts_qa cannot be both True.")

        self.images_qa = images_qa
        self.texts_qa = texts_qa
        self.caption_source = caption_source
        self.dataset_path_or_name = dataset_path_or_name
        # Random number generator for reproducibility and stable one-to-one
        # question-answer pair selection between images_qa and texts_qa datasets
        self.rng = np.random.RandomState(seed=seed)

        # Load dataset from Hugging Face Hub or Cache
        try:
            dataset = load_dataset(
                dataset_path_or_name, split=split, cache_dir=cache_dir
            )
        except Exception as e:
            # Try loading from disk if it's a local path
            try:
                dataset = load_from_disk(dataset_path_or_name)
            except Exception as disk_e:
                raise RuntimeError(
                    f"Failed to load dataset '{dataset_path_or_name}'. "
                    f"HF Hub error: {e}, Disk error: {disk_e}"
                )

        # Add an idx column to the dataset
        dataset = dataset.add_column("idx", list(range(len(dataset))))

        if allowed_indices and dataset_split:
            raise ValueError(
                "allowed_indices and dataset_split are mutually exclusive!"
            )

        # Optional semantic filtering by row index — applied BEFORE downsampling
        if allowed_indices is not None:
            valid = sorted(allowed_indices & set(range(len(dataset))))
            before = len(dataset)
            dataset = dataset.select(valid)
            print(
                f"Applied allowed_indices filter "
                f"(from {before} to {len(dataset)} samples)"
            )

        # Detect schema: check for 'questions_and_answers' field
        self.has_qa_list = "questions_and_answers" in dataset.features

        # Filter Yes/No questions if requested
        if remove_yes_no:
            print("Filtering out Yes/No questions from OpenVQADataset...")

            # Determine num_proc based on CPU count (using half as requested in general context)
            num_proc = max(1, (os.cpu_count() or 2) // 2)

            if self.has_qa_list:

                def filter_yes_no_answers_list(example):
                    qa_list = example["questions_and_answers"]
                    # Keep only QA pairs where answer is NOT yes/no
                    filtered_qa = [
                        qa
                        for qa in qa_list
                        if qa["answer"].strip().lower() not in ["yes", "no"]
                    ]
                    example["questions_and_answers"] = filtered_qa
                    return example

                # 1. Filter the internal list of questions
                dataset = dataset.map(
                    filter_yes_no_answers_list,
                    num_proc=num_proc,
                    desc="Filtering Yes/No answers",
                )

                # 2. Remove samples that have no questions left
                initial_len = len(dataset)
                dataset = dataset.filter(
                    lambda x: len(x["questions_and_answers"]) > 0,
                    desc="Removing empty samples",
                )
                print(
                    f"Filtered {initial_len - len(dataset)} empty samples (from {initial_len} to {len(dataset)})"
                )

            else:
                # Flat schema (e.g. GQA)
                # Check 'answer' field
                initial_len = len(dataset)
                dataset = dataset.filter(
                    lambda x: str(x["answer"]).strip().lower() not in ["yes", "no"],
                    num_proc=num_proc,
                    desc="Filtering Yes/No answers (flat)",
                )
                print(
                    f"Filtered {initial_len - len(dataset)} samples (from {initial_len} to {len(dataset)})"
                )

        # Select subset: precomputed split OR seed-based downsampling (mutually exclusive)
        if dataset_split is not None:
            split_path = _resolve_split_path(dataset_path_or_name, dataset_split)
            with open(split_path) as f:
                split_data = json.load(f)
            split_indices = split_data["indices"]
            before = len(dataset)
            dataset = dataset.select(split_indices)
            print(
                f"Applied dataset split '{dataset_split}': "
                f"{before} -> {len(dataset)} samples (from {split_path})"
            )
        elif downsample_size and downsample_size < len(dataset):
            print(
                f"Downsampling dataset from {len(dataset)} to {downsample_size} samples"
            )
            dataset = dataset.shuffle(seed=seed).select(range(downsample_size))

        # Filter images or captions based on keep_images flag
        if images_qa:
            # GQA might name it 'caption' (singular) or 'captions' (plural)
            # COCOQA uses 'captions'
            # Safe remove if present
            cols_to_remove = [
                c
                for c in ["captions", "caption", "synthetic-caption"]
                if c in dataset.column_names
            ]
            if cols_to_remove:
                dataset = dataset.remove_columns(cols_to_remove)
        elif texts_qa:
            dataset = dataset.remove_columns("image")
            # Remove unused caption columns based on caption_source selection
            if caption_source == "synthetic":
                cols_to_remove = [c for c in ["captions"] if c in dataset.column_names]
            else:
                cols_to_remove = [
                    c for c in ["synthetic-caption"] if c in dataset.column_names
                ]
            if cols_to_remove:
                dataset = dataset.remove_columns(cols_to_remove)

        # Sort the dataset by idx to guarantee images_qa and texts_qa are in the same order
        dataset = dataset.sort("idx")

        self.dataset = dataset

        # Pre-compute deterministic question-answer selection for each sample
        # Only needed if we have a list of QA pairs
        if self.has_qa_list:
            self._precompute_qa_selection()

    def _hash_qa(self, qa_dict):
        """
        Create a SHA-256 hash for a question-answer dict.
        Normalizes by stripping whitespace and converting to lowercase.
        Returns the hex digest string.
        """
        question = qa_dict["question"].strip().lower()
        answer = qa_dict["answer"].strip().lower()
        combined = f"{question}|||{answer}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _precompute_qa_selection(self):
        """
        Pre-compute which question-answer pair to use for each dataset sample.
        This ensures deterministic behavior: same idx always returns same Q&A.
        """
        seen_hashes = set()
        self._selected_qa_indices = {}

        for sample_idx in range(len(self.dataset)):
            question_data = self.dataset[sample_idx]
            questions_and_answers = question_data["questions_and_answers"]

            # Create deterministic shuffle based on dataset idx and seed
            shuffled_indices = list(range(len(questions_and_answers)))
            self.rng.shuffle(shuffled_indices)

            # Find first unique Q&A pair
            selected_idx = None
            for shuffled_idx in shuffled_indices:
                qa_dict = questions_and_answers[shuffled_idx]
                hash_qa = self._hash_qa(qa_dict)
                if hash_qa not in seen_hashes:
                    seen_hashes.add(hash_qa)
                    selected_idx = shuffled_idx
                    break

            # If no unique Q&A found, use first from shuffled list
            if selected_idx is None:
                selected_idx = shuffled_indices[0]

            self._selected_qa_indices[sample_idx] = selected_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question_data = self.dataset[idx]

        # Get Question and Answer
        if self.has_qa_list:
            # Use pre-computed selection for deterministic behavior
            questions_and_answers = question_data["questions_and_answers"]
            selected_qa_idx = self._selected_qa_indices[idx]
            unique_qa_dict = questions_and_answers[selected_qa_idx]

            q_text = unique_qa_dict["question"]
            a_text = unique_qa_dict["answer"]
        else:
            # Flat schema (GQA, TextVQA, CountBench)
            q_text = question_data["question"]
            if (
                "textvqa" in self.dataset_path_or_name.lower()
                and "answers" in question_data
            ):
                # TextVQA has 'answers' list. Pick the most frequent one (majority voting)
                # or just the first one if we need a single string.
                # For logit lens, we need a single target. TextVQA usually has 10 answers.
                # We'll pick the most frequent one.
                answers = question_data["answers"]
                if isinstance(answers, list) and len(answers) > 0:
                    # Simple majority
                    from collections import Counter

                    c = Counter(answers)
                    a_text = c.most_common(1)[0][0]
                else:
                    a_text = ""
            elif "countbench" in self.dataset_path_or_name.lower():
                a_text = str(question_data["number"])
            else:
                a_text = question_data["answer"]

        # construct the full question prompt with optional context and captions based on the dataset and configuration
        full_question_prompt = ""

        # Handle Captions
        if self.texts_qa:
            if self.caption_source == "synthetic":
                # Use synthetic-caption (single string)
                caption = question_data["synthetic-caption"]
                full_question_prompt += f"Below is a description of an image. Use it to answer the question.\n\n{caption}\n\n"
            elif "captions" in question_data:
                captions = question_data["captions"]
                full_question_prompt += (
                    f"Below are {len(captions)} descriptions of an image. "
                    f"Use them to answer the question.\n\n"
                )

                lines = [f"{i + 1}: {caption}" for i, caption in enumerate(captions)]
                full_question_prompt += "\n".join(lines) + "\n\n"
            else:
                raise ValueError("No captions found in the dataset")

        full_question_prompt += f"{q_text}"
        if "cocoqa" in self.dataset_path_or_name.lower():
            full_question_prompt += "?"

        # Image ID extraction (handle different field names)
        # GQA uses 'imageId', COCOQA uses 'image_id'
        img_id_raw = question_data.get("image_id") or question_data.get("imageId")

        sample = {
            "idx": question_data.get("idx", None),
            "question": full_question_prompt,
            "answer_letter": a_text,
            "image_id": extract_image_id(str(img_id_raw)) if img_id_raw else None,
        }

        if self.images_qa:
            sample["image"] = question_data.get("image", None)

        return sample


class ImageCaptioningDataset(Dataset):
    """LLaVA-Instruct captioning dataset for text-vs-image comparisons."""

    def __init__(
        self,
        dataset_path_or_name: str,
        downsample_size: Optional[int] = None,
        images_qa: bool = False,
        texts_qa: bool = False,
        image_dir: Optional[str] = None,
    ):
        if images_qa == texts_qa:
            raise ValueError("Exactly one of images_qa or texts_qa must be True.")
        if not is_llava_instruct_150k(dataset_path_or_name):
            raise ValueError(
                f"Captioning only supports {LLAVA_INSTRUCT_150K_NAME}; "
                f"got {dataset_path_or_name!r}."
            )

        self.dataset_path_or_name = dataset_path_or_name
        self.images_qa = images_qa
        self.texts_qa = texts_qa
        self.image_dir = (
            Path(image_dir).expanduser() if image_dir else DEFAULT_COCO_TRAIN2014_DIR
        )

        raw_dataset = load_llava_instruct_detail_records()
        self.dataset = _select_llava_caption_records(
            dataset=raw_dataset,
            image_dir=self.image_dir,
            downsample_size=downsample_size,
        )
        print(
            f"Loaded {len(self.dataset)} LLaVA-Instruct caption records "
            f"from {LLAVA_INSTRUCT_DETAIL_FILE!r}"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_textual_caption_prompt(self, caption: str) -> str:
        return f"{TEXTUAL_CAPTION_PREAMBLE} {caption}\n\n{CAPTIONING_PROMPT}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.dataset[idx]

        caption = data["caption"]
        prompt = (
            self._build_textual_caption_prompt(caption)
            if self.texts_qa
            else f"\n{CAPTIONING_PROMPT}"
        )
        sample = {
            "idx": data["idx"],
            "user_prompt": prompt,
            "captions": [caption],
            "image_id": data.get("id") or Path(data["image_name"]).stem,
            "image_path": data["image_path"],
            "gpt_response_length": data["gpt_response_length"],
        }
        if self.images_qa:
            sample["image"] = Image.open(data["image_path"]).convert("RGB")

        return sample


_DEFAULT_SYSTEM_CACHE: dict[str, bool] = {}


def _is_llava_family_processor(processor: Any) -> bool:
    """Return True for LLaVA-family processors, including OneVision (Qwen2.5-VL).

    Why this exists:
    - LLaVA-1.x processors are usually named like ``LlavaProcessor``.
    - LLaVA-OneVision uses ``Qwen2_5_VLProcessor`` via ``AutoProcessor`` and
      does not contain "llava" in the processor class name.
    """
    if processor is None:
        return False

    proc_name = processor.__class__.__name__.lower()
    tokenizer_name = ""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer_name = tokenizer.__class__.__name__.lower()

    llava_markers = ("llava", "qwen2_5_vl", "qwen2vl")
    return any(marker in proc_name for marker in llava_markers) or any(
        marker in tokenizer_name for marker in llava_markers
    )


def _processor_has_default_system(processor: Any) -> bool:
    """Return True if callers should skip injecting SYSTEM_ROLE into conversations.

    This is True when either (a) the chat template already injects a system
    message by default (LLaVA-OneVision's Qwen2 tokenizer), or (b) the chat
    template cannot accept SYSTEM_ROLE's list-of-dicts content format
    (Pixtral's Mistral-style template concatenates system content directly onto
    a string). In both cases the caller should NOT prepend SYSTEM_ROLE.

    Result is cached per processor class so the probe runs at most once per type.
    """
    cls_name = processor.__class__.__name__
    if cls_name in _DEFAULT_SYSTEM_CACHE:
        return _DEFAULT_SYSTEM_CACHE[cls_name]

    # Prefer the processor's own chat template — for some multimodal processors
    # (e.g. Pixtral) the template lives on the processor, not the tokenizer.
    proc_apply = getattr(processor, "apply_chat_template", None)
    tokenizer = getattr(processor, "tokenizer", None)
    tok_apply = getattr(tokenizer, "apply_chat_template", None) if tokenizer is not None else None
    apply_fn = proc_apply or tok_apply
    if apply_fn is None:
        _DEFAULT_SYSTEM_CACHE[cls_name] = False
        return False

    user_msg = {"role": "user", "content": [{"type": "text", "text": "?"}]}
    try:
        probe = apply_fn([user_msg], tokenize=False, add_generation_prompt=False)
    except Exception:
        probe = ""
    if "system" in probe.lower():
        _DEFAULT_SYSTEM_CACHE[cls_name] = True
        return True

    # Template didn't auto-inject a system message. Check whether it can handle
    # our SYSTEM_ROLE format (list-of-dicts content). If it can't, return True
    # so callers skip injection rather than crashing.
    system_role = {
        "role": "system",
        "content": [{"type": "text", "text": "probe"}],
    }
    try:
        apply_fn([system_role, user_msg], tokenize=False, add_generation_prompt=False)
    except Exception:
        _DEFAULT_SYSTEM_CACHE[cls_name] = True
        return True

    _DEFAULT_SYSTEM_CACHE[cls_name] = False
    return False


def _build_llava_prompts(
    questions: List[str],
    images: List[Optional[Image.Image]],
    processor: Any,
    guide_text: str = "",
    assistant_prefix: str = "",
) -> List[str]:
    """Format questions + (optional) images into chat-template strings for LLaVA models.

    Injects SYSTEM_ROLE only when the processor has no built-in default system
    message (LLaVA-1.5).  LLaVA-OneVision's Qwen2 tokenizer already includes
    one, so no injection is needed there.

    When ``assistant_prefix`` is non-empty, it is appended as a partial assistant
    turn so the model continues from that point (continuation mode).
    """
    include_system = not _processor_has_default_system(processor)
    if not images:
        images = [None] * len(questions)
    conversations = []
    for question, image in zip(questions, images):
        content: List[Dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image"})
        text = format_user_turn_text(question, guide_text)
        content.append({"type": "text", "text": text})
        conv: List[Dict[str, Any]] = [{"role": "user", "content": content}]
        if include_system:
            conv.insert(0, SYSTEM_ROLE)
        if assistant_prefix:
            conv.append({"role": "assistant", "content": [{"type": "text", "text": assistant_prefix}]})
        conversations.append(conv)
    if assistant_prefix:
        return processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False, continue_final_message=True,
        )
    return processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True,
    )


def collate_vqa_batch(
    batch,
    processor,
    guide_text,
    answer_letters_with_processed_batch,
    model: Optional[PreTrainedModel] = None,
):
    # Custom collate function to handle images and text
    questions = [item["question"] for item in batch]
    answer_letters = [item["answer_letter"] for item in batch]
    images = [item.get("image", None) for item in batch]
    image_ids = [item.get("image_id", None) for item in batch]

    if processor is not None:
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": (True if len(questions) > 1 else False),
        }

        if _is_llava_family_processor(processor):
            processor_kwargs["text"] = _build_llava_prompts(
                questions=questions,
                images=images,
                processor=processor,
                guide_text=guide_text,
            )
            if any(image is not None for image in images):
                processor_kwargs["images"] = images

        elif model and "intern" in model.__class__.__name__.lower():
            has_images = any(image is not None for image in images)
            if has_images:
                pixel_values, num_patches_list = get_pixel_values(images)
                queries = apply_internvl_chat_template(
                    model=model,
                    text=questions,
                    num_patches_list=num_patches_list,
                    guide_text=guide_text,
                )
            else:
                # Text-only mode: format with chat template but no image tokens
                pixel_values = None
                num_patches_list = None
                queries = apply_internvl_chat_template(
                    model=model, text=questions, guide_text=guide_text
                )
            processor_kwargs["text"] = queries

        elif model and is_cambrian_model(model=model):
            has_images = any(image is not None for image in images)
            prepared = prepare_cambrian_batch_inputs(
                model=model,
                tokenizer=processor,
                questions=questions,
                images=images if has_images else None,
                guide_text=guide_text,
            )
            tokenized = prepared["tokenized"]
            prompts = prepared["prompts"]
            image_tensors = prepared["image_tensors"]
            image_sizes = prepared["image_sizes"]

        if not (model and is_cambrian_model(model=model)):
            tokenized = processor(**processor_kwargs)

        outputs = (tokenized,)
        if model and "intern" in model.__class__.__name__.lower():
            if pixel_values is not None:
                outputs += (pixel_values, processor_kwargs["text"])
            else:
                outputs += (processor_kwargs["text"],)
        if model and is_cambrian_model(model=model):
            if image_tensors is not None:
                outputs += (image_tensors, image_sizes, prompts)
            else:
                outputs += (prompts,)
        if answer_letters_with_processed_batch:
            # !not so neat but it's a backward compatibility feature
            return outputs + (answer_letters,)
        return outputs

    # Raw mode (default)
    return {
        "questions": questions,
        "answer_letters": answer_letters,
        "images": images,
        "image_ids": image_ids,
    }


def collate_captioning_batch(
    batch,
    processor,
    model: Optional[PreTrainedModel] = None,
):
    user_prompts = [item["user_prompt"] for item in batch]
    captions = [item["captions"] for item in batch]
    images = [item.get("image", None) for item in batch]
    image_ids = [item["image_id"] for item in batch]

    if processor is not None:
        has_images = any(image is not None for image in images)
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": (True if len(user_prompts) > 1 else False),
        }

        if _is_llava_family_processor(processor):
            processor_kwargs["text"] = _build_llava_prompts(
                questions=user_prompts,
                images=images,
                processor=processor,
            )
            if has_images:
                processor_kwargs["images"] = images

        elif model and "intern" in model.__class__.__name__.lower():
            if has_images:
                pixel_values, num_patches_list = get_pixel_values(images)
                queries = apply_internvl_chat_template(
                    model=model,
                    text=user_prompts,
                    num_patches_list=num_patches_list,
                )
            else:
                pixel_values = None
                queries = apply_internvl_chat_template(model=model, text=user_prompts)
            processor_kwargs["text"] = queries

        elif model and is_cambrian_model(model=model):
            prepared = prepare_cambrian_batch_inputs(
                model=model,
                tokenizer=processor,
                questions=user_prompts,
                images=images if has_images else None,
            )
            tokenized = prepared["tokenized"]
            prompts = prepared["prompts"]
            image_tensors = prepared["image_tensors"]
            image_sizes = prepared["image_sizes"]

        if not (model and is_cambrian_model(model=model)):
            tokenized = processor(**processor_kwargs)

        # Build return tuple based on requested data
        result = [tokenized]
        if model and "intern" in model.__class__.__name__.lower():
            if pixel_values is not None:
                result.append(pixel_values)
            else:
                result.append(processor_kwargs["text"])
        if model and is_cambrian_model(model=model):
            if image_tensors is not None:
                result.extend([image_tensors, image_sizes, prompts])
            else:
                result.append(prompts)

        return tuple(result) if len(result) > 1 else result[0]

    # Raw mode (default)
    return {
        "user_prompts": user_prompts,
        "captions": captions,
        "images": images,
        "image_ids": image_ids,
    }


def get_dataloader(
    dataset_path_or_name: str,
    cache_dir: Optional[str] = None,
    split: Optional[str] = None,
    texts_qa: Optional[bool] = None,
    images_qa: Optional[bool] = None,
    question_instruction_type: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    downsample_size: Optional[int] = None,
    seed: int = 42,
    processor: Optional[ProcessorType] = None,
    guide_text: str = "Answer the question using a single word or phrase.\n",
    answer_letters_with_processed_batch: bool = False,
    caption_source: Optional[str] = None,
    remove_yes_no: bool = False,
    filter_mmbench_4_choices: bool = False,
    model: Optional[PreTrainedModel] = None,
    allowed_indices: Optional[set[int]] = None,
    unique_images: bool = False,
    open_ended: bool = False,
    dataset_split: Optional[str] = None,
):
    """
    Get a DataLoader for either multiple choice benchmarks or open-ended VQA datasets.
    For convenience, if a processor is provided, the dataset will be preprocessed and returned as
    model-ready tensors for residual stream tracing, remaining to move to the model device.
    If answer_letters_with_processed_batch is True, in addition to the processed batch, the raw answer letters
    will be returned as a list of strings in a tuple with the processed batch.
    Otherwise, the dataset will be returned as a dictionary with "questions", "answer_letters",
    and "images" keys.

    Args:
        dataset_path_or_name: Path to dataset or HuggingFace dataset name.
        cache_dir: Cache directory for HuggingFace datasets (required for benchmark datasets).
        split: Dataset split to use.
        texts_qa: Whether to use text-only samples (for multimodal datasets).
        images_qa: Whether to use image samples (for multimodal datasets).
        question_instruction_type: Instruction type for multiple choice questions.
        batch_size: Batch size for DataLoader.
        num_workers: Number of workers for DataLoader.
        downsample_size: Maximum number of samples to use.
        seed: Random seed for reproducibility.
        processor: A processor used to tokenize and encode text+image inputs.
        guide_text: Optional text to append to questions during prompt formatting.
        answer_letters_with_processed_batch: Whether to include answer letters in the processed batch,
            it just applies when a processor is provided.
        chat_template_exists: Whether to use the chat template for formatting prompts.
        remove_yes_no: Whether to remove Yes/No questions from the dataset.
        filter_mmbench_4_choices: Whether to filter MMBench samples to only those with 4 choices.
    """

    print("\nLoading dataloader...")

    # Determine dataset type based on name/path
    is_multiple_choice_benchmark = any(
        benchmark in dataset_path_or_name.lower()
        for benchmark in MULTIPLE_CHOICE_BENCHMARKS
    )

    if is_multiple_choice_benchmark:
        if cache_dir is None:
            raise ValueError("cache_dir is required for benchmark datasets")
        dataset = MultipleChoiceDatasetBenchmark(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split=split,
            texts_qa=texts_qa,
            images_qa=images_qa,
            question_instruction_type=question_instruction_type,
            downsample_size=downsample_size,
            seed=seed,
            remove_yes_no=remove_yes_no,
            filter_mmbench_4_choices=filter_mmbench_4_choices,
            allowed_indices=allowed_indices,
            unique_images=unique_images,
            open_ended=open_ended,
        )
    elif (
        "vqav2" in dataset_path_or_name.lower()
        or "cocoqa" in dataset_path_or_name.lower()
        or "gqa" in dataset_path_or_name.lower()
        or "textvqa" in dataset_path_or_name.lower()
        or "countbench" in dataset_path_or_name.lower()
    ):
        dataset = OpenVQADataset(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split=split or "train",
            downsample_size=downsample_size,
            seed=seed,
            images_qa=images_qa,
            texts_qa=texts_qa,
            remove_yes_no=remove_yes_no,
            caption_source=caption_source,
            allowed_indices=allowed_indices,
            dataset_split=dataset_split,
        )
    else:
        raise ValueError(
            f"Dataset {dataset_path_or_name} is not supported. "
            "Only multiple choice benchmarks and open-ended VQA datasets are supported."
        )

    collate_fn = partial(
        collate_vqa_batch,
        processor=processor,
        guide_text=guide_text,
        answer_letters_with_processed_batch=answer_letters_with_processed_batch,
        model=model,
    )

    pin_memory = not (model is not None and is_cambrian_model(model=model))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def get_dataloader_for_captioning(
    dataset_path_or_name: str,
    batch_size: int = 1,
    num_workers: int = 4,
    downsample_size: Optional[int] = None,
    processor: Optional[ProcessorType] = None,
    images_qa: bool = False,
    texts_qa: bool = False,
    image_dir: Optional[str] = None,
    model: Optional[PreTrainedModel] = None,
):
    dataset = ImageCaptioningDataset(
        dataset_path_or_name=dataset_path_or_name,
        downsample_size=downsample_size,
        images_qa=images_qa,
        texts_qa=texts_qa,
        image_dir=image_dir,
    )

    collate_fn = partial(
        collate_captioning_batch,
        processor=processor,
        model=model,
    )

    pin_memory = not (model is not None and is_cambrian_model(model=model))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# ClassificationDataset — for ScienceQA, MMMU, mini-imagenet probing
# ---------------------------------------------------------------------------

SCIENCEQA_MIN_TOPIC_SAMPLES = 70
MMMU_IMAGE_COLUMNS = [f"image_{i}" for i in range(1, 8)]
_IDENTIFY_QUESTION = (
    "You are a precision image classifier. Your task is to identify the primary entity "
    "in the image using the most specific name possible. Rules:\n"
    "1. Output exactly one identifier.\n"
    "2. Use snake_case for multi-word names (e.g., golden_retriever).\n"
    "3. Do not use articles (a, an, the).\n"
    "4. Do not provide descriptions or sentences."
)

_DESCRIBE_QUESTION = "Describe this image in detail."
_DESCRIBE_STARTS = (
    "A picture of a",
    "The scene shows",
    "Portrayed in the figure, there is",
    "I see an image with",
    "Of course! Above, there is a photo displaying",
)

_PROMPT_MODES: dict[str, tuple[str, str]] = {
    "describe":    (_DESCRIBE_QUESTION, ""),
    "continuation": (
        "In one sentence, describe the contents of the image.",
        "In the image, there is a ",
    ),
    "mcq":                ("", ""),  # mini-imagenet, category-based, N-choice
    "mcq_nocat":          ("", ""),  # mini-imagenet, category-free, N-choice (new)
    "mcq_nocat_filter":   ("", ""),  # mini-imagenet, category-free + filter to N classes
    "category_id":        ("", ""),  # mini-imagenet, open id with category hint
    "identify":           (_IDENTIFY_QUESTION, "Identity: "),
    "animal_mcq":         ("", ""),  # imagenet_short_text, N-choice from all 100 classes
    "animal_mcq_filter":  ("", ""),  # imagenet_short_text, filter to N classes + N-choice
    "animal_id":          ("This animal is a ", ""),
    "generic_mcq":            ("", ""),  # generic HF classification datasets
    "generic_mcq_filter":     ("", ""),
    "generic_mcq_top_filter": ("", ""),
}

MCQ_MODES: frozenset[str] = frozenset({
    "mcq", "mcq_nocat", "mcq_nocat_filter",
    "animal_mcq", "animal_mcq_filter",
    "generic_mcq", "generic_mcq_filter", "generic_mcq_top_filter",
})
MCQ_FILTER_MODES: frozenset[str] = frozenset({
    "mcq_nocat_filter", "animal_mcq_filter",
    "generic_mcq_filter", "generic_mcq_top_filter",
})
MCQ_FREE_MODES: frozenset[str] = frozenset({
    "mcq_nocat", "mcq_nocat_filter",
    "animal_mcq", "animal_mcq_filter",
    "generic_mcq", "generic_mcq_filter", "generic_mcq_top_filter",
})

# Dataset-specific MCQ question text for generic HF classification datasets.
_GENERIC_MCQ_QUESTIONS: dict[str, str] = {
    "food101":           "What food is shown in this image?",
    "sun397":            "What scene is shown in this image?",
    "caltech101":        "What object is shown in this image?",
    "oxford_flowers102": "What flower is shown in this image?",
    "places365":         "What place or scene is shown in this image?",
    "dtd":               "What texture is shown in this image?",
}

_GENERIC_DATASETS: tuple[str, ...] = (
    "food101", "sun397", "caltech101",
    "oxford_flowers102", "places365", "dtd",
)


def _detect_label_col(column_names: list[str]) -> str:
    for name in ("label", "labels"):
        if name in column_names:
            return name
    raise ValueError(
        f"No label column found in dataset. Columns: {column_names}"
    )

_MCQ_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "mini-imagenet" / "mini-imagenet_mcq_categorization.csv"
)


def _load_mcq_catalog(path: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Returns (wnid_to_entry, category_to_class_names)."""
    wnid_to_entry: dict[str, dict] = {}
    category_to_classes: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wnid_to_entry[row["wnid"]] = {
                "class_name": row["class_name"],
                "question_category": row["question_category"],
            }
            category_to_classes.setdefault(row["question_category"], []).append(row["class_name"])
    return wnid_to_entry, category_to_classes


def _format_mmmu_question(instance: dict) -> str:
    """Format an MMMU instance as 'question\\nA. opt\\n...' with image tags stripped."""
    question_text = re.sub(r"<image[^>]*>", " ", str(instance["question"]))
    question_text = re.sub(r"\s+", " ", question_text).strip()
    raw_options = instance.get("options", [])
    if isinstance(raw_options, str):
        try:
            options = ast.literal_eval(raw_options)
        except (ValueError, SyntaxError):
            options = []
    else:
        options = list(raw_options) if raw_options is not None else []
    choices_lines = []
    for letter_ord, choice_text in zip(range(ord("A"), ord("Z") + 1), options):
        normalized = re.sub(r"<image[^>]*>", " ", str(choice_text))
        normalized = re.sub(r"\s+", " ", normalized).strip()
        choices_lines.append(f"{chr(letter_ord)}. {normalized}")
    if choices_lines:
        return f"{question_text}\n" + "\n".join(choices_lines)
    return question_text


def _balanced_describe_starts(num_samples: int, seed: int) -> list[str]:
    repeats = (num_samples + len(_DESCRIBE_STARTS) - 1) // len(_DESCRIBE_STARTS)
    starts = list(_DESCRIBE_STARTS) * repeats
    starts = starts[:num_samples]
    rng = random.Random(seed)
    rng.shuffle(starts)
    return starts


def _format_describe_prompt(start: str) -> str:
    return f'{_DESCRIBE_QUESTION} Start the description with: "{start}"'


class ClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "derek-thomas/ScienceQA",
        split: str = "test",
        max_samples: int | None = None,
        label_granularity: str = "intermediate",
        prompt_mode: str = "describe",
        seed: int = SEED,
        n_choices: int = 10,
        subsample_size: int | None = None,
        label_mode: str = "class_name",
    ) -> None:
        if subsample_size is not None and prompt_mode in MCQ_FILTER_MODES:
            raise ValueError(
                f"--subsample-size is disjoint from filter modes; got "
                f"prompt_mode={prompt_mode!r} with subsample_size={subsample_size}. "
                f"Filter modes already subsample by class."
            )
        if prompt_mode in MCQ_MODES and n_choices < 2:
            raise ValueError(f"n_choices must be >= 2 for MCQ modes, got {n_choices}")
        if label_mode == "option_letter" and prompt_mode not in MCQ_MODES:
            raise ValueError(
                f"label_mode='option_letter' requires an MCQ prompt_mode; "
                f"got prompt_mode={prompt_mode!r}"
            )

        dataset = load_dataset(dataset_name, split=split)
        self.dataset_name_lower = dataset_name.lower()
        self.label_granularity = label_granularity
        self.prompt_mode = prompt_mode
        self.n_choices = n_choices
        self.subsample_size = subsample_size
        self.label_mode = label_mode

        if "scienceqa" in self.dataset_name_lower:
            topics_counts = Counter(dataset["topic"])

            def filter_min_counts_and_image(example):
                return (
                    topics_counts[example["topic"]] >= SCIENCEQA_MIN_TOPIC_SAMPLES
                    and example["image"] is not None
                )

            filtered_dataset = dataset.filter(filter_min_counts_and_image, num_proc=8)
            topics_counts_filtered = Counter(filtered_dataset["topic"])

            def filter_min_counts(example):
                return topics_counts_filtered[example["topic"]] >= SCIENCEQA_MIN_TOPIC_SAMPLES

            self.dataset = filtered_dataset.filter(filter_min_counts, num_proc=8)
        elif "mmmu" in self.dataset_name_lower:
            self.dataset = dataset.filter(
                lambda example: sum(
                    example.get(column) is not None for column in MMMU_IMAGE_COLUMNS
                ) <= 1,
                num_proc=8,
            )
            self.dataset = self.dataset.filter(
                lambda example: str(example.get("question_type", "")).strip().lower()
                == "multiple-choice",
                num_proc=8,
            )
            if label_granularity == "detailed":
                all_subfields = [str(sf).strip() for sf in self.dataset["subfield"]]
                subfield_counts = Counter(sf for sf in all_subfields if sf)
                if subfield_counts:
                    mean_count = sum(subfield_counts.values()) / len(subfield_counts)
                    valid_subfields = {
                        sf for sf, cnt in subfield_counts.items() if cnt >= mean_count
                    }
                    self.dataset = self.dataset.filter(
                        lambda example: str(example.get("subfield", "")).strip()
                        in valid_subfields,
                        num_proc=8,
                    )
        elif "mini-imagenet" in self.dataset_name_lower:
            lf = dataset.features["label"]
            self._class_names: list[str] | None = lf.names if hasattr(lf, "names") else None
            self._mini_imagenet_question = _PROMPT_MODES.get(prompt_mode, _PROMPT_MODES["describe"])[0]
            self._mcq_catalog, self._category_classes = _load_mcq_catalog(_MCQ_CATALOG_PATH)
            self._all_catalog_classes: list[str] = sorted({
                entry["class_name"] for entry in self._mcq_catalog.values()
            })
            self._mcq_selected: set[str] = set()
            if prompt_mode == "mcq_nocat_filter":
                rng = random.Random(seed)
                self._mcq_selected = set(rng.sample(self._all_catalog_classes, n_choices))
                dataset = dataset.filter(
                    lambda x: self._mcq_catalog[self._class_names[x["label"]]]["class_name"]
                              in self._mcq_selected
                )
            self.dataset = dataset
        elif "imagenet_short_text" in self.dataset_name_lower:
            # Build offset→class_name map from first unique 100 rows.
            # synset like "cocker_spaniel.n.01" → "cocker_spaniel"
            seen: dict[str, str] = {}
            for row in dataset:
                if row["offset"] not in seen:
                    seen[row["offset"]] = row["synset"].split(".")[0]
                if len(seen) == 100:
                    break
            self._imagenet_offset_to_class: dict[str, str] = seen
            self._imagenet_all_classes: list[str] = sorted(seen.values())
            self._imagenet_animal_question: str = _PROMPT_MODES.get(
                prompt_mode, _PROMPT_MODES["animal_id"]
            )[0]
            self._animal_selected: set[str] = set()
            if prompt_mode == "animal_mcq_filter":
                rng = random.Random(seed)
                self._animal_selected = set(rng.sample(self._imagenet_all_classes, n_choices))
                dataset = dataset.filter(
                    lambda x: self._imagenet_offset_to_class[x["offset"]] in self._animal_selected
                )
            self.dataset = dataset
        elif any(ds in self.dataset_name_lower for ds in _GENERIC_DATASETS):
            self._label_col: str = _detect_label_col(dataset.column_names)
            lf = dataset.features[self._label_col]
            self._class_names_list: list[str] = (
                lf.names if hasattr(lf, "names") else [str(i) for i in range(lf.num_classes)]
            )
            self._generic_mcq_question: str = next(
                q for k, q in _GENERIC_MCQ_QUESTIONS.items() if k in self.dataset_name_lower
            )
            dataset = self._prepare_generic_classification_dataset(
                dataset=dataset,
                dataset_name=dataset_name,
                prompt_mode=prompt_mode,
                n_choices=n_choices,
                seed=seed,
            )
            self.dataset = dataset
        else:
            raise ValueError(
                "Unsupported dataset for classification pipeline: "
                f"{dataset_name!r}. Supported: ScienceQA, MMMU, mini-imagenet, "
                "imagenet_short_text, food101, sun397, caltech101, "
                "oxford_flowers102, places365, dtd."
            )

        # Disable image decoding for all init loops — images are ZIP-backed on CephFS
        # and decoding each costs ~0.3s (open zip + extract + decode JPEG). None of
        # the loops below need the actual pixel data; they only read label/metadata fields.
        from datasets import Image as HFImage
        _has_img = "image" in self.dataset.column_names
        if _has_img:
            self.dataset = self.dataset.cast_column("image", HFImage(decode=False))

        # Compute labels before any subsampling (needed to stratify).
        pre_labels = [self._extract_label(instance) for instance in self.dataset]

        # Seeded class-stratified subsample (disjoint from *_filter modes).
        if subsample_size is not None:
            from .subsample import stratified_indices
            keep = stratified_indices(pre_labels, subsample_size, seed=seed)
            self.dataset = self.dataset.select(keep)
            self.labels = [pre_labels[i] for i in keep]
        else:
            self.labels = pre_labels

        # _within_class_idx on the final (post-subsample) dataset.
        self._within_class_idx = self._build_within_class_indices(self.labels)

        # _option_letters: cycle A..chr(A+k-1) per class via (within_class_idx % k).
        self._option_letters: list[str | None] = []
        for i, instance in enumerate(self.dataset):
            if self.prompt_mode == "mcq":
                wnid = self._class_names[instance["label"]]
                cat_size = len(
                    self._category_classes[self._mcq_catalog[wnid]["question_category"]]
                )
                k = min(cat_size, self.n_choices)
            elif self.prompt_mode in MCQ_FREE_MODES:
                k = self.n_choices
            else:
                self._option_letters.append(None)
                continue
            pos = self._within_class_idx[i] % k
            self._option_letters.append(chr(ord("A") + pos))

        # Remap public labels to option letters if requested.
        # Stratification above always used class-name labels (correct); this only
        # affects the externally visible self.labels after all data shaping is done.
        if label_mode == "option_letter":
            self.labels = [
                letter if letter is not None else self.labels[i]
                for i, letter in enumerate(self._option_letters)
            ]

        # max_samples: last-resort prefix cap (debug). Applied after stratified sample.
        if max_samples is not None and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
            self.labels = self.labels[:max_samples]
            self._within_class_idx = self._within_class_idx[:max_samples]
            self._option_letters = self._option_letters[:max_samples]

        self._describe_starts: list[str] = []
        if (
            self.prompt_mode == "describe"
            and (
                "mini-imagenet" in self.dataset_name_lower
                or "imagenet_short_text" in self.dataset_name_lower
            )
        ):
            self._describe_starts = _balanced_describe_starts(len(self.dataset), seed)

        self.sample_ids = [
            self._extract_sample_id(index=idx, instance=instance)
            for idx, instance in enumerate(self.dataset)
        ]

        self.prompt_signature = self._compute_prompt_signature()

        # Re-enable image decoding so __getitem__ returns PIL Images for model inference.
        if _has_img:
            self.dataset = self.dataset.cast_column("image", HFImage(decode=True))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, str | Image.Image]:
        instance = self.dataset[idx]
        image = self._extract_image(instance)
        prompt, option_letter = self._build_prompt(instance, idx)
        return {
            "question": prompt,
            "image": image,
            "label": self.labels[idx],
            "option_letter": option_letter,
            "_raw": dict(instance),
        }

    @staticmethod
    def _build_within_class_indices(labels: list[str]) -> list[int]:
        class_seen: dict[str, int] = {}
        within_class_idx: list[int] = []
        for label in labels:
            n = class_seen.get(label, 0)
            within_class_idx.append(n)
            class_seen[label] = n + 1
        return within_class_idx

    def _generic_label(self, instance: dict) -> str:
        return self._class_names_list[instance[self._label_col]]

    def _generic_labels(self, dataset: datasets.Dataset) -> list[str]:
        from datasets import Image as HFImage
        _ds = dataset.cast_column("image", HFImage(decode=False)) if "image" in dataset.column_names else dataset
        return [self._generic_label(instance) for instance in _ds]

    @staticmethod
    def _filter_no_decode(
        dataset: datasets.Dataset,
        fn: Any,
        *,
        num_proc: int = 8,
    ) -> datasets.Dataset:
        """Filter without decoding Image columns (avoids ZIP/network I/O for label-only filters)."""
        from datasets import Image as HFImage
        has_image = "image" in dataset.column_names
        if has_image:
            dataset = dataset.cast_column("image", HFImage(decode=False))
        result = dataset.filter(fn, num_proc=num_proc)
        if has_image:
            result = result.cast_column("image", HFImage(decode=True))
        return result

    def _prepare_generic_classification_dataset(
        self,
        *,
        dataset: datasets.Dataset,
        dataset_name: str,
        prompt_mode: str,
        n_choices: int,
        seed: int,
    ) -> datasets.Dataset:
        """Apply generic-dataset shaping before prompt construction.

        ``generic_mcq`` now has dataset-specific default semantics:
        - Food101: keep all 101 classes, cap to 100 samples per class.
        - SUN397: keep a seeded subset of 200 classes, all 50 samples each.
        - Caltech101: keep the full imbalanced train split.

        ``generic_mcq_filter`` and ``generic_mcq_top_filter`` keep their existing
        behavior; this helper simply centralizes the shared class-pool setup.
        """
        self._all_generic_classes: list[str] = sorted(self._class_names_list)
        self._generic_pool_classes: list[str] = list(self._all_generic_classes)
        self._generic_selected: set[str] = set()

        if prompt_mode == "generic_mcq":
            if "food101" in self.dataset_name_lower:
                from .subsample import stratified_indices

                raw_labels = self._generic_labels(dataset)
                target_size = 100 * len(self._all_generic_classes)
                keep = stratified_indices(raw_labels, target_size, seed=seed)
                dataset = dataset.select(keep)
                self._generic_pool_classes = sorted(set(raw_labels[i] for i in keep))
            elif "sun397" in self.dataset_name_lower:
                if len(self._all_generic_classes) < 200:
                    raise ValueError(
                        f"generic_mcq for {dataset_name!r} requires at least 200 classes, "
                        f"but only {len(self._all_generic_classes)} were found."
                    )
                rng = random.Random(seed)
                self._generic_pool_classes = sorted(
                    rng.sample(self._all_generic_classes, 200)
                )
                selected = set(self._generic_pool_classes)
                dataset = self._filter_no_decode(dataset, lambda x: self._generic_label(x) in selected)
            return dataset

        if prompt_mode == "generic_mcq_filter":
            if len(self._all_generic_classes) < n_choices:
                raise ValueError(
                    f"generic_mcq_filter requires at least n_choices={n_choices} classes "
                    f"in {dataset_name!r}, but only {len(self._all_generic_classes)} found."
                )
            rng = random.Random(seed)
            self._generic_selected = set(
                rng.sample(self._all_generic_classes, n_choices)
            )
            self._generic_pool_classes = sorted(self._generic_selected)
            return self._filter_no_decode(dataset, lambda x: self._generic_label(x) in self._generic_selected)

        if prompt_mode == "generic_mcq_top_filter":
            from .subsample import stratified_indices

            raw_labels = self._generic_labels(dataset)
            counts = Counter(raw_labels)
            top_n = sorted(
                counts,
                key=lambda class_name: (-counts[class_name], class_name),
            )[:n_choices]
            if len(top_n) < n_choices:
                raise ValueError(
                    f"generic_mcq_top_filter requires at least n_choices={n_choices} classes "
                    f"in {dataset_name!r}, but only {len(top_n)} found."
                )
            min_count = min(counts[class_name] for class_name in top_n)
            self._generic_selected = set(top_n)
            self._generic_pool_classes = sorted(self._generic_selected)
            dataset = self._filter_no_decode(dataset, lambda x: self._generic_label(x) in self._generic_selected)
            filtered_labels = self._generic_labels(dataset)
            keep = stratified_indices(
                filtered_labels,
                n_choices * min_count,
                seed=seed,
            )
            return dataset.select(keep)

        return dataset

    def _extract_image(self, instance: dict) -> Image.Image | None:
        image = instance.get("image_1") if "mmmu" in self.dataset_name_lower else instance["image"]
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return image

    @staticmethod
    def _extract_mmmu_subject(example_id: str) -> str:
        return "_".join(example_id.split("_")[1:-1])

    def _extract_label(self, instance: dict) -> str:
        if "scienceqa" in self.dataset_name_lower:
            return instance["topic"]
        if "mini-imagenet" in self.dataset_name_lower:
            names = getattr(self, "_class_names", None)
            wnid = names[instance["label"]] if names is not None else str(instance["label"])
            return self._mcq_catalog[wnid]["class_name"]   # always class_name
        if "imagenet_short_text" in self.dataset_name_lower:
            return self._imagenet_offset_to_class[instance["offset"]]
        if any(ds in self.dataset_name_lower for ds in _GENERIC_DATASETS):
            return self._class_names_list[instance[self._label_col]]
        if self.label_granularity == "detailed":
            return str(instance.get("subfield", "")).strip()
        return self._extract_mmmu_subject(instance["id"])

    def _extract_sample_id(self, index: int, instance: dict) -> str:
        if "id" in instance:
            return str(instance["id"])
        if "pid" in instance:
            return str(instance["pid"])
        question = str(instance.get("question", ""))
        label = self._extract_label(instance)
        return f"{self.dataset_name_lower}:{index}:{label}:{question}"

    def _build_prompt(self, instance: dict, idx: int) -> tuple[str, str | None]:
        if "scienceqa" in self.dataset_name_lower:
            context_text = instance.get("hint", None)
            context_text = "" if context_text == "nan" else context_text
            choices_list = instance["choices"]
            prompt = ""
            if context_text:
                prompt += f"Context: {context_text} "
            if choices_list is not None:
                last_choice = choices_list[-1]
                for letter_ord, choice_text in zip(range(ord("A"), ord("Z") + 1), choices_list):
                    prompt += f"{chr(letter_ord)}. {choice_text}"
                    prompt += "\n" if choice_text != last_choice else ""
            return prompt, None
        if "mini-imagenet" in self.dataset_name_lower:
            if self.prompt_mode == "mcq":
                return self._build_mini_imagenet_category_mcq(instance, idx)
            if self.prompt_mode in ("mcq_nocat", "mcq_nocat_filter"):
                return self._build_mini_imagenet_free_mcq(instance, idx)
            if self.prompt_mode == "category_id":
                return self._build_mini_imagenet_category_id_prompt(instance), None
            if self.prompt_mode == "describe":
                return _format_describe_prompt(self._describe_starts[idx]), None
            return self._mini_imagenet_question, None  # continuation / identify
        if "imagenet_short_text" in self.dataset_name_lower:
            if self.prompt_mode in ("animal_mcq", "animal_mcq_filter"):
                return self._build_animal_mcq(instance, idx)
            if self.prompt_mode == "describe":
                return _format_describe_prompt(self._describe_starts[idx]), None
            return self._imagenet_animal_question, None  # continuation / animal_id
        if any(ds in self.dataset_name_lower for ds in _GENERIC_DATASETS):
            if self.prompt_mode in ("generic_mcq", "generic_mcq_filter", "generic_mcq_top_filter"):
                return self._build_generic_mcq(instance, idx)
            return self._generic_mcq_question, None
        return _format_mmmu_question(instance), None

    def _compute_prompt_signature(self) -> str:
        prompts = [
            self._build_prompt(instance, idx)[0]
            for idx, instance in enumerate(self.dataset)
        ]
        return hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()

    def _build_mini_imagenet_category_mcq(self, instance: dict, idx: int) -> tuple[str, str]:
        """Category-based MCQ: distractors drawn from the correct class's category."""
        from .mcq_utils import build_mcq_prompt
        names = self._class_names
        wnid = names[instance["label"]] if names is not None else str(instance["label"])
        entry = self._mcq_catalog[wnid]
        category = entry["question_category"]
        correct = entry["class_name"]
        pool = [c for c in self._category_classes[category] if c != correct]
        k = min(len(pool) + 1, self.n_choices)
        correct_pos = self._within_class_idx[idx] % k
        return build_mcq_prompt(
            question=f"What {category} is shown in this image?",
            correct=correct,
            pool=pool,
            n_choices=k,
            correct_pos=correct_pos,
            shuffle_seed=wnid,
        )

    def _build_mini_imagenet_free_mcq(self, instance: dict, idx: int) -> tuple[str, str]:
        """Category-free MCQ for mini-imagenet.

        ``mcq_nocat``         -> pool is all catalog class names (minus correct).
        ``mcq_nocat_filter``  -> pool is the ``n_choices`` filtered classes (minus correct).
        """
        from .mcq_utils import build_mcq_prompt
        names = self._class_names
        wnid = names[instance["label"]] if names is not None else str(instance["label"])
        correct = self._mcq_catalog[wnid]["class_name"]
        if self.prompt_mode == "mcq_nocat_filter":
            pool = [c for c in sorted(self._mcq_selected) if c != correct]
        else:
            pool = [c for c in self._all_catalog_classes if c != correct]
        correct_pos = self._within_class_idx[idx] % self.n_choices
        return build_mcq_prompt(
            question="What is shown in this image?",
            correct=correct,
            pool=pool,
            n_choices=self.n_choices,
            correct_pos=correct_pos,
            shuffle_seed=wnid,
        )

    def _build_mini_imagenet_category_id_prompt(self, instance: dict) -> str:
        names = self._class_names
        wnid = names[instance["label"]] if names is not None else str(instance["label"])
        category = self._mcq_catalog[wnid]["question_category"]
        return f"Provide the specific name of the {category} shown.\nOutput exactly one word."

    def _build_animal_mcq(self, instance: dict, idx: int) -> tuple[str, str]:
        """Category-free MCQ for imagenet_short_text.

        ``animal_mcq``         -> pool is all 100 class names (minus correct).
        ``animal_mcq_filter``  -> pool is the ``n_choices`` filtered classes (minus correct).
        """
        from .mcq_utils import build_mcq_prompt
        offset = instance["offset"]
        correct = self._imagenet_offset_to_class[offset]
        if self.prompt_mode == "animal_mcq_filter":
            pool = [c for c in sorted(self._animal_selected) if c != correct]
        else:
            pool = [c for c in self._imagenet_all_classes if c != correct]
        correct_pos = self._within_class_idx[idx] % self.n_choices
        return build_mcq_prompt(
            question="What animal is shown in this image?",
            correct=correct,
            pool=pool,
            n_choices=self.n_choices,
            correct_pos=correct_pos,
            shuffle_seed=offset,
            letter_sep=": ",  # preserved from prior animal_mcq formatting
        )

    def _build_generic_mcq(self, instance: dict, idx: int) -> tuple[str, str]:
        """Category-free MCQ for generic HF classification datasets (food101, sun397, caltech101).

        ``generic_mcq``            -> pool is the shaped generic-MCQ class set (minus correct).
        ``generic_mcq_filter``     -> pool is the N randomly selected classes (minus correct).
        ``generic_mcq_top_filter`` -> pool is the top-N most-populated classes (minus correct).
        """
        from .mcq_utils import build_mcq_prompt
        correct = self._class_names_list[instance["label"]]
        if self.prompt_mode in ("generic_mcq_filter", "generic_mcq_top_filter"):
            pool = [c for c in sorted(self._generic_selected) if c != correct]
        else:
            pool = [c for c in self._generic_pool_classes if c != correct]
        correct_pos = self._within_class_idx[idx] % self.n_choices
        return build_mcq_prompt(
            question=self._generic_mcq_question,
            correct=correct,
            pool=pool,
            n_choices=self.n_choices,
            correct_pos=correct_pos,
            shuffle_seed=correct,
        )
