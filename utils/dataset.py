import base64
import hashlib
import json
import os
import re
from functools import partial
from io import BytesIO
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
    SHORT_CAPTION_PROMPT,
    SQA_ANSWER_CHOICES,
    SYSTEM_ROLE,
    POPE_ANSWER_CHOICES,
    AI2D_ANSWER_CHOICES,
)
from .internvl_utils import apply_chat_template as apply_internvl_chat_template
from .internvl_utils import get_pixel_values
from .model_utils import ProcessorType

MULTIPLE_CHOICE_BENCHMARKS = [
    "mmlu",
    "scienceqa",
    "mmbench",
    "mme",
    "ai2d",
    "pope",
    "seed",
]


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
        seed: int = 42,
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
            # Flat schema (GQA, TextVQA)
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
    """
    Dataset for captioning datasets.
    Unnecessary to load it as a class but used for simplicity.

    Args:
        dataset_path_or_name: Path to the dataset directory or Hugging Face dataset identifier.
        downsample_size: Maximum number of samples to use
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset_path_or_name: str,
        downsample_size: Optional[int] = None,
        seed: int = 42,
        concatenate_captions_and_add_prompt: bool = False,
        concatenate_captions: bool = False,
    ):
        dataset_dir = os.path.expanduser(dataset_path_or_name)
        self.concatenate_captions_and_add_prompt = concatenate_captions_and_add_prompt
        self.concatenate_captions = concatenate_captions

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        try:
            # Load the initial dataset
            dataset = load_from_disk(dataset_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {dataset_dir}: {e}")

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
        data = self.dataset[idx]
        captions = data["captions"]
        prompt = ""
        if self.concatenate_captions_and_add_prompt:
            prompt = f"Below are {len(captions)} numbered descriptions of the image. Please generate an additional description of the image. The caption should be a single sentence, not multiple sentences!!!\n\n"
            for idx, caption in zip(range(len(captions)), captions):
                prompt += f"{idx + 1}: {caption}\n"
        elif self.concatenate_captions:
            prompt = " ".join(captions)
        else:
            prompt = SHORT_CAPTION_PROMPT

        sample = {
            # for simplicity I add the prompt here
            "user_prompt": prompt,
            "captions": captions,
            "image_id": extract_image_id(data["image_id"]),
            "image": data["image"],
        }

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
    """Return True if the processor's chat template injects a system message by default.

    Probes by rendering a minimal user-only conversation and checking whether
    the word "system" appears in the output.  Result is cached per processor
    class so the probe runs at most once per type.
    """
    cls_name = processor.__class__.__name__
    if cls_name in _DEFAULT_SYSTEM_CACHE:
        return _DEFAULT_SYSTEM_CACHE[cls_name]
    tokenizer = getattr(processor, "tokenizer", processor)
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if apply_fn is None:
        _DEFAULT_SYSTEM_CACHE[cls_name] = False
        return False
    try:
        probe = apply_fn(
            [{"role": "user", "content": [{"type": "text", "text": "?"}]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        result = "system" in probe.lower()
    except Exception:
        result = False
    _DEFAULT_SYSTEM_CACHE[cls_name] = result
    return result


def _build_llava_prompts(
    questions: List[str],
    images: List[Optional[Image.Image]],
    processor: Any,
    guide_text: str = "",
) -> List[str]:
    """Format questions + (optional) images into chat-template strings for LLaVA models.

    Injects SYSTEM_ROLE only when the processor has no built-in default system
    message (LLaVA-1.5).  LLaVA-OneVision's Qwen2 tokenizer already includes
    one, so no injection is needed there.
    """
    include_system = not _processor_has_default_system(processor)
    if not images:
        images = [None] * len(questions)
    conversations = []
    for question, image in zip(questions, images):
        content: List[Dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image"})
        text = f"{question}\n{guide_text}\n" if guide_text else question
        content.append({"type": "text", "text": text})
        conv: List[Dict[str, Any]] = [{"role": "user", "content": content}]
        if include_system:
            conv.insert(0, SYSTEM_ROLE)
        conversations.append(conv)
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
    return_captions=False,
    return_image_ids=False,
    return_images=False,
    model: Optional[PreTrainedModel] = None,
):
    # Sanity check
    if return_image_ids and not return_captions:
        raise ValueError(
            "return_image_ids can only be True if return_captions is also True"
        )

    # Custom collate function to handle images and text
    user_prompts = [item["user_prompt"] for item in batch]
    captions = [item["captions"] for item in batch]
    images = [item["image"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    if processor is not None:
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
            processor_kwargs["images"] = images

        elif model and "intern" in model.__class__.__name__.lower():
            pixel_values, num_patches_list = get_pixel_values(images)
            queries = apply_internvl_chat_template(
                model=model,
                text=user_prompts,
                num_patches_list=num_patches_list,
            )
            processor_kwargs["text"] = queries

        elif model and is_cambrian_model(model=model):
            prepared = prepare_cambrian_batch_inputs(
                model=model,
                tokenizer=processor,
                questions=user_prompts,
                images=images,
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
            result.append(pixel_values)
        if model and is_cambrian_model(model=model):
            result.extend([image_tensors, image_sizes, prompts])
        if return_captions:
            result.append(captions)
        if return_image_ids:
            result.append(image_ids)
        if return_images:
            result.append(images)

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
    seed: int = 42,
    processor: Optional[ProcessorType] = None,
    return_captions: bool = False,
    return_image_ids: bool = False,
    return_images: bool = False,
    concatenate_captions_and_add_prompt: bool = False,
    concatenate_captions: bool = False,
    model: Optional[PreTrainedModel] = None,
):
    dataset = ImageCaptioningDataset(
        dataset_path_or_name=dataset_path_or_name,
        downsample_size=downsample_size,
        seed=seed,
        concatenate_captions_and_add_prompt=concatenate_captions_and_add_prompt,
        concatenate_captions=concatenate_captions,
    )

    collate_fn = partial(
        collate_captioning_batch,
        processor=processor,
        return_captions=return_captions,
        return_image_ids=return_image_ids,
        return_images=return_images,
        model=model,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
