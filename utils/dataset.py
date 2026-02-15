import argparse
import base64
import hashlib
import os
import re
from functools import partial
from io import BytesIO
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel

from .constants import (
    ANSWER_TEXT,
    ASSISTANT_ROLE,
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
    ):
        self.dataset_path_or_name = dataset_path_or_name.lower()

        if (
            "scienceqa" not in dataset_path_or_name.lower()
            and "mmlu" not in dataset_path_or_name.lower()
            and "mmbench" not in dataset_path_or_name.lower()
            and "mme" not in dataset_path_or_name.lower()
            and "ai2d" not in dataset_path_or_name.lower()
            and "pope" not in dataset_path_or_name.lower()
        ):
            raise ValueError(
                f"Dataset {dataset_path_or_name} is not supported. "
                "Only 'scienceqa', 'mmlu', 'mmbench', 'ai2d', 'pope', and 'mme' datasets are supported."
            )

        if texts_qa and images_qa:
            raise ValueError("texts_qa and images_qa cannot be both True.")

        self.question_instruction_type = question_instruction_type
        self.texts_qa = texts_qa
        self.images_qa = images_qa

        # Load the initial dataset
        dataset = get_hf_dataset_split(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split=split,
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

        mm_dataset = "image" in dataset.features

        # Apply filters if not mm_dataset mode
        if mm_dataset:
            if images_qa:
                dataset = dataset.filter(lambda example: example["image"] is not None)
            if texts_qa:
                # If using texts_qa, we need captions.
                # If the dataset was patched, it should have a 'caption' field.
                if "caption" not in dataset.features:
                    raise ValueError(
                        f"Dataset {dataset_path_or_name} does not have a 'caption' field required for texts_qa=True."
                    )

                # We do NOT filter out images here because the dataset technically has images,
                # we just choose not to return them in __getitem__.
                # However, we should ensure the caption is not None.
                dataset = dataset.filter(lambda example: example["caption"] is not None)

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question_data = self.dataset[idx]
        question_text = question_data["question"]

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

        context_text = question_data.get("hint", None)
        context_text = "" if context_text == "nan" else context_text
        subject = question_data.get("subject", question_data.get("category", None))

        def format_subject(subject):
            return subject.replace("_", " ")

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
            # A/B/C/D for MMBench
            answer_letter = question_data["answer"]

        full_question_prompt = ""

        # Add caption context if texts_qa is True
        if self.texts_qa:
            caption = question_data.get("caption")
            if caption:
                full_question_prompt += f"Below is a description of an image. Use it to answer the question.\n\n{caption}\n\n"
            else:
                # Should have been filtered out, but just in case
                pass

        if subject is not None and "mmlu" in self.dataset_path_or_name:
            if self.question_instruction_type == "singular":
                full_question_prompt += f"The following is a multiple choice question (with answers) about {format_subject(subject)}.\n\n"
            elif self.question_instruction_type == "plural":
                full_question_prompt += f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"

        if context_text is not None and context_text != "":
            if "scienceqa" in self.dataset_path_or_name:
                full_question_prompt += "Context: "
            full_question_prompt += f"{context_text} "
        full_question_prompt += f"{question_text} "

        if (
            "mme" not in self.dataset_path_or_name
            and "pope" not in self.dataset_path_or_name
        ):
            if "mmbench" in self.dataset_path_or_name:
                full_question_prompt += "There are several options:\n"
            else:
                full_question_prompt += "\n"

            if choices_list is not None:
                last_choice = choices_list[-1]
                for letter_ord, choice_text in zip(
                    range(ord("A"), ord("Z") + 1), choices_list
                ):
                    full_question_prompt += f"{chr(letter_ord)}: {choice_text}"
                    full_question_prompt += "\n" if choice_text != last_choice else ""

        sample = {
            "question": full_question_prompt,
            "answer_letter": answer_letter,
        }

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

        # Apply downsampling if requested and if dataset is larger than requested size
        if downsample_size and downsample_size < len(dataset):
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
                c for c in ["captions", "caption", "synthetic-caption"] if c in dataset.column_names
            ]
            if cols_to_remove:
                dataset = dataset.remove_columns(cols_to_remove)
        elif texts_qa:
            dataset = dataset.remove_columns("image")
            # Remove unused caption columns based on caption_source selection
            if caption_source == "synthetic":
                cols_to_remove = [c for c in ["captions"] if c in dataset.column_names]
            else:
                cols_to_remove = [c for c in ["synthetic-caption"] if c in dataset.column_names]
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
        full_question_prompt = ""

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

        # Handle Captions
        if self.texts_qa:
            if self.caption_source == "synthetic":
                # Use synthetic-caption (single string)
                caption = question_data["synthetic-caption"]
                full_question_prompt += f"Below is a description of the image. Use it to answer the question.\n\n{caption}\n"
            elif "captions" in question_data:
                captions = question_data["captions"]
                if isinstance(captions, list):
                    full_question_prompt += f"Below are {len(captions)} descriptions of an image. Use them to answer the question.\n\n"
                    for i, caption in enumerate(captions):
                        full_question_prompt += f"{i + 1}: {caption}\n"
                else:
                    # Fallback if it's somehow a string
                    full_question_prompt += f"Below is a description of the image. Use it to answer the question.\n\n{captions}\n"
            elif "caption" in question_data:
                caption = question_data["caption"]
                full_question_prompt += f"Below is a description of the image. Use it to answer the question.\n\n{caption}\n"
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


def format_prompts(
    questions: List[str],
    args: argparse.Namespace,
    processor: ProcessorType,
    images: Optional[List[Image.Image]] = None,
    chat_template_exists: bool = False,
) -> List[str]:
    """
    Format input questions and images into structured prompts for a model.

    Parameters:
        questions: List of question and options in string format.
        images: List of images associated with each question.
        args: Arguments namespace with a `chat_mode` boolean attribute.
        processor: A processor used to tokenize and encode text+image inputs.
        chat_template_exists: Whether the chat template exists.
    Returns:
        list: Formatted prompts either as conversations (dict format) or plain strings.
    """
    if args.chat_mode and chat_template_exists:
        conversations_list = _format_as_conversations(questions, images, args)
        kwargs_chat_template = {
            "conversation": conversations_list,
            "tokenize": False,
        }
        if args.continue_final_message:
            kwargs_chat_template["continue_final_message"] = True
            kwargs_chat_template["add_generation_prompt"] = False
        else:
            kwargs_chat_template["add_generation_prompt"] = True
        # Apply the chat template to format the conversations
        return processor.apply_chat_template(**kwargs_chat_template)

    elif args.chat_mode and not chat_template_exists:
        # for simplicity I use these parameters for the custom prompt
        return _format_as_custom_prompts_text_image(questions, images, args)
    elif not args.chat_mode:
        return _format_as_plain_prompts_forward_pass(
            questions, images, args.image_first
        )
    else:
        return _format_as_plain_prompts(questions, images)


def _format_as_conversations(questions, images, args):
    formatted_conversations = []
    continue_final_message = (
        hasattr(args, "continue_final_message") and args.continue_final_message
    )
    guide_text: str = ""
    if hasattr(args, "guide_text") and args.guide_text:
        guide_text = f" {args.guide_text}"

    if not images:
        images = [None] * len(questions)
    for question, image in zip(questions, images):
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": f"{question}\n{guide_text}\n"})

        conversation = [{"role": "user", "content": content}]
        conversation.insert(0, SYSTEM_ROLE)
        if continue_final_message:
            conversation.append(ASSISTANT_ROLE)
        formatted_conversations.append(conversation)
    return formatted_conversations


def _format_as_plain_prompts(questions, images):
    formatted_prompts = []
    for image, question in zip(images, questions):
        prompt_prefix = "<image>\n" if image is not None else ""
        prompt = f"{prompt_prefix}{question}{ANSWER_TEXT}"
        formatted_prompts.append(prompt)
    return formatted_prompts


def _format_as_custom_prompts_text_image(questions, images, args):
    system_message = SYSTEM_ROLE["content"][0]["text"]

    continue_final_message = (
        hasattr(args, "continue_final_message") and args.continue_final_message
    )
    formatted_prompts = []
    for image, question_text in zip(images, questions):
        prompt_parts = []

        # System message
        prompt_parts.append(f"{system_message}")

        # User message with image token after text
        user_message = f"USER: {question_text}"
        user_message += "\n<image>\n" if image is not None else ""
        prompt_parts.append(user_message)

        # Assistant prefix (optional)
        if continue_final_message:
            prompt_parts.append("ASSISTANT: Answer:")
        else:
            prompt_parts.append("ASSISTANT:")

        # Join with newlines
        prompt = "".join(prompt_parts)

        formatted_prompts.append(prompt)
    return formatted_prompts


def _format_as_plain_prompts_forward_pass(questions, images, image_first: bool = True):
    formatted_prompts = []
    for image, question_text in zip(images, questions):
        image_text = "<image>" if image is not None else ""
        if image_first:
            prompt = f"{image_text} {question_text}"
        else:
            prompt = f"{question_text} {image_text}"
        formatted_prompts.append(prompt)
    return formatted_prompts


def collate_vqa_batch(
    batch,
    processor,
    guide_text,
    answer_letters_with_processed_batch,
    chat_template_exists=True,
    chat_mode=True,
    model: Optional[PreTrainedModel] = None,
):
    # Custom collate function to handle images and text
    questions = [item["question"] for item in batch]
    answer_letters = [item["answer_letter"] for item in batch]
    images = [item.get("image", None) for item in batch]
    indices = [item.get("indices", None) for item in batch]
    image_ids = [item.get("image_id", None) for item in batch]

    if processor is not None:
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": (True if len(questions) > 1 else False),
        }

        if "llava" in processor.__class__.__name__.lower():
            # For simplicity, define args for backward code compatibility
            # todo: redesign the interface if used in the future
            args = argparse.Namespace(
                chat_mode=chat_mode,
                continue_final_message=False,
                guide_text=guide_text,
                # !"enforced" but consistent for these experiments
            )

            prompts = format_prompts(
                questions=questions,
                images=images,
                args=args,
                processor=processor,
                chat_template_exists=chat_template_exists,
                # !"enforced" but consistent for these experiments
            )
            processor_kwargs["text"] = prompts

            # todo: improve this, it is not neat
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

        tokenized = processor(**processor_kwargs)

        outputs = (tokenized,)
        if model and "intern" in model.__class__.__name__.lower():
            if pixel_values is not None:
                outputs += (pixel_values, processor_kwargs["text"])
            else:
                outputs += (processor_kwargs["text"],)
        if answer_letters_with_processed_batch:
            # !not so neat but it's a backward compatibility feature
            return outputs + (answer_letters,)
        return outputs

    # Raw mode (default)
    return {
        "questions": questions,
        "answer_letters": answer_letters,
        "images": images,
        "indices": indices,
        "image_ids": image_ids,
    }


def collate_captioning_batch(
    batch,
    processor,
    chat_template_exists=True,
    return_captions=False,
    return_image_ids=False,
    return_images=False,
    chat_mode=True,
    image_first=True,
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

        if "llava" in processor.__class__.__name__.lower():
            # For simplicity, define args for backward code compatibility
            # todo: redesign the interface if used in the future
            args = argparse.Namespace(
                chat_mode=chat_mode,
                continue_final_message=False,
                # !"enforced" but consistent for these experiments
                image_first=image_first,
            )

            prompts = format_prompts(
                # todo: rename questions to user_prompts
                # this is a backward compatibility feature
                questions=user_prompts,
                images=images,
                args=args,
                processor=processor,
                chat_template_exists=chat_template_exists,
            )

            processor_kwargs["text"] = prompts
            processor_kwargs["images"] = images

        elif model and "intern" in model.__class__.__name__.lower():
            pixel_values, num_patches_list = get_pixel_values(images)
            queries = apply_internvl_chat_template(
                model=model,
                text=user_prompts,
                num_patches_list=num_patches_list,
            )
            processor_kwargs["text"] = queries

        tokenized = processor(**processor_kwargs)
        # Build return tuple based on requested data
        result = [tokenized]
        if model and "intern" in model.__class__.__name__.lower():
            result.append(pixel_values)
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
    chat_template_exists: bool = True,
    caption_source: Optional[str] = None,
    remove_yes_no: bool = False,
    filter_mmbench_4_choices: bool = False,
    model: Optional[PreTrainedModel] = None,
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
        chat_template_exists=chat_template_exists,
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


def get_dataloader_for_captioning(
    dataset_path_or_name: str,
    batch_size: int = 1,
    num_workers: int = 4,
    downsample_size: Optional[int] = None,
    seed: int = 42,
    processor: Optional[ProcessorType] = None,
    chat_template_exists: bool = True,
    return_captions: bool = False,
    return_image_ids: bool = False,
    return_images: bool = False,
    concatenate_captions_and_add_prompt: bool = False,
    concatenate_captions: bool = False,
    chat_mode: bool = True,
    image_first: bool = True,
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
        chat_template_exists=chat_template_exists,
        return_captions=return_captions,
        return_image_ids=return_image_ids,
        return_images=return_images,
        chat_mode=chat_mode,
        image_first=image_first,
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
