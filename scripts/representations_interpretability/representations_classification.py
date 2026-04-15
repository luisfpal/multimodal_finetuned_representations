import argparse
import ast
import gc
import hashlib
import json
import math
import os
import re
import time
import warnings
from collections import Counter
from functools import partial, wraps
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from dadapy import Data
from datasets import load_dataset
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    homogeneity_completeness_v_measure,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.residual_stream_tracer import residual_stream_tracer
from utils import load_hf_model_and_processor_or_tokenizer, seed_all
from utils.cambrian_utils import prepare_cambrian_batch_inputs, resolve_cambrian_conv_mode
from utils.constants import SYSTEM_ROLE
from utils.dataset import _build_llava_prompts, _processor_has_default_system
from utils.internvl_utils import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    apply_chat_template as internvl_apply_chat_template,
    get_pixel_values,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "flash_attention_2" if DEVICE.type == "cuda" else "sdpa"
HOME_DIR = Path.home()
RESULTS_BASE = HOME_DIR / "multimodal_finetuned_representations" / "results"
REPRESENTATIONS_BASE = HOME_DIR / "scratch"
MIN_TOPIC_SAMPLES = 70
MMMU_IMAGE_COLUMNS = [f"image_{index}" for index in range(1, 8)]
RESIDUAL_STREAM_TYPE = "output_layer"

SUPPORTED_METRICS = ("logistic_regression", "knn", "label_overlap", "homogeneity")
METRIC_TO_KEYS: dict[str, list[str]] = {
    "logistic_regression": ["f1_weighted", "f1_weighted_train", "mcc"],
    "knn": ["knn_f1_weighted", "knn_mcc"],
    "label_overlap": ["label_overlap"],
    "homogeneity": ["homogeneity_score", "adjusted_rand_score", "n_clusters_ratio"],
}


def format_time_smart(seconds: int) -> str:
    if seconds < 0:
        return "0s"

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def timed_metric(metric_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            layer_results = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            print(
                f"Metric {metric_name} completed in "
                f"{format_time_smart(int(elapsed_time))}."
            )
            return layer_results

        return wrapper

    return decorator


class ClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "derek-thomas/ScienceQA",
        split: str = "test",
        max_samples: int | None = None,
        label_granularity: str = "intermediate",
    ) -> None:
        dataset = load_dataset(dataset_name, split=split)
        self.dataset_name_lower = dataset_name.lower()
        self.label_granularity = label_granularity

        if "scienceqa" in self.dataset_name_lower:
            topics_counts = Counter(dataset["topic"])

            def filter_min_counts_and_image(example):
                return (
                    topics_counts[example["topic"]] >= MIN_TOPIC_SAMPLES
                    and example["image"] is not None
                )

            filtered_dataset = dataset.filter(filter_min_counts_and_image, num_proc=8)
            topics_counts_filtered = Counter(filtered_dataset["topic"])

            def filter_min_counts(example):
                return topics_counts_filtered[example["topic"]] >= MIN_TOPIC_SAMPLES

            self.dataset = filtered_dataset.filter(filter_min_counts, num_proc=8)
        elif "mmmu" in self.dataset_name_lower:
            # Layer 1: remove instances with >1 image
            self.dataset = dataset.filter(
                lambda example: sum(
                    example.get(column) is not None for column in MMMU_IMAGE_COLUMNS
                ) <= 1,
                num_proc=8,
            )
            # Layer 2: keep only multiple-choice
            self.dataset = self.dataset.filter(
                lambda example: str(example.get("question_type", "")).strip().lower()
                == "multiple-choice",
                num_proc=8,
            )
            # Layer 3 (detailed): keep only subfields with count >= mean count
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
            self.dataset = dataset
        else:
            raise ValueError(
                "Unsupported dataset for classification pipeline: "
                f"{dataset_name}. Supported: ScienceQA, MMMU, timm/mini-imagenet"
            )

        if max_samples is not None:
            self.dataset = self.dataset.select(
                range(min(max_samples, len(self.dataset)))
            )

        self.labels = [self._extract_label(instance) for instance in self.dataset]
        self.sample_ids = [
            self._extract_sample_id(index=idx, instance=instance)
            for idx, instance in enumerate(self.dataset)
        ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, str | Image.Image]:
        instance = self.dataset[idx]
        image = self._extract_image(instance)
        prompt = self._build_prompt(instance)

        return {
            "question": prompt,
            "image": image,
            "label": self._extract_label(instance),
            "_raw": dict(instance),  # needed for ICL collator
        }

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
            return names[instance["label"]] if names is not None else str(instance["label"])
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

    def _build_prompt(self, instance: dict) -> str:
        if "scienceqa" in self.dataset_name_lower:
            context_text = instance.get("hint", None)
            context_text = "" if context_text == "nan" else context_text
            choices_list = instance["choices"]
            prompt = ""
            if context_text:
                prompt += f"Context: {context_text} "
            if choices_list is not None:
                last_choice = choices_list[-1]
                for letter_ord, choice_text in zip(
                    range(ord("A"), ord("Z") + 1), choices_list
                ):
                    prompt += f"{chr(letter_ord)}. {choice_text}"
                    prompt += "\n" if choice_text != last_choice else ""
            return prompt

        if "mini-imagenet" in self.dataset_name_lower:
            return "Describe this image in detail."
            # return "In one sentence, describe the contents of the image."

        return _format_mmmu_question(instance)


# ---------------------------------------------------------------------------
# In-context learning helpers (MMMU only)
# ---------------------------------------------------------------------------

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


def _load_mmmu_dev_pool(label_granularity: str) -> dict[str, list[dict]]:
    """Load MMMU dev split, filter to single-image MC instances, group by category key.

    Groups by subject (intermediate) or subfield (detailed) — the same key used
    as the classification label — so shots always come from the same category
    as the test instance.
    """
    dev = load_dataset("lmms-lab/MMMU", split="dev")
    dev = dev.filter(
        lambda ex: (
            sum(ex.get(col) is not None for col in MMMU_IMAGE_COLUMNS) == 1
            and str(ex.get("question_type", "")).strip().lower() == "multiple-choice"
        ),
        num_proc=4,
    )
    pool: dict[str, list[dict]] = {}
    for instance in dev:
        if label_granularity == "detailed":
            key = str(instance.get("subfield", "")).strip()
        else:
            parts = str(instance["id"]).split("_")
            key = "_".join(parts[1:-1]) if len(parts) >= 3 else ""
        if key:
            pool.setdefault(key, []).append(dict(instance))
    return pool


def _build_icl_conversation(
    shots: list[dict],
    test_instance: dict,
    processor,
) -> tuple[list[dict], list[Image.Image]]:
    """Build a single-user-turn ICL conversation: N shots then the test query.

    Each shot contributes an image block + 'question\\nAnswer: X' text block.
    The test instance ends with '\\nAnswer:' so the model predicts the letter.
    """
    content: list[dict] = []
    images: list[Image.Image] = []

    for shot in shots:
        img = shot.get("image_1")
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        content.append({"type": "image"})
        content.append({
            "type": "text",
            "text": f"{_format_mmmu_question(shot)}\nAnswer: {shot['answer']}\n\n",
        })
        images.append(img)

    test_img = test_instance.get("image_1")
    if isinstance(test_img, Image.Image):
        test_img = test_img.convert("RGB")
    content.append({"type": "image"})
    content.append({"type": "text", "text": f"{_format_mmmu_question(test_instance)}\nAnswer:"})
    images.append(test_img)

    conv: list[dict] = [{"role": "user", "content": content}]
    if not _processor_has_default_system(processor):
        conv.insert(0, SYSTEM_ROLE)

    return conv, images


def _llava_zero_shot_collate_fn(
    batch: list[dict],
    processor,
    guide_text: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    texts = _build_llava_prompts(
        questions=[item["question"] for item in batch],
        images=[item["image"] for item in batch],
        processor=processor,
        guide_text=guide_text,
    )
    if prompt_log is not None:
        prompt_log.append(texts[0])
    return (
        processor(return_tensors="pt", text=texts, images=[item["image"] for item in batch]),
        [item["label"] for item in batch],
    )


def _icl_collate_fn(
    batch: list[dict],
    processor,
    shots_pool: dict[str, list[dict]],
    n_shots: int,
    guide_text: str,
    label_granularity: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    """Collator for ICL: builds multi-image conversation from dev shots + test instance.

    Falls back to zero-shot for instances whose category has no dev shots.
    """
    item = batch[0]  # batch_size=1 always
    label = item["label"]
    test_instance = item["_raw"]

    # Determine category key (must match the label used for classification)
    if label_granularity == "detailed":
        key = str(test_instance.get("subfield", "")).strip()
    else:
        parts = str(test_instance["id"]).split("_")
        key = "_".join(parts[1:-1]) if len(parts) >= 3 else ""

    shots = shots_pool.get(key, [])[:n_shots]

    if shots:
        conv, images = _build_icl_conversation(shots, test_instance, processor)
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        if prompt_log is not None:
            prompt_log.append(text)
        tokenized = processor(text=text, images=images, return_tensors="pt")
    else:
        # Zero-shot fallback for categories with no dev examples
        text = _build_llava_prompts(
            questions=[item["question"]],
            images=[item["image"]],
            processor=processor,
            guide_text=guide_text,
        )
        if prompt_log is not None:
            prompt_log.append(text[0])
        tokenized = processor(text=text, images=[item["image"]], return_tensors="pt")

    return tokenized, [label]


# ---------------------------------------------------------------------------
# Model-type detection
# ---------------------------------------------------------------------------


def _is_internvl(model_name: str) -> bool:
    return "intern" in model_name.lower()


def _is_cambrian(model_name: str) -> bool:
    return "cambrian" in model_name.lower()


# ---------------------------------------------------------------------------
# InternVL2 collate functions
# ---------------------------------------------------------------------------


def _build_internvl_icl_query(
    shots: list[dict],
    test: dict,
    model,
    tokenizer,
) -> tuple[dict, torch.Tensor, str]:
    """Build InternVL2 multi-image ICL prompt.

    Each shot and the test instance each get one <image> placeholder, which is
    replaced with its specific image tokens (patch count varies per image).

    Returns (tokenized_dict, pixel_values, query) where query is the fully
    formatted string with image tokens replaced (for logging purposes).
    """
    import copy as _copy

    all_images: list[Image.Image] = []
    for s in shots:
        img = s.get("image_1")
        all_images.append(img.convert("RGB") if isinstance(img, Image.Image) else img)
    test_img = test.get("image_1")
    all_images.append(test_img.convert("RGB") if isinstance(test_img, Image.Image) else test_img)

    pixel_values, num_patches_list = get_pixel_values(all_images)

    parts: list[str] = []
    for s in shots:
        parts.append(f"<image>\n{_format_mmmu_question(s)}\nAnswer: {s['answer']}\n\n")
    parts.append(f"<image>\n{_format_mmmu_question(test)}\nAnswer:")
    full_text = "".join(parts)

    template = _copy.deepcopy(model.conv_template)
    template.system_message = model.system_message
    template.messages = []
    template.append_message(template.roles[0], full_text)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for n_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * model.num_image_token * n_patches
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

    tokenized = tokenizer(query, return_tensors="pt", padding=False)
    return tokenized, pixel_values, query


def _internvl_zero_shot_collate_fn(
    batch: list[dict],
    model,
    tokenizer,
    guide_text: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    item = batch[0]
    img = item["image"]
    pixel_values, num_patches_list = get_pixel_values([img])
    query = internvl_apply_chat_template(model, [item["question"]], num_patches_list, guide_text)[0]
    if prompt_log is not None:
        prompt_log.append(query)
    tokenized = tokenizer(query, return_tensors="pt", padding=False)
    return tokenized, pixel_values, [item["label"]]


def _internvl_icl_collate_fn(
    batch: list[dict],
    model,
    tokenizer,
    shots_pool: dict[str, list[dict]],
    n_shots: int,
    label_granularity: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    item = batch[0]
    label = item["label"]
    test_instance = item["_raw"]

    if label_granularity == "detailed":
        key = str(test_instance.get("subfield", "")).strip()
    else:
        parts = str(test_instance["id"]).split("_")
        key = "_".join(parts[1:-1]) if len(parts) >= 3 else ""

    shots = shots_pool.get(key, [])[:n_shots]

    if shots:
        tokenized, pixel_values, query = _build_internvl_icl_query(shots, test_instance, model, tokenizer)
        if prompt_log is not None:
            prompt_log.append(query)
    else:
        # Zero-shot fallback
        img = item["image"]
        pixel_values, num_patches_list = get_pixel_values([img])
        query = internvl_apply_chat_template(model, [item["question"]], num_patches_list)[0]
        if prompt_log is not None:
            prompt_log.append(query)
        tokenized = tokenizer(query, return_tensors="pt", padding=False)

    return tokenized, pixel_values, [label]


# ---------------------------------------------------------------------------
# Cambrian collate functions
# ---------------------------------------------------------------------------


def _cambrian_zero_shot_collate_fn(
    batch: list[dict],
    model,
    tokenizer,
    guide_text: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    item = batch[0]
    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=[item["question"]],
        images=[item["image"]],
        guide_text=guide_text,
    )
    if prompt_log is not None:
        prompt_log.append(prepared["prompts"][0])
    return (
        prepared["tokenized"],
        prepared["image_tensors"],
        prepared["image_sizes"],
        prepared["prompts"],
        [item["label"]],
    )


def _cambrian_icl_collate_fn(
    batch: list[dict],
    model,
    tokenizer,
    shots_pool: dict[str, list[dict]],
    n_shots: int,
    label_granularity: str,
    guide_text: str,
    prompt_log: list[str] | None = None,
) -> tuple:
    """Cambrian ICL is NOT SUPPORTED — SVA is hardwired to single-image position.

    Cambrian's Spatial Vision Aggregator (SVA) cross-attention layer reads image
    features from a hardcoded position (config.image_position=91) at every 3rd
    decoder layer. This is by architectural design, not a bug.

    For proper ICL, we would need:
    1. N image tokens at variable positions
    2. SVA to track which position has the actual image
    3. Residual stream to reflect N separate visual inputs

    None of these are true. SVA always looks at position 91, regardless of where
    the image actually is. Text-only shots + single test image at position 91 is
    not true ICL — it's zero-shot with label context in text.

    Fallback: always use zero-shot, regardless of shots_pool.
    """
    item = batch[0]
    label = item["label"]

    # Always zero-shot, ignoring shots_pool for Cambrian
    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=[item["question"]],
        images=[item["image"]],
        guide_text=guide_text,
    )
    if prompt_log is not None:
        prompt_log.append(prepared["prompts"][0])
    return (
        prepared["tokenized"],
        prepared["image_tensors"],
        prepared["image_sizes"],
        prepared["prompts"],
        [label],
    )


def _build_sample_signature(sample_ids: list[str], labels: np.ndarray) -> str:
    if len(sample_ids) != len(labels):
        raise ValueError(
            f"Sample IDs and labels length mismatch: {len(sample_ids)} vs {len(labels)}"
        )
    joined = "\n".join(
        f"{sample_id}\t{label}" for sample_id, label in zip(sample_ids, labels)
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def get_representations_and_labels(
    model_name: str,
    dataset_name: str,
    split: str,
    guide_text: str,
    tokens_pooling_method: str,
    representation_cache_path: Path | None,
    force_recompute_representations: bool,
    save_representations: bool,
    sample_set_tag: str,
    label_granularity: str = "intermediate",
    max_samples: int | None = None,
    n_shots: int = 0,
    save_prompts_path: Path | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    # mini-imagenet: override dataset-specific parameters unconditionally so this
    # dataset can coexist with MMMU/ScienceQA in the same shell script run.
    if "mini-imagenet" in dataset_name.lower():
        if split != "validation":
            print(f"  [mini-imagenet] Overriding split '{split}' → 'validation'")
            split = "validation"
        guide_text = ""   # captioning prompt — no MCQ suffix
        n_shots = 0       # ICL not supported for non-MMMU datasets

    dataset = ClassificationDataset(
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        label_granularity=label_granularity,
    )
    labels = np.array(dataset.labels)
    sample_ids = list(dataset.sample_ids)
    sample_signature = _build_sample_signature(sample_ids=sample_ids, labels=labels)

    if (
        representation_cache_path is not None
        and representation_cache_path.exists()
        and not force_recompute_representations
    ):
        print(f"Loading cached representations: {representation_cache_path}")
        cached_representations: torch.Tensor | None = None
        metadata = {}
        with safe_open(str(representation_cache_path), framework="pt") as handle:
            metadata = handle.metadata() or {}
            if "residual_stream" in handle.keys():
                cached_representations = handle.get_tensor("residual_stream")

        if cached_representations is not None:
            cache_signature = metadata.get("sample_signature", "")
            cache_num_samples = metadata.get("num_samples", "")
            cache_pooling = metadata.get("tokens_pooling_method", "")
            cache_split = metadata.get("split", "")
            cache_sample_set = metadata.get("sample_set_tag", "")

            is_valid_cache = (
                cached_representations.shape[1] == len(labels)
                and cache_num_samples == str(len(labels))
                and cache_signature == sample_signature
                and cache_pooling == tokens_pooling_method
                and cache_split == split
                and cache_sample_set == sample_set_tag
            )

            if is_valid_cache:
                if save_prompts_path is not None:
                    print(
                        f"  [prompts] Representations loaded from cache — "
                        f"prompts were saved in the original run. "
                        f"Re-run with --force-recompute-representations to regenerate."
                    )
                return cached_representations, labels

            print(
                "Cached representations mismatch (shape/metadata/signature). Recomputing..."
            )

    # Captioning datasets don't use MCQ guide text
    if "mini-imagenet" in dataset_name.lower():
        guide_text = ""

    # Mutable list captured by collate closures to record formatted prompts.
    _prompt_log: list[str] = []

    # Load dev shots before model (no GPU needed; fail fast if unavailable)
    shots_pool: dict[str, list[dict]] | None = None
    if n_shots > 0 and "mmmu" in dataset_name.lower():
        print(f"Loading MMMU dev pool for {n_shots}-shot ICL ...")
        shots_pool = _load_mmmu_dev_pool(label_granularity)
        print(f"  Dev pool: {len(shots_pool)} categories, "
              f"{sum(len(v) for v in shots_pool.values())} instances.")

    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=model_name,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE,
    )

    if _is_internvl(model_name):
        if shots_pool is not None:
            collate_fn = partial(
                _internvl_icl_collate_fn,
                model=model,
                tokenizer=processor,
                shots_pool=shots_pool,
                n_shots=n_shots,
                label_granularity=label_granularity,
                prompt_log=_prompt_log,
            )
        else:
            collate_fn = partial(
                _internvl_zero_shot_collate_fn,
                model=model,
                tokenizer=processor,
                guide_text=guide_text,
                prompt_log=_prompt_log,
            )
    elif _is_cambrian(model_name):
        if shots_pool is not None:
            collate_fn = partial(
                _cambrian_icl_collate_fn,
                model=model,
                tokenizer=processor,
                shots_pool=shots_pool,
                n_shots=n_shots,
                label_granularity=label_granularity,
                guide_text=guide_text,
                prompt_log=_prompt_log,
            )
        else:
            collate_fn = partial(
                _cambrian_zero_shot_collate_fn,
                model=model,
                tokenizer=processor,
                guide_text=guide_text,
                prompt_log=_prompt_log,
            )
    elif shots_pool is not None:
        collate_fn = partial(
            _icl_collate_fn,
            processor=processor,
            shots_pool=shots_pool,
            n_shots=n_shots,
            guide_text=guide_text,
            label_granularity=label_granularity,
            prompt_log=_prompt_log,
        )
    else:
        collate_fn = partial(
            _llava_zero_shot_collate_fn,
            processor=processor,
            guide_text=guide_text,
            prompt_log=_prompt_log,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    representations = residual_stream_tracer(
        model=model,
        processed_dataloader=dataloader,
        processor=processor,
        return_dtype=torch.float32,
        tokens_pooling_method=tokens_pooling_method,
        residual_stream_type="output_layer",
        return_deepcopy=True,
        save_representations=False,
    )

    if not isinstance(representations, torch.Tensor):
        raise ValueError(
            "Expected tensor representations from residual_stream_tracer. "
            f"Got: {type(representations)}"
        )

    if representation_cache_path is not None and save_representations:
        os.makedirs(representation_cache_path.parent, exist_ok=True)
        save_file(
            {"residual_stream": representations.cpu().to(torch.float32)},
            str(representation_cache_path),
            metadata={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "split": split,
                "tokens_pooling_method": tokens_pooling_method,
                "num_samples": str(len(labels)),
                "sample_signature": sample_signature,
                "sample_set_tag": sample_set_tag,
            },
        )
        print(f"Saved representations cache: {representation_cache_path}")

    if save_prompts_path is not None and _prompt_log:
        records = [
            {
                "idx": i,
                "sample_id": sample_ids[i],
                "label": str(labels[i]),
                "formatted_prompt": prompt,
            }
            for i, prompt in enumerate(_prompt_log)
        ]
        save_prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_prompts_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Saved prompts: {save_prompts_path}")

    result = representations.cpu().to(torch.float32)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result, labels


def _remove_duplicate_rows(
    data: torch.Tensor,
    labels: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """Remove exact duplicate representation vectors before k-NN based metrics.

    Duplicate rows have zero pairwise distance, making them trivial nearest
    neighbours of each other. This inflates label-overlap scores in early layers
    where many token representations are identical (no attention mixing yet).
    """
    np_data = data.detach().cpu().to(torch.float32).numpy()
    _, unique_indices = np.unique(np_data, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)  # preserve original order
    return data[unique_indices], labels[unique_indices]


def compute_label_overlap(
    data: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    maxk: int = 30,
    use_dadapy: bool = True,
    metric: str = "euclidean",
    class_fraction: float | None = 0.1,
    weighted: bool = True,
) -> float:
    if isinstance(data, torch.Tensor):
        data_t = data.detach().to(torch.float32)
    else:
        data_t = torch.from_numpy(np.asarray(data, dtype=np.float32))

    if isinstance(labels, torch.Tensor):
        labels_t = labels.detach().long()
    else:
        labels_t = torch.from_numpy(np.asarray(labels)).long()

    data_t = data_t.to(DEVICE)
    labels_t = labels_t.to(DEVICE)

    n_samples = data_t.shape[0]
    effective_k = min(maxk, n_samples - 1)
    if effective_k <= 0:
        return 0.0

    unique_labels, class_counts = np.unique(labels_t.detach().cpu().numpy(), return_counts=True)
    class_count_map = {
        int(label): int(count) for label, count in zip(unique_labels, class_counts)
    }
    class_weight_map = {
        label: 1.0 / count for label, count in class_count_map.items()
    }

    if class_fraction is not None:
        k_per_class: dict[int, int] = {}
        for label, count in class_count_map.items():
            class_k = int(count * class_fraction)
            if class_k == 0:
                class_k = 1
                warnings.warn(
                    (
                        f"max_k < 1 for label {label}. max_k set to 1. "
                        "Consider increasing class_fraction."
                    ),
                    stacklevel=2,
                )
            k_per_class[label] = class_k

        effective_k = min(max(k_per_class.values()), n_samples - 1)
        k_per_sample_np = np.array(
            [min(k_per_class[int(label)], effective_k) for label in labels_t.detach().cpu().numpy()]
        )
    else:
        k_per_sample_np = np.full((n_samples,), effective_k, dtype=np.int64)

    if use_dadapy:
        with torch.no_grad():
            if metric == "cosine":
                normed = F.normalize(data_t, p=2, dim=1)
                sim = normed @ normed.T
                sim_no_self = sim.clone()
                sim_no_self.fill_diagonal_(-float("inf"))
                nn_sim, nn_idx = torch.topk(sim_no_self, effective_k, dim=1, largest=True)
                nn_dist = torch.clamp(1.0 - nn_sim, min=0.0, max=2.0)
            else:
                dist = torch.cdist(data_t, data_t)
                dist_no_self = dist.clone()
                dist_no_self.fill_diagonal_(float("inf"))
                nn_dist, nn_idx = torch.topk(
                    dist_no_self,
                    effective_k,
                    dim=1,
                    largest=False,
                )

        self_idx = torch.arange(n_samples, device=DEVICE).unsqueeze(1)
        self_dist = torch.zeros((n_samples, 1), dtype=nn_dist.dtype, device=DEVICE)

        dist_for_dadapy = torch.cat([self_dist, nn_dist], dim=1).cpu().numpy()
        idx_for_dadapy = torch.cat([self_idx, nn_idx], dim=1).cpu().numpy()

        clean_labels = labels_t.detach().cpu().numpy().astype(int)
        d = Data(distances=(dist_for_dadapy, idx_for_dadapy), maxk=effective_k)
        overlap = d.return_label_overlap(
            clean_labels,
            k=effective_k,
            class_fraction=class_fraction,
            weighted=weighted,
        )
        return float(overlap)

    with torch.no_grad():
        if metric == "cosine":
            normed = F.normalize(data_t, p=2, dim=1)
            sim = normed @ normed.T
            sim.fill_diagonal_(-float("inf"))
            neighbors = torch.topk(sim, effective_k, dim=1, largest=True).indices
        else:
            dist = torch.cdist(data_t, data_t)
            dist.fill_diagonal_(float("inf"))
            neighbors = torch.topk(
                dist,
                effective_k,
                dim=1,
                largest=False,
            ).indices

        target_labels = labels_t.view(n_samples, 1)
        neighbor_labels = labels_t[neighbors]
        overlaps = (neighbor_labels == target_labels)

        if class_fraction is not None:
            nearest_neighbor_rank = torch.arange(effective_k, device=DEVICE).unsqueeze(0)
            k_per_sample_t = torch.from_numpy(k_per_sample_np).to(DEVICE).unsqueeze(1)
            mask = nearest_neighbor_rank >= k_per_sample_t
            overlaps = overlaps & (~mask)

        overlaps_per_sample = overlaps.sum(dim=1).to(torch.float32) / torch.from_numpy(
            k_per_sample_np
        ).to(DEVICE)

        if weighted:
            sample_weights_np = np.array(
                [class_weight_map[int(label)] for label in labels_t.detach().cpu().numpy()],
                dtype=np.float32,
            )
            sample_weights_t = torch.from_numpy(sample_weights_np).to(DEVICE)
            weighted_overlap = (overlaps_per_sample * sample_weights_t).sum() / sample_weights_t.sum()
            return float(weighted_overlap.item())

        return float(overlaps_per_sample.mean().item())


@timed_metric("knn")
def classify_per_layer_knn(
    representations: torch.Tensor,
    y: np.ndarray,
    n_neighbors: int = 30,
) -> dict[int, dict[str, float]]:
    num_layers, num_samples, _ = representations.shape
    use_stratify = bool(np.all(np.bincount(y) >= 2))
    layer_results: dict[int, dict[str, float]] = {}

    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric knn", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            X_train_gpu = representations[layer_idx, train_idx].to(
                device=DEVICE,
                dtype=torch.float32,
            )
            X_test_gpu = representations[layer_idx, test_idx].to(
                device=DEVICE,
                dtype=torch.float32,
            )

            X_train_gpu = F.normalize(X_train_gpu, p=2, dim=1)
            X_test_gpu = F.normalize(X_test_gpu, p=2, dim=1)

            train_sim = torch.mm(X_train_gpu, X_train_gpu.t())
            test_sim = torch.mm(X_test_gpu, X_train_gpu.t())

            D_train = torch.clamp(1.0 - train_sim, min=0.0, max=2.0).cpu().numpy()
            D_test = torch.clamp(1.0 - test_sim, min=0.0, max=2.0).cpu().numpy()
            np.fill_diagonal(D_train, 0.0)
            D_train = np.ascontiguousarray(D_train, dtype=np.float32)
            D_test = np.ascontiguousarray(D_test, dtype=np.float32)

            n_neighbors_effective = min(n_neighbors, len(train_idx))
            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors_effective,
                metric="precomputed",
                weights="distance",
                algorithm="brute",
            )
            knn.fit(D_train, y[train_idx])
            y_pred = knn.predict(D_test)
            y_true = y[test_idx]

            layer_results[layer_idx] = {
                "knn_f1_weighted": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "knn_mcc": float(matthews_corrcoef(y_true, y_pred)),
            }

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results


@timed_metric("label_overlap")
def classify_per_layer_label_overlap(
    representations: torch.Tensor,
    y: np.ndarray,
    maxk: int = 30,
) -> dict[int, dict[str, float]]:
    num_layers = representations.shape[0]
    layer_results: dict[int, dict[str, float]] = {}

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric label_overlap", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            layer_data, y_dedup = _remove_duplicate_rows(representations[layer_idx], y)
            overlap = compute_label_overlap(
                data=layer_data,
                labels=y_dedup,
                maxk=maxk,
                use_dadapy=True,
                metric="euclidean",
            )
            layer_results[layer_idx] = {"label_overlap": float(overlap)}

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results


@timed_metric("logistic_regression")
def classify_per_layer_logistic_regression(
    representations: torch.Tensor,
    y: np.ndarray,
) -> dict[int, dict[str, float]]:
    num_layers, num_samples, _ = representations.shape
    use_stratify = bool(np.all(np.bincount(y) >= 2))
    layer_results: dict[int, dict[str, float]] = {}

    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    if np.unique(y[train_idx]).shape[0] < 2:
        warnings.warn(
            "Logistic regression requires at least 2 classes in train split.",
            stacklevel=2,
        )
        return {i: {"f1_weighted": 0.0, "f1_weighted_train": 0.0, "mcc": 0.0} for i in range(num_layers)}

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric logistic_regression", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            X = representations[layer_idx].numpy()
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            n_train_samples, n_features = X_train.shape
            # Underdetermined (n_train < n_features): L1 + liblinear promotes sparsity.
            # Overdetermined (n_train >= n_features): L2 + lbfgs, fast multinomial solver.
            if n_train_samples < n_features:
                classifier = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=0.1,
                    tol=1e-2,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                )
            else:
                classifier = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                )

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(X_train, y_train)
                if captured:
                    print(f"Warning: Layer {layer_idx} failed to converge.")

            y_pred = classifier.predict(X_test)
            y_pred_train = classifier.predict(X_train)
            layer_results[layer_idx] = {
                "f1_weighted": float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                "f1_weighted_train": float(
                    f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
                ),
                "mcc": float(matthews_corrcoef(y_test, y_pred)),
            }

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results


@timed_metric("homogeneity")
def homogeneity_score_per_layer(
    representations: torch.Tensor,
    y: np.ndarray,
    K: int = 16,
    Z: float = 1.65,
) -> dict[int, dict[str, float]]:
    num_layers, n_samples, _ = representations.shape
    layer_results: dict[int, dict[str, float]] = {}

    if n_samples <= K:
        warnings.warn(
            f"n_samples ({n_samples}) <= K ({K}). Returning 0.0 for all layers.",
            stacklevel=2,
        )
        return {
            layer_idx: {"homogeneity_score": 0.0, "adjusted_rand_score": 0.0, "n_clusters_ratio": 0.0}
            for layer_idx in range(num_layers)
        }

    n_labels = len(np.unique(y))
    # id_index=4 is valid: default range_max=64 → id_list has int(log2(64))=6 elements (0-5)
    id_index = int(math.log2(K))  # = 4 for K=16

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric homogeneity", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            try:
                layer_data = F.normalize(
                    representations[layer_idx].to(DEVICE, dtype=torch.float32), p=2, dim=1
                )
                with torch.no_grad():
                    distances = torch.cdist(layer_data, layer_data).cpu().numpy().astype(np.float64)

                d = Data(distances=distances)
                d.distances = d.remove_zero_dists(d.distances)
                id_list, _, _ = d.return_id_scaling_gride()
                d.set_id(id_list[id_index])
                d.compute_density_kNN(k=K)
                cluster_labels = d.compute_clustering_ADP(Z=Z)

                score, _, _ = homogeneity_completeness_v_measure(y, cluster_labels)
                layer_results[layer_idx] = {
                    "homogeneity_score": float(score),
                    "adjusted_rand_score": float(adjusted_rand_score(y, cluster_labels)),
                    "n_clusters_ratio": float(d.N_clusters / n_labels),
                }
            except Exception as exc:
                print(f"Warning: Homogeneity failed at layer {layer_idx}: {exc}")
                layer_results[layer_idx] = {
                    "homogeneity_score": 0.0,
                    "adjusted_rand_score": 0.0,
                    "n_clusters_ratio": 0.0,
                }

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="lbasile/llava_7b")
    parser.add_argument("--dataset-name", type=str, default="derek-thomas/ScienceQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--guide-text",
        type=str,
        default="Answer with the option's letter from the given choices directly.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help=(
            "Root directory for saving results. Use default to save under "
            "~/multimodal_finetuned_representations/results."
        ),
    )
    parser.add_argument(
        "--representations-dir",
        type=str,
        default=str(REPRESENTATIONS_BASE),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--tokens-pooling-method",
        type=str,
        default="last",
        choices=["last", "mean_image"],
    )
    parser.add_argument(
        "--save-representations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force-recompute-representations",
        action="store_true",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["logistic_regression"],
        choices=list(SUPPORTED_METRICS),
    )
    parser.add_argument(
        "--mmmu-label-granularity",
        type=str,
        default="intermediate",
        choices=["intermediate", "detailed"],
        help=(
            "Label granularity for MMMU. "
            "'intermediate': subjects (~30 classes, default). "
            "'detailed': subfields with count >= mean count."
        ),
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=0,
        help=(
            "Number of in-context learning shots from the MMMU dev split. "
            "0 = zero-shot (default). Shots are matched by subject/subfield. "
            "Results and representations are saved with a _{n}shot suffix."
        ),
    )
    parser.add_argument(
        "--save-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save formatted prompts (after chat-template application) to "
            "prompts{icl_suffix}.json in the same directory as metric files."
        ),
    )
    return parser.parse_args()


def _resolve_results_base(results_dir_arg: str) -> Path:
    if results_dir_arg == "results":
        return RESULTS_BASE
    return Path(results_dir_arg).expanduser().resolve()


def _build_output_path(
    *,
    results_base: Path,
    dataset_name: str,
    split: str,
    model_name: str,
    pooling_method: str,
    metric_name: str,
    suffix: str = "",
    extension: str = "safetensors",
) -> Path:
    dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
    model_slug = Path(model_name).name
    out_dir = (
        results_base
        / "representations_classification"
        / dataset_slug
        / split
        / model_slug
        / pooling_method
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{metric_name}{suffix}.{extension}"


def _build_representation_cache_path(
    *,
    representations_base: Path,
    dataset_name: str,
    model_name: str,
    pooling_method: str,
    sample_set_tag: str,
) -> Path:
    dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
    model_slug = model_name.split("/")[-1]
    out_dir = (
        representations_base
        / "representations"
        / model_slug
        / dataset_slug
        / RESIDUAL_STREAM_TYPE
        / pooling_method
        / sample_set_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "representations.safetensors"


def _save_metric_results(
    *,
    out_path: Path,
    metric_name: str,
    layer_results: dict[int, dict[str, float]],
    num_layers: int,
    args: argparse.Namespace,
    num_samples: int,
    label_classes: np.ndarray,
) -> None:
    keys = METRIC_TO_KEYS[metric_name]
    tensors = {
        key: torch.tensor(
            [layer_results[i][key] for i in range(num_layers)], dtype=torch.float32
        )
        for key in keys
    }
    metadata = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "pooling_method": args.tokens_pooling_method,
        "guide_text": args.guide_text,
        "num_samples": str(num_samples),
        "classes": ", ".join(str(c) for c in label_classes),
        "metric": metric_name,
        "metric_keys": ", ".join(keys),
    }
    save_file(tensors, str(out_path), metadata=metadata)
    print(f"Saved {metric_name}: {out_path}")


def main() -> None:
    args = parse_args()
    results_base = _resolve_results_base(args.results_dir)
    representations_base = Path(args.representations_dir).expanduser().resolve()

    # ICL suffix differentiates n-shot runs from zero-shot without touching any
    # zero-shot paths.  Both cache path and metric filenames carry the suffix.
    icl_suffix = f"_{args.n_shots}shot" if args.n_shots > 0 else ""
    sample_set_tag = f"unique_images{icl_suffix}"

    representation_cache_path = _build_representation_cache_path(
        representations_base=representations_base,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        pooling_method=args.tokens_pooling_method,
        sample_set_tag=sample_set_tag,
    )

    prompts_save_path = (
        _build_output_path(
            results_base=results_base,
            dataset_name=args.dataset_name,
            split=args.split,
            model_name=args.model_name,
            pooling_method=args.tokens_pooling_method,
            metric_name="prompts",
            suffix=icl_suffix,
            extension="json",
        )
        if args.save_prompts
        else None
    )

    representations, labels = get_representations_and_labels(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        guide_text=args.guide_text,
        tokens_pooling_method=args.tokens_pooling_method,
        representation_cache_path=representation_cache_path,
        force_recompute_representations=args.force_recompute_representations,
        save_representations=args.save_representations,
        sample_set_tag=sample_set_tag,
        label_granularity=args.mmmu_label_granularity,
        max_samples=args.max_samples,
        n_shots=args.n_shots,
        save_prompts_path=prompts_save_path,
    )

    encoded_labels = np.asarray(labels)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(encoded_labels)
    print(f'Number of unique labels: {label_encoder.classes_.size}')
    print(f'First 5 classes: {label_encoder.classes_[:5]}')

    num_layers, num_samples, _ = representations.shape

    for metric_name in args.metrics:
        print(f"\nComputing metric: {metric_name}")

        if metric_name == "logistic_regression":
            layer_results = classify_per_layer_logistic_regression(representations, y)
        elif metric_name == "knn":
            layer_results = classify_per_layer_knn(representations, y)
        elif metric_name == "label_overlap":
            layer_results = classify_per_layer_label_overlap(representations, y)
        else:  # homogeneity
            layer_results = homogeneity_score_per_layer(representations, y)

        out_path = _build_output_path(
            results_base=results_base,
            dataset_name=args.dataset_name,
            split=args.split,
            model_name=args.model_name,
            pooling_method=args.tokens_pooling_method,
            metric_name=metric_name,
            suffix=icl_suffix,
        )

        _save_metric_results(
            out_path=out_path,
            metric_name=metric_name,
            layer_results=layer_results,
            num_layers=num_layers,
            args=args,
            num_samples=num_samples,
            label_classes=label_encoder.classes_,
        )


if __name__ == "__main__":
    seed_all(42)
    main()
