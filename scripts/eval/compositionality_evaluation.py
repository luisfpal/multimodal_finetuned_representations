"""
Compositionality evaluation for generative VLMs on ARO and BiVLC benchmarks.

Evaluates VLMs using two complementary methods:
  - Log-likelihood scoring: P(caption | image) via model logits (primary)
  - MCQ prompting: multiple-choice question answering (secondary)

Benchmarks:
  - ARO (Attribution, Relation, Order): 50k+ test cases for compositional failures
    Subtasks: VG-Relation, VG-Attribution, COCO-Order, Flickr30k-Order
  - BiVLC: Bidirectional compositionality (I2T + T2I + Group score)

Supports: LLaVA 1.5/v1.6, LLaVA-OneVision, InternVL2, InternVL3.5

Setup:
  ARO data: Clone https://github.com/mertyg/vision-language-models-are-bows
            into third_party/ and run download. See --aro-data-root.
  BiVLC:    Auto-downloads from HuggingFace (imirandam/BiVLC).

Usage:
    python scripts/eval/compositionality_evaluation.py \\
        --model-name-or-path lbasile/llava_7b \\
        --output-dir evals/llava_7b/compositionality \\
        --benchmark bivlc \\
        --limit 100
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from rich import print
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.constants import SYSTEM_ROLE
from utils.dataset import _processor_has_default_system
from utils.internvl_utils import (
    apply_chat_template as apply_internvl_chat_template,
    get_combined_embeddings,
    get_pixel_values,
)
from utils.model_utils import load_hf_model_and_processor_or_tokenizer

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 42
DEFAULT_DATASETS_CACHE = os.environ.get(
    "HF_DATASETS_CACHE",
    os.path.expanduser("~/scratch/huggingface/datasets"),
)
DEFAULT_ARO_REPO = str(
    Path.home() / "scratch" / "datasets" / "vision-language-models-are-bows"
)

ARO_SUBTASKS = ("vg_relation", "vg_attribution", "coco_order", "flickr30k_order")
BIVLC_DATASET_ID = "imirandam/BiVLC"

# Neutral prompt for log-likelihood scoring
SCORING_PROMPT = "Describe this image."

# MCQ prompt templates
MCQ_BINARY_PROMPT = (
    "Which caption better describes this image?\n"
    "A) {caption_a}\n"
    "B) {caption_b}\n"
    "Answer with the letter only."
)
MCQ_5WAY_PROMPT = (
    "Which caption best describes this image?\n"
    "A) {caption_a}\n"
    "B) {caption_b}\n"
    "C) {caption_c}\n"
    "D) {caption_d}\n"
    "E) {caption_e}\n"
    "Answer with the letter only."
)


# =============================================================================
# MODEL TYPE DETECTION
# =============================================================================


def detect_model_type(model_name_or_path: str) -> str:
    """Detect model family from model name/path string."""
    name_lower = model_name_or_path.lower()
    if "internvl3" in name_lower:
        return "internvl3"
    if "internvl" in name_lower:
        return "internvl2"
    # LLaVA family (including OneVision, Qwen2-VL variants)
    return "llava"


def is_internvl_model(model: torch.nn.Module) -> bool:
    """Check if model is InternVL by class name."""
    return "intern" in model.__class__.__name__.lower()


# =============================================================================
# LOG-LIKELIHOOD SCORING
# =============================================================================


def _build_llava_conversation(
    caption: str,
    has_image: bool,
    processor: Any,
    include_caption: bool = True,
) -> list:
    """Build a LLaVA-style conversation for log-likelihood scoring."""
    include_system = not _processor_has_default_system(processor)

    user_content: list = []
    if has_image:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": SCORING_PROMPT})

    conv: list = [{"role": "user", "content": user_content}]
    if include_system:
        conv.insert(0, SYSTEM_ROLE)

    if include_caption:
        conv.append(
            {"role": "assistant", "content": [{"type": "text", "text": caption}]}
        )

    return conv


def _compute_log_likelihood_llava(
    model: torch.nn.Module,
    processor: Any,
    image: Optional[Image.Image],
    caption: str,
    normalize_by_length: bool = True,
) -> float:
    """Compute log-likelihood of caption given image for LLaVA models."""
    device = model.device
    has_image = image is not None

    # Build full conversation (with caption as assistant response)
    full_conv = _build_llava_conversation(caption, has_image, processor, include_caption=True)
    full_text = processor.apply_chat_template(
        full_conv, tokenize=False, add_generation_prompt=False
    )

    # Build prompt only (without caption)
    prompt_conv = _build_llava_conversation(caption, has_image, processor, include_caption=False)
    prompt_text = processor.apply_chat_template(
        prompt_conv, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    proc_kwargs = {"return_tensors": "pt", "padding": False}
    if has_image:
        proc_kwargs["images"] = image
    full_inputs = processor(text=full_text, **proc_kwargs)
    prompt_inputs = processor(text=prompt_text, **proc_kwargs)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # Forward pass
    full_inputs = {k: v.to(device) for k, v in full_inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        outputs = model(**full_inputs)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Extract log-probs for caption tokens only
    # Logits at position t predict token at position t+1
    caption_logits = logits[0, prompt_len - 1 : -1]  # (caption_len, vocab_size)
    caption_token_ids = full_inputs["input_ids"][0, prompt_len:]  # (caption_len,)

    log_probs = F.log_softmax(caption_logits, dim=-1)
    token_log_probs = log_probs.gather(1, caption_token_ids.unsqueeze(1)).squeeze(1)
    total_log_prob = token_log_probs.sum().item()

    if normalize_by_length and len(caption_token_ids) > 0:
        total_log_prob /= len(caption_token_ids)

    return total_log_prob


def _build_internvl_full_query(
    model: torch.nn.Module,
    question: str,
    caption: str,
    num_patches: Optional[int] = None,
) -> str:
    """Build InternVL query with caption as assistant response."""
    from utils.internvl_utils import IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN

    has_image = num_patches is not None
    txt = question
    if has_image and "<image>" not in txt:
        txt = "<image>\n" + txt

    # Full prompt with assistant response
    template = copy.deepcopy(model.conv_template)
    template.system_message = model.system_message
    template.messages = []
    template.append_message(template.roles[0], txt)
    template.append_message(template.roles[1], caption)
    full_query = template.get_prompt()

    # Replace <image> with IMG_CONTEXT tokens
    if has_image:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        full_query = full_query.replace("<image>", image_tokens, 1)

    return full_query


def _build_internvl_prompt_query(
    model: torch.nn.Module,
    question: str,
    num_patches: Optional[int] = None,
) -> str:
    """Build InternVL query without assistant response (prompt only)."""
    from utils.internvl_utils import IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN

    has_image = num_patches is not None
    txt = question
    if has_image and "<image>" not in txt:
        txt = "<image>\n" + txt

    template = copy.deepcopy(model.conv_template)
    template.system_message = model.system_message
    template.messages = []
    template.append_message(template.roles[0], txt)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if has_image:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

    return query


def _get_internvl_img_context_token_id(tokenizer: Any) -> int:
    """Get the img_context_token_id for InternVL models."""
    from utils.internvl_utils import IMG_CONTEXT_TOKEN

    return tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)


def _compute_log_likelihood_internvl(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    caption: str,
    normalize_by_length: bool = True,
) -> float:
    """Compute log-likelihood of caption given image for InternVL models."""
    device = model.device

    # Process image
    pixel_values = None
    num_patches = None
    if image is not None:
        pv, num_patches_list = get_pixel_values(image)
        pixel_values = pv.to(device)
        num_patches = num_patches_list[0]

    # Build full query (with caption) and prompt query (without)
    full_query = _build_internvl_full_query(model, SCORING_PROMPT, caption, num_patches)
    prompt_query = _build_internvl_prompt_query(model, SCORING_PROMPT, num_patches)

    # Tokenize
    full_ids = tokenizer(full_query, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt_query, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # Forward pass
    img_ctx_id = _get_internvl_img_context_token_id(tokenizer)
    with torch.no_grad():
        if pixel_values is not None:
            inputs_embeds = get_combined_embeddings(
                model, full_ids, pixel_values, img_context_token_id=img_ctx_id
            )
            outputs = model.language_model(inputs_embeds=inputs_embeds)
        else:
            inputs_embeds = model.language_model.get_input_embeddings()(full_ids)
            outputs = model.language_model(inputs_embeds=inputs_embeds)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Extract caption token log-probs
    caption_logits = logits[0, prompt_len - 1 : -1]
    caption_token_ids = full_ids[0, prompt_len:]

    log_probs = F.log_softmax(caption_logits, dim=-1)
    token_log_probs = log_probs.gather(1, caption_token_ids.unsqueeze(1)).squeeze(1)
    total_log_prob = token_log_probs.sum().item()

    if normalize_by_length and len(caption_token_ids) > 0:
        total_log_prob /= len(caption_token_ids)

    return total_log_prob


def compute_caption_log_likelihood(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    image: Optional[Image.Image],
    caption: str,
    normalize_by_length: bool = True,
) -> float:
    """Compute log-likelihood of caption given image.

    Dispatches to model-specific implementation based on model type.
    """
    if is_internvl_model(model):
        return _compute_log_likelihood_internvl(
            model, processor_or_tokenizer, image, caption, normalize_by_length
        )
    return _compute_log_likelihood_llava(
        model, processor_or_tokenizer, image, caption, normalize_by_length
    )


# =============================================================================
# MCQ EVALUATION
# =============================================================================


def _generate_response_llava(
    model: torch.nn.Module,
    processor: Any,
    image: Optional[Image.Image],
    question: str,
    max_new_tokens: int = 16,
) -> str:
    """Generate a text response from LLaVA model."""
    include_system = not _processor_has_default_system(processor)
    user_content: list = []
    if image is not None:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": question})

    conv: list = [{"role": "user", "content": user_content}]
    if include_system:
        conv.insert(0, SYSTEM_ROLE)

    prompt_text = processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True
    )

    proc_kwargs = {"text": prompt_text, "return_tensors": "pt", "padding": False}
    if image is not None:
        proc_kwargs["images"] = image

    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Strip input tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def _generate_response_internvl(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    max_new_tokens: int = 16,
) -> str:
    """Generate a text response from InternVL model."""
    from utils.internvl_utils import generate_responses

    device = model.device
    pixel_values = None
    num_patches_list = None

    if image is not None:
        pv, num_patches_list = get_pixel_values(image)
        pixel_values = pv.to(device)

    # Build query using standard chat template
    queries = apply_internvl_chat_template(
        model=model,
        text=[question],
        num_patches_list=num_patches_list,
    )
    query = queries[0]

    # Tokenize
    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    # Get combined embeddings if we have images
    img_ctx_id = _get_internvl_img_context_token_id(tokenizer)
    if pixel_values is not None:
        inputs_embeds = get_combined_embeddings(
            model, input_ids, pixel_values, img_context_token_id=img_ctx_id
        )
        responses = generate_responses(
            model, tokenizer,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
    else:
        responses = generate_responses(
            model, tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    return responses[0]


def generate_response(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    max_new_tokens: int = 16,
) -> str:
    """Generate response, dispatching to model-specific implementation."""
    if is_internvl_model(model):
        return _generate_response_internvl(
            model, processor_or_tokenizer, image, question, max_new_tokens
        )
    return _generate_response_llava(
        model, processor_or_tokenizer, image, question, max_new_tokens
    )


def parse_mcq_answer(response: str, num_choices: int = 2) -> Optional[int]:
    """Parse MCQ answer from model response. Returns 0-indexed choice or None."""
    import re

    valid_letters = [chr(ord("A") + i) for i in range(num_choices)]
    response = response.strip()

    # 1. Exact single letter (e.g., "B")
    if response.upper() in valid_letters:
        return valid_letters.index(response.upper())

    # 2. Letter followed by punctuation (e.g., "B.", "B)", "B:")
    match = re.match(r"^([A-Z])[.):\s]", response.upper())
    if match and match.group(1) in valid_letters:
        return valid_letters.index(match.group(1))

    # 3. "answer is X" or "answer: X" pattern
    match = re.search(r"answer\s*(?:is|:)\s*([A-Z])\b", response, re.IGNORECASE)
    if match and match.group(1).upper() in valid_letters:
        return valid_letters.index(match.group(1).upper())

    # 4. Last standalone letter in response (most likely the answer)
    matches = re.findall(r"\b([A-Z])\b", response.upper())
    for letter in reversed(matches):
        if letter in valid_letters:
            return valid_letters.index(letter)

    return None


# =============================================================================
# ARO DATASETS
# =============================================================================


def _ensure_aro_repo_on_path(aro_repo_root: str) -> str:
    """Add ARO repo to sys.path and return the resolved path."""
    aro_path = os.path.expanduser(aro_repo_root)
    if not os.path.isdir(os.path.join(aro_path, "dataset_zoo")):
        raise ImportError(
            f"ARO repo not found at {aro_path}. "
            f"Clone https://github.com/mertyg/vision-language-models-are-bows "
            f"to {aro_path}"
        )
    if aro_path not in sys.path:
        sys.path.insert(0, aro_path)
    return aro_path


class AROVGDataset(Dataset):
    """ARO VG-Relation or VG-Attribution dataset.

    Data (annotations + images) is downloaded into the repo root.
    """

    def __init__(
        self,
        subtask: str,
        aro_repo_root: str,
        limit: Optional[int] = None,
    ):
        if subtask not in ("vg_relation", "vg_attribution"):
            raise ValueError(f"Invalid ARO VG subtask: {subtask}")

        self.subtask = subtask
        aro_path = _ensure_aro_repo_on_path(aro_repo_root)

        from dataset_zoo import VG_Relation, VG_Attribution

        DatasetClass = VG_Relation if subtask == "vg_relation" else VG_Attribution
        self.dataset = DatasetClass(
            image_preprocess=None, download=True, root_dir=aro_path
        )
        self.limit = min(limit, len(self.dataset)) if limit else len(self.dataset)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image = item["image_options"][0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        return {
            "image": image,
            "caption_options": item["caption_options"],
            "correct_index": 1,  # true caption is at index 1
        }


class AROOrderDataset(Dataset):
    """ARO COCO-Order or Flickr30k-Order dataset.

    COCO-Order needs COCO test2014 images at `root_dir`.
    Flickr30k-Order needs flickr30k-images at `root_dir`.
    Both are downloaded/referenced relative to the repo root by default.
    """

    def __init__(
        self,
        subtask: str,
        aro_repo_root: str,
        limit: Optional[int] = None,
    ):
        if subtask not in ("coco_order", "flickr30k_order"):
            raise ValueError(f"Invalid ARO Order subtask: {subtask}")

        self.subtask = subtask
        aro_path = _ensure_aro_repo_on_path(aro_repo_root)

        from dataset_zoo import COCO_Order, Flickr30k_Order

        DatasetClass = COCO_Order if subtask == "coco_order" else Flickr30k_Order
        # Order datasets need image dirs; pass repo root and let them resolve
        self.dataset = DatasetClass(
            image_preprocess=None, download=True, root_dir=aro_path
        )
        self.limit = min(limit, len(self.dataset)) if limit else len(self.dataset)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image = item["image_options"][0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        return {
            "image": image,
            "caption_options": item["caption_options"],
            "correct_index": 0,  # original caption is at index 0
        }


def load_aro_dataset(
    subtask: str,
    aro_repo_root: str,
    limit: Optional[int] = None,
) -> Dataset:
    """Load an ARO subtask dataset."""
    if subtask in ("vg_relation", "vg_attribution"):
        return AROVGDataset(subtask, aro_repo_root, limit)
    return AROOrderDataset(subtask, aro_repo_root, limit)


# =============================================================================
# BiVLC DATASET
# =============================================================================


def load_bivlc_dataset(
    cache_dir: str,
    limit: Optional[int] = None,
) -> list:
    """Load BiVLC dataset from HuggingFace. Returns list of dicts."""
    dataset = load_dataset(BIVLC_DATASET_ID, split="test", cache_dir=cache_dir)
    if limit and limit < len(dataset):
        dataset = dataset.select(range(limit))
    return dataset


# =============================================================================
# BENCHMARK EVALUATION — LOG-LIKELIHOOD
# =============================================================================


def evaluate_aro_loglikelihood(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    subtask: str,
    aro_repo_root: str,
    limit: Optional[int] = None,
    normalize_by_length: bool = True,
) -> Dict[str, Any]:
    """Evaluate a single ARO subtask using log-likelihood scoring."""
    dataset = load_aro_dataset(subtask, aro_repo_root, limit)

    correct = 0
    total = 0
    per_sample: list = []

    for idx in tqdm(range(len(dataset)), desc=f"ARO {subtask} (loglik)"):
        item = dataset[idx]
        image = item["image"]
        captions = item["caption_options"]
        correct_idx = item["correct_index"]

        # Score each caption
        scores = []
        for caption in captions:
            try:
                score = compute_caption_log_likelihood(
                    model, processor_or_tokenizer, image, caption, normalize_by_length
                )
            except Exception as e:
                print(f"[yellow]⚠ Error scoring sample {idx}: {e}[/yellow]")
                score = float("-inf")
            scores.append(score)

        predicted_idx = scores.index(max(scores))
        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append({
            "idx": idx,
            "scores": scores,
            "predicted": predicted_idx,
            "correct": correct_idx,
            "is_correct": is_correct,
        })

        # Periodic memory cleanup
        if (idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "per_sample": per_sample,
    }


def evaluate_bivlc_loglikelihood(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    cache_dir: str,
    limit: Optional[int] = None,
    normalize_by_length: bool = True,
) -> Dict[str, Any]:
    """Evaluate BiVLC using log-likelihood scoring."""
    dataset = load_bivlc_dataset(cache_dir, limit)

    i2t_correct = 0
    t2i_correct = 0
    group_correct = 0
    total = 0
    per_type: Dict[str, Dict[str, int]] = {}
    per_sample: list = []

    for idx in tqdm(range(len(dataset)), desc="BiVLC (loglik)"):
        item = dataset[idx]
        image = item["image"]
        caption = item["caption"]
        neg_caption = item["negative_caption"]
        neg_image = item["negative_image"]
        comp_type = item.get("type", "unknown")
        comp_subtype = item.get("subtype", "unknown")

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        if isinstance(neg_image, Image.Image):
            neg_image = neg_image.convert("RGB")

        try:
            # I2T: score caption vs neg_caption conditioned on image
            score_cap = compute_caption_log_likelihood(
                model, processor_or_tokenizer, image, caption, normalize_by_length
            )
            score_neg_cap = compute_caption_log_likelihood(
                model, processor_or_tokenizer, image, neg_caption, normalize_by_length
            )
            i2t_ok = score_cap > score_neg_cap

            # T2I: score caption conditioned on image vs neg_image
            score_img = score_cap  # already computed above
            score_neg_img = compute_caption_log_likelihood(
                model, processor_or_tokenizer, neg_image, caption, normalize_by_length
            )
            t2i_ok = score_img > score_neg_img

            group_ok = i2t_ok and t2i_ok

        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            i2t_ok = t2i_ok = group_ok = False
            score_cap = score_neg_cap = score_neg_img = float("nan")

        if i2t_ok:
            i2t_correct += 1
        if t2i_ok:
            t2i_correct += 1
        if group_ok:
            group_correct += 1
        total += 1

        # Per-type tracking
        type_key = f"{comp_type}/{comp_subtype}"
        if type_key not in per_type:
            per_type[type_key] = {"i2t": 0, "t2i": 0, "group": 0, "total": 0}
        per_type[type_key]["total"] += 1
        if i2t_ok:
            per_type[type_key]["i2t"] += 1
        if t2i_ok:
            per_type[type_key]["t2i"] += 1
        if group_ok:
            per_type[type_key]["group"] += 1

        per_sample.append({
            "idx": idx,
            "i2t_correct": i2t_ok,
            "t2i_correct": t2i_ok,
            "group_correct": group_ok,
            "scores": {
                "cap_given_img": score_cap,
                "negcap_given_img": score_neg_cap,
                "cap_given_negimg": score_neg_img,
            },
            "type": comp_type,
            "subtype": comp_subtype,
        })

        if (idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute per-type accuracies
    per_type_acc = {}
    for key, counts in per_type.items():
        n = counts["total"]
        per_type_acc[key] = {
            "i2t": counts["i2t"] / n if n > 0 else 0.0,
            "t2i": counts["t2i"] / n if n > 0 else 0.0,
            "group": counts["group"] / n if n > 0 else 0.0,
            "total": n,
        }

    return {
        "i2t_accuracy": i2t_correct / total if total > 0 else 0.0,
        "t2i_accuracy": t2i_correct / total if total > 0 else 0.0,
        "group_accuracy": group_correct / total if total > 0 else 0.0,
        "num_total": total,
        "per_type": per_type_acc,
        "per_sample": per_sample,
    }


# =============================================================================
# BENCHMARK EVALUATION — MCQ
# =============================================================================


def evaluate_aro_mcq(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    subtask: str,
    aro_repo_root: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a single ARO subtask using MCQ prompting."""
    dataset = load_aro_dataset(subtask, aro_repo_root, limit)

    correct = 0
    total = 0
    per_sample: list = []

    for idx in tqdm(range(len(dataset)), desc=f"ARO {subtask} (mcq)"):
        item = dataset[idx]
        image = item["image"]
        captions = item["caption_options"]
        correct_idx = item["correct_index"]
        num_choices = len(captions)

        # Build MCQ prompt
        if num_choices == 2:
            prompt = MCQ_BINARY_PROMPT.format(
                caption_a=captions[0], caption_b=captions[1]
            )
        elif num_choices == 5:
            prompt = MCQ_5WAY_PROMPT.format(
                caption_a=captions[0],
                caption_b=captions[1],
                caption_c=captions[2],
                caption_d=captions[3],
                caption_e=captions[4],
            )
        else:
            # Fallback: build dynamically
            lines = [f"{chr(ord('A') + i)}) {cap}" for i, cap in enumerate(captions)]
            prompt = (
                "Which caption best describes this image?\n"
                + "\n".join(lines)
                + "\nAnswer with the letter only."
            )

        try:
            response = generate_response(
                model, processor_or_tokenizer, image, prompt
            )
            predicted_idx = parse_mcq_answer(response, num_choices)
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            response = ""
            predicted_idx = None

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append({
            "idx": idx,
            "response": response,
            "predicted": predicted_idx,
            "correct": correct_idx,
            "is_correct": is_correct,
        })

        if (idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "per_sample": per_sample,
    }


def evaluate_bivlc_mcq(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    cache_dir: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate BiVLC I2T direction using MCQ prompting.

    T2I is skipped for MCQ because it requires multi-image prompting
    which most VLMs don't support natively. Use log-likelihood for T2I.
    """
    dataset = load_bivlc_dataset(cache_dir, limit)

    i2t_correct = 0
    total = 0
    per_type: Dict[str, Dict[str, int]] = {}
    per_sample: list = []

    for idx in tqdm(range(len(dataset)), desc="BiVLC I2T (mcq)"):
        item = dataset[idx]
        image = item["image"]
        caption = item["caption"]
        neg_caption = item["negative_caption"]
        comp_type = item.get("type", "unknown")
        comp_subtype = item.get("subtype", "unknown")

        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Randomize order to avoid position bias, but track correct answer
        import random

        rng = random.Random(RANDOM_SEED + idx)
        if rng.random() > 0.5:
            options = [caption, neg_caption]
            correct_idx = 0
        else:
            options = [neg_caption, caption]
            correct_idx = 1

        prompt = MCQ_BINARY_PROMPT.format(
            caption_a=options[0], caption_b=options[1]
        )

        try:
            response = generate_response(model, processor_or_tokenizer, image, prompt)
            predicted_idx = parse_mcq_answer(response, 2)
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            response = ""
            predicted_idx = None

        i2t_ok = predicted_idx == correct_idx
        if i2t_ok:
            i2t_correct += 1
        total += 1

        type_key = f"{comp_type}/{comp_subtype}"
        if type_key not in per_type:
            per_type[type_key] = {"i2t": 0, "total": 0}
        per_type[type_key]["total"] += 1
        if i2t_ok:
            per_type[type_key]["i2t"] += 1

        per_sample.append({
            "idx": idx,
            "response": response,
            "predicted": predicted_idx,
            "correct": correct_idx,
            "is_correct": i2t_ok,
            "type": comp_type,
            "subtype": comp_subtype,
        })

        if (idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_type_acc = {}
    for key, counts in per_type.items():
        n = counts["total"]
        per_type_acc[key] = {
            "i2t": counts["i2t"] / n if n > 0 else 0.0,
            "total": n,
        }

    return {
        "i2t_accuracy": i2t_correct / total if total > 0 else 0.0,
        "num_total": total,
        "per_type": per_type_acc,
        "per_sample": per_sample,
        "note": "T2I not evaluated with MCQ (requires multi-image support). Use loglikelihood.",
    }


# =============================================================================
# RESULT SAVING
# =============================================================================


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time as Xh Ym Zs."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m > 0 else f"{seconds:.1f}s"


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    model_identifier: str,
    benchmark: str,
    eval_method: str,
    elapsed_sec: float,
) -> None:
    """Save evaluation results to JSONL (metrics) and JSON (per-sample)."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract per-sample data and build summary
    per_sample = results.pop("per_sample", [])
    summary = {
        "experiment_name": f"single/{model_identifier}",
        "eval_method": eval_method,
        **results,
        "elapsed_time": format_elapsed_time(elapsed_sec),
    }

    # Append metrics to JSONL
    jsonl_path = os.path.join(output_dir, f"{benchmark}_results.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[green]  Metrics → {jsonl_path}[/green]")

    # Save per-sample responses
    suffix = f"_{eval_method}" if eval_method != "loglikelihood" else ""
    json_path = os.path.join(
        output_dir, f"{model_identifier}{suffix}_responses.json"
    )
    with open(json_path, "w") as f:
        json.dump(per_sample, f, indent=2, default=str)
    print(f"[green]  Samples → {json_path}[/green]")


# =============================================================================
# ORCHESTRATION
# =============================================================================


def run_aro_evaluation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    args: argparse.Namespace,
    model_identifier: str,
) -> None:
    """Run full ARO evaluation."""
    print("[bold magenta]=== ARO Compositionality Benchmark ===[/bold magenta]")

    all_results: Dict[str, Dict[str, Any]] = {}
    eval_methods = _get_eval_methods(args.eval_method)

    for subtask in args.aro_subtasks:
        print(f"\n[cyan]▶ {subtask}[/cyan]")
        output_dir = os.path.join(args.output_dir, "aro")

        for method in eval_methods:
            print(f"[dim]  Method: {method}[/dim]")
            start = time.time()

            try:
                if method == "loglikelihood":
                    results = evaluate_aro_loglikelihood(
                        model, processor_or_tokenizer,
                        subtask, args.aro_data_root,
                        limit=args.limit,
                        normalize_by_length=args.normalize_by_length,
                    )
                else:
                    results = evaluate_aro_mcq(
                        model, processor_or_tokenizer,
                        subtask, args.aro_data_root,
                        limit=args.limit,
                    )

                elapsed = time.time() - start
                acc_key = "accuracy"
                print(
                    f"[green]  ✓ {subtask} ({method}): "
                    f"{results[acc_key]:.4f} ({format_elapsed_time(elapsed)})[/green]"
                )

                save_results(
                    results, output_dir, model_identifier,
                    f"aro_{subtask}", method, elapsed,
                )
                all_results[f"{subtask}_{method}"] = results

            except Exception as e:
                print(f"[red]  ✗ {subtask} ({method}): {e}[/red]")

    # Print unified summary
    print("\n[bold cyan]── ARO Summary ──[/bold cyan]")
    for key, res in all_results.items():
        acc = res.get("accuracy", 0.0)
        print(f"  {key}: {acc:.4f}")

    # Compute mean per method
    for method in eval_methods:
        method_results = [
            v["accuracy"] for k, v in all_results.items() if k.endswith(f"_{method}")
        ]
        if method_results:
            mean_acc = sum(method_results) / len(method_results)
            print(f"  [bold]mean ({method}): {mean_acc:.4f}[/bold]")


def run_bivlc_evaluation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    args: argparse.Namespace,
    model_identifier: str,
) -> None:
    """Run full BiVLC evaluation."""
    print("[bold magenta]=== BiVLC Compositionality Benchmark ===[/bold magenta]")

    output_dir = os.path.join(args.output_dir, "bivlc")
    eval_methods = _get_eval_methods(args.eval_method)

    for method in eval_methods:
        print(f"\n[cyan]▶ BiVLC ({method})[/cyan]")
        start = time.time()

        try:
            if method == "loglikelihood":
                results = evaluate_bivlc_loglikelihood(
                    model, processor_or_tokenizer,
                    cache_dir=args.datasets_cache_dir,
                    limit=args.limit,
                    normalize_by_length=args.normalize_by_length,
                )
                elapsed = time.time() - start
                print(
                    f"[green]  ✓ I2T: {results['i2t_accuracy']:.4f} | "
                    f"T2I: {results['t2i_accuracy']:.4f} | "
                    f"Group: {results['group_accuracy']:.4f} "
                    f"({format_elapsed_time(elapsed)})[/green]"
                )
            else:
                results = evaluate_bivlc_mcq(
                    model, processor_or_tokenizer,
                    cache_dir=args.datasets_cache_dir,
                    limit=args.limit,
                )
                elapsed = time.time() - start
                print(
                    f"[green]  ✓ I2T: {results['i2t_accuracy']:.4f} "
                    f"({format_elapsed_time(elapsed)})[/green]"
                )

            save_results(
                results, output_dir, model_identifier,
                "bivlc", method, elapsed,
            )

        except Exception as e:
            print(f"[red]  ✗ BiVLC ({method}): {e}[/red]")


def _get_eval_methods(eval_method: str) -> list:
    """Resolve eval method choice to list of methods to run."""
    if eval_method == "both":
        return ["loglikelihood", "mcq"]
    return [eval_method]


# =============================================================================
# MAIN
# =============================================================================


def cleanup_memory() -> None:
    """Release GPU memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on ARO and BiVLC compositionality benchmarks",
    )
    parser.add_argument(
        "--model-name-or-path", type=str, required=True,
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--benchmark", type=str, default="all", choices=["aro", "bivlc", "all"],
        help="Which benchmark to evaluate (default: all)",
    )
    parser.add_argument(
        "--aro-subtasks", type=str, nargs="+", default=list(ARO_SUBTASKS),
        choices=list(ARO_SUBTASKS),
        help="ARO subtasks to evaluate (default: all 4)",
    )
    parser.add_argument(
        "--aro-data-root", type=str,
        default=DEFAULT_ARO_REPO,
        help="Path to cloned ARO repo (vision-language-models-are-bows)",
    )
    parser.add_argument(
        "--datasets-cache-dir", type=str, default=DEFAULT_DATASETS_CACHE,
        help="Cache directory for datasets (default: $HF_DATASETS_CACHE)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples per benchmark (for debugging)",
    )
    parser.add_argument(
        "--eval-method", type=str, default="both",
        choices=["loglikelihood", "mcq", "both"],
        help="Evaluation method (default: both)",
    )
    parser.add_argument(
        "--normalize-by-length", action="store_true", default=True,
        help="Normalize log-likelihood by caption length (default: True)",
    )
    parser.add_argument(
        "--no-normalize-by-length", action="store_false", dest="normalize_by_length",
        help="Disable length normalization",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (currently only batch_size=1 is supported)",
    )

    args = parser.parse_args()

    # Print configuration
    print("[bold cyan]╔══════════════════════════════════════════════════════════╗[/bold cyan]")
    print("[bold cyan]║         COMPOSITIONALITY EVALUATION                      ║[/bold cyan]")
    print("[bold cyan]╚══════════════════════════════════════════════════════════╝[/bold cyan]")
    print(f"[cyan]Model:     {args.model_name_or_path}[/cyan]")
    print(f"[cyan]Benchmark: {args.benchmark}[/cyan]")
    print(f"[cyan]Method:    {args.eval_method}[/cyan]")
    if args.limit:
        print(f"[yellow]Limit:     {args.limit} samples[/yellow]")
    print()

    # Load model
    print("[dim]Loading model...[/dim]")
    model, processor_or_tokenizer = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
    )
    model.eval()
    model_identifier = args.model_name_or_path.replace("/", "_")
    print(f"[green]✓ Model loaded: {model.__class__.__name__}[/green]")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    overall_start = time.time()

    # Run benchmarks
    if args.benchmark in ("aro", "all"):
        try:
            run_aro_evaluation(model, processor_or_tokenizer, args, model_identifier)
        except Exception as e:
            print(f"[red]✗ ARO evaluation failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    if args.benchmark in ("bivlc", "all"):
        try:
            run_bivlc_evaluation(model, processor_or_tokenizer, args, model_identifier)
        except Exception as e:
            print(f"[red]✗ BiVLC evaluation failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    overall_elapsed = time.time() - overall_start
    print()
    print(f"[bold green]✓ All evaluations complete ({format_elapsed_time(overall_elapsed)})[/bold green]")
    print(f"[cyan]  Results: {args.output_dir}[/cyan]")

    cleanup_memory()


if __name__ == "__main__":
    main()
