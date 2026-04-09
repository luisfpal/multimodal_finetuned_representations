"""
Compositionality evaluation for generative VLMs on ARO and BiVLC benchmarks.

Evaluates VLMs using three complementary methods:
  - Log-likelihood scoring: P(caption | image) via model logits (primary)
  - MCQ prompting: generate single-token answer to MCQ prompt
  - MCQ logits: constrained comparison of answer letter token logits

Benchmarks:
  - ARO (Attribution, Relation, Order): 50k+ test cases for compositional failures
    Subtasks: VG-Relation, VG-Attribution, COCO-Order, Flickr30k-Order
  - BiVLC: Bidirectional compositionality (I2T + T2I + Group score)

Supports: LLaVA 1.5/v1.6, LLaVA-OneVision, InternVL2, InternVL3.5, Cambrian

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
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import nltk
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from rich import print
from torch.utils.data import Dataset
from tqdm import tqdm

# ARO Order subtasks use nltk.word_tokenize (via TextShuffler) which needs punkt_tab.
# Auto-download once; subsequent calls are no-ops.
nltk.download("punkt_tab", quiet=True)

from utils.cambrian_utils import (
    is_cambrian_model,
    apply_chat_template as apply_cambrian_chat_template,
    generate_cambrian_batch,
    prepare_cambrian_batch_inputs,
    preprocess_cambrian_images,
    resolve_cambrian_conv_mode,
    get_cambrian_device,
)
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

ARO_SUBTASKS = ("flickr30k_order", "coco_order", "vg_relation", "vg_attribution")
BIVLC_DATASET_ID = "imirandam/BiVLC"

# SugarCrepe — 7 categories across Add / Replace / Swap perturbation types.
# Images: COCO val2017. JSONs from github.com/RAIVNLab/sugar-crepe.
SUGARCREPE_CATEGORIES = (
    "add_obj",
    "add_att",
    "replace_obj",
    "replace_att",
    "replace_rel",
    "swap_obj",
    "swap_att",
)
# Paper-standard display labels (SugarCrepe NeurIPS 2023, Table 3 column order)
SUGARCREPE_LABELS: dict[str, str] = {
    "add_obj": "Add-Obj",
    "add_att": "Add-Att",
    "replace_obj": "Rep-Obj",
    "replace_att": "Rep-Att",
    "replace_rel": "Rep-Rel",
    "swap_obj": "Swap-Obj",
    "swap_att": "Swap-Att",
}
# Type groupings for aggregate reporting
SUGARCREPE_GROUPS: dict[str, tuple[str, ...]] = {
    "Add": ("add_obj", "add_att"),
    "Replace": ("replace_obj", "replace_att", "replace_rel"),
    "Swap": ("swap_obj", "swap_att"),
}
DEFAULT_SUGARCREPE_ROOT = str(Path.home() / "scratch" / "datasets" / "sugar-crepe")

# Default sample limits per subtask (full dataset sizes in comments).
# CLI --limit overrides these.
TASK_LIMITS = {
    # "flickr30k_order": 5000,  # 4885 samples
    # "coco_order": 5000,       # 25010 samples
    # "vg_relation": 5000,      # 23937 samples
    # "vg_attribution": 5000,   # 28748 samples
    # "bivlc": 5000,            # 2933 samples
    # SugarCrepe per-category limits (comment out to use full dataset)
    # "add_obj": 2062,          # 2062 samples
    # "add_att": 692,           # 692 samples
    # "replace_obj": 1652,      # 1652 samples
    # "replace_att": 788,       # 788 samples
    # "replace_rel": 1406,      # 1406 samples
    # "swap_obj": 245,          # 245 samples
    # "swap_att": 666,          # 666 samples
}

# Neutral prompt for log-likelihood scoring
SCORING_PROMPT = "Describe this image."

# MCQ prompt templates
MCQ_BINARY_PROMPT = (
    "Which caption better describes this image?\n"
    "A) {caption_a}\n"
    "B) {caption_b}\n"
    "Answer with the option's letter from the given choices directly."
)
MCQ_5WAY_PROMPT = (
    "Which caption best describes this image?\n"
    "A) {caption_a}\n"
    "B) {caption_b}\n"
    "C) {caption_c}\n"
    "D) {caption_d}\n"
    "E) {caption_e}\n"
    "Answer with the option's letter from the given choices directly."
)


def build_mcq_prompt(captions: list[str]) -> str:
    """Build MCQ prompt from a list of captions."""
    num_choices = len(captions)
    if num_choices == 2:
        return MCQ_BINARY_PROMPT.format(caption_a=captions[0], caption_b=captions[1])
    if num_choices == 5:
        return MCQ_5WAY_PROMPT.format(
            caption_a=captions[0],
            caption_b=captions[1],
            caption_c=captions[2],
            caption_d=captions[3],
            caption_e=captions[4],
        )
    # Fallback: build dynamically
    lines = [f"{chr(ord('A') + i)}) {cap}" for i, cap in enumerate(captions)]
    return (
        "Which caption best describes this image?\n"
        + "\n".join(lines)
        + "\nAnswer with the letter only."
    )


def _shuffle_captions(
    captions: list[str],
    correct_idx: int,
    idx: int,
) -> tuple[list[str], int, list[int]]:
    """Deterministically shuffle captions to avoid position bias.

    Returns (shuffled_captions, new_correct_idx, shuffle_order).
    shuffle_order[i] = original index that ended up at position i.
    """
    rng = random.Random(RANDOM_SEED + idx)
    order = list(range(len(captions)))
    rng.shuffle(order)
    shuffled = [captions[i] for i in order]
    new_correct = order.index(correct_idx)
    return shuffled, new_correct, order


# =============================================================================
# MODEL TYPE DETECTION
# =============================================================================


def detect_model_type(model_name_or_path: str) -> str:
    """Detect model family from model name/path string."""
    name_lower = model_name_or_path.lower()
    if "cambrian" in name_lower:
        return "cambrian"
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
    full_conv = _build_llava_conversation(
        caption, has_image, processor, include_caption=True
    )
    full_text = processor.apply_chat_template(
        full_conv, tokenize=False, add_generation_prompt=False
    )

    # Build prompt only (without caption)
    prompt_conv = _build_llava_conversation(
        caption, has_image, processor, include_caption=False
    )
    prompt_text = processor.apply_chat_template(
        prompt_conv, tokenize=False, add_generation_prompt=False
    )

    # Tokenize
    proc_kwargs = {"return_tensors": "pt", "padding": False}
    if has_image:
        proc_kwargs["images"] = image
    full_inputs = processor(text=full_text, **proc_kwargs)
    prompt_inputs = processor(text=prompt_text, **proc_kwargs)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # Forward pass
    full_inputs = {
        k: v.to(device) for k, v in full_inputs.items() if isinstance(v, torch.Tensor)
    }
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


def _compute_log_likelihood_cambrian(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    caption: str,
    normalize_by_length: bool = True,
) -> float:
    """Compute log-likelihood of caption given image for Cambrian models."""
    from utils.cambrian_bridge import _ensure_cambrian_importable

    _ensure_cambrian_importable()
    from cambrian.constants import IMAGE_TOKEN_INDEX
    from cambrian.mm_utils import tokenizer_image_token_llama3

    device = get_cambrian_device(model)
    has_image = image is not None
    conv_mode = resolve_cambrian_conv_mode(model)

    # Build full conversation: user asks SCORING_PROMPT, assistant replies with caption
    from cambrian.conversation import conv_templates

    use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))

    def _build_cambrian_prompt(question: str, response: Optional[str] = None) -> str:
        from cambrian.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )

        q = question
        if has_image:
            if use_im_start_end:
                image_prefix = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
            else:
                image_prefix = DEFAULT_IMAGE_TOKEN
            q = f"{image_prefix}\n{q}"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], response)
        return conv.get_prompt()

    # Full prompt (with caption as assistant response)
    full_text = _build_cambrian_prompt(SCORING_PROMPT, caption)
    # Prompt only (without caption, for measuring prompt length)
    prompt_text = _build_cambrian_prompt(SCORING_PROMPT, None)

    # Tokenize using Cambrian's image-token-aware tokenizer
    tokenize_fn = (
        tokenizer_image_token_llama3 if conv_mode in {"llama_3", "llama_v3"} else None
    )
    if tokenize_fn is None:
        from cambrian.mm_utils import tokenizer_image_token

        tokenize_fn = tokenizer_image_token

    full_ids = tokenize_fn(
        full_text, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    prompt_ids = tokenize_fn(
        prompt_text, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
    )

    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    full_ids = full_ids.to(device)
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # Forward pass
    forward_kwargs: dict[str, Any] = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
        "use_cache": False,
    }
    if has_image:
        image_tensors = preprocess_cambrian_images(
            [image], tokenizer.image_processor, model, device, torch.float16
        )
        forward_kwargs["images"] = image_tensors
        forward_kwargs["image_sizes"] = [image.size]

    with torch.no_grad():
        outputs = model(**forward_kwargs)
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
    if is_cambrian_model(model):
        return _compute_log_likelihood_cambrian(
            model, processor_or_tokenizer, image, caption, normalize_by_length
        )
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


def _build_llava_prompt(
    processor: Any,
    image: Optional[Image.Image],
    question: str,
) -> tuple[str, dict]:
    """Build LLaVA prompt and processor inputs. Returns (prompt_text, inputs_dict)."""
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
    inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    return prompt_text, inputs


def _generate_response_llava(
    model: torch.nn.Module,
    processor: Any,
    image: Optional[Image.Image],
    question: str,
    max_new_tokens: int = 1,
) -> tuple[str, str]:
    """Generate a text response from LLaVA model. Returns (response, prompt)."""
    prompt_text, inputs = _build_llava_prompt(processor, image, question)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    # Strip input tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response, prompt_text


def _clean_internvl_prompt_for_saving(query: str) -> str:
    """Remove InternVL image placeholder tokens for human-readable saved prompt."""
    from utils.internvl_utils import IMG_CONTEXT_TOKEN

    cleaned = query.replace(IMG_CONTEXT_TOKEN, "")
    cleaned = cleaned.replace("</img>", "").replace("<img>", "<image>")
    return cleaned


def _build_internvl_inputs(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
) -> tuple[str, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Build InternVL inputs. Returns (cleaned_prompt, input_ids, attention_mask, inputs_embeds_or_None)."""
    device = model.device
    pixel_values = None
    num_patches_list = None

    if image is not None:
        pv, num_patches_list = get_pixel_values(image)
        pixel_values = pv.to(device)

    queries = apply_internvl_chat_template(
        model=model,
        text=[question],
        num_patches_list=num_patches_list,
    )
    query = queries[0]
    cleaned_prompt = _clean_internvl_prompt_for_saving(query)

    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    img_ctx_id = _get_internvl_img_context_token_id(tokenizer)
    inputs_embeds = None
    if pixel_values is not None:
        inputs_embeds = get_combined_embeddings(
            model, input_ids, pixel_values, img_context_token_id=img_ctx_id
        )

    return cleaned_prompt, input_ids, attention_mask, inputs_embeds


def _generate_response_internvl(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    max_new_tokens: int = 1,
) -> tuple[str, str]:
    """Generate a text response from InternVL model. Returns (response, prompt)."""
    from utils.internvl_utils import generate_responses

    cleaned_prompt, input_ids, attention_mask, inputs_embeds = _build_internvl_inputs(
        model, tokenizer, image, question
    )

    if inputs_embeds is not None:
        responses = generate_responses(
            model,
            tokenizer,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
    else:
        responses = generate_responses(
            model,
            tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    return responses[0], cleaned_prompt


def _generate_response_cambrian(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    max_new_tokens: int = 1,
) -> tuple[str, str]:
    """Generate a text response from Cambrian model. Returns (response, prompt)."""
    # Build prompt for saving (same template logic as generate_cambrian_batch)
    prompts = apply_cambrian_chat_template(
        model=model,
        text=[question],
        has_image=image is not None,
    )
    prompt_text = prompts[0]

    images = [image] if image is not None else None
    responses = generate_cambrian_batch(
        model=model,
        tokenizer=tokenizer,
        questions=[question],
        images=images,
        max_new_tokens=max_new_tokens,
    )
    return responses[0], prompt_text


def generate_response(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    max_new_tokens: int = 1,
) -> tuple[str, str]:
    """Generate response, dispatching to model-specific implementation.

    Returns (response_text, prompt_text).
    """
    if is_cambrian_model(model):
        return _generate_response_cambrian(
            model, processor_or_tokenizer, image, question, max_new_tokens
        )
    if is_internvl_model(model):
        return _generate_response_internvl(
            model, processor_or_tokenizer, image, question, max_new_tokens
        )
    return _generate_response_llava(
        model, processor_or_tokenizer, image, question, max_new_tokens
    )


def parse_mcq_answer(response: str, num_choices: int = 2) -> Optional[int]:
    """Parse MCQ answer from model response. Returns 0-indexed choice or None.

    Designed for single-token generation (max_new_tokens=1). The decoded token
    is typically a bare letter ("A"), possibly with leading space (" A"), or
    letter + punctuation ("B."). We intentionally avoid heuristic fallbacks
    (e.g., regex for "answer is X") that inflate accuracy with verbose output.
    """
    import re

    valid_letters = [chr(ord("A") + i) for i in range(num_choices)]
    response = response.strip().upper()

    # 1. Exact single letter (e.g., "A", "B")
    if response in valid_letters:
        return valid_letters.index(response)

    # 2. Letter followed by punctuation (e.g., "B.", "B)", "B:")
    match = re.match(r"^([A-Z])[.):\s]", response)
    if match and match.group(1) in valid_letters:
        return valid_letters.index(match.group(1))

    return None


# =============================================================================
# CONSTRAINED LOGITS MCQ
# =============================================================================


def _get_answer_token_ids(
    tokenizer: Any,
    num_choices: int,
) -> list[int]:
    """Get token IDs for answer letters A, B, C, ... up to num_choices."""
    ids = []
    for i in range(num_choices):
        letter = chr(ord("A") + i)
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        # Single letters are single tokens in all supported tokenizers
        ids.append(encoded[0])
    return ids


def _pick_answer_from_logits(
    logits: torch.Tensor,
    answer_token_ids: list[int],
) -> tuple[int, str]:
    """Pick the answer letter with highest logit among answer tokens.

    Args:
        logits: logits at the last input position, shape (vocab_size,)
        answer_token_ids: token IDs for answer letters [A, B, ...]

    Returns:
        (predicted_index, predicted_letter) — 0-indexed answer and letter string.
    """
    answer_logits = logits[answer_token_ids]
    predicted = int(answer_logits.argmax().item())
    letter = chr(ord("A") + predicted)
    return predicted, letter


def _get_mcq_logits_llava(
    model: torch.nn.Module,
    processor: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ for LLaVA. Returns (predicted_idx, predicted_letter, prompt)."""
    prompt_text, inputs = _build_llava_prompt(processor, image, question)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at last input position
    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, prompt_text


def _get_mcq_logits_internvl(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ for InternVL. Returns (predicted_idx, predicted_letter, prompt)."""
    cleaned_prompt, input_ids, attention_mask, inputs_embeds = _build_internvl_inputs(
        model, tokenizer, image, question
    )

    with torch.no_grad():
        if inputs_embeds is not None:
            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        else:
            outputs = model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, cleaned_prompt


def _get_mcq_logits_cambrian(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ for Cambrian. Returns (predicted_idx, predicted_letter, prompt)."""
    sample_images = [image] if image is not None else None
    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=[question],
        images=sample_images,
    )
    prompt_text = prepared["prompts"][0]

    input_ids = prepared["tokenized"]["input_ids"]
    attention_mask = prepared["tokenized"]["attention_mask"]

    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if prepared["image_tensors"] is not None:
        forward_kwargs["images"] = prepared["image_tensors"]
        forward_kwargs["image_sizes"] = prepared["image_sizes"]

    with torch.inference_mode():
        outputs = model(**forward_kwargs)

    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, prompt_text


def get_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ, dispatching to model-specific implementation.

    Returns (predicted_index, predicted_letter, prompt_text).
    """
    if is_cambrian_model(model):
        return _get_mcq_logits_cambrian(
            model, processor_or_tokenizer, image, question, answer_token_ids
        )
    if is_internvl_model(model):
        return _get_mcq_logits_internvl(
            model, processor_or_tokenizer, image, question, answer_token_ids
        )
    return _get_mcq_logits_llava(
        model, processor_or_tokenizer, image, question, answer_token_ids
    )


# =============================================================================
# LIMIT RESOLUTION
# =============================================================================


def _resolve_limit(subtask: str, cli_limit: Optional[int]) -> Optional[int]:
    """Resolve sample limit: CLI --limit overrides per-subtask defaults."""
    if cli_limit is not None:
        return cli_limit
    return TASK_LIMITS.get(subtask)


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
        n = len(self.dataset)
        if limit and limit < n:
            rng = random.Random(RANDOM_SEED)
            self.indices = sorted(rng.sample(range(n), limit))
        else:
            self.indices = list(range(n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[idx]]
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
        # Order datasets need split="test"; Flickr30k_Order requires it as positional arg
        self.dataset = DatasetClass(
            image_preprocess=None, split="test", download=True, root_dir=aro_path
        )
        n = len(self.dataset)
        if limit and limit < n:
            rng = random.Random(RANDOM_SEED)
            self.indices = sorted(rng.sample(range(n), limit))
        else:
            self.indices = list(range(n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[idx]]
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


_ARO_ORDER_IMAGE_DIRS: dict[str, tuple[str, str]] = {
    "coco_order": ("val2014", "COCO val2014"),
    "flickr30k_order": ("flickr30k-images", "Flickr30k"),
}


def validate_aro_image_dirs(subtasks: list[str], aro_data_root: str) -> None:
    """Fail-fast: verify image directories exist before loading model."""
    aro_path = os.path.expanduser(aro_data_root)
    missing = []
    for st in subtasks:
        if st in _ARO_ORDER_IMAGE_DIRS:
            subdir, name = _ARO_ORDER_IMAGE_DIRS[st]
            if not os.path.isdir(os.path.join(aro_path, subdir)):
                missing.append((st, name, os.path.join(aro_path, subdir)))
    if missing:
        lines = ["Missing image directories for ARO order subtasks:"]
        for st, name, path in missing:
            lines.append(f"  - {st}: {name} images not found at {path}")
        lines.append("")
        lines.append("Download them with:")
        lines.append(
            f"  python scripts/data/download_aro_images.py --aro-data-root {aro_path}"
        )
        raise FileNotFoundError("\n".join(lines))


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
        rng = random.Random(RANDOM_SEED)
        indices = sorted(rng.sample(range(len(dataset)), limit))
        dataset = dataset.select(indices)
    return dataset


# =============================================================================
# SUGARCREPE DATASET
# =============================================================================


def load_sugarcrepe_data(
    data_root: str,
    category: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """Load one SugarCrepe category JSON. Returns list of dicts with filename/caption/negative_caption."""
    json_path = os.path.join(data_root, "data", f"{category}.json")
    with open(json_path) as f:
        raw = json.load(f)
    samples = list(raw.values())
    if limit and limit < len(samples):
        rng = random.Random(RANDOM_SEED)
        samples = [samples[i] for i in sorted(rng.sample(range(len(samples)), limit))]
    return samples


def validate_sugarcrepe_data(data_root: str) -> None:
    """Fail fast if SugarCrepe JSON files or COCO val2017 images directory are missing."""
    data_dir = os.path.join(data_root, "data")
    img_dir = os.path.join(data_root, "val2017")
    missing = [
        c
        for c in SUGARCREPE_CATEGORIES
        if not os.path.isfile(os.path.join(data_dir, f"{c}.json"))
    ]
    if missing:
        raise FileNotFoundError(
            f"SugarCrepe JSON files missing for: {missing}\n"
            f"  Expected in: {data_dir}\n"
            f"  Run: bash shell/data/download_sugarcrepe.sh"
        )
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"COCO val2017 image directory not found: {img_dir}\n"
            f"  Run: bash shell/data/download_sugarcrepe.sh"
        )


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

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(total=n_samples, desc=f"ARO {subtask} (loglik)", unit="sample")
    update_count = 0

    skipped = 0
    for idx in range(n_samples):
        try:
            item = dataset[idx]
        except (FileNotFoundError, OSError):
            skipped += 1
            update_count += 1
            if update_count >= update_every or idx == n_samples - 1:
                progress_bar.update(update_count)
                update_count = 0
            continue

        image = item["image"]
        captions = item["caption_options"]
        correct_idx = item["correct_index"]

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

        per_sample.append(
            {
                "idx": idx,
                "scores": scores,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()
    if skipped:
        print(
            f"[yellow]  ⚠ Skipped {skipped}/{n_samples} samples (missing images)[/yellow]"
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
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

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(total=n_samples, desc="BiVLC (loglik)", unit="sample")
    update_count = 0

    for idx in range(n_samples):
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
            score_pos_cap_pos_img = compute_caption_log_likelihood(
                model, processor_or_tokenizer, image, caption, normalize_by_length
            )
            score_pos_cap_neg_img = compute_caption_log_likelihood(
                model, processor_or_tokenizer, neg_image, caption, normalize_by_length
            )
            score_neg_cap_pos_img = compute_caption_log_likelihood(
                model, processor_or_tokenizer, image, neg_caption, normalize_by_length
            )
            score_neg_cap_neg_img = compute_caption_log_likelihood(
                model,
                processor_or_tokenizer,
                neg_image,
                neg_caption,
                normalize_by_length,
            )
            Ipos2T = score_pos_cap_pos_img > score_neg_cap_pos_img
            InegT = score_neg_cap_neg_img > score_pos_cap_neg_img
            TposI = score_pos_cap_pos_img > score_pos_cap_neg_img
            TnegI = score_neg_cap_neg_img > score_neg_cap_pos_img

            i2t_ok = Ipos2T and InegT
            t2i_ok = TposI and TnegI

            group_ok = i2t_ok and t2i_ok

        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            i2t_ok = t2i_ok = group_ok = False
            score_pos_cap_pos_img = score_pos_cap_neg_img = score_neg_cap_pos_img = (
                score_neg_cap_neg_img
            ) = float("nan")

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

        per_sample.append(
            {
                "idx": idx,
                "caption": caption,
                "neg_caption": neg_caption,
                "i2t_correct": i2t_ok,
                "t2i_correct": t2i_ok,
                "group_correct": group_ok,
                "scores": {
                    "pos_cap_pos_img": score_pos_cap_pos_img,
                    "pos_cap_neg_img": score_pos_cap_neg_img,
                    "neg_cap_pos_img": score_neg_cap_pos_img,
                    "neg_cap_neg_img": score_neg_cap_neg_img,
                },
                "type": comp_type,
                "subtype": comp_subtype,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()

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

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(total=n_samples, desc=f"ARO {subtask} (mcq)", unit="sample")
    update_count = 0

    skipped = 0
    for idx in range(n_samples):
        try:
            item = dataset[idx]
        except (FileNotFoundError, OSError):
            skipped += 1
            update_count += 1
            if update_count >= update_every or idx == n_samples - 1:
                progress_bar.update(update_count)
                update_count = 0
            continue

        image = item["image"]
        captions = item["caption_options"]
        original_correct_idx = item["correct_index"]
        num_choices = len(captions)

        # Shuffle to avoid position bias (ARO has fixed correct positions)
        shuffled, correct_idx, shuffle_order = _shuffle_captions(
            captions, original_correct_idx, idx
        )
        prompt = build_mcq_prompt(shuffled)

        try:
            response, prompt_text = generate_response(
                model, processor_or_tokenizer, image, prompt
            )
            predicted_idx = parse_mcq_answer(response, num_choices)
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            response = ""
            prompt_text = prompt
            predicted_idx = None

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append(
            {
                "idx": idx,
                "prompt": prompt_text,
                "response": response,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "shuffle_order": shuffle_order,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()
    if skipped:
        print(
            f"[yellow]  ⚠ Skipped {skipped}/{n_samples} samples (missing images)[/yellow]"
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
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

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(total=n_samples, desc="BiVLC I2T (mcq)", unit="sample")
    update_count = 0

    for idx in range(n_samples):
        item = dataset[idx]
        image = item["image"]
        caption = item["caption"]
        neg_caption = item["negative_caption"]
        comp_type = item.get("type", "unknown")
        comp_subtype = item.get("subtype", "unknown")

        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Randomize order to avoid position bias, but track correct answer
        options = [caption, neg_caption]
        options, correct_idx, shuffle_order = _shuffle_captions(options, 0, idx)

        prompt = build_mcq_prompt(options)

        try:
            response, prompt_text = generate_response(
                model, processor_or_tokenizer, image, prompt
            )
            predicted_idx = parse_mcq_answer(response, 2)
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            response = ""
            prompt_text = prompt
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

        per_sample.append(
            {
                "idx": idx,
                "prompt": prompt_text,
                "response": response,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": i2t_ok,
                "type": comp_type,
                "subtype": comp_subtype,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()

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
# BENCHMARK EVALUATION — MCQ LOGITS (CONSTRAINED)
# =============================================================================


def evaluate_aro_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    subtask: str,
    aro_repo_root: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a single ARO subtask using constrained logits MCQ.

    Compares logits of answer letter tokens (A, B, ...) at the last input
    position. The letter with the highest logit wins. No generation or parsing.
    """
    dataset = load_aro_dataset(subtask, aro_repo_root, limit)

    # Determine tokenizer for answer token IDs
    tokenizer = (
        processor_or_tokenizer.tokenizer
        if hasattr(processor_or_tokenizer, "tokenizer")
        else processor_or_tokenizer
    )
    # Pre-compute answer token IDs (max possible choices across ARO subtasks)
    max_choices = 5
    all_answer_ids = _get_answer_token_ids(tokenizer, max_choices)

    correct = 0
    total = 0
    per_sample: list = []

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(
        total=n_samples, desc=f"ARO {subtask} (mcq_logits)", unit="sample"
    )
    update_count = 0

    skipped = 0
    for idx in range(n_samples):
        try:
            item = dataset[idx]
        except (FileNotFoundError, OSError):
            skipped += 1
            update_count += 1
            if update_count >= update_every or idx == n_samples - 1:
                progress_bar.update(update_count)
                update_count = 0
            continue

        image = item["image"]
        captions = item["caption_options"]
        original_correct_idx = item["correct_index"]
        num_choices = len(captions)
        answer_token_ids = all_answer_ids[:num_choices]

        # Shuffle to avoid position bias
        shuffled, correct_idx, shuffle_order = _shuffle_captions(
            captions, original_correct_idx, idx
        )
        prompt = build_mcq_prompt(shuffled)

        try:
            predicted_idx, predicted_token, prompt_text = get_mcq_logits(
                model, processor_or_tokenizer, image, prompt, answer_token_ids
            )
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            predicted_idx = None
            predicted_token = ""
            prompt_text = prompt

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append(
            {
                "idx": idx,
                "prompt": prompt_text,
                "predicted_token": predicted_token,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "shuffle_order": shuffle_order,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()
    if skipped:
        print(
            f"[yellow]  ⚠ Skipped {skipped}/{n_samples} samples (missing images)[/yellow]"
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
        "per_sample": per_sample,
    }


def evaluate_bivlc_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    cache_dir: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate BiVLC I2T using constrained logits MCQ.

    T2I is skipped (requires multi-image prompting). Use log-likelihood for T2I.
    """
    dataset = load_bivlc_dataset(cache_dir, limit)

    tokenizer = (
        processor_or_tokenizer.tokenizer
        if hasattr(processor_or_tokenizer, "tokenizer")
        else processor_or_tokenizer
    )
    answer_token_ids = _get_answer_token_ids(tokenizer, 2)

    i2t_correct = 0
    total = 0
    per_type: Dict[str, Dict[str, int]] = {}
    per_sample: list = []

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(total=n_samples, desc="BiVLC I2T (mcq_logits)", unit="sample")
    update_count = 0

    for idx in range(n_samples):
        item = dataset[idx]
        image = item["image"]
        caption = item["caption"]
        neg_caption = item["negative_caption"]
        comp_type = item.get("type", "unknown")
        comp_subtype = item.get("subtype", "unknown")

        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Randomize order to avoid position bias
        options = [caption, neg_caption]
        options, correct_idx, shuffle_order = _shuffle_captions(options, 0, idx)

        prompt = build_mcq_prompt(options)

        try:
            predicted_idx, predicted_token, prompt_text = get_mcq_logits(
                model, processor_or_tokenizer, image, prompt, answer_token_ids
            )
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            predicted_idx = None
            predicted_token = ""
            prompt_text = prompt

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

        per_sample.append(
            {
                "idx": idx,
                "prompt": prompt_text,
                "predicted_token": predicted_token,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": i2t_ok,
                "type": comp_type,
                "subtype": comp_subtype,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()

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
# BENCHMARK EVALUATION — SUGARCREPE
# =============================================================================


def evaluate_sugarcrepe_mcq(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    data_root: str,
    category: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one SugarCrepe category using MCQ prompting.

    Binary choice: positive caption vs. hard negative caption.
    Captions are shuffled to eliminate position bias.
    """
    samples = load_sugarcrepe_data(data_root, category, limit)
    img_dir = os.path.join(data_root, "val2017")

    correct = 0
    total = 0
    per_sample: list = []

    n_samples = len(samples)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(
        total=n_samples, desc=f"SugarCrepe {category} (mcq)", unit="sample"
    )
    update_count = 0

    for idx, sample in enumerate(samples):
        image = Image.open(os.path.join(img_dir, sample["filename"])).convert("RGB")
        caption = sample["caption"]
        neg_caption = sample["negative_caption"]

        options, correct_idx, shuffle_order = _shuffle_captions(
            [caption, neg_caption], 0, idx
        )
        prompt = build_mcq_prompt(options)

        try:
            response, prompt_text = generate_response(
                model, processor_or_tokenizer, image, prompt
            )
            predicted_idx = parse_mcq_answer(response, 2)
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            response = ""
            prompt_text = prompt
            predicted_idx = None

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append(
            {
                "idx": idx,
                "filename": sample["filename"],
                "prompt": prompt_text,
                "response": response,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "shuffle_order": shuffle_order,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "num_correct": correct,
        "num_total": total,
        "per_sample": per_sample,
    }


def evaluate_sugarcrepe_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    data_root: str,
    category: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one SugarCrepe category using constrained logits MCQ.

    Compares logits of answer letter tokens (A, B) at the last input position.
    No generation or parsing — always produces a valid answer.
    """
    samples = load_sugarcrepe_data(data_root, category, limit)
    img_dir = os.path.join(data_root, "val2017")

    tokenizer = (
        processor_or_tokenizer.tokenizer
        if hasattr(processor_or_tokenizer, "tokenizer")
        else processor_or_tokenizer
    )
    answer_token_ids = _get_answer_token_ids(tokenizer, 2)

    correct = 0
    total = 0
    per_sample: list = []

    n_samples = len(samples)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(
        total=n_samples, desc=f"SugarCrepe {category} (mcq_logits)", unit="sample"
    )
    update_count = 0

    for idx, sample in enumerate(samples):
        image = Image.open(os.path.join(img_dir, sample["filename"])).convert("RGB")
        caption = sample["caption"]
        neg_caption = sample["negative_caption"]

        options, correct_idx, shuffle_order = _shuffle_captions(
            [caption, neg_caption], 0, idx
        )
        prompt = build_mcq_prompt(options)

        try:
            predicted_idx, predicted_token, prompt_text = get_mcq_logits(
                model, processor_or_tokenizer, image, prompt, answer_token_ids
            )
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            predicted_idx = None
            predicted_token = ""
            prompt_text = prompt

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append(
            {
                "idx": idx,
                "filename": sample["filename"],
                "prompt": prompt_text,
                "predicted_token": predicted_token,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "shuffle_order": shuffle_order,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "num_correct": correct,
        "num_total": total,
        "per_sample": per_sample,
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

    # Save per-sample responses (include benchmark in filename to avoid overwrites)
    suffix = f"_{eval_method}" if eval_method != "loglikelihood" else ""
    json_path = os.path.join(
        output_dir, f"{benchmark}_{model_identifier}{suffix}_responses.json"
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
        resolved_limit = _resolve_limit(subtask, args.limit)
        print(f"\n[cyan]▶ {subtask}[/cyan]")
        if resolved_limit:
            print(f"[yellow]  Sample limit: {resolved_limit}[/yellow]")
        output_dir = os.path.join(args.output_dir, "aro")

        for method in eval_methods:
            print(f"[dim]  Method: {method}[/dim]")
            start = time.time()

            try:
                if method == "loglikelihood":
                    results = evaluate_aro_loglikelihood(
                        model,
                        processor_or_tokenizer,
                        subtask,
                        args.aro_data_root,
                        limit=resolved_limit,
                        normalize_by_length=args.normalize_by_length,
                    )
                elif method == "mcq_logits":
                    results = evaluate_aro_mcq_logits(
                        model,
                        processor_or_tokenizer,
                        subtask,
                        args.aro_data_root,
                        limit=resolved_limit,
                    )
                else:
                    results = evaluate_aro_mcq(
                        model,
                        processor_or_tokenizer,
                        subtask,
                        args.aro_data_root,
                        limit=resolved_limit,
                    )

                elapsed = time.time() - start
                acc_key = "accuracy"
                print(
                    f"[green]  ✓ {subtask} ({method}): "
                    f"{results[acc_key]:.4f} ({format_elapsed_time(elapsed)})[/green]"
                )

                save_results(
                    results,
                    output_dir,
                    model_identifier,
                    f"aro_{subtask}",
                    method,
                    elapsed,
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
    resolved_limit = _resolve_limit("bivlc", args.limit)

    if resolved_limit:
        print(f"[yellow]  Sample limit: {resolved_limit}[/yellow]")

    for method in eval_methods:
        print(f"\n[cyan]▶ BiVLC ({method})[/cyan]")
        start = time.time()

        try:
            if method == "loglikelihood":
                results = evaluate_bivlc_loglikelihood(
                    model,
                    processor_or_tokenizer,
                    cache_dir=args.datasets_cache_dir,
                    limit=resolved_limit,
                    normalize_by_length=args.normalize_by_length,
                )
                elapsed = time.time() - start
                print(
                    f"[green]  ✓ I2T: {results['i2t_accuracy']:.4f} | "
                    f"T2I: {results['t2i_accuracy']:.4f} | "
                    f"Group: {results['group_accuracy']:.4f} "
                    f"({format_elapsed_time(elapsed)})[/green]"
                )
            elif method == "mcq_logits":
                results = evaluate_bivlc_mcq_logits(
                    model,
                    processor_or_tokenizer,
                    cache_dir=args.datasets_cache_dir,
                    limit=resolved_limit,
                )
                elapsed = time.time() - start
                print(
                    f"[green]  ✓ I2T: {results['i2t_accuracy']:.4f} "
                    f"({format_elapsed_time(elapsed)})[/green]"
                )
            else:
                results = evaluate_bivlc_mcq(
                    model,
                    processor_or_tokenizer,
                    cache_dir=args.datasets_cache_dir,
                    limit=resolved_limit,
                )
                elapsed = time.time() - start
                print(
                    f"[green]  ✓ I2T: {results['i2t_accuracy']:.4f} "
                    f"({format_elapsed_time(elapsed)})[/green]"
                )

            save_results(
                results,
                output_dir,
                model_identifier,
                "bivlc",
                method,
                elapsed,
            )

        except Exception as e:
            print(f"[red]  ✗ BiVLC ({method}): {e}[/red]")


def _get_eval_methods(eval_method: str) -> list:
    """Resolve eval method choice to list of methods to run."""
    if eval_method == "all":
        return ["loglikelihood", "mcq", "mcq_logits"]
    return [eval_method]


def run_sugarcrepe_evaluation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    args: argparse.Namespace,
    model_identifier: str,
) -> None:
    """Run full SugarCrepe evaluation.

    Supports mcq and mcq_logits only. loglikelihood raises NotImplementedError.
    Results are saved as:
      - Per-category response JSONs (one per category per method)
      - One aggregate JSONL entry per method with per_category, per_type, and mean accuracy
        in paper-standard column order (Add-Obj/Add-Att/Rep-Obj/Rep-Att/Rep-Rel/Swap-Obj/Swap-Att).
    """
    print("[bold magenta]=== SugarCrepe Compositionality Benchmark ===[/bold magenta]")

    output_dir = os.path.join(args.output_dir, "sugarcrepe")
    os.makedirs(output_dir, exist_ok=True)

    eval_methods = [
        m for m in _get_eval_methods(args.eval_method) if m != "loglikelihood"
    ]
    if "loglikelihood" in _get_eval_methods(args.eval_method):
        print(
            "[yellow]  ⚠ loglikelihood skipped — not implemented for SugarCrepe[/yellow]"
        )

    for method in eval_methods:
        print(f"\n[cyan]▶ SugarCrepe ({method})[/cyan]")
        method_start = time.time()

        category_accs: dict[str, float] = {}
        total_samples = 0

        for category in args.sugarcrepe_categories:
            limit = _resolve_limit(category, args.limit)
            start = time.time()
            try:
                if method == "mcq_logits":
                    results = evaluate_sugarcrepe_mcq_logits(
                        model,
                        processor_or_tokenizer,
                        args.sugarcrepe_data_root,
                        category,
                        limit,
                    )
                else:
                    results = evaluate_sugarcrepe_mcq(
                        model,
                        processor_or_tokenizer,
                        args.sugarcrepe_data_root,
                        category,
                        limit,
                    )
                elapsed = time.time() - start
                acc = results["accuracy"]
                category_accs[category] = acc
                total_samples += results["num_total"]

                print(
                    f"[green]  ✓ {SUGARCREPE_LABELS[category]}: "
                    f"{acc * 100:.1f}% "
                    f"({results['num_correct']}/{results['num_total']}, "
                    f"{format_elapsed_time(elapsed)})[/green]"
                )

                # Save per-category responses
                per_sample = results.pop("per_sample", [])
                suffix = f"_{method}" if method != "loglikelihood" else ""
                resp_path = os.path.join(
                    output_dir,
                    f"sugarcrepe_{category}_{model_identifier}{suffix}_responses.json",
                )
                with open(resp_path, "w") as f:
                    json.dump(per_sample, f, indent=2, default=str)
                print(f"[green]    Samples → {resp_path}[/green]")

            except Exception as e:
                print(f"[red]  ✗ {category} ({method}): {e}[/red]")
                import traceback

                traceback.print_exc()

        if not category_accs:
            continue

        # Build paper-standard aggregate (per_category, per_type, mean)
        per_category = {
            SUGARCREPE_LABELS[c]: category_accs[c]
            for c in SUGARCREPE_CATEGORIES
            if c in category_accs
        }
        per_type: dict[str, float] = {}
        for grp, cats in SUGARCREPE_GROUPS.items():
            done = [c for c in cats if c in category_accs]
            if done:
                per_type[grp] = sum(category_accs[c] for c in done) / len(done)
        mean_acc = sum(per_category.values()) / len(per_category)
        total_elapsed = time.time() - method_start

        # Print paper-ready table
        col_order = [
            SUGARCREPE_LABELS[c] for c in SUGARCREPE_CATEGORIES if c in category_accs
        ]
        header = "  ".join(f"{lbl:>7}" for lbl in col_order)
        values = "  ".join(f"{per_category[lbl] * 100:>7.1f}" for lbl in col_order)
        grp_header = "  ".join(f"{g:>7}" for g in per_type)
        grp_values = "  ".join(f"{v * 100:>7.1f}" for v in per_type.values())
        print(f"\n[bold cyan]  SugarCrepe ({method}) — paper table[/bold cyan]")
        print(f"  {header}  │  {grp_header}  │    Mean")
        print(f"  {values}  │  {grp_values}  │  {mean_acc * 100:>6.2f}%")

        # Append aggregate to JSONL
        aggregate = {
            "experiment_name": f"single/{model_identifier}",
            "eval_method": method,
            "per_category": per_category,
            "per_type": per_type,
            "mean": mean_acc,
            "num_total": total_samples,
            "elapsed_time": format_elapsed_time(total_elapsed),
        }
        jsonl_path = os.path.join(output_dir, "sugarcrepe_results.jsonl")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(aggregate) + "\n")
        print(f"[green]  Aggregate → {jsonl_path}[/green]")


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
        description="Evaluate VLMs on ARO, BiVLC, and SugarCrepe compositionality benchmarks",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["aro", "bivlc", "sugarcrepe", "all"],
        help="Which benchmark to evaluate (default: all)",
    )
    parser.add_argument(
        "--aro-subtasks",
        type=str,
        nargs="+",
        default=list(ARO_SUBTASKS),
        choices=list(ARO_SUBTASKS),
        help="ARO subtasks to evaluate (default: all 4)",
    )
    parser.add_argument(
        "--aro-data-root",
        type=str,
        default=DEFAULT_ARO_REPO,
        help="Path to cloned ARO repo (vision-language-models-are-bows)",
    )
    parser.add_argument(
        "--sugarcrepe-data-root",
        type=str,
        default=DEFAULT_SUGARCREPE_ROOT,
        help="Path to SugarCrepe data dir (must contain data/*.json and val2017/)",
    )
    parser.add_argument(
        "--sugarcrepe-categories",
        type=str,
        nargs="+",
        default=list(SUGARCREPE_CATEGORIES),
        choices=list(SUGARCREPE_CATEGORIES),
        help="SugarCrepe categories to evaluate (default: all 7)",
    )
    parser.add_argument(
        "--datasets-cache-dir",
        type=str,
        default=DEFAULT_DATASETS_CACHE,
        help="Cache directory for datasets (default: $HF_DATASETS_CACHE)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per subtask/category (for debugging)",
    )
    parser.add_argument(
        "--eval-method",
        type=str,
        default="all",
        choices=["loglikelihood", "mcq", "mcq_logits", "all"],
        help="Evaluation method (default: all)",
    )
    parser.add_argument(
        "--normalize-by-length",
        action="store_true",
        default=True,
        help="Normalize log-likelihood by caption length (default: True)",
    )
    parser.add_argument(
        "--no-normalize-by-length",
        action="store_false",
        dest="normalize_by_length",
        help="Disable length normalization",
    )

    args = parser.parse_args()

    # Print configuration
    print(
        "[bold cyan]╔══════════════════════════════════════════════════════════╗[/bold cyan]"
    )
    print(
        "[bold cyan]║         COMPOSITIONALITY EVALUATION                      ║[/bold cyan]"
    )
    print(
        "[bold cyan]╚══════════════════════════════════════════════════════════╝[/bold cyan]"
    )
    print(f"[cyan]Model:     {args.model_name_or_path}[/cyan]")
    print(f"[cyan]Benchmark: {args.benchmark}[/cyan]")
    print(f"[cyan]Method:    {args.eval_method}[/cyan]")
    if args.benchmark in ("sugarcrepe", "all"):
        print(f"[cyan]SugarCrepe: {args.sugarcrepe_data_root}[/cyan]")
    if args.limit:
        print(f"[yellow]Limit:     {args.limit} samples[/yellow]")
    print()

    # Validate data availability and method compatibility before expensive model load
    if args.benchmark in ("aro", "all"):
        validate_aro_image_dirs(args.aro_subtasks, args.aro_data_root)
    if args.benchmark in ("sugarcrepe", "all"):
        validate_sugarcrepe_data(args.sugarcrepe_data_root)
    # Block loglikelihood early (before model load) only when explicitly requested for sugarcrepe
    if args.benchmark == "sugarcrepe" and args.eval_method == "loglikelihood":
        raise NotImplementedError(
            "SugarCrepe log-likelihood is not implemented. "
            "Use --eval-method mcq or mcq_logits."
        )

    # Load model
    print("[dim]Loading model...[/dim]")
    model, processor_or_tokenizer = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
        # attn_implementation="sdpa",
        # dtype=torch.float16,
    )
    model.eval()
    model_identifier = args.model_name_or_path.replace("/", "_")
    print(f"[green]✓ Model loaded: {model.__class__.__name__}[/green]")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    overall_start = time.time()

    # Run benchmarks
    if args.benchmark in ("bivlc", "all"):
        try:
            run_bivlc_evaluation(model, processor_or_tokenizer, args, model_identifier)
        except Exception as e:
            print(f"[red]✗ BiVLC evaluation failed: {e}[/red]")
            import traceback

            traceback.print_exc()

    if args.benchmark in ("aro", "all"):
        try:
            run_aro_evaluation(model, processor_or_tokenizer, args, model_identifier)
        except Exception as e:
            print(f"[red]✗ ARO evaluation failed: {e}[/red]")
            import traceback

            traceback.print_exc()

    if args.benchmark in ("sugarcrepe", "all"):
        try:
            run_sugarcrepe_evaluation(
                model, processor_or_tokenizer, args, model_identifier
            )
        except Exception as e:
            print(f"[red]✗ SugarCrepe evaluation failed: {e}[/red]")
            import traceback

            traceback.print_exc()

    overall_elapsed = time.time() - overall_start
    print()
    print(
        f"[bold green]✓ All evaluations complete ({format_elapsed_time(overall_elapsed)})[/bold green]"
    )
    print(f"[cyan]  Results: {args.output_dir}[/cyan]")

    cleanup_memory()


if __name__ == "__main__":
    main()
