"""
Utilities for captioned task variants in lmms-eval.

These tasks replace images with text captions, enabling text-only VLM evaluation
while reusing the original scoring/metrics pipelines.

Architecture:
    - doc_to_visual returns [] → models operate in text-only mode
    - doc_to_text prepends a caption context block before the original question
    - process_results / aggregation functions are re-exported from the library

No library code is modified.
"""

from typing import List, Optional

import pandas as pd

# =============================================================================
# CAPTION FORMATTING
# =============================================================================

SINGLE_CAPTION_PREFIX = (
    "\nBelow is a description of an image. Use it to answer the question.\n\n"
)

MULTI_CAPTION_PREFIX_TEMPLATE = (
    "\nBelow are {n} descriptions of an image. Use them to answer the question.\n\n"
)


def format_single_caption(caption: str) -> str:
    """Format a single caption as context prefix for the question."""
    return f"{SINGLE_CAPTION_PREFIX}{caption}\n\n"


def format_multi_captions(captions: List[str]) -> str:
    """Format multiple captions as numbered context prefix for the question."""
    prefix = MULTI_CAPTION_PREFIX_TEMPLATE.format(n=len(captions))
    numbered = "\n".join(f"{i + 1}: {cap}" for i, cap in enumerate(captions))
    return f"{prefix}{numbered}\n\n"


# =============================================================================
# doc_to_visual: CAPTIONED MODE (no image)
# =============================================================================


def captioned_doc_to_visual(doc):
    """Return empty list — models operate in text-only mode."""
    return []


# =============================================================================
# doc_to_visual: GQA VAL-BALANCED VISUAL MODE (with image)
# =============================================================================


def gqa_val_balanced_doc_to_visual(doc):
    """Return image from the unified GQA val-balanced dataset.

    Unlike the original GQA task which loads images from a separate dataset
    subset (val_balanced_images), this dataset already has images merged in.
    No cross-dataset lookup needed.
    """
    return [doc["image"].convert("RGB")]


# =============================================================================
# process_docs: MME — Sort by question_id to restore pair-consecutive ordering
# =============================================================================


def mme_sort_by_question_id(dataset):
    """Sort MME dataset by question_id so paired questions are consecutive.

    The original lmms-lab/MME dataset has pairs (2 questions per image) stored
    consecutively. The captioned variant (l11p/MME-test-captioned) may have
    them scattered. This restores the pairing order required by
    mme_aggregate_results (which asserts len(scores) == 2 per question_id).
    """
    return dataset.sort("question_id")


# =============================================================================
# process_docs: MMBench — Normalize column names to match lmms-eval expectations
# =============================================================================


def mmbench_normalize_columns(dataset):
    """Rename 'l2-category' → 'L2-category' to match lmms-eval's en_utils.py.

    The original lmms-lab/MMBench dataset uses 'L2-category' (uppercase L),
    but the captioned variant (l11p/MMBench_EN-dev-captioned) has 'l2-category'
    (lowercase). The built-in mmbench_process_results reads doc['L2-category'],
    so we must normalise before evaluation.
    """
    if (
        "l2-category" in dataset.column_names
        and "L2-category" not in dataset.column_names
    ):
        dataset = dataset.rename_column("l2-category", "L2-category")
    return dataset


# =============================================================================
# doc_to_text: MME
# =============================================================================

_MME_REPLACE_PROMPT = " Please answer yes or no."


def mme_captioned_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """MME question with caption replacing image.

    Logic mirrors the original mme_doc_to_text but prepends the caption
    context block so the model uses text description instead of visual input.
    """
    caption = doc["caption"]
    question = doc["question"].strip()

    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if pre_prompt:
            question = question.replace(_MME_REPLACE_PROMPT, "")
            question = f"{pre_prompt}{question}"
        if post_prompt:
            question = question.replace(_MME_REPLACE_PROMPT, "")
            question = f"{question}{post_prompt}"

    return f"{format_single_caption(caption)}{question}"


# =============================================================================
# doc_to_text: MMBENCH
# =============================================================================


def _build_mmbench_options_prompt(doc):
    """Build options prompt matching the original mmbench evaluator logic."""
    option_candidates = ["A", "B", "C", "D", "E"]
    prompt = ""
    for c in option_candidates:
        val = doc.get(c, None)
        if val is not None and str(val).lower() != "nan":
            prompt += f"{c}. {val}\n"
    return prompt


def mmbench_captioned_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """MMBench question with caption replacing image.

    Logic mirrors the original mmbench_doc_to_text but prepends the caption
    context block.
    """
    caption = doc["caption"]
    options_prompt = _build_mmbench_options_prompt(doc)

    hint = doc.get("hint", "")
    question = doc["question"]

    if pd.notna(hint) and hint != "nan" and hint != "":
        query = f"{hint} {question} {options_prompt}"
    else:
        query = f"{question} {options_prompt}"

    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if post_prompt:
            query = f"{query}\n{post_prompt}"

    return f"{format_single_caption(caption)}{query}"


# =============================================================================
# doc_to_text: VQAv2 (SYNTHETIC CAPTION)
# =============================================================================


def vqav2_captioned_synthetic_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """VQAv2 question with single synthetic caption replacing image."""
    caption = doc["caption"]
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = f"{pre_prompt}{doc['question']}{post_prompt}"
    return f"{format_single_caption(caption)}{question}"


# =============================================================================
# doc_to_text: VQAv2 (HUMAN CAPTIONS)
# =============================================================================


def vqav2_captioned_human_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """VQAv2 question with multiple human captions replacing image."""
    captions = doc["captions"]
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = f"{pre_prompt}{doc['question']}{post_prompt}"
    return f"{format_multi_captions(captions)}{question}"


# =============================================================================
# doc_to_text: GQA VAL-BALANCED (visual and captioned)
# =============================================================================


def gqa_val_balanced_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """GQA val-balanced question — same text format as original gqa_doc_to_text."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"


def gqa_val_balanced_captioned_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """GQA val-balanced question with caption replacing image."""
    caption = doc["caption"]
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{format_single_caption(caption)}{pre_prompt}{question}{post_prompt}"


# =============================================================================
# PROCESS RESULTS & AGGREGATION — Re-exported from the library
# =============================================================================
# These are imported directly from lmms-eval's built-in task utils.
# No code duplication — the exact same scoring logic is used.

from lmms_eval.tasks.mme.utils import (  # noqa: E402
    mme_aggregate_results,
    mme_process_results,
)

from lmms_eval.tasks.mmbench.en_utils import (  # noqa: E402
    mmbench_aggregate_dev_results_eval,
    mmbench_aggregate_dev_results_submission,
    mmbench_process_results,
)

from lmms_eval.tasks.vqav2.utils import (  # noqa: E402
    vqav2_process_results_val,
)
