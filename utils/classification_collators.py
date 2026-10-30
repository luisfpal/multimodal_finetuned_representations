"""Collate functions and ICL helpers for the representations classification pipeline.

Groups collators by model family (LLaVA, InternVL, Cambrian) plus shared utilities
(_is_internvl, _is_cambrian, _build_sample_signature, _load_mmmu_dev_pool).
"""
import copy
import hashlib

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from .constants import SYSTEM_ROLE
from .dataset import (
    MMMU_IMAGE_COLUMNS,
    _build_llava_prompts,
    _format_mmmu_question,
    _processor_has_default_system,
)
from .cambrian_utils import prepare_cambrian_batch_inputs
from .internvl_utils import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    apply_chat_template as internvl_apply_chat_template,
    get_pixel_values,
)
from .prompt_formatting import format_user_turn_text


# ---------------------------------------------------------------------------
# Model-type detection
# ---------------------------------------------------------------------------


def _is_internvl(model_name: str) -> bool:
    return "intern" in model_name.lower()


def _is_cambrian(model_name: str) -> bool:
    return "cambrian" in model_name.lower()


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _build_sample_signature(sample_ids: list[str], labels: np.ndarray) -> str:
    if len(sample_ids) != len(labels):
        raise ValueError(
            f"Sample IDs and labels length mismatch: {len(sample_ids)} vs {len(labels)}"
        )
    joined = "\n".join(
        f"{sample_id}\t{label}" for sample_id, label in zip(sample_ids, labels)
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _load_mmmu_dev_pool(label_granularity: str) -> dict[str, list[dict]]:
    """Load MMMU dev split, filter to single-image MC instances, group by category key."""
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


# ---------------------------------------------------------------------------
# LLaVA collate functions
# ---------------------------------------------------------------------------


def _build_icl_conversation(
    shots: list[dict],
    test_instance: dict,
    processor,
) -> tuple[list[dict], list[Image.Image]]:
    """Build a single-user-turn ICL conversation: N shots then the test query."""
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
    content.append({
        "type": "text",
        "text": format_user_turn_text(f"{_format_mmmu_question(test_instance)}\nAnswer:"),
    })
    images.append(test_img)

    conv: list[dict] = [{"role": "user", "content": content}]
    if not _processor_has_default_system(processor):
        conv.insert(0, SYSTEM_ROLE)

    return conv, images


def _llava_zero_shot_collate_fn(
    batch: list[dict],
    processor,
    guide_text: str,
    assistant_prefix: str = "",
    prompt_log: list[str] | None = None,
) -> tuple:
    texts = _build_llava_prompts(
        questions=[item["question"] for item in batch],
        images=[item["image"] for item in batch],
        processor=processor,
        guide_text=guide_text,
        assistant_prefix=assistant_prefix,
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
    assistant_prefix: str = "",
    prompt_log: list[str] | None = None,
) -> tuple:
    """Collator for LLaVA ICL. Falls back to zero-shot for categories with no dev shots."""
    item = batch[0]  # batch_size=1 always
    label = item["label"]
    test_instance = item["_raw"]

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
        text = _build_llava_prompts(
            questions=[item["question"]],
            images=[item["image"]],
            processor=processor,
            guide_text=guide_text,
            assistant_prefix=assistant_prefix,
        )
        if prompt_log is not None:
            prompt_log.append(text[0])
        tokenized = processor(text=text, images=[item["image"]], return_tensors="pt")

    return tokenized, [label]


# ---------------------------------------------------------------------------
# InternVL2 collate functions
# ---------------------------------------------------------------------------


def _build_internvl_icl_query(
    shots: list[dict],
    test: dict,
    model,
    tokenizer,
) -> tuple[dict, torch.Tensor, str]:
    """Build InternVL2 multi-image ICL prompt."""
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
    full_text = format_user_turn_text("".join(parts))

    template = copy.deepcopy(model.conv_template)
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
    assistant_prefix: str = "",
    prompt_log: list[str] | None = None,
) -> tuple:
    item = batch[0]
    img = item["image"]
    pixel_values, num_patches_list = get_pixel_values([img])
    query = internvl_apply_chat_template(
        model, [item["question"]], num_patches_list, guide_text, assistant_prefix
    )[0]
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
    assistant_prefix: str = "",
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
        tokenized, pixel_values, query = _build_internvl_icl_query(
            shots, test_instance, model, tokenizer
        )
        if prompt_log is not None:
            prompt_log.append(query)
    else:
        img = item["image"]
        pixel_values, num_patches_list = get_pixel_values([img])
        query = internvl_apply_chat_template(
            model, [item["question"]], num_patches_list, assistant_prefix=assistant_prefix
        )[0]
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
    assistant_prefix: str = "",
    prompt_log: list[str] | None = None,
) -> tuple:
    item = batch[0]
    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=[item["question"]],
        images=[item["image"]],
        guide_text=guide_text,
        assistant_prefix=assistant_prefix,
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
    assistant_prefix: str = "",
    prompt_log: list[str] | None = None,
) -> tuple:
    """Cambrian ICL is NOT SUPPORTED — SVA is hardwired to single-image position.

    Cambrian's Spatial Vision Aggregator (SVA) cross-attention layer reads image
    features from a hardcoded position (config.image_position=91) at every 3rd
    decoder layer. ICL would require N images at variable positions, which SVA
    cannot handle. Fallback: always zero-shot.
    """
    if n_shots > 0:
        import warnings

        warnings.warn(
            f"Cambrian does not support in-context examples: requested n_shots={n_shots}, "
            "running zero-shot instead.",
            stacklevel=2,
        )
    item = batch[0]
    label = item["label"]
    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=[item["question"]],
        images=[item["image"]],
        guide_text=guide_text,
        assistant_prefix=assistant_prefix,
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
