"""
COCOQA evaluation script — standalone VQA evaluation for l11p/COCOQA-restval-ICQA.

Supports 3 evaluation modes:
  - visual:            Use original images as context
  - human_captions:    Replace images with human captions (list)
  - synthetic_caption: Replace images with a single synthetic caption

Supports 2 QA selection strategies:
  - all:    Evaluate ALL question-answer pairs per image
  - single: Deterministic SHA-256 dedup → 1 QA per image

Works with both LLaVA and InternVL2 models via load_hf_model_and_processor_or_tokenizer.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(PROJECT_ROOT))

from tasks.captioned.utils import format_multi_captions, format_single_caption
from utils.internvl_utils import batch_generate as internvl_batch_generate
from utils.internvl_utils import get_pixel_values
from utils.model_utils import load_hf_model_and_processor_or_tokenizer
from utils.vqa_normalizer import normalize_vqa_answer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID = "l11p/COCOQA-restval-ICQA"
VQA_GUIDE_TEXT = "Answer the question using a single word or phrase."

# Regex to strip image tokens from saved prompts (both LLaVA and InternVL)
_IMAGE_TOKEN_RE = re.compile(
    r"<img>.*?</img>|<IMG_CONTEXT>|<image>\n?", flags=re.DOTALL
)


# ═══════════════════════════════════════════════════════════════════════════
# QA SELECTION
# ═══════════════════════════════════════════════════════════════════════════


def _hash_qa(question: str, answer: str) -> str:
    combined = f"{question.strip().lower()}|||{answer.strip().lower()}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def flatten_samples(
    dataset, qa_selection: str, limit: int | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    """Convert HF dataset rows into flat (image_id, question, answer, image, captions, synthetic_caption) samples.

    Args:
        dataset: HF dataset split with COCOQA-restval-ICQA schema.
        qa_selection: 'all' (every QA pair) or 'single' (SHA-256 dedup, 1 per image).
        limit: Max number of dataset rows to process (for quick testing).
        seed: RNG seed for deterministic single-QA selection.
    """
    import random

    rng = random.Random(seed)
    rows = dataset.select(range(min(limit, len(dataset)))) if limit else dataset

    samples: list[dict[str, Any]] = []

    if qa_selection == "all":
        for row in rows:
            for qa in row["questions_and_answers"]:
                samples.append(
                    {
                        "image_id": row["image_id"],
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "image": row["image"],
                        "captions": row["captions"],
                        "synthetic_caption": row["synthetic-caption"],
                    }
                )
    else:  # single
        seen_hashes: set[str] = set()
        for row in rows:
            qas = row["questions_and_answers"]
            indices = list(range(len(qas)))
            rng.shuffle(indices)
            selected_idx = indices[0]  # fallback
            for idx in indices:
                h = _hash_qa(qas[idx]["question"], qas[idx]["answer"])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    selected_idx = idx
                    break
            qa = qas[selected_idx]
            samples.append(
                {
                    "image_id": row["image_id"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "image": row["image"],
                    "captions": row["captions"],
                    "synthetic_caption": row["synthetic-caption"],
                }
            )

    return samples


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════


def build_questions(samples: list[dict], mode: str) -> list[str]:
    """Build the text question for each sample, prepending caption context when needed."""
    questions: list[str] = []
    for s in samples:
        q = s["question"]
        if mode == "human_captions":
            caption_ctx = format_multi_captions(s["captions"])
            questions.append(f"{caption_ctx}{q}\n{VQA_GUIDE_TEXT}")
        elif mode == "synthetic_caption":
            caption_ctx = format_single_caption(s["synthetic_caption"])
            questions.append(f"{caption_ctx}{q}\n{VQA_GUIDE_TEXT}")
        else:  # visual
            questions.append(f"{q}\n{VQA_GUIDE_TEXT}")
    return questions


# ═══════════════════════════════════════════════════════════════════════════
# LLAVA GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def _build_llava_conversation(question: str, has_image: bool) -> list[dict]:
    """Build a single conversation in the OpenAI-style format for LLaVA."""
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "A chat between a curious human and an artificial intelligence assistant. "
                        "The assistant gives helpful, detailed, and polite answers to the human's questions."
                    ),
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def generate_llava_batch(
    model,
    processor,
    questions: list[str],
    images: list[Image.Image] | None,
    max_new_tokens: int,
) -> tuple[list[str], list[str]]:
    """Generate responses for a batch of questions with LLaVA.

    Returns:
        (responses, prompts) — prompts are the formatted text strings sent to the model.
    """
    has_images = images is not None
    conversations = [_build_llava_conversation(q, has_images) for q in questions]
    prompts = processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )

    processor_kwargs: dict[str, Any] = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": True,
    }
    if has_images:
        processor_kwargs["images"] = images

    inputs = processor(**processor_kwargs).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    generated_ids = output_ids[:, input_len:]
    responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
    responses = [r.strip() for r in responses]

    return responses, prompts


# ═══════════════════════════════════════════════════════════════════════════
# INTERNVL GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def generate_internvl_batch(
    model,
    tokenizer,
    questions: list[str],
    images: list[Image.Image] | None,
    max_new_tokens: int,
) -> tuple[list[str], list[str]]:
    """Generate responses for a batch of questions with InternVL2.

    Returns:
        (responses, prompts) — prompts are reconstructed from the chat template.
    """
    from utils.internvl_utils import apply_chat_template

    # Build prompts for logging (before batch_chat does its own templating)
    if images is not None:
        _, num_patches_list = get_pixel_values(images)
        prompts = apply_chat_template(
            model, questions, num_patches_list=num_patches_list
        )
    else:
        prompts = apply_chat_template(model, questions, num_patches_list=None)

    responses = internvl_batch_generate(
        model, tokenizer, questions, images=images, max_new_tokens=max_new_tokens
    )

    return responses, prompts


# ═══════════════════════════════════════════════════════════════════════════
# SAVING
# ═══════════════════════════════════════════════════════════════════════════


def strip_image_tokens(prompt: str) -> str:
    """Remove model-specific image tokens from a prompt for readable logs."""
    return _IMAGE_TOKEN_RE.sub("", prompt).strip()


def save_responses(
    output_dir: str,
    model_id: str,
    records: list[dict],
    mode: str,
    qa_selection: str,
) -> None:
    """Save per-sample response log as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_id}_responses_{qa_selection}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"✓ Responses saved to {path}")


def save_metrics(
    output_dir: str,
    model_id: str,
    accuracy: float,
    n_correct: int,
    n_total: int,
    mode: str,
    qa_selection: str,
    elapsed_sec: float,
) -> None:
    """Append accuracy metric to a JSONL results file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"cocoqa_{mode}_{qa_selection}_results.jsonl"
    path = os.path.join(output_dir, filename)

    minutes = int(elapsed_sec // 60)
    seconds = int(elapsed_sec % 60)
    record = {
        "experiment_name": model_id,
        "mode": mode,
        "qa_selection": qa_selection,
        "accuracy": round(accuracy, 4),
        "n_correct": n_correct,
        "n_total": n_total,
        "elapsed_time": f"{minutes}m {seconds}s",
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"✓ Metrics appended to {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COCOQA VQA Evaluation")
    p.add_argument("--model-name-or-path", required=True, help="HF model identifier")
    p.add_argument(
        "--mode",
        required=True,
        choices=["visual", "human_captions", "synthetic_caption"],
        help="Evaluation mode: visual | human_captions | synthetic_caption",
    )
    p.add_argument(
        "--qa-selection",
        default="all",
        choices=["all", "single"],
        help="QA selection strategy: all pairs or 1 per image (SHA-256 dedup)",
    )
    p.add_argument("--output-dir", required=True, help="Directory for results")
    p.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    p.add_argument(
        "--limit", type=int, default=None, help="Limit dataset rows (for testing)"
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=50, help="Max generation tokens"
    )
    p.add_argument(
        "--model-cache-dir",
        default=os.environ.get("HF_HUB_CACHE"),
        help="HF model cache directory",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Model loading ---
    print(f"\n{'=' * 60}")
    print(f"  COCOQA Evaluation")
    print(f"  Model: {args.model_name_or_path}")
    print(f"  Mode:  {args.mode} | QA: {args.qa_selection} | BS: {args.batch_size}")
    print(f"{'=' * 60}\n")

    model, proc_or_tok = load_hf_model_and_processor_or_tokenizer(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir,
        device_map="auto",
    )
    model.eval()

    is_internvl = "intern" in model.__class__.__name__.lower()
    is_visual = args.mode == "visual"
    model_id = args.model_name_or_path.replace("/", "_")

    # --- Dataset loading ---
    print(f"\nLoading dataset: {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train")
    samples = flatten_samples(ds, args.qa_selection, limit=args.limit)
    print(f"✓ {len(samples)} QA samples prepared ({args.qa_selection} mode)")

    # --- Build questions ---
    questions = build_questions(samples, args.mode)

    # --- Inference loop ---
    records: list[dict] = []
    n_correct = 0
    start_time = time.time()

    n_batches = (len(samples) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"Evaluating ({args.mode})"):
        lo = batch_idx * args.batch_size
        hi = min(lo + args.batch_size, len(samples))
        batch_questions = questions[lo:hi]
        batch_samples = samples[lo:hi]

        # Prepare images (only for visual mode)
        batch_images: list[Image.Image] | None = None
        if is_visual:
            batch_images = [s["image"] for s in batch_samples]

        # Generate
        if is_internvl:
            responses, prompts = generate_internvl_batch(
                model, proc_or_tok, batch_questions, batch_images, args.max_new_tokens
            )
        else:
            responses, prompts = generate_llava_batch(
                model, proc_or_tok, batch_questions, batch_images, args.max_new_tokens
            )

        # Evaluate
        for i, (sample, response, prompt) in enumerate(
            zip(batch_samples, responses, prompts)
        ):
            norm_pred = normalize_vqa_answer(response)
            norm_gt = normalize_vqa_answer(sample["answer"])
            correct = norm_pred == norm_gt

            if correct:
                n_correct += 1

            records.append(
                {
                    "image_id": sample["image_id"],
                    "question": sample["question"],
                    "ground_truth": sample["answer"],
                    "prompt": strip_image_tokens(prompt),
                    "response": response,
                    "correct": correct,
                }
            )

    elapsed = time.time() - start_time
    accuracy = n_correct / len(records) if records else 0.0

    # --- Report ---
    print(f"\n{'=' * 60}")
    print(f"  Results: {n_correct}/{len(records)} correct = {accuracy:.4f}")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"{'=' * 60}\n")

    # --- Save ---
    save_responses(args.output_dir, model_id, records, args.mode, args.qa_selection)
    save_metrics(
        args.output_dir,
        model_id,
        accuracy,
        n_correct,
        len(records),
        args.mode,
        args.qa_selection,
        elapsed,
    )


if __name__ == "__main__":
    main()
