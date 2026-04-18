"""
MMMU standalone evaluation script — zero-shot and n-shot ICL accuracy.

Evaluates MMMU `validation` split. Shots come from `dev` split.
Restricted to single-image, multiple-choice instances.
ICL shots are matched by subcategory (subject parsed from `id`).

Usage:
    python scripts/eval/mmmu_evaluation.py \
        --model-name lbasile/llava_7b \
        --n-shots 3 \
        --output-dir results/mmmu_evaluation \
        [--max-samples 20]
"""

import argparse
import ast
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.constants import SYSTEM_ROLE
from utils.dataset import _build_llava_prompts, _processor_has_default_system
from utils.model_utils import load_hf_model_and_processor_or_tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOME_DIR = Path.home()
HF_HOME = HOME_DIR / "scratch" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))

MMMU_IMAGE_COLUMNS = [f"image_{i}" for i in range(1, 8)]
GUIDE_TEXT = "Answer with the option's letter from the given choices directly."
_LETTER_RE = re.compile(r"\b([A-E])\b")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _keep(ex: dict) -> bool:
    """Single-image + multiple-choice filter."""
    return (
        sum(ex.get(col) is not None for col in MMMU_IMAGE_COLUMNS) == 1
        and str(ex.get("question_type", "")).strip().lower() == "multiple-choice"
    )


def _extract_subject(example_id: str) -> str:
    """'validation_Accounting_1' → 'Accounting'."""
    parts = str(example_id).split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else ""


def _format_question(instance: dict) -> str:
    """Format question + options (A./B./...), stripping any <image> refs."""
    question = re.sub(r"<image[^>]*>", " ", str(instance.get("question", "")))
    question = re.sub(r"\s+", " ", question).strip()

    raw = instance.get("options", [])
    try:
        options = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
    except (ValueError, SyntaxError):
        options = []

    lines = [question]
    for letter_ord, opt in zip(range(ord("A"), ord("Z") + 1), options):
        text = re.sub(r"<image[^>]*>", " ", str(opt))
        text = re.sub(r"\s+", " ", text).strip()
        lines.append(f"{chr(letter_ord)}. {text}")
    return "\n".join(lines)


def _build_shot_pool(dev_ds) -> dict[str, list[dict]]:
    """Filter dev split and group by subject."""
    pool: dict[str, list[dict]] = {}
    for instance in dev_ds:
        if not _keep(instance):
            continue
        key = _extract_subject(str(instance["id"]))
        if key:
            pool.setdefault(key, []).append(dict(instance))
    return pool


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_icl_prompt(
    shots: list[dict],
    test: dict,
    processor: Any,
) -> tuple[str, list[Image.Image]]:
    """Build ICL conversation: N shots + test query with guide text.

    Shots: [img, "Q\\nAnswer: X\\n\\n"]
    Test:  [img, "Q\\n{GUIDE_TEXT}\\nAnswer:"]
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
            "text": f"{_format_question(shot)}\nAnswer: {shot['answer']}\n\n",
        })
        images.append(img)

    test_img = test.get("image_1")
    if isinstance(test_img, Image.Image):
        test_img = test_img.convert("RGB")
    content.append({"type": "image"})
    content.append({
        "type": "text",
        # "text": f"{_format_question(test)}\n{GUIDE_TEXT}\nAnswer:",
        "text": f"{_format_question(test)}\nAnswer:",
    })
    images.append(test_img)

    conv: list[dict] = [{"role": "user", "content": content}]
    if not _processor_has_default_system(processor):
        conv.insert(0, SYSTEM_ROLE)

    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    return text, images


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _extract_answer(response: str) -> str:
    """Return first A–E letter found in response, else first uppercase char."""
    m = _LETTER_RE.search(response.strip())
    return m.group(1) if m else response.strip()[:1].upper()


def _decode_raw_generated_tokens(processor: Any, token_ids: torch.Tensor) -> list[str]:
    """Decode generated token ids into raw token strings (without post-processing)."""
    ids = token_ids.detach().cpu().tolist()
    tokenizer = getattr(processor, "tokenizer", processor)

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return [str(tok) for tok in tokenizer.convert_ids_to_tokens(ids)]

    if hasattr(tokenizer, "batch_decode"):
        return [
            str(tok)
            for tok in tokenizer.batch_decode(
                [[token_id] for token_id in ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        ]

    return [str(token_id) for token_id in ids]


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_responses(output_dir: Path, model_slug: str, n_shots: int, records: list[dict]) -> None:
    model_dir = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"mmmu_{n_shots}shot_responses.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Responses saved: {path}")


def save_metrics(
    output_dir: Path,
    model_name: str,
    n_shots: int,
    accuracy: float,
    n_correct: int,
    n_total: int,
    n_fallback: int,
    elapsed: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mmmu_results.jsonl"
    minutes, seconds = divmod(int(elapsed), 60)
    record = {
        "model": model_name,
        "n_shots": n_shots,
        "accuracy": round(accuracy, 4),
        "n_correct": n_correct,
        "n_total": n_total,
        "n_zero_shot_fallback": n_fallback,
        "elapsed_time": f"{minutes}m {seconds}s",
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Metrics appended: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMMU n-shot ICL evaluation")
    p.add_argument("--model-name", required=True, help="HF model identifier")
    p.add_argument("--n-shots", type=int, default=0, help="Number of ICL shots (0 = zero-shot)")
    p.add_argument("--output-dir", required=True, help="Directory for results")
    p.add_argument("--max-new-tokens", type=int, default=16, help="Max generation tokens")
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print(f"\n{'=' * 60}")
    print("  MMMU Evaluation")
    print(f"  Model:   {args.model_name}")
    print(f"  N-shots: {args.n_shots}")
    print(f"{'=' * 60}\n")

    # --- Load model ---
    cache_dir = str(HF_HOME / "hub")
    model, processor = load_hf_model_and_processor_or_tokenizer(
        args.model_name,
        cache_dir=cache_dir,
        device_map="auto",
    )
    model.eval()

    # --- Load datasets ---
    print("Loading MMMU validation split ...")
    val_ds = load_dataset("lmms-lab/MMMU", split="validation", cache_dir=str(HF_HOME / "datasets"))
    val_filtered = [dict(ex) for ex in val_ds if _keep(ex)]
    if args.max_samples:
        val_filtered = val_filtered[: args.max_samples]
    print(f"  Validation (filtered): {len(val_filtered)} instances")

    shot_pool: dict[str, list[dict]] = {}
    if args.n_shots > 0:
        print("Loading MMMU dev split for shots ...")
        dev_ds = load_dataset("lmms-lab/MMMU", split="dev", cache_dir=str(HF_HOME / "datasets"))
        shot_pool = _build_shot_pool(dev_ds)
        print(f"  Dev pool subjects: {len(shot_pool)}")

    # --- Evaluation loop ---
    records: list[dict] = []
    n_correct = 0
    n_fallback = 0
    start_time = time.time()

    for sample in tqdm(val_filtered, desc=f"Evaluating ({args.n_shots}-shot)"):
        subject = _extract_subject(str(sample["id"]))
        shots = shot_pool.get(subject, [])[: args.n_shots] if args.n_shots > 0 else []
        used_icl = len(shots) > 0

        if used_icl:
            text, images = _build_icl_prompt(shots, sample, processor)
            inputs = processor(text=text, images=images, return_tensors="pt").to(model.device)
        else:
            if args.n_shots > 0:
                n_fallback += 1
            img = sample.get("image_1")
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            text = _build_llava_prompts(
                [_format_question(sample)],
                [img],
                processor,
                # guide_text=f"{GUIDE_TEXT}\nAnswer:",
                guide_text=f"{GUIDE_TEXT}",
            )[0]
            inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        generated_token_ids = output_ids[0, input_len:]
        response = processor.decode(generated_token_ids, skip_special_tokens=True).strip()
        raw_decoded_tokens = _decode_raw_generated_tokens(processor, generated_token_ids)
        predicted = _extract_answer(response)
        correct = predicted == str(sample.get("answer", "")).strip()
        if correct:
            n_correct += 1

        records.append({
            "id": sample["id"],
            "subject": subject,
            "question": text,
            "ground_truth": sample.get("answer", ""),
            "response": response,
            "raw_decoded_tokens": raw_decoded_tokens,
            "predicted": predicted,
            "correct": correct,
            "used_icl": used_icl,
        })

    elapsed = time.time() - start_time
    accuracy = n_correct / len(records) if records else 0.0

    # --- Report ---
    print(f"\n{'=' * 60}")
    print(f"  Accuracy: {n_correct}/{len(records)} = {accuracy:.4f}")
    if n_fallback:
        print(f"  Zero-shot fallbacks: {n_fallback} (no dev shots for subject)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 60}\n")

    # --- Save ---
    output_dir = Path(args.output_dir)
    model_slug = args.model_name.replace("/", "_")
    save_responses(output_dir, model_slug, args.n_shots, records)
    save_metrics(output_dir, args.model_name, args.n_shots, accuracy, n_correct, len(records), n_fallback, elapsed)


if __name__ == "__main__":
    main()
