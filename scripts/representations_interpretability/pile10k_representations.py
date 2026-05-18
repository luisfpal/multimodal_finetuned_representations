"""Extract residual-stream representations for NeelNanda/pile-10k (text-only).

Two pooling methods:
  last          — last token of the full prompt sequence (model-agnostic)
  last_user_text — last token of {text_context} inside the structured prompt

Prompt structure applied to every sample (after per-model chat-template):
  SYSTEM (model default)
  USER: Continue this text:\\n\\n\\n{text_context}\\n\\n\\n
  ASSISTANT:

The triple-newline delimiter is unique across all model chat templates
(LLaVA 1.5, LLaVA OV, InternVL2, Cambrian all use at most \\n\\n internally),
so find_user_text_span() can locate {text_context} unambiguously in the decoded
token sequence.

NOTE: Prompt building bypasses format_user_turn_text() (which does rstrip) so
that the trailing \\n\\n\\n delimiter is preserved in the tokenized sequence.

Inspection workflow (run before the full extraction to verify prompt format):
  python pile10k_representations.py \\
    --model-name-or-path lbasile/llava_7b \\
    --inspect-only \\
    --n-prompt-samples 5 \\
    --save-prompts-dir /path/to/scratch/pile10k_prompt_logs
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding

from src.residual_stream_tracer import residual_stream_tracer
from utils import load_hf_model_and_processor_or_tokenizer, seed_all
from utils.cambrian_utils import (
    is_cambrian_model,
    resolve_cambrian_conv_mode,
    tokenize_cambrian_prompts,
)
from utils.constants import LLaVA1_5_SYSTEM_INSTRUCTION
from utils.token_analysis import find_user_text_span

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "flash_attention_2" if DEVICE.type == "cuda" else "sdpa"
RESIDUAL_STREAM_TYPE = "output_layer"
DATASET_TAG = "pile10k_train_txt"

# The unique delimiter pair that brackets {text_context} in every prompt.
_PREAMBLE = "Continue this text:\n\n\n"
_DELIMITER = "\n\n\n"

POOLING_METHODS_DEFAULT = ["last", "last_user_text"]
RESIDUAL_STREAM_TYPES = ["output_layer", "post_input_layernorm"]
PROMPT_FORMATS = ["wrapped", "plain", "raw"]


def _get_user_content(text: str, prompt_format: str) -> str:
    if prompt_format == "plain":
        return text
    return f"{_PREAMBLE}{text}{_DELIMITER}"


def _n_assistant_tokens(processor: Any, model: Any) -> int:
    if _is_llava_family(processor):
        return 5  # LLaVA 1.5 and LLaVA OV both use 5
    if model is not None and "intern" in model.__class__.__name__.lower():
        return 5
    if model is not None and is_cambrian_model(model=model):
        return 4
    raise ValueError(
        f"Cannot determine assistant token count for model {model.__class__.__name__}"
    )


def build_pooling_variant(
    pooling_method: str,
    prompt_format: str = "wrapped",
    remove_assistant_tokens: bool = False,
) -> str:
    parts = [pooling_method]
    if prompt_format in ("plain", "raw"):
        parts.append(prompt_format)
    if remove_assistant_tokens:
        parts.append("no_asst")
    return "_".join(parts)


# ── Dataset ───────────────────────────────────────────────────────────────────


class Pile10kDataset(Dataset):
    """NeelNanda/pile-10k with token-based text truncation.

    Each sample is truncated to at most `max_tokens` tokens (encode → take
    first N ids → decode).  This is precise and model-specific — the tokenizer
    already available after model loading is reused here.
    """

    def __init__(self, tokenizer: Any, max_tokens: int = 1024, min_tokens: int = 0):
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        self._contexts: list[str] = []
        n_skipped = 0
        for row in ds:
            ids = tokenizer.encode(row["text"], add_special_tokens=False)[:max_tokens]
            if len(ids) < min_tokens:
                n_skipped += 1
                continue
            self._contexts.append(tokenizer.decode(ids, skip_special_tokens=True))
        if n_skipped:
            print(f"   Filtered out {n_skipped} samples with < {min_tokens} tokens")

    def __len__(self) -> int:
        return len(self._contexts)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {"text_context": self._contexts[idx]}


# ── Prompt builders (per model family) ───────────────────────────────────────
# These bypass format_user_turn_text() intentionally so that the trailing \n\n\n
# delimiter is preserved in the tokenized sequence.


def _has_default_system(processor: Any) -> bool:
    """True if the processor's chat template injects a system message by default."""
    tok = getattr(processor, "tokenizer", processor)
    apply_fn = getattr(tok, "apply_chat_template", None)
    if apply_fn is None:
        return False
    try:
        probe = apply_fn(
            [{"role": "user", "content": [{"type": "text", "text": "?"}]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return "system" in probe.lower()
    except Exception:
        return False


def _is_llava_family(processor: Any) -> bool:
    if processor is None:
        return False
    name = processor.__class__.__name__.lower()
    tok_name = getattr(
        getattr(processor, "tokenizer", None), "__class__", type(None)
    ).__name__.lower()
    markers = ("llava", "qwen2_5_vl", "qwen2vl")
    return any(m in name for m in markers) or any(m in tok_name for m in markers)


def _build_llava_prompts(
    texts: list[str], processor: Any, prompt_format: str = "wrapped"
) -> list[str]:
    """Build LLaVA-family chat-template strings without rstrip side-effects."""
    include_system = not _has_default_system(processor)
    tok = getattr(processor, "tokenizer", processor)
    prompts = []
    for text in texts:
        user_content = _get_user_content(text, prompt_format)
        conv: list[dict] = []
        if include_system:
            conv.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": LLaVA1_5_SYSTEM_INSTRUCTION}],
                }
            )
        conv.append(
            {"role": "user", "content": [{"type": "text", "text": user_content}]}
        )
        prompts.append(
            tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def _build_internvl_prompts(
    texts: list[str], model: Any, prompt_format: str = "wrapped"
) -> list[str]:
    """Build InternVL2 chat-template strings without rstrip side-effects."""
    prompts = []
    for text in texts:
        user_content = _get_user_content(text, prompt_format)
        template = copy.deepcopy(model.conv_template)
        template.system_message = model.system_message
        template.messages = []
        template.append_message(template.roles[0], user_content)
        template.append_message(template.roles[1], None)
        prompts.append(template.get_prompt())
    return prompts


def _build_cambrian_prompts(
    texts: list[str], model: Any, prompt_format: str = "wrapped"
) -> list[str]:
    """Build Cambrian chat-template strings without rstrip side-effects."""
    try:
        from cambrian.conversation import conv_templates
    except ImportError as exc:
        raise ImportError("Cambrian is not installed.") from exc

    selected_mode = resolve_cambrian_conv_mode(model=model)
    prompts = []
    for text in texts:
        user_content = _get_user_content(text, prompt_format)
        conv = conv_templates[selected_mode].copy()
        conv.append_message(conv.roles[0], user_content)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())
    return prompts


def _build_prompts(
    texts: list[str], *, processor: Any, model: Any, prompt_format: str = "wrapped"
) -> list[str]:
    """Dispatch to the right prompt builder for this model type."""
    if _is_llava_family(processor):
        return _build_llava_prompts(texts, processor, prompt_format)
    if model is not None and "intern" in model.__class__.__name__.lower():
        return _build_internvl_prompts(texts, model, prompt_format)
    if model is not None and is_cambrian_model(model=model):
        return _build_cambrian_prompts(texts, model, prompt_format)
    raise ValueError(
        f"Unsupported model type: {model.__class__.__name__}. "
        "Expected a LLaVA, InternVL, or Cambrian model."
    )


def _tokenize_single_prompt(
    formatted_prompt: str,
    *,
    processor: Any,
    model: Any,
) -> np.ndarray:
    """Tokenize one prompt → 1-D numpy int array of token IDs."""
    if _is_llava_family(processor):
        tok = getattr(processor, "tokenizer", processor)
        ids = tok([formatted_prompt], return_tensors="pt")["input_ids"][0]
        return ids.numpy()

    if model is not None and "intern" in model.__class__.__name__.lower():
        ids = processor([formatted_prompt], return_tensors="pt")["input_ids"][0]
        return ids.numpy()

    if model is not None and is_cambrian_model(model=model):
        conv_mode = resolve_cambrian_conv_mode(model=model)
        tokenized = tokenize_cambrian_prompts(
            prompts=[formatted_prompt],
            tokenizer=processor,
            conv_mode=conv_mode,
            device=torch.device("cpu"),
        )
        return tokenized["input_ids"][0].numpy()

    raise ValueError(f"Unsupported model type: {model.__class__.__name__}")


# ── Collate ───────────────────────────────────────────────────────────────────


def collate_pile10k_batch(
    batch: list[dict[str, str]],
    *,
    processor: Any,
    model: Any,
    prompt_format: str = "wrapped",
    remove_assistant_tokens: bool = False,
) -> tuple[BatchEncoding]:
    """Tokenize a pile-10k batch into a 1-tuple compatible with residual_stream_tracer.

    The tracer's batch-unpacking logic expects:
      LLaVA:    batch[0] = BatchEncoding with input_ids, attention_mask
      InternVL: batch[0] = BatchEncoding with input_ids, attention_mask
      Cambrian: batch[0] = BatchEncoding with input_ids, attention_mask
    A 1-tuple satisfies all three paths.

    Args:
        prompt_format: "wrapped" (default) uses the preamble+delimiter structure;
            "plain" passes {text_context} directly to the chat template.
        remove_assistant_tokens: when True, strips the trailing assistant-role tokens
            from every sequence in the batch after tokenization.
    """
    texts = [item["text_context"] for item in batch]
    do_pad = len(texts) > 1

    if prompt_format == "raw":
        tok = getattr(processor, "tokenizer", processor)
        tok.padding_side = "left"
        tokenized = tok(texts, return_tensors="pt", padding=do_pad)
        return (tokenized,)

    if _is_llava_family(processor):
        prompts = _build_llava_prompts(texts, processor, prompt_format)
        tok = getattr(processor, "tokenizer", processor)
        tok.padding_side = "left"
        tokenized = tok(prompts, return_tensors="pt", padding=do_pad)
        if remove_assistant_tokens:
            n = _n_assistant_tokens(processor, model)
            tokenized["input_ids"] = tokenized["input_ids"][:, :-n]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :-n]
        return (tokenized,)

    if model is not None and "intern" in model.__class__.__name__.lower():
        prompts = _build_internvl_prompts(texts, model, prompt_format)
        # processor IS the tokenizer for InternVL
        processor.padding_side = "left"
        tokenized = processor(prompts, return_tensors="pt", padding=do_pad)
        if remove_assistant_tokens:
            n = _n_assistant_tokens(processor, model)
            tokenized["input_ids"] = tokenized["input_ids"][:, :-n]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :-n]
        return (tokenized,)

    if model is not None and is_cambrian_model(model=model):
        prompts = _build_cambrian_prompts(texts, model, prompt_format)
        conv_mode = resolve_cambrian_conv_mode(model=model)
        # tokenize_cambrian_prompts handles left-padding internally
        tokenized = tokenize_cambrian_prompts(
            prompts=prompts,
            tokenizer=processor,
            conv_mode=conv_mode,
            device=torch.device("cpu"),
        )
        if remove_assistant_tokens:
            n = _n_assistant_tokens(processor, model)
            tokenized["input_ids"] = tokenized["input_ids"][:, :-n]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :-n]
        return (tokenized,)

    raise ValueError(
        f"Unsupported model type: {model.__class__.__name__}. "
        "Expected a LLaVA, InternVL, or Cambrian model."
    )


# ── Prompt inspection ─────────────────────────────────────────────────────────


def inspect_prompts(
    processor: Any,
    model: Any,
    dataset: Pile10kDataset,
    *,
    n_samples: int,
    save_dir: str,
    model_name_or_path: str,
    prompt_format: str = "wrapped",
) -> None:
    """Format, tokenize, and span-verify N pile-10k prompts; save a JSON report.

    For each sample the report contains:
      formatted_prompt — the exact string the model sees after chat-template
      span             — token and character positions of {text_context} found by
                         find_user_text_span(), plus a decoded preview and a
                         verification flag (does span text contain text_context?)

    This is the cheapest decisive check that:
      1. The prompt format is correct (system/user/assistant structure)
      2. The trailing \\n\\n\\n delimiter is preserved (not stripped)
      3. find_user_text_span() locates {text_context} correctly

    Output path: {save_dir}/{model_slug}_prompt_inspection.json
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    n_samples = min(n_samples, len(dataset))
    records: list[dict] = []

    for idx in range(n_samples):
        text_context = dataset[idx]["text_context"]
        record: dict = {
            "idx": idx,
            "text_context_chars": len(text_context),
            "text_context": text_context,
        }

        if prompt_format == "raw":
            tok = getattr(processor, "tokenizer", processor)
            ids = tok.encode(text_context, add_special_tokens=True)
            record.update({
                "formatted_prompt": text_context,
                "n_tokens": len(ids),
                "span": {"note": "Span check skipped: raw format has no structure"},
            })
            status = "OK"
        else:
            try:
                formatted_prompt = _build_prompts(
                    [text_context], processor=processor, model=model,
                    prompt_format=prompt_format,
                )[0]
                record["formatted_prompt"] = formatted_prompt

                input_ids_np = _tokenize_single_prompt(
                    formatted_prompt, processor=processor, model=model
                )
                record["n_tokens"] = int(len(input_ids_np))

                # Span verification via find_user_text_span (only valid for wrapped format)
                if prompt_format == "plain":
                    record["span"] = {
                        "note": "Span check skipped: plain format has no preamble/delimiter"
                    }
                    status = "OK"
                else:
                    try:
                        tok_start, tok_end = find_user_text_span(tokenizer, input_ids_np)
                        span_ids = input_ids_np[tok_start:tok_end]
                        span_text = tokenizer.decode(
                            span_ids.tolist(), skip_special_tokens=False
                        )
                        # Normalise whitespace slightly for comparison
                        tc_stripped = text_context.strip()
                        span_stripped = span_text.strip()
                        matches = tc_stripped in span_stripped or span_stripped in tc_stripped
                        record["span"] = {
                            "tok_start": int(tok_start),
                            "tok_end": int(tok_end),
                            "n_span_tokens": int(tok_end - tok_start),
                            "span_text_chars": len(span_text),
                            "span_text": span_text,
                            "text_context_found_in_span": bool(matches),
                        }
                        status = "OK" if matches else "MISMATCH"
                    except RuntimeError as exc:
                        record["span"] = {"error": str(exc)}
                        status = "SPAN_FAILED"

            except Exception as exc:
                record["error"] = str(exc)
                status = "BUILD_FAILED"

        record["status"] = status
        icon = "✅" if status == "OK" else "❌"
        n_tok = record.get("n_tokens", "?")
        n_span = record.get("span", {}).get("n_span_tokens", "?")
        print(
            f"  [{idx:4d}] {icon} {status} | total={n_tok} tokens | span={n_span} tokens"
        )
        records.append(record)

    model_slug = model_name_or_path.rstrip("/").split("/")[-1]
    output = {
        "model_name": model_name_or_path,
        "model_slug": model_slug,
        "timestamp": datetime.now().isoformat(),
        "preamble": _PREAMBLE,
        "delimiter": _DELIMITER,
        "n_samples": len(records),
        "samples": records,
    }

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{model_slug}_prompt_inspection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Saved inspection: {out_path}")


# ── Path builder ──────────────────────────────────────────────────────────────


def build_repr_path(
    representations_dir: str,
    model_name_or_path: str,
    pooling_variant: str,
    max_tokens: int,
    min_tokens: int,
    residual_stream_type: str = RESIDUAL_STREAM_TYPE,
) -> str:
    """Canonical output path for a model × pooling_variant × token-range combination.

    Structure:
      {representations_dir}/representations/{model_slug}/{dataset_tag}/
        tokens_{min}_{max}/{residual_stream_type}/{pooling_variant}/
        all_samples/representations.safetensors

    ``pooling_variant`` encodes the pooling method plus any prompt/token flags,
    e.g. "last", "last_plain", "last_no_asst", "last_plain_no_asst", "last_user_text".
    Default args reproduce the original path for backward compatibility.
    """
    model_slug = model_name_or_path.rstrip("/").split("/")[-1]
    token_tag = f"tokens_{min_tokens}_{max_tokens}"
    return os.path.join(
        os.path.expanduser(representations_dir),
        "representations",
        model_slug,
        DATASET_TAG,
        token_tag,
        residual_stream_type,
        pooling_variant,
        "all_samples",
        "representations.safetensors",
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract residual-stream representations for NeelNanda/pile-10k (text-only). "
            "Representations are saved per pooling method and cached — rerun is a no-op "
            "unless --force is specified."
        )
    )
    parser.add_argument(
        "--representations-dir",
        type=str,
        default=os.path.expanduser("~/scratch"),
        help="Base directory for representation cache (default: ~/scratch)",
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Token budget for text_context truncation (default: 1024)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Minimum token count to keep a sample; shorter samples are discarded (default: 0)",
    )
    parser.add_argument(
        "--pooling-methods",
        nargs="+",
        default=POOLING_METHODS_DEFAULT,
        help=f"Pooling methods to extract (default: {POOLING_METHODS_DEFAULT})",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if cache exists"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print paths without computing"
    )
    # ── Inspection args ──────────────────────────────────────────────────────
    parser.add_argument(
        "--save-prompts-dir",
        type=str,
        default=None,
        help=(
            "Directory to save per-model prompt inspection JSON files. "
            "If set, inspection runs before extraction. "
            "Defaults to {representations_dir}/pile10k_prompt_logs when --inspect-only is used."
        ),
    )
    parser.add_argument(
        "--n-prompt-samples",
        type=int,
        default=5,
        help="Number of samples to inspect and save (default: 5)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help=(
            "Format and verify N prompts, save inspection JSON, then exit — "
            "skip representation extraction entirely. "
            "Useful for verifying prompt format and span detection before submitting the full job."
        ),
    )
    parser.add_argument(
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="wrapped",
        help=(
            "How to format {text_context} inside the chat template. "
            "'wrapped' (default) uses 'Continue this text:\\n\\n\\n{text}\\n\\n\\n'; "
            "'plain' passes {text_context} directly. "
            "plain is incompatible with last_user_text pooling."
        ),
    )
    parser.add_argument(
        "--remove-assistant-tokens",
        action="store_true",
        help=(
            "Strip the trailing assistant-role tokens from input_ids after tokenization. "
            "Only valid with 'last' pooling."
        ),
    )
    parser.add_argument(
        "--residual-stream-type",
        choices=RESIDUAL_STREAM_TYPES,
        default="output_layer",
        help="Residual stream extraction point (default: output_layer)",
    )
    args = parser.parse_args()

    # Validate incompatible combinations upfront
    if args.prompt_format == "plain" and "last_user_text" in args.pooling_methods:
        parser.error(
            "prompt_format='plain' is incompatible with 'last_user_text' pooling: "
            "find_user_text_span() requires the preamble/delimiter structure."
        )
    if args.remove_assistant_tokens and "last_user_text" in args.pooling_methods:
        parser.error(
            "--remove-assistant-tokens is only supported with 'last' pooling; "
            "'last_user_text' requires the assistant suffix to locate the text span."
        )
    if args.prompt_format == "raw" and "last_user_text" in args.pooling_methods:
        parser.error(
            "prompt_format='raw' is incompatible with 'last_user_text' pooling: "
            "raw format has no structure for find_user_text_span()."
        )
    if args.prompt_format == "raw" and args.remove_assistant_tokens:
        parser.error(
            "prompt_format='raw' is incompatible with --remove-assistant-tokens: "
            "no chat template is applied so there are no assistant tokens."
        )

    seed_all(args.seed)

    # Resolve inspection output directory
    run_inspection = args.save_prompts_dir is not None or args.inspect_only
    prompts_dir = args.save_prompts_dir or os.path.join(
        os.path.expanduser(args.representations_dir), "pile10k_prompt_logs"
    )

    # Determine which (pooling_method, pooling_variant, save_path) triples need computation
    methods_to_run: list[tuple[str, str, str]] = []
    if not args.inspect_only:
        for pm in args.pooling_methods:
            pv = build_pooling_variant(pm, args.prompt_format, args.remove_assistant_tokens)
            path = build_repr_path(
                args.representations_dir, args.model_name_or_path, pv,
                max_tokens=args.max_tokens, min_tokens=args.min_tokens,
                residual_stream_type=args.residual_stream_type,
            )
            if os.path.exists(path) and not args.force:
                print(f"  ✅ Cache hit [{pv}]: {path}")
            else:
                print(f"  🔄 Will compute [{pv}]: {path}")
                methods_to_run.append((pm, pv, path))

        if not methods_to_run and not run_inspection:
            print("All methods already cached. Use --force to recompute.")
            return

        if args.dry_run:
            print("Dry run — exiting without computation.")
            return

    # Load model + processor
    print(f"\n🤖 Loading: {args.model_name_or_path}")
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    model.eval()

    # Build dataset once — shared by inspection and dataloader
    print(f"\n📂 Loading pile-10k (max_tokens={args.max_tokens})...")
    tokenizer = getattr(processor, "tokenizer", processor)
    dataset = Pile10kDataset(
        tokenizer=tokenizer, max_tokens=args.max_tokens, min_tokens=args.min_tokens
    )
    print(f"   {len(dataset)} samples loaded")

    # Prompt inspection (before extraction — validates format and span logic)
    if run_inspection:
        print(f"\n🔬 Inspecting {args.n_prompt_samples} prompts → {prompts_dir}")
        inspect_prompts(
            processor=processor,
            model=model,
            dataset=dataset,
            n_samples=args.n_prompt_samples,
            save_dir=prompts_dir,
            model_name_or_path=args.model_name_or_path,
            prompt_format=args.prompt_format,
        )
        if args.inspect_only:
            print("\n--inspect-only: exiting after prompt inspection.")
            return

    if not methods_to_run:
        return

    # Build DataLoader (wraps the already-constructed dataset)
    collate_fn = partial(
        collate_pile10k_batch,
        processor=processor,
        model=model,
        prompt_format=args.prompt_format,
        remove_assistant_tokens=args.remove_assistant_tokens,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    print(f"   {len(dataloader)} batches | batch_size={args.batch_size}")

    # Extract and save per pooling method
    for pm, pv, save_path in methods_to_run:
        print(f"\n{'=' * 64}")
        print(f"🔍 Pooling: {pv}  (method={pm}, format={args.prompt_format}, "
              f"remove_asst={args.remove_assistant_tokens})")
        print(f"{'=' * 64}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        residual_stream = residual_stream_tracer(
            model=model,
            processed_dataloader=dataloader,
            processor=processor,
            return_dtype=MODEL_DTYPE,
            tokens_pooling_method=pm,
            residual_stream_type=args.residual_stream_type,
            return_deepcopy=True,
            save_path=None,
            save_representations=False,
        )

        print(f"   Shape: {tuple(residual_stream.shape)}")
        save_file({"residual_stream": residual_stream}, save_path)
        print(f"   💾 Saved: {save_path}")

        del residual_stream
        gc.collect()
        torch.cuda.empty_cache()

    # Cleanup
    del model, processor, dataloader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
