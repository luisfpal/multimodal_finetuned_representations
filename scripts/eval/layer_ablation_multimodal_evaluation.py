#!/usr/bin/env python3
"""
LMMS Evaluation with layer-skipping ablations.

Evaluates multimodal models by masking image attention in selected layers,
allowing ablation studies on the role of individual layers.

Based on lm_harness.py but adapted for lmms_eval library.
"""

import argparse
import json
import os
import sys
import time
import warnings
from typing import Optional

import torch
from lmms_eval import evaluator
from lmms_eval.models.chat.llava_hf import LlavaHf
from lmms_eval.models.chat.llava_onevision1_5 import Llava_OneVision1_5
from lmms_eval.models.simple.internvl2 import InternVL2
from lmms_eval.models.simple.llama_vision import LlamaVision

from utils.internvl2_compat import (
    patch_internvl_model_post_instantiation,
)
from utils.internvl_load_helpers import force_internvl_attn_implementation
from utils.vlm_layer_skipping.cambrian_custom import CambrianCustom
from utils.vlm_layer_skipping.patch_cambrian import (
    configure_layer_skip as configure_cambrian_layer_skip,
)
from utils.vlm_layer_skipping.patch_cambrian import (
    install_compat as install_cambrian_compat,
)
from utils.vlm_layer_skipping.patch_internvl import patch as patch_internvl
from utils.vlm_layer_skipping.patch_llama_vision import patch as patch_llama

# import patch_functions
from utils.vlm_layer_skipping.patch_llava import patch as patch_llava
from utils.vlm_layer_skipping.patch_onevision import patch as patch_onevision

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

from utils.llm_judge import ensure_llm_judge_env

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# =============================================================================
# Prompt + answer-extraction overrides (without modifying lmms-eval source)
#
# Stock vqav2/gqa/mme utils give a soft "Answer with a single word or phrase"
# post-prompt and strict scoring (exact-match for vqav2/gqa, yes/no parser for
# mme).  Large instruction-tuned models (e.g. Llama-3.2-90B-Vision) ignore the
# instruction and reply with full sentences ("The walls are white.", "No, it
# is not."), which the strict scorers count as 0 even when the substring is
# correct.  Affects ~25 pts on gqa and ~50% of mme responses.
#
# We intercept the YAML !function loader to swap:
#   - vqav2_doc_to_text       -> stronger short-answer prompt
#   - vqav2_process_results_val -> extract short answer before scoring
#   - gqa_doc_to_text         -> stronger short-answer prompt
#   - mme_doc_to_text         -> stronger yes/no prompt
#   - mme_process_results     -> extract short answer before scoring
#
# GQA has no custom process_results (uses framework's exact_match), so for GQA
# we additionally wrap lm.generate_until at model creation time -- see
# _install_response_extractor() called from evaluate_with_ablation().
#
# Tasks not listed above pass through unchanged.
# =============================================================================
import re

import lmms_eval.utils as _lmms_utils

_VQA_PROMPT_TAIL = (
    "\nAnswer with a single word or short phrase only. "
    "Do not write a full sentence. Do not repeat the question."
)
_MME_PROMPT_TAIL = '\nAnswer with only "Yes" or "No". Do not add anything else.'

_VERBOSE_PREFIX_RE = re.compile(
    r"^(?:the|this|that|it|they|there|these|those)\s+(?:\S+\s+)*?"
    r"(?:is|are|was|were)\s+",
    re.IGNORECASE,
)
_YN_PREFIX_RE = re.compile(r"^(yes|no)\b", re.IGNORECASE)
# "Right side" / "Top side" -> direction word.  Only directional+side, never
# bare "side", to avoid over-cropping legitimate two-word GTs like "tank top".
_TRAILING_SIDE_RE = re.compile(
    r"^(left|right|top|bottom|middle|front|back|center|upper|lower)\s+side$",
    re.IGNORECASE,
)
# "To the right" / "On the floor" / "In a corner" -> drop the prep + optional
# article.  Empirically all GQA / VQAv2 GTs in our sample are bare nouns, so
# stripping is safe; we additionally require the remainder to be short to
# avoid over-stripping things like "on the way to the airport".
_LEADING_PREP_RE = re.compile(
    r"^(?:to|on|in|at|by|with|of)\s+(?:the|a|an)?\s*",
    re.IGNORECASE,
)
# "A woman" / "The beach" -> "woman" / "beach".  Confirmed safe: 0 / 500
# GQA validation GTs start with an article (and EvalAI strips them on VQAv2
# anyway, so this is redundant-but-harmless there).
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def _extract_short_answer(resp: str) -> str:
    s = resp.strip().rstrip(".").strip()
    if not s:
        return s
    # Yes / No first: catches "No, it does not." -> "No".
    yn = _YN_PREFIX_RE.match(s)
    if yn:
        return s[: yn.end()]
    # "The X is Y." -> "Y".
    m = _VERBOSE_PREFIX_RE.match(s)
    if m:
        s = s[m.end() :].strip().rstrip(".").strip()
    # Specific compound: "Right side" -> "right".  Done before the article /
    # prep rules so we don't accidentally lose the directional word.
    m = _TRAILING_SIDE_RE.match(s)
    if m:
        return m.group(1)
    # "To the right" / "On the floor" -> "right" / "floor", but only if the
    # remainder is a short noun phrase (<=3 words).  Skips edge cases like
    # full sentences that happen to start with a preposition.
    m = _LEADING_PREP_RE.match(s)
    if m:
        rest = s[m.end() :].strip().rstrip(".").strip()
        if rest and len(rest.split()) <= 3:
            s = rest
    # "A woman" / "The chair" -> "woman" / "chair".
    m = _LEADING_ARTICLE_RE.match(s)
    if m:
        rest = s[m.end() :].strip().rstrip(".").strip()
        if rest:
            s = rest
    # Slash alternatives: "Bedroom/office" -> "Bedroom".  Both VQAv2's
    # EvalAI processor and GQA's exact_match strip "/" as punctuation, so
    # the unsplit form ("Bedroomoffice") cannot match either alternative.
    # Take the leftmost item — typically the model's primary guess.
    if "/" in s:
        first = s.split("/", 1)[0].strip().rstrip(".").strip()
        if first:
            s = first
    return s


# --- vqav2 ---
def _patched_vqav2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return f"{doc['question']}{_VQA_PROMPT_TAIL}"


_orig_vqav2_process_results_val = None


def _patched_vqav2_process_results_val(doc, result):
    extracted = [_extract_short_answer(r) if isinstance(r, str) else r for r in result]
    return _orig_vqav2_process_results_val(doc, extracted)


# --- gqa (no custom process_results; extraction installed via lm.generate_until) ---
def _patched_gqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return f"{doc['question']}{_VQA_PROMPT_TAIL}"


# --- mme ---
def _patched_mme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return f"{doc['question'].strip()}{_MME_PROMPT_TAIL}"


_orig_mme_process_results = None


def _patched_mme_process_results(doc, results):
    extracted = [_extract_short_answer(r) if isinstance(r, str) else r for r in results]
    return _orig_mme_process_results(doc, extracted)


_orig_import_function = _lmms_utils.import_function


def _import_function_with_overrides(loader, node):
    fn = _orig_import_function(loader, node)
    global _orig_vqav2_process_results_val, _orig_mme_process_results
    name = getattr(fn, "__name__", "")
    if name == "vqav2_doc_to_text":
        return _patched_vqav2_doc_to_text
    if name == "vqav2_process_results_val":
        _orig_vqav2_process_results_val = fn
        return _patched_vqav2_process_results_val
    if name == "gqa_doc_to_text":
        return _patched_gqa_doc_to_text
    if name == "mme_doc_to_text":
        return _patched_mme_doc_to_text
    if name == "mme_process_results":
        _orig_mme_process_results = fn
        return _patched_mme_process_results
    return fn


# Tasks that need response extraction at lm.generate_until time (i.e. tasks
# that have no !function process_results hook we can intercept).  GQA scores
# via the framework's default exact_match metric, so we strip verbose prefixes
# from the response *before* it reaches that scorer.  vqav2 and mme already
# have their extraction wired through process_results overrides above.
_TASKS_NEEDING_GENERATE_UNTIL_EXTRACTION = {"gqa"}


def _install_response_extractor(lm, active_tasks: set[str]) -> None:
    """Wrap lm.generate_until so responses for `active_tasks` are extracted
    before the framework hands them to the metric.  No-op if no overlap."""
    if not active_tasks:
        return
    original = lm.generate_until

    def patched(requests):
        responses = original(requests)
        out = []
        for req, resp in zip(requests, responses):
            tname = getattr(req, "task_name", None)
            if tname in active_tasks and isinstance(resp, str):
                resp = _extract_short_answer(resp)
            out.append(resp)
        return out

    lm.generate_until = patched


def _enable_short_answer_overrides() -> None:
    """Activate the YAML !function overrides that swap stock vqav2/gqa/mme
    prompts and process_results for short-answer-friendly versions.

    Called only for Llama-family models, since other multimodal models in
    this repo (LlavaHf, OneVision, InternVL2, Cambrian) already obey the
    'single word or phrase' instruction and don't need the patch.  Idempotent.
    """
    if getattr(_enable_short_answer_overrides, "_done", False):
        return
    _lmms_utils.import_function = _import_function_with_overrides
    _enable_short_answer_overrides._done = True
    print(
        "[lmm_eval] llama short-answer overrides ACTIVE: vqav2_val, gqa, mme "
        "(stronger prompts + verbose-prefix / yes-no extractor)."
    )


# =============================================================================
# CONSTANTS
# =============================================================================

# Tasks whose aggregate metrics require the full dataset; CLI --limit is
# ignored for these (the lmms-eval aggregator asserts a fixed sample count).
TASKS_REQUIRE_FULL = {"naturalbench"}

# Reproducibility
RANDOM_SEED = NUMPY_RANDOM_SEED = TORCH_RANDOM_SEED = FEWSHOT_RANDOM_SEED = 42

# Tasks requiring LLM judge (need API key)
LLM_JUDGE_TASKS = {
    "mmvet",
    "mmbench_en_dev",
    "mmbench_cn_dev",
    "mmbench_en_test",
    "mmbench_cn_test",
    "llava_in_the_wild",
}
LLAVA_SYSTEM_INSTRUCTION = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
INTERNVL2_SYSTEM_INSTRUCTION = (
    "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，"
    "英文名叫InternVL, 是一个有用无害的人工智能助手。"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_cli_args(output_path: str) -> argparse.Namespace:
    """Create cli_args for tasks that generate submission files."""
    return argparse.Namespace(output_path=output_path, process_with_media=True)


def get_num_layers(model, model_path: str) -> int:
    """Return the number of LLM decoder layers for the loaded multimodal model."""
    mp = model_path.lower()
    if "llava" in mp and "onevision" not in mp:
        # LlavaForConditionalGeneration
        return model.config.text_config.num_hidden_layers
    if "internvl" in mp:
        return model.language_model.model.config.num_hidden_layers
    if "onevision" in mp:
        return model.config.text_config.num_hidden_layers
    if "cambrian" in mp:
        return model.config.num_hidden_layers
    if "llama" in mp:
        # MllamaForConditionalGeneration
        return model.config.text_config.num_hidden_layers
    raise ValueError(f"Cannot determine num_layers for model: {model_path}")


def validate_task_requirements(tasks: list) -> None:
    """Validate environment variables for tasks requiring LLM judge.

    Loads .env and acquires an Orfeo JWT (via utils.llm_judge) if
    OPENAI_API_KEY isn't already set, instead of requiring a shell wrapper
    to mint it beforehand.
    """
    requires_judge = any(task in LLM_JUDGE_TASKS for task in tasks)

    if requires_judge:
        if not ensure_llm_judge_env():
            eval_logger.error("✗ ERROR: LLM judge API key missing!")
            eval_logger.error(
                f"  Tasks {[t for t in tasks if t in LLM_JUDGE_TASKS]} require OPENAI_API_KEY"
            )
            sys.exit(1)
        eval_logger.info(
            f"✓ LLM Judge: {os.environ.get('MODEL_VERSION')} @ {os.environ.get('OPENAI_API_URL')}"
        )


def compute_text_special_ids(model_path: str, lm) -> list[int]:
    """Return tokenizer special-token ids minus the image / EOI / start-of-
    image markers (model-specific). These are the chat-structure tokens
    that wrap the user turn and the generation prompt — the same set used
    for ``text_special`` pooling in extract_activations.

    Cambrian's image marker is a synthetic -200 sentinel outside the
    tokenizer vocabulary, so no exclusions are needed there: the entire
    ``tokenizer.all_special_ids`` (Llama-3's ``<|eot_id|>``,
    ``<|start_header_id|>``, ``<|end_header_id|>``, ``<|begin_of_text|>``,
    ...) is the chat-structure set.
    """
    mp = model_path.lower()
    if "internvl" in mp:
        tok = lm._tokenizer
        extras_drop = {
            tok.convert_tokens_to_ids("<IMG_CONTEXT>"),
            tok.convert_tokens_to_ids("</img>"),
            tok.convert_tokens_to_ids("<img>"),
        }
    elif "onevision" in mp:
        tok = lm.processor.tokenizer
        extras_drop = {
            tok.convert_tokens_to_ids("<|image_pad|>"),
            tok.convert_tokens_to_ids("<|vision_end|>"),
            tok.convert_tokens_to_ids("<|vision_start|>"),
            tok.convert_tokens_to_ids("<|video_pad|>"),
        }
    elif "cambrian" in mp:
        tok = lm._tokenizer
        extras_drop = set()
    else:
        raise ValueError(
            f"text_special ablation is only wired up for InternVL, "
            f"OneVision, and Cambrian; got model_path={model_path!r}."
        )
    extras_drop.discard(None)
    return [tid for tid in tok.all_special_ids if tid not in extras_drop]


def attach_system_instruction(lm, system_text: str) -> None:
    """Prepend a system message to every prompt by wrapping the model's
    ``processor.apply_chat_template``.

    Required for models whose ``generate_until`` builds prompts via
    ``apply_chat_template`` but does not forward ``simple_evaluate``'s
    ``system_instruction`` arg (notably ``LlamaVision``, which hardcodes a
    single user-only message).

    Useful for A/B testing whether an authorization-style preamble lifts
    Llama-3.2-Vision-Instruct's RLHF refusals on MME artwork / celebrity
    subtasks (~40 % refusal rate without one).

    No-op if ``system_text`` is falsy.  Safe to compose with other wrappers
    on the same processor (e.g. ``attach_prompt_logger``)."""
    if not system_text:
        return
    if not hasattr(lm, "processor") or not hasattr(lm.processor, "apply_chat_template"):
        eval_logger.warning(
            f"--llama_system_instruction set, but {type(lm).__name__} has no "
            "processor.apply_chat_template; skipping injection."
        )
        return

    original_apply = lm.processor.apply_chat_template

    def patched_apply(messages, **kwargs):
        if isinstance(messages, list) and any(
            isinstance(m, dict) and m.get("role") == "system" for m in messages
        ):
            return original_apply(messages, **kwargs)
        new_messages = [{"role": "system", "content": system_text}] + list(messages)
        return original_apply(new_messages, **kwargs)

    lm.processor.apply_chat_template = patched_apply
    print(f"[lmm_eval] system instruction injected ({len(system_text)} chars).")


def attach_prompt_logger(lm, n_samples: int, output_path: str) -> None:
    """Wrap lm.generate_until to capture prompts+responses for the first n_samples."""
    original_generate_until = lm.generate_until

    def patched_generate_until(requests):
        captured_prompts: list[str] = []
        original_apply = lm.processor.apply_chat_template

        def capturing_apply(messages, **kwargs):
            result = original_apply(messages, **kwargs)
            if len(captured_prompts) < n_samples:
                captured_prompts.append(result)
            return result

        lm.processor.apply_chat_template = capturing_apply
        try:
            responses = original_generate_until(requests)
        finally:
            lm.processor.apply_chat_template = original_apply

        samples = [
            {"sample_index": i, "prompt": p, "response": r}
            for i, (p, r) in enumerate(zip(captured_prompts, responses[:n_samples]))
        ]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"Debug prompts/responses saved to {output_path}")
        return responses

    lm.generate_until = patched_generate_until


def evaluate_with_ablation(
    model_path: str,
    tasks: list[str],
    start_layer: int = 0,
    window_size: Optional[int] = None,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    limit: Optional[int] = 2000,
    sliding_window: bool = False,
    log_samples: bool = False,
    lorenzo_skipper: bool = False,
    max_new_tokens: int = 32,
    ablation_mode: str = "image",
    trust_remote_code: bool = False,
    batch_size: int = 1,
    attn_implementation: Optional[str] = None,
    dtype: str = "bfloat16",
    evaluate_full_sliding_window: bool = False,
    evaluate_full_progressive_ablation: bool = False,
    evaluate_full_reverse_progressive_ablation: bool = False,
    evaluate_complementary_intervals: bool = False,
    end_layer: Optional[int] = None,
    ablate_all_layers: bool = False,
    text_special_span: bool = False,
    interval: int = 1,
    debug_samples: int = 0,
    restart_layer: int = 0,
    llama_system_instruction: Optional[str] = None,
):
    """
    Evaluate a multimodal model with image attention masked in specific layers.

    Args:
        model_path: HuggingFace model path
        tasks: List of task names to evaluate
        start_layer: First layer to mask image attention (0-indexed)
        window_size: Number of layers to mask. None = all from start_layer
        batch_size: Batch size for evaluation
        device: Device to use
        output_dir: Base directory for results
        limit: Limit number of samples per task (for testing)

    Returns:
        Dictionary of evaluation results
    """

    # Llama 3.2 Vision ignores the stock 'single word or phrase' post-prompt
    # and answers in full sentences, which the strict vqav2 / gqa / mme
    # scorers count as 0.  Activate short-answer overrides ONLY for the
    # llama family; other models (LlavaHf, OneVision, InternVL2, Cambrian)
    # already obey the instruction and stay on the unpatched pipeline.
    # Must fire before simple_evaluate(), which loads task YAMLs.
    _is_llama = "llama" in model_path.lower()
    if _is_llama:
        _enable_short_answer_overrides()

    if "llava" in model_path.lower() and "onevision" not in model_path.lower():
        print(f"loading llava model {model_path}")

        # lm = LlavaHf(
        #     pretrained=model_path,
        #     device=device,
        #     lorenzo_skipper=lorenzo_skipper,
        #     batch_size=batch_size,
        #     start_layer=start_layer,
        #     window_size=window_size,
        #     device_map="auto",
        #     max_new_tokens=max_new_tokens,
        # )

        system_instruction = LLAVA_SYSTEM_INSTRUCTION
        # eager is required by patch_llava: the 4D additive mask is needed to
        # ablate only the cross-modal part of the attention matrix.
        lm = LlavaHf(
            pretrained=model_path,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        )

    elif "internvl" in model_path.lower():

        # lm = InternVL2Custom(
        #     pretrained=model_path,
        #     device_map="auto",
        #     batch_size="1",
        #     max_new_tokens=max_new_tokens,
        # )

        system_instruction = INTERNVL2_SYSTEM_INSTRUCTION
        # eager is required by patch_internvl for the 4D additive mask.
        # InternVL2.__init__ upstream doesn't expose attn_implementation, so
        # monkey-patch AutoModel.from_pretrained for the duration of the ctor
        # instead of editing the vendored lmms-eval source.
        with force_internvl_attn_implementation("eager"):
            lm = InternVL2(
                pretrained=model_path,
                device_map="auto",
            )
        # InternLM2ForCausalLM in the remote code doesn't inherit from
        # GenerationMixin; on transformers >= 4.50 PreTrainedModel doesn't
        # either, so .generate() goes missing. Inject it post-load.
        patch_internvl_model_post_instantiation(lm._model)
        # InternLM2's tokenizer has pad_token_id=None, so generate() falls
        # back to EOS and prints a warning every call. Set it once here to
        # silence ~2000 log lines per task; safe because batch_size=1.
        # Use the tokenizer's eos id (92542): the language_model's default
        # GenerationConfig (created by patch_internvl_model_post_instantiation)
        # has eos_token_id=None.
        eos_id = lm._tokenizer.eos_token_id
        lm._model.language_model.generation_config.pad_token_id = eos_id

    elif "onevision" in model_path.lower():
        system_instruction = None
        # eager is required by patch_onevision (same rationale as LLaVA).
        # Import-time compat shim for the 8B repo's flash_attn_varlen_func
        # import lives at the top of patch_onevision (already imported above).
        lm = Llava_OneVision1_5(
            pretrained=model_path,
            device_map="auto",
            attn_implementation="eager",
        )

    elif "cambrian" in model_path.lower():
        system_instruction = None
        # eager is required by configure_cambrian_layer_skip (same rationale as LLaVA).
        lm = CambrianCustom(
            pretrained=model_path,
            device_map="auto",
            attn_implementation="eager",
        )
        # Modern-transformers compat is required regardless of whether layer
        # skipping is later configured. Install once here so the no-skip
        # baseline iteration also runs correctly.
        install_cambrian_compat(lm.model)

    elif "llama" in model_path.lower():
        system_instruction = None
        llama_models_dir = os.environ.get(
            "LLAMA_VISION_MODELS_DIR",
            os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"),
        )
        lm = LlamaVision(
            pretrained=f"{llama_models_dir}/{model_path}",
            device_map="auto",
            attn_implementation=attn_implementation,
        )

    else:
        raise ValueError(f"Unsupported model_path: {model_path}")

    # Prevent the base-class clean() from deleting _model between iterations.
    lm.clean = lambda: torch.cuda.empty_cache()

    # For tasks that score via the framework's default metric (no !function
    # process_results hook), apply response extraction at the model output
    # level.  Currently: gqa.  Llama-only, mirroring the YAML overrides above.
    if _is_llama:
        _install_response_extractor(
            lm, _TASKS_NEEDING_GENERATE_UNTIL_EXTRACTION.intersection(tasks)
        )

    # Inject a system message into Llama-3.2-Vision prompts (LlamaVision's
    # generate_until ignores simple_evaluate's system_instruction arg).
    # Use to A/B test whether an authorization preamble unlocks the RLHF
    # refusals on MME artwork / celebrity subtasks.
    if llama_system_instruction and "llama" in model_path.lower():
        attach_system_instruction(lm, llama_system_instruction)

    if debug_samples > 0 and "llama" in model_path.lower():
        debug_out = os.path.join(
            output_dir or ".", model_path.split("/")[-1], "debug_samples.json"
        )
        attach_prompt_logger(lm, debug_samples, debug_out)

    num_layers = get_num_layers(lm._model, model_path)
    print(f"Detected {num_layers} LLM decoder layers for {model_path}")

    layers_to_skip_list: list = []
    if evaluate_full_sliding_window:
        if window_size is None or window_size <= 0:
            raise ValueError(
                "--window_size > 0 is required for --evaluate_full_sliding_window"
            )
        for s in range(restart_layer, num_layers - window_size + 2, interval):
            layers_to_skip_list.append(list(range(s, s + window_size)))

    elif evaluate_full_progressive_ablation:
        # Include the no-skip baseline (w=0) and all progressive windows up to num_layers.
        for w in range(restart_layer, num_layers + 1, interval):
            layers_to_skip_list.append(list(range(0, w)))

    elif evaluate_full_reverse_progressive_ablation:
        # Symmetric to --evaluate_full_progressive_ablation but grows the
        # ablated window from the output side: w=0 -> baseline, w=1 -> last
        # layer, ..., w=num_layers -> all layers.
        for w in range(restart_layer, num_layers + 1, interval):
            layers_to_skip_list.append(list(range(num_layers - w, num_layers)))

    elif evaluate_complementary_intervals:
        # Llama-only: evaluate two complementary cross-attention configurations
        # over a window [start_layer, end_layer).
        #   Iter 1 ("keep"):   keep cross-attn ONLY inside the window
        #                      (skip every cross-attn layer outside it)
        #   Iter 2 ("remove"): remove cross-attn inside the window
        #                      (skip the window, keep the rest)
        # patch_llama only acts on indices that are actually cross-attention
        # layers, so passing the full index range is safe.
        assert "llama" in model_path.lower(), (
            "--evaluate_complementary_intervals is only supported for llama models."
        )
        assert end_layer is not None, (
            "--end_layer is required for --evaluate_complementary_intervals."
        )
        assert 0 <= start_layer < end_layer <= num_layers, (
            f"Need 0 <= start_layer ({start_layer}) < end_layer ({end_layer}) "
            f"<= num_layers ({num_layers})."
        )

        layers_to_skip_list.append(
            list(range(0, start_layer)) + list(range(end_layer, num_layers))
        )
        layers_to_skip_list.append(list(range(start_layer, end_layer)))

    elif ablate_all_layers:
        # Single run with every decoder layer included in layers_to_skip.
        # Works for any model (the per-model patch handlers iterate
        # `layers_to_skip` uniformly).
        layers_to_skip_list = [list(range(num_layers))]

    else:
        assert start_layer is not None and window_size is not None
        single = list(range(start_layer, start_layer + window_size))
        layers_to_skip_list = [single]

    for iter_idx, layers_to_skip in enumerate(layers_to_skip_list):

        # text_special* and all_special modes need the per-model tokenizer to
        # enumerate chat-structure token ids; resolve once per outer iteration.
        ts_ids = None
        if ablation_mode in ("text_special", "text_special_restore", "all_special"):
            ts_ids = compute_text_special_ids(model_path, lm)

        if "llava" in model_path.lower() and "onevision" not in model_path.lower():
            patch_llava(
                model=lm.model,
                layers_to_skip=layers_to_skip,
                ablation_mode=ablation_mode,
            )

        elif "internvl" in model_path.lower():
            patch_internvl(
                model=lm.model,
                layers_to_skip=layers_to_skip,
                ablation_mode=ablation_mode,
                text_special_ids=ts_ids,
                text_special_span=text_special_span,
            )

        elif "onevision" in model_path.lower():
            patch_onevision(
                model=lm.model,
                layers_to_skip=layers_to_skip,
                ablation_mode=ablation_mode,
                text_special_ids=ts_ids,
                text_special_span=text_special_span,
            )

        elif "cambrian" in model_path.lower():
            configure_cambrian_layer_skip(
                model=lm.model,
                layers_to_skip=layers_to_skip,
                ablation_mode=ablation_mode,
                text_special_ids=ts_ids,
                text_special_span=text_special_span,
            )

        elif "llama" in model_path.lower():
            patch_llama(model=lm.model, layers_to_skip=layers_to_skip)

        if not layers_to_skip:
            cur_window = 0
            # In reverse mode anchor the filename at num_layers so the series
            # reads L{num_layers}_W0, L{num_layers}_W{w}, ... with L fixed.
            cur_start = num_layers if evaluate_full_reverse_progressive_ablation else None
            print("Evaluating model with no layers skipped (baseline)")
        else:
            cur_window = len(layers_to_skip)
            cur_start = (
                num_layers
                if evaluate_full_reverse_progressive_ablation
                else layers_to_skip[0]
            )
            if evaluate_complementary_intervals:
                # iter 0 keeps cross-attn inside [start, end); iter 1 removes it.
                if iter_idx == 0:
                    print(
                        f"[complementary_intervals] KEEP cross-attn in "
                        f"[{start_layer},{end_layer}); skip outside "
                        f"({len(layers_to_skip)} layer indices)."
                    )
                else:
                    print(
                        f"[complementary_intervals] REMOVE cross-attn in "
                        f"[{start_layer},{end_layer}); keep outside "
                        f"({len(layers_to_skip)} layer indices)."
                    )
            else:
                print(
                    f"Evaluating model with {ablation_mode} attention masked in layers "
                    f"{layers_to_skip[0]}-{layers_to_skip[-1]}"
                )

        # Skip --limit for tasks that require the full dataset (their
        # aggregator asserts a fixed sample count). simple_evaluate's `limit`
        # is a single scalar applied to every task — no per-task override —
        # so when a full-required task is present we drop --limit globally.
        full_tasks = [t for t in tasks if t in TASKS_REQUIRE_FULL]
        if limit is not None and full_tasks:
            non_full = [t for t in tasks if t not in TASKS_REQUIRE_FULL]
            if non_full:
                eval_logger.warning(
                    f"--limit={limit} dropped for the whole run: {full_tasks} "
                    f"requires the full dataset and lmms-eval applies --limit "
                    f"globally. {non_full} will also run unlimited; run them "
                    f"separately if you need a sample cap."
                )
            else:
                eval_logger.warning(
                    f"Ignoring --limit={limit} for {full_tasks}: aggregate "
                    f"metric requires the full dataset."
                )
            effective_limit = None
        else:
            effective_limit = limit

        print(tasks)
        start = time.time()
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            limit=effective_limit,
            log_samples=log_samples,
            system_instruction=system_instruction,
            cli_args=create_cli_args("./tmp"),
            random_seed=RANDOM_SEED,
            numpy_random_seed=NUMPY_RANDOM_SEED,
            torch_random_seed=TORCH_RANDOM_SEED,
            fewshot_random_seed=FEWSHOT_RANDOM_SEED,
            gen_kwargs=f"max_new_tokens={max_new_tokens}",
        )

        end = time.time()
        print(f"duration: {(end-start)/60}")

        if output_dir and results:
            model_name = model_path.split("/")[-1]
            model_name += f"_{ablation_mode}"
            if sliding_window or evaluate_full_sliding_window:
                model_name += f"_sw{cur_window}"
            if evaluate_full_reverse_progressive_ablation:
                model_name += "_reverse"
            if evaluate_complementary_intervals:
                model_name += f"_win{start_layer}_{end_layer}"

            model_output_dir = os.path.join(output_dir, model_name)

            all_results = results.get("results", {})
            for task in tasks:
                task_results = {
                    k: v
                    for k, v in all_results.items()
                    if k == task or k.startswith(f"{task}_")
                }

                if task_results:
                    if evaluate_complementary_intervals:
                        tag = "keep" if iter_idx == 0 else "remove"
                        filename = f"{task}_{tag}.json"
                    elif cur_start is None:
                        filename = f"{task}_L0_W0.json"
                    elif lorenzo_skipper:
                        filename = f"{task}_L{cur_start}_W{cur_window}_lore.json"
                    else:
                        filename = f"{task}_L{cur_start}_W{cur_window}.json"

                    task_output_dir = os.path.join(model_output_dir, task)
                    os.makedirs(task_output_dir, exist_ok=True)
                    output_path = os.path.join(task_output_dir, filename)
                    with open(output_path, "w") as f:
                        serializable = {
                            "config": {
                                "model_path": model_path,
                                "start_layer": cur_start,
                                "window_size": cur_window,
                                "task": task,
                            },
                            "results": task_results,
                        }
                        json.dump(serializable, f, indent=2, default=str)
                    print(f"Results for {task} saved to {output_path}")

                    if log_samples:
                        task_samples = results.get("samples", {}).get(task, [])
                        if task_samples:
                            samples_filename = filename.replace(
                                ".json", "_samples.jsonl"
                            )
                            samples_path = os.path.join(
                                task_output_dir, samples_filename
                            )
                            with open(samples_path, "w") as f:
                                for sample in task_samples:
                                    f.write(
                                        json.dumps(
                                            sample, default=str, ensure_ascii=False
                                        )
                                        + "\n"
                                    )
                            print(f"Per-sample log for {task} saved to {samples_path}")

        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multimodal LLM with layer-skipping ablations using lmms-eval"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--skip_layers",
        type=int,
        default=None,
        help="Number of early layers to skip. If not set, runs sweep over predefined layers.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["vqav2"],
        help="Tasks to evaluate (e.g., vqav2, mme, seedbench, gqa, textvqa)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/layer_skip_eval_mm",
        help="Output directory for results",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=0,
        help="Size of the window to skip starting at skip_layers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for testing)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to evaluate (overrides default sweep)",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=0,
        help="Start layer for image attention masking. If set, masks text attention "
        "to image tokens starting from this layer.",
    )
    parser.add_argument(
        "--restart_layer",
        type=int,
        default=0,
        help="Start layer for image attention masking. If set, masks text attention "
        "to image tokens starting from this layer.",
    )
    parser.add_argument("--sliding_window", action="store_true")
    parser.add_argument("--lorenzo_skipper", action="store_true")
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log individual samples during evaluation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Hard cap on max new tokens for generation (default: 32)",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="image",
        help="Hard cap on max new tokens for generation (default: 32)",
        choices=[
            "image",
            "cross_modal",
            "eoi",
            "eoi_restore",
            "eoi_query",
            "text_special",
            "text_special_restore",
            "all_special",
            "layer",
        ],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU batch size for generate_until. >1 only takes effect on "
        "models whose generate_until actually batches (LlamaVision does).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="HF attn_implementation, e.g. 'flash_attention_2' or 'sdpa'. "
        "Big speedup on long multimodal prompts.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model compute dtype: bfloat16 / float16 / float32 / auto.",
    )
    parser.add_argument(
        "--evaluate_full_sliding_window",
        action="store_true",
        help="Sweep a fixed-size window across all layers (requires --window_size > 0).",
    )
    parser.add_argument(
        "--evaluate_full_progressive_ablation",
        action="store_true",
        help="Progressively skip layers [0:w] for w in range(0, num_layers+1, interval).",
    )
    parser.add_argument(
        "--evaluate_full_reverse_progressive_ablation",
        action="store_true",
        help="Symmetric to --evaluate_full_progressive_ablation but grows from "
        "the output side: progressively skip layers [num_layers-w : num_layers] "
        "for w in range(0, num_layers+1, interval).",
    )
    parser.add_argument(
        "--ablate_all_layers",
        action="store_true",
        help="Single run with --ablation_mode applied to every decoder layer "
        "(layers_to_skip = [0, 1, ..., num_layers-1]). Works across all models.",
    )
    parser.add_argument(
        "--evaluate_complementary_intervals",
        action="store_true",
        help="Llama-only: run two complementary configurations over the window "
        "[--start_layer, --end_layer): one keeping cross-attention only inside "
        "the window (skip outside), one removing cross-attention inside the "
        "window (skip inside). Use to A/B a contiguous block of cross-attn "
        "layers, e.g. --start_layer 15 --end_layer 50 on Llama-3.2-90B-Vision.",
    )
    parser.add_argument(
        "--end_layer",
        type=int,
        default=None,
        help="Exclusive upper bound for --evaluate_complementary_intervals.",
    )
    parser.add_argument(
        "--text_special_span",
        action="store_true",
        help="For --ablation_mode text_special / text_special_restore: extend "
        "ablated positions to the contiguous range from the first to the "
        "last post-EOI text-special token, so glue tokens between them "
        "(typically '\\n' between <|im_end|> and <|im_start|>) get ablated "
        "too. Single-turn prompts only — multi-turn spans may swallow "
        "assistant content.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Step size for sliding-window / progressive sweeps (default: 1).",
    )
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=0,
        help="If > 0, save the prompt and response for the first N samples to "
        "<output_dir>/<model>/debug_samples.json (llama models only).",
    )
    parser.add_argument(
        "--llama_system_instruction",
        type=str,
        default=None,
        help="System message to prepend to every prompt for Llama-3.2-Vision "
        "(injected via processor.apply_chat_template; LlamaVision's "
        "generate_until otherwise builds user-only messages and ignores "
        "simple_evaluate's system_instruction). Useful to test whether an "
        "authorization preamble unlocks RLHF refusals on MME artwork / "
        "celebrity subtasks. Example: 'You are an AI assistant evaluating "
        "images for academic research. Answer every question concisely "
        "based solely on what you observe in the image.'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate task requirements (e.g., LLM judge API)
    validate_task_requirements(args.tasks)

    start = time.time()
    evaluate_with_ablation(
        model_path=args.model_path,
        tasks=args.tasks,
        start_layer=args.start_layer,
        window_size=args.window_size,
        device=args.device,
        output_dir=args.output_dir,
        limit=args.limit,
        sliding_window=args.sliding_window,
        log_samples=args.log_samples,
        lorenzo_skipper=args.lorenzo_skipper,
        max_new_tokens=args.max_new_tokens,
        ablation_mode=args.ablation_mode,
        batch_size=args.batch_size,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
        evaluate_full_sliding_window=args.evaluate_full_sliding_window,
        evaluate_full_progressive_ablation=args.evaluate_full_progressive_ablation,
        evaluate_full_reverse_progressive_ablation=args.evaluate_full_reverse_progressive_ablation,
        evaluate_complementary_intervals=args.evaluate_complementary_intervals,
        end_layer=args.end_layer,
        ablate_all_layers=args.ablate_all_layers,
        text_special_span=args.text_special_span,
        interval=args.interval,
        debug_samples=args.debug_samples,
        restart_layer=args.restart_layer,
        llama_system_instruction=args.llama_system_instruction,
    )
    print(f"Evaluation took: {(time.time()-start)/60:.2f} min")
