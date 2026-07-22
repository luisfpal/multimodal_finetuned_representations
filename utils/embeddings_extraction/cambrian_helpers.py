"""Cambrian-specific helpers for embedding extraction.

Cambrian doesn't fit the AutoProcessor / AutoModel.from_pretrained pattern
the rest of embeddings_vlms.py is built around:

  - Loading goes through ``cambrian.model.builder.load_pretrained_model``
    (we reuse the existing utils/cambrian_model bridge for cache-compat
    patches and consistent behavior with the lmm_eval path).
  - Prompts are built with ``cambrian.conversation.conv_templates`` and
    tokenized via ``tokenizer_image_token`` — ``<image>`` becomes a single
    ``IMAGE_TOKEN_INDEX = -200`` *sentinel* in input_ids, NOT a real token id.
  - The model expands that one sentinel into ``image_token_len`` (576 for
    cambrian-8b) image embeddings inside ``prepare_inputs_labels_for_multimodal``,
    so hidden-state length is ``T_orig - 1 + 576 != T_orig``. The
    ``attention_mask`` we hand the model gets re-shaped inside that call.

Implications for extract_activations:

  - The activation hooks see the *expanded* sequence. Pooling can't use the
    original ``attention_mask`` directly; we capture the post-expansion mask
    by patching ``prepare_inputs_labels_for_multimodal``.
  - Image-avg / EOI pooling can't key off real token ids in input_ids
    (the sentinel is gone after expansion). We synthesize a fake input_ids
    tensor in expanded space marking image-token positions with -200 and
    the position immediately after the image span with -201 — these are
    then matched the same way as InternVL's <IMG_CONTEXT> / </img>.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# Sentinel stashed in the synthetic pool input_ids at image-span positions.
# -200 reuses Cambrian's IMAGE_TOKEN_INDEX (the model can't see this
# synthetic tensor anyway — it's only consumed by extract_activations.
# _update_activation_dict to find image positions for _imgavg pooling).
# Cambrian has no end-of-image / image–text separator token, so we don't
# inject an EOI sentinel: extract_activations falls back to the last image
# position as the post-image boundary for _textavg / _textspec, and the
# _eoi / _imgeoi extras are intentionally not allocated for Cambrian.
IMAGE_TOKEN_SENTINEL = -200
# Cambrian's cmm appends one image_newline embedding (a learned LM-domain
# parameter, not a vision-projector output) after every row of image
# features. We mark those positions with a separate sentinel so the
# default _imgavg pool can exclude them — pooling the newline parameter
# in pulls the image-pool toward LM-token distribution and inflates
# imgavg↔textavg retrieval at early layers. _imgavg_with_newline keeps
# them included for comparison.
IMAGE_NEWLINE_SENTINEL = -201


def load_cambrian(model_path: str, attn_implementation: str = "eager"):
    """Load Cambrian model + tokenizer + image processors via the bridge.

    Returns (model, tokenizer, image_processor, conv_mode).
    """
    from utils.cambrian_bridge import load_cambrian_pretrained
    from utils.cambrian_utils import resolve_cambrian_conv_mode

    model, tokenizer = load_cambrian_pretrained(
        model_path, attn_implementation=attn_implementation
    )
    image_processor = getattr(tokenizer, "image_processor", None)
    if image_processor is None:
        raise RuntimeError(
            "Cambrian tokenizer has no image_processor attached after "
            "load_cambrian_pretrained — bridge state is inconsistent."
        )
    conv_mode = resolve_cambrian_conv_mode(
        model=model, model_name_or_path=model_path
    )
    return model, tokenizer, image_processor, conv_mode


def _build_prompt(
    question: str, image_only: bool, conv_mode: str, model_config
) -> str:
    from cambrian.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
    )
    from cambrian.conversation import conv_templates

    use_im_start_end = bool(getattr(model_config, "mm_use_im_start_end", False))
    img_prefix = (
        f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
        if use_im_start_end
        else DEFAULT_IMAGE_TOKEN
    )
    qs = img_prefix if image_only else f"{img_prefix}\n{question}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _tokenize_prompt(prompt: str, tokenizer, conv_mode: str) -> torch.Tensor:
    from cambrian.constants import IMAGE_TOKEN_INDEX
    from cambrian.mm_utils import (
        tokenizer_image_token,
        tokenizer_image_token_llama3,
    )

    tokenize_fn = (
        tokenizer_image_token_llama3
        if conv_mode in {"llama_3", "llama_v3"}
        else tokenizer_image_token
    )
    return tokenize_fn(
        prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )


def _process_image_cpu(image: Image.Image, image_processor) -> List[torch.Tensor]:
    """CPU-only single-image variant of cambrian.mm_utils.process_images.

    Cambrian-8b has multiple aux vision towers (~4); each gets its own
    preprocessor with its own crop_size. Returns a list of per-tower
    tensors of shape (C, H_i, W_i). The original process_images calls
    .half().cuda() at the end which we avoid so this can run in
    DataLoader workers without CUDA context.
    """
    from cambrian.mm_utils import expand2square

    if image.mode != "RGB":
        image = image.convert("RGB")
    out: List[torch.Tensor] = []
    for proc in image_processor:
        cur = image
        if hasattr(proc, "image_mean"):
            target = proc.crop_size["height"]
            cur = expand2square(
                cur, tuple(int(x * 255) for x in proc.image_mean)
            ).resize((target, target))
        cur = proc.preprocess(cur, return_tensors="pt")["pixel_values"][0]
        out.append(cur)
    return out


class CambrianImageDataset(Dataset):
    """Dataset for Cambrian: builds a -200-sentinel input_ids and per-tower
    image tensors. Returns plain Python objects from __getitem__; the
    Cambrian collator below assembles them into a batch (forced batch_size=1).

    image_only=True keeps only the <image> placeholder in the user message.
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        image_processor,
        model_config,
        conv_mode: str,
        image_only: bool = False,
        image_dtype: torch.dtype = torch.float16,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.image_only = image_only
        # Cambrian's load_pretrained_model returns the model in float16; the
        # vision towers and projector expect inputs in the same dtype.
        self.image_dtype = image_dtype
        # Human-readable per-sample prompts saved as metadata downstream.
        # Built once with the <image> text placeholder still in place — the
        # tokenizer_image_token call in __getitem__ replaces that with the
        # -200 sentinel, which would not be useful for analysis.
        self.text_prompts = [
            _build_prompt(
                r.get("question", ""), self.image_only, self.conv_mode, self.model_config
            )
            for r in records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        question = record.get("question", "")

        prompt = _build_prompt(
            question, self.image_only, self.conv_mode, self.model_config
        )
        input_ids = _tokenize_prompt(prompt, self.tokenizer, self.conv_mode)
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
        attention_mask = torch.ones_like(input_ids)

        image_aux_list = [
            t.to(self.image_dtype)
            for t in _process_image_cpu(image, self.image_processor)
        ]

        return {
            "text_prompts": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images_aux_list": image_aux_list,
            "image_size": image.size,  # (W, H)
        }


def cambrian_collator(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cambrian collator. Forces batch_size=1 because:

      - input_ids carries one -200 per sample at a sample-dependent position,
        and the post-expansion T_exp = T_orig - 1 + image_token_len differs
        across samples whenever T_orig differs;
      - per-aux image tensors have different per-tower crop sizes that the
        model expects as a list of (B, C, H_i, W_i) — at B>1 we'd need to
        pad image_size lists and reconcile per-sample expansion offsets in
        the pool_info synthesis. Not needed for embedding extraction.
    """
    if len(items) != 1:
        raise ValueError(
            "CambrianImageDataset requires batch_size=1 — see "
            "cambrian_helpers.py docstring for the constraint."
        )
    item = items[0]
    return {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "images": [aux.unsqueeze(0) for aux in item["images_aux_list"]],
        "image_sizes": [item["image_size"]],
    }


def wrap_cambrian_for_pooling(model) -> None:
    """Patch model.prepare_inputs_labels_for_multimodal so each forward
    leaves on the model a ``_cambrian_pool_aux`` dict with:

      - ``attention_mask_expanded``: (B, T_exp) post-expansion mask, on CPU.
      - ``synthetic_input_ids``: (B, T_exp) Long tensor on CPU with
        IMAGE_TOKEN_SENTINEL across the image span and the original
        tokenizer ids preserved everywhere else. extract_activations uses
        the image sentinel for _imgavg pooling; the real ids at other
        positions feed _textspec's all_special_ids matching.

    No EOI sentinel is injected: Cambrian has no image–text separator
    token, so eoi_token_id is left unset by the caller and the post-image
    boundary for _textavg / _textspec falls back to the last image
    position.
    """
    original = model.prepare_inputs_labels_for_multimodal

    def patched(input_ids, position_ids, attention_mask, *args, **kwargs):
        # Capture image positions and the original input_ids BEFORE
        # expansion. With batch_size=1 each sample has one (or zero) -200.
        orig_input_ids = None
        image_positions: List[Optional[int]] = []
        if input_ids is not None:
            orig_input_ids = input_ids.detach().cpu()
            for b in range(input_ids.shape[0]):
                pos = (input_ids[b] == IMAGE_TOKEN_SENTINEL).nonzero(as_tuple=True)[0]
                image_positions.append(pos[0].item() if len(pos) > 0 else None)

        result = original(
            input_ids, position_ids, attention_mask, *args, **kwargs
        )

        expanded_mask = result[2]
        inputs_embeds = result[4]
        # cmm returns final_size at index 8: list of (cur_h, cur_w) per
        # sample after unpad_image, used here to locate the per-row
        # image_newline positions inside the image span.
        final_size = result[8] if len(result) > 8 else None
        if (
            expanded_mask is not None
            and inputs_embeds is not None
            and image_positions
            and orig_input_ids is not None
        ):
            B, T_exp = inputs_embeds.shape[:2]
            # The actual inserted image span length is variable per sample
            # (depends on aspect ratio after unpad_image, plus one
            # image_newline per row), NOT model.config.image_token_len.
            # Recover it from the expansion delta. Requires B==1, which the
            # cambrian_collator already enforces.
            assert B == 1, (
                "wrap_cambrian_for_pooling assumes batch_size=1; the cambrian "
                "collator enforces this — image span length recovery is "
                "ambiguous at B>1 because each sample expands by a different "
                "amount."
            )
            t_orig_total = orig_input_ids.shape[1]
            actual_image_len = T_exp - (t_orig_total - 1)
            synthetic = torch.zeros((B, T_exp), dtype=torch.long)
            for b, p in enumerate(image_positions):
                orig = orig_input_ids[b]
                T_orig = orig.shape[0]
                if p is None:
                    # No image — expanded sequence equals the original.
                    copy_len = min(T_orig, T_exp)
                    synthetic[b, :copy_len] = orig[:copy_len]
                    continue
                # Pre-image text: copy original ids verbatim.
                synthetic[b, :p] = orig[:p]
                # Image span: -200 sentinel.
                end_image = min(p + actual_image_len, T_exp)
                synthetic[b, p:end_image] = IMAGE_TOKEN_SENTINEL
                # Overwrite per-row image_newline positions with -201.
                # cmm flattens (cur_h, cur_w+1, D) → (cur_h*(cur_w+1), D),
                # so within the image span the newline lands at the last
                # column of every row: offset r*(cur_w+1) + cur_w.
                if final_size is not None and b < len(final_size):
                    cur_h, cur_w = final_size[b]
                    if cur_h * (cur_w + 1) != actual_image_len:
                        # Sanity check: derived span must match cmm's output.
                        # If this fires, cmm's layout assumption changed
                        # upstream and the offsets below would be wrong.
                        raise RuntimeError(
                            f"cmm final_size {(cur_h, cur_w)} implies image "
                            f"span {cur_h * (cur_w + 1)} but expansion delta "
                            f"is {actual_image_len}"
                        )
                    for r in range(cur_h):
                        nl_pos = p + r * (cur_w + 1) + cur_w
                        if nl_pos < end_image:
                            synthetic[b, nl_pos] = IMAGE_NEWLINE_SENTINEL
                # Post-image text: copy original ids[p+1..] (skip the one
                # -200 in the original) into expanded positions [p+actual_image_len..].
                tail_dst_start = p + actual_image_len
                tail_src_start = p + 1
                tail_len = min(T_orig - tail_src_start, T_exp - tail_dst_start)
                if tail_len > 0:
                    synthetic[b, tail_dst_start:tail_dst_start + tail_len] = (
                        orig[tail_src_start:tail_src_start + tail_len]
                    )
            model._cambrian_pool_aux = {
                "attention_mask_expanded": expanded_mask.detach().cpu(),
                "synthetic_input_ids": synthetic,
            }
        return result

    model.prepare_inputs_labels_for_multimodal = patched


def make_cambrian_pool_info_provider(model):
    """Return a callable suitable for extract_activations(pool_info_provider=...).

    Reads what wrap_cambrian_for_pooling stashed during the most recent forward
    and returns it shaped how _update_activation_dict expects.
    """

    def provider(batch):
        aux = getattr(model, "_cambrian_pool_aux", None)
        if aux is None:
            raise RuntimeError(
                "model._cambrian_pool_aux missing — call "
                "wrap_cambrian_for_pooling(model) before extraction."
            )
        return {
            "mask": aux["attention_mask_expanded"] != 0,
            "input_ids": aux["synthetic_input_ids"],
        }

    return provider


# Layer paths for get_target_layers. Cambrian's loaded class is
# CambrianLlamaForCausalLM (no AutoModel wrapper, no separate language_model
# attribute) — decoder layers live at "model.layers.{i}" and the final
# norm at "model.norm".
CAMBRIAN_LAYER_PREFIX = "model.layers"
CAMBRIAN_NORM_PATH = "model.norm"
