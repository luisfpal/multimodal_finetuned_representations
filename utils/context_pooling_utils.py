from __future__ import annotations

from typing import Any, Optional

import torch

from .internvl_utils import IMG_CONTEXT_TOKEN
from .token_analysis import find_caption_token_span, find_user_text_span

# Per-model anchor sequences at the end of the user role (immediately before the
# assistant turn).  Each entry specifies the full tail token-ID sequence and how
# many of those tokens (counting from the start of the anchor) to include in the
# pool.  The last anchor tokens (e.g. ":" in LLaVA 1.5) mark role boundaries and
# are excluded from the pooled representation.
#
# Detection is done by matching the tail of the content-stripped input_ids against
# these anchors in order.  The list is checked top-to-bottom; first match wins.
END_USER_ROLE_CONFIGS: list[dict] = [
    {   # LLaVA 1.5 — ASSISTANT: (▁A SS IST ANT :)
        "anchor_ids": [319, 1799, 9047, 13566, 29901],
        "n_pool": 4,   # exclude the final ':'
    },
    {   # LLaVA OneVision — <|im_end|> Ċ <|im_start|> assistant Ċ
        "anchor_ids": [151645, 198, 151644, 77091, 198],
        "n_pool": 3,   # first 3 of 5
    },
    {   # Cambrian-8B — <|eot_id|> <|start_header_id|> assistant <|end_header_id|>
        "anchor_ids": [128009, 128006, 78191, 128007],
        "n_pool": 2,   # first 2 of 4
    },
    {   # InternVL2-8B — <|im_end|> <|im_start|> ass istant (space)
        "anchor_ids": [92542, 92543, 525, 11353, 364],
        "n_pool": 2,   # first 2 of 5
    },
]


def get_tokenizer_from_processor(processor: Optional[Any]) -> Optional[Any]:
    """Extract tokenizer from processor/tokenizer object."""
    if processor is None:
        return None
    if hasattr(processor, "tokenizer"):
        # Qwen2TokenizerFast for LLaVA-OneVision
        # LlamaTokenizerFast for LLaVA 1.5
        return processor.tokenizer
    return processor


def resolve_image_token_ids(model: Any, processor: Optional[Any] = None) -> list[int]:
    """Resolve image-token ids from model/processor without hardcoding."""
    token_ids: set[int] = set()

    image_token_index = getattr(
        getattr(model, "config", None), "image_token_index", None
    )
    if isinstance(image_token_index, int):
        token_ids.add(image_token_index)

    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if isinstance(model_type, str) and "cambrian" in model_type.lower():
        try:
            from cambrian.constants import IMAGE_TOKEN_INDEX as CAMBRIAN_IMAGE_TOKEN_INDEX

            if isinstance(CAMBRIAN_IMAGE_TOKEN_INDEX, int):
                token_ids.add(CAMBRIAN_IMAGE_TOKEN_INDEX)
        except Exception:
            pass

    img_context_token_id = getattr(model, "img_context_token_id", None)
    if isinstance(img_context_token_id, int) and img_context_token_id != -1:
        token_ids.add(img_context_token_id)

    tokenizer = get_tokenizer_from_processor(processor)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        for token in (IMG_CONTEXT_TOKEN, "<image>", "<|image_pad|>"):
            token_id = convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.add(token_id)

    token_ids.discard(0)
    return sorted(token_ids)


def get_image_token_indices(
    input_ids_1d: torch.Tensor,
    image_token_ids: list[int],
) -> torch.Tensor:
    """Get token positions for any image token id in one sequence."""
    if input_ids_1d.ndim != 1:
        raise ValueError(
            f"Expected 1D input_ids, got shape={tuple(input_ids_1d.shape)}"
        )

    if not image_token_ids:
        return torch.empty(0, dtype=torch.long, device=input_ids_1d.device)

    token_ids = torch.tensor(
        image_token_ids,
        dtype=input_ids_1d.dtype,
        device=input_ids_1d.device,
    )
    return torch.nonzero(torch.isin(input_ids_1d, token_ids), as_tuple=True)[0]


def get_end_user_role_indices(input_ids_1d: torch.Tensor) -> torch.Tensor:
    """Return pooling indices for the end-of-user-role anchor of the current model.

    Auto-detects the model by matching the tail of ``input_ids_1d`` against the
    known per-model anchor sequences in ``END_USER_ROLE_CONFIGS``.  Raises
    ``ValueError`` if no anchor matches, which catches wrong model or prompt-format
    changes at extraction time rather than silently producing nonsense embeddings.

    Args:
        input_ids_1d: 1-D content-stripped token-ID tensor (padding already removed).

    Returns:
        1-D ``torch.long`` tensor of ``n_pool`` indices into ``input_ids_1d``.
    """
    if input_ids_1d.ndim != 1:
        raise ValueError(
            f"Expected 1D input_ids, got shape={tuple(input_ids_1d.shape)}"
        )
    for config in END_USER_ROLE_CONFIGS:
        anchor = config["anchor_ids"]
        n = len(anchor)
        if len(input_ids_1d) < n:
            continue
        tail = input_ids_1d[-n:]
        expected = torch.tensor(
            anchor, dtype=input_ids_1d.dtype, device=input_ids_1d.device
        )
        if torch.equal(tail, expected):
            n_pool = config["n_pool"]
            start = len(input_ids_1d) - n
            return torch.arange(
                start, start + n_pool, dtype=torch.long, device=input_ids_1d.device
            )
    raise ValueError(
        "end_user_role: could not find any known anchor sequence in the tail of "
        f"input_ids.  Tail (last 6 tokens): {input_ids_1d[-6:].tolist()}.  "
        "Supported models: LLaVA 1.5, LLaVA OneVision, Cambrian-8B, InternVL2-8B."
    )


def get_text_context_indices(
    input_ids_1d: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    """Get caption token positions for one sequence via regex + binary search.

    Model-agnostic: works for any tokenizer (LLaVA, InternVL, etc.) because
    the caption boundary is detected in the decoded text, not in token IDs.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for mean_text_context pooling")

    ids_np = input_ids_1d.detach().cpu().numpy()

    try:
        start_idx, end_idx = find_caption_token_span(tokenizer, ids_np)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc

    return torch.arange(
        start_idx, end_idx, dtype=torch.long, device=input_ids_1d.device
    )


def get_user_text_indices(
    input_ids_1d: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    """Token indices spanning {text_context} in a pile-10k 'Continue this text' prompt.

    Model-agnostic: the boundary is found in decoded text via substring search,
    not in token IDs.  See ``find_user_text_span`` for the delimiter strategy.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for last_user_text pooling")

    ids_np = input_ids_1d.detach().cpu().numpy()

    try:
        start_idx, end_idx = find_user_text_span(tokenizer, ids_np)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc

    return torch.arange(
        start_idx, end_idx, dtype=torch.long, device=input_ids_1d.device
    )
