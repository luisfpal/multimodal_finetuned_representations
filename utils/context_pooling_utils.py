from __future__ import annotations

from typing import Any, Optional

import torch

from .internvl_utils import IMG_CONTEXT_TOKEN
from .token_analysis import find_caption_token_span


def get_tokenizer_from_processor(processor: Optional[Any]) -> Optional[Any]:
    """Extract tokenizer from processor/tokenizer object."""
    if processor is None:
        return None
    if hasattr(processor, "tokenizer"):
        return processor.tokenizer
    return processor


def resolve_image_token_ids(model: Any, processor: Optional[Any] = None) -> list[int]:
    """Resolve image-token ids from model/processor without hardcoding."""
    token_ids: set[int] = set()

    image_token_index = getattr(
        getattr(model, "config", None), "image_token_index", None
    )
    if isinstance(image_token_index, int) and image_token_index >= 0:
        token_ids.add(image_token_index)

    img_context_token_id = getattr(model, "img_context_token_id", None)
    if isinstance(img_context_token_id, int) and img_context_token_id >= 0:
        token_ids.add(img_context_token_id)

    tokenizer = get_tokenizer_from_processor(processor)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        for token in (IMG_CONTEXT_TOKEN, "<image>"):
            token_id = convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.add(token_id)

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
