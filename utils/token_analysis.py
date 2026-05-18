"""Extract caption token positions from tokenized VQA prompts.

Strategy: regex on decoded text + binary-search char→token mapping.

The prompt structure (identical for LLaVA and InternVL after tokenization) is::

    [chat prefix]...answer the question.\\n\\n{CAPTION}\\n\\n{question}\\n{guide}...[chat suffix]

Steps:
    1. Decode full ``input_ids`` to plain text.
    2. Regex captures caption **character** boundaries.
    3. Binary search on cumulative decode lengths maps char position → token index.
    4. Verify: decoded token span contains the regex-extracted caption.

The approach is **model-agnostic** — the same regex and binary search work for
any tokenizer (LLaVA Vicuna, InternVL InternLM2, etc.) because the caption
boundary pattern lives in the *decoded text*, not in the token IDs.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# The preamble ALWAYS ends with "answer the question.\n\n".
# The caption ALWAYS ends with "\n\n" before the question.
# Lazy (.+?) stops at the FIRST \n\n after preamble — caption text never
# contains \n\n (multi-caption uses single \n between entries).
_CAPTION_RE = re.compile(
    r"answer the question\.\n\n"  # left boundary (preamble tail)
    r"(.+?)"  # caption (lazy — stops at first \n\n)
    r"\n\n",  # right boundary (before question)
    re.DOTALL,
)


def _decode_length(tokenizer: Any, input_ids: np.ndarray, n_tokens: int) -> int:
    """Return the character length of decoding the first *n_tokens*."""
    if n_tokens == 0:
        return 0
    return len(
        tokenizer.decode(input_ids[:n_tokens].tolist(), skip_special_tokens=False)
    )


def _char_to_token(
    tokenizer: Any, input_ids: np.ndarray, char_pos: int, full_len: int
) -> int:
    """Binary search: find the token index whose decode produces the character at *char_pos*.

    Cumulative decode lengths are monotonically non-decreasing (adding a token
    adds >= 0 characters), so binary search is valid.  O(log n) decode calls.
    """
    lo, hi = 0, full_len - 1
    while lo < hi:
        mid = (lo + hi) // 2
        length = _decode_length(tokenizer, input_ids, mid + 1)
        if length <= char_pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


def find_caption_token_span(
    tokenizer: Any,
    input_ids: np.ndarray,
) -> tuple[int, int]:
    """Find the (start, end) token indices of the caption in a tokenized VQA prompt.

    Args:
        tokenizer: Any HuggingFace tokenizer with ``.decode()``.
        input_ids: 1-D array of token IDs for a single sequence.

    Returns:
        ``(start, end)`` — exclusive end, Python-slice convention.
        ``input_ids[start:end]`` are the caption tokens.

    Raises:
        RuntimeError: If the caption pattern is not found or verification fails.
    """
    input_ids = np.asarray(input_ids)
    n = len(input_ids)

    # Stage 1: Decode → regex-find caption character boundaries
    full_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

    match = _CAPTION_RE.search(full_text)
    if not match:
        raise RuntimeError(
            f"Caption regex did not match in decoded text.\n"
            f"  Pattern: {_CAPTION_RE.pattern!r}\n"
            f"  Text (first 500 chars): {full_text[:500]!r}"
        )

    caption_char_start = match.start(1)
    caption_char_end = match.end(1)  # exclusive

    # Stage 2: Binary search for token boundaries — O(log n) decode calls
    start_tok = _char_to_token(tokenizer, input_ids, caption_char_start, n)
    end_tok = _char_to_token(tokenizer, input_ids, caption_char_end - 1, n) + 1

    # Stage 3: Verify via cumulative decode (not subsequence decode,
    # because SentencePiece decodes subsequences differently from context)
    tok_char_start = _decode_length(tokenizer, input_ids, start_tok)
    tok_char_end = _decode_length(tokenizer, input_ids, end_tok)
    actual_text = full_text[tok_char_start:tok_char_end]
    caption_from_regex = match.group(1)

    if caption_from_regex not in actual_text:
        raise RuntimeError(
            f"Verification failed: regex caption not in token span text.\n"
            f"  start_tok={start_tok}, end_tok={end_tok}\n"
            f"  Regex caption: {caption_from_regex[:80]!r}...\n"
            f"  Token span text: {actual_text[:80]!r}..."
        )

    return start_tok, end_tok


# Delimiter for pile-10k "Continue this text" prompts.
# Three newlines are used so the boundary is unambiguous across all model chat
# templates (LLaVA 1.5, LLaVA OV, InternVL2, Cambrian all use at most \n\n).
_USER_TEXT_PREAMBLE = "Continue this text:\n\n\n"
_USER_TEXT_DELIMITER = "\n\n\n"


def find_user_text_span(
    tokenizer: Any,
    input_ids: np.ndarray,
) -> tuple[int, int]:
    """Find the (start, end) token indices of {text_context} in a pile-10k prompt.

    The prompt is expected to contain the unique substring
    ``"Continue this text:\\n\\n\\n{text_context}\\n\\n\\n"``.
    The function locates the preamble by substring search, then finds the next
    ``\\n\\n\\n`` delimiter.  Both char positions are mapped to token indices via
    binary search — identical strategy to ``find_caption_token_span``.

    Args:
        tokenizer: Any HuggingFace tokenizer with ``.decode()``.
        input_ids: 1-D array of token IDs for a single sequence.

    Returns:
        ``(start, end)`` — exclusive end, Python-slice convention.

    Raises:
        RuntimeError: If the preamble or closing delimiter is not found.
    """
    input_ids = np.asarray(input_ids)
    decoded = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

    preamble_pos = decoded.find(_USER_TEXT_PREAMBLE)
    if preamble_pos == -1:
        raise RuntimeError(
            f"Preamble {_USER_TEXT_PREAMBLE!r} not found in decoded sequence.\n"
            f"  Text (first 300 chars): {decoded[:300]!r}"
        )
    char_start = preamble_pos + len(_USER_TEXT_PREAMBLE)

    # Use rfind to locate the CLOSING delimiter — the last \n\n\n before the assistant
    # turn.  find(char_start) would match the FIRST \n\n\n after char_start, which can
    # be INSIDE text_context when the tokenizer's truncation boundary leaves text_context
    # starting with \n\n\n (e.g. a sample that begins with blank lines).  rfind is safe
    # because model templates never produce \n\n\n in the assistant suffix.
    char_end = decoded.rfind(_USER_TEXT_DELIMITER)
    if char_end <= char_start:
        raise RuntimeError(
            f"Closing delimiter {_USER_TEXT_DELIMITER!r} not found after preamble.\n"
            f"  char_start={char_start}, rfind returned char_end={char_end}\n"
            f"  Text around preamble: {decoded[preamble_pos:preamble_pos + 200]!r}"
        )

    full_len = len(input_ids)
    tok_start = _char_to_token(tokenizer, input_ids, char_start, full_len)

    T_end = _char_to_token(tokenizer, input_ids, char_end - 1, full_len)
    # BPE tokenizers can merge the last chars of text_context with the leading \n\n\n
    # delimiter into a single token (e.g., "incomes.\n\n\n"). If so, decode(0..T_end)
    # overshoots char_end. Exclude that boundary-spanning token so the span stays
    # within text_context. This is model-agnostic: no class-name checks needed.
    if _decode_length(tokenizer, input_ids, T_end + 1) > char_end:
        tok_end = T_end
    else:
        tok_end = T_end + 1

    # Safety guard: T_end + 1 can still be <= tok_start when T_end < tok_start (e.g.
    # empty text_context after tokenizer truncation). tok_start + 1 guarantees a
    # non-empty span in all cases since tok_start + 1 > tok_start unconditionally.
    if tok_end <= tok_start:
        tok_end = tok_start + 1

    return tok_start, tok_end
