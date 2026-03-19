"""
Compatibility patch for LLaVA-OneVision-1.5 remote code on transformers >= 4.47.

The model's remote ``modeling_llavaonevision1_5.py`` imports
``flash_attn_varlen_func`` from ``transformers.modeling_flash_attention_utils``,
but that symbol was removed from the transformers public API — it only lives in
the ``flash_attn`` package itself.

This patch injects the symbol back into the module namespace so the remote
import succeeds without touching the cached file on disk.

Safe to call unconditionally:
  - No-op when flash-attention is not installed.
  - No-op when the symbol is already present (future transformers versions).
  - No-op for every other model that does not use this import path.

Upstream fix (merged but not yet synced to HF hub as of 2026-03):
  https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5/pull/33
"""

import logging

logger = logging.getLogger(__name__)


def patch_flash_attn_varlen_import() -> None:
    """Inject ``flash_attn_varlen_func`` into ``transformers.modeling_flash_attention_utils``.

    Must be called before ``AutoModelForCausalLM.from_pretrained`` for any
    model that uses ``trust_remote_code=True`` and imports this symbol (e.g.
    LLaVA-OneVision-1.5). Idempotent — safe to call multiple times.
    """
    try:
        import transformers.modeling_flash_attention_utils as _fa_utils
    except ImportError:
        return

    if hasattr(_fa_utils, "flash_attn_varlen_func"):
        return  # already present — no-op

    try:
        from transformers.modeling_flash_attention_utils import is_flash_attn_available
    except ImportError:
        return

    if not is_flash_attn_available():
        return

    try:
        from flash_attn import flash_attn_varlen_func

        _fa_utils.flash_attn_varlen_func = flash_attn_varlen_func
        logger.info(
            "llavaonevision_compat: injected flash_attn_varlen_func into "
            "transformers.modeling_flash_attention_utils"
        )
    except ImportError:
        logger.warning(
            "llavaonevision_compat: flash_attn not installed --"
            " flash_attn_varlen_func could not be injected;"
            " LLaVA-OneVision-1.5 will fail to load."
        )
