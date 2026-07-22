"""
Load helpers for InternVL2 that avoid editing the vendored lmms-eval source.

Problem: `lmms_eval.models.simple.internvl2.InternVL2.__init__` calls
`AutoModel.from_pretrained(...)` without exposing `attn_implementation` or
`use_flash_attn`. The cross-modal ablation patch in `patch_internvl_lorenzo.py`
requires eager attention so that a 4D additive mask is available to edit.

Solution: a context manager that monkey-patches `AutoModel.from_pretrained`
for the duration of the `InternVL2(...)` constructor. Inside the context:
- `attn_implementation` is injected into the kwargs;
- when "eager" is requested, `use_flash_attn=False` is also injected, because
  `InternVLChatModel.__init__` (custom trust_remote_code code) sets
  `config.llm_config.attn_implementation` from `use_flash_attn` at construction
  time, and the attention module class is chosen at that moment.

Outside the `with` block, `AutoModel.from_pretrained` is restored unchanged,
so nothing else in the process is affected.
"""

from contextlib import contextmanager

from transformers import AutoModel


@contextmanager
def force_internvl_attn_implementation(attn_implementation):
    """Temporarily inject ``attn_implementation`` (and ``use_flash_attn`` when
    eager) into every call to ``AutoModel.from_pretrained`` inside the block.

    Usage::

        from utils.internvl_load_helpers import (
            force_internvl_attn_implementation,
        )

        with force_internvl_attn_implementation("eager"):
            lm = InternVL2(pretrained=model_path, device_map="auto")
    """
    if attn_implementation is None:
        yield
        return

    original_from_pretrained = AutoModel.from_pretrained

    def wrapper(*args, **kwargs):
        kwargs["attn_implementation"] = attn_implementation
        if attn_implementation == "eager":
            # InternVLChatModel.__init__ reads `use_flash_attn` to decide
            # the inner InternLM2 attention class; without this the decoder
            # layers would still be instantiated as flash attention.
            kwargs["use_flash_attn"] = False
        elif attn_implementation == "flash_attention_2":
            kwargs["use_flash_attn"] = True
        return original_from_pretrained(*args, **kwargs)

    AutoModel.from_pretrained = wrapper
    try:
        yield
    finally:
        AutoModel.from_pretrained = original_from_pretrained
