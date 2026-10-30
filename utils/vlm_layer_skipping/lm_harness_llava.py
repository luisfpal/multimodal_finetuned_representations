"""lm-evaluation-harness HFLM wrapper for LlavaForConditionalGeneration."""

import torch
from transformers import AutoProcessor

# Monkeypatch: older custom model files import flash_attn_varlen_func directly from
# transformers.modeling_flash_attention_utils, but newer transformers removed that re-export.
# Restore the symbol so the import succeeds even when flash_attention_2 is not used.
try:
    import transformers.modeling_flash_attention_utils as _flash_utils

    if not hasattr(_flash_utils, "flash_attn_varlen_func"):
        try:
            from flash_attn import flash_attn_varlen_func as _favf

            _flash_utils.flash_attn_varlen_func = _favf
        except ImportError:
            _flash_utils.flash_attn_varlen_func = None
except ImportError:
    pass

from typing import Optional

from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import stop_sequences_criteria
from transformers import LlavaForConditionalGeneration

from utils.constants import (
    LLAVA_CHAT_TEMPLATE,
    LLaVA1_5_SYSTEM_INSTRUCTION,
)


class LlavaHFLM(HFLM):
    """
    Thin lm-evaluation-harness ``HFLM`` wrapper around
    ``LlavaForConditionalGeneration``.

    Routes evaluation through the full model (which handles text-only
    gracefully when pixel_values is None). No layer-skipping or
    image-ablation patching is done here — callers that want either
    should apply it externally via
    ``utils.vlm_layer_skipping.patch_llava.patch``.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
        system_prompt: Optional[str] = None,
        max_gen_toks: int = 256,
        **kwargs,
    ):
        revision = kwargs.pop("revision", None)

        self._system_prompt = system_prompt or LLaVA1_5_SYSTEM_INSTRUCTION

        _lv_kw = dict(torch_dtype=dtype, device_map=device)
        if revision is not None:
            _lv_kw["revision"] = revision
        base_model = LlavaForConditionalGeneration.from_pretrained(
            pretrained, **_lv_kw
        ).eval()
        print(f"[SkipLayersHFLM] Loaded LLaVA model: {pretrained}")

        _p_kw = {}
        if revision is not None:
            _p_kw["revision"] = revision
        processor = AutoProcessor.from_pretrained(pretrained, **_p_kw)
        tokenizer = processor.tokenizer
        tokenizer.chat_template = LLAVA_CHAT_TEMPLATE

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Bypass HFLM's full __init__ (we loaded the model ourselves)
        super(HFLM, self).__init__()

        self._model = base_model
        self.tokenizer = tokenizer
        self._device = torch.device(device if isinstance(device, str) else "cuda")
        self.backend = "causal"

        self._max_length = base_model.config.text_config.max_position_embeddings

        self.truncation = False

        self.logits_cache = False
        self._rank = 0
        self._world_size = 1
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = batch_size
        self.batch_size_per_gpu = batch_size

        self._config = base_model.config

        self.mixed_precision_dtype = None
        self.softmax_dtype = torch.float32
        self.add_bos_token = False
        self.custom_prefix_token_id = tokenizer.bos_token_id
        self._max_gen_toks = max_gen_toks if max_gen_toks is not None else 256
        self.AUTO_MODEL_CLASS = LlavaForConditionalGeneration
        self.revision = revision if revision is not None else "main"
        self.pretrained = pretrained
        self.peft = None
        self.delta = None
        self.think_end_token = None
        self.chat_template_args = {}

    # ------------------------------------------------------------------
    # Properties that differ from the parent HFLM defaults
    # ------------------------------------------------------------------

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    # ------------------------------------------------------------------
    # Chat template
    #
    # LLAVA_CHAT_TEMPLATE is the multimodal-typed schema used by HF's
    # llava processor (``content`` is a list of ``{"type": ..., "text"|...}``
    # dicts). lm-eval-harness, however, hands us chat histories with plain
    # string ``content``. Calling the template with strings makes the
    # template's ``selectattr('type','equalto','text')`` filter select
    # nothing — the entire user/system text is silently dropped and the
    # model is asked an empty prompt (`'USER: ASSISTANT:'`), producing
    # chance-level results on every task. Promote string content into the
    # typed-list form before delegating to the tokenizer template.
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        chat_history,
        add_generation_prompt: bool = True,
    ) -> str:
        normalized = []
        for msg in chat_history:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": msg["role"], "content": content})
        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # ------------------------------------------------------------------
    # Model call / generate – route through the full model which handles
    # text-only gracefully (pixel_values defaults to None, skipping vision).
    #
    # _model_call is overridden because the parent asserts AUTO_MODEL_CLASS
    # is AutoModelForCausalLM or AutoModelForVision2Seq, but we use
    # LlavaForConditionalGeneration.
    # ------------------------------------------------------------------

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._model(input_ids=inps).logits

    def _model_generate(self, context, max_length, stop, **kwargs):
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        kwargs["temperature"] = kwargs.get("temperature", 0.0)
        do_sample = kwargs.get("do_sample")
        if kwargs.get("temperature") == 0.0 and do_sample is None:
            kwargs["do_sample"] = False
        if kwargs.get("do_sample") is False and kwargs.get("temperature") == 0.0:
            kwargs.pop("temperature", None)

        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **kwargs,
        )
