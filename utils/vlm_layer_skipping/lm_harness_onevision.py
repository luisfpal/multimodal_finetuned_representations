"""lm-evaluation-harness HFLM subclass for evaluating LLaVA-OneVision-1.5 models."""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

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


class OnevisionHFLM(HFLM):
    """
    Thin lm-evaluation-harness ``HFLM`` wrapper around
    LLaVA-OneVision-1.5 models.

    Routes evaluation through the full ForConditionalGeneration model
    (which handles text-only gracefully when pixel_values is None).
    No layer-skipping or image-ablation patching is done here — callers
    that want either should apply it externally via
    ``utils.vlm_layer_skipping.patch_onevision.patch``.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        batch_size: int = 1,
        system_prompt: Optional[str] = None,
        max_gen_toks: int = 64,
        **kwargs,
    ):
        revision = kwargs.pop("revision", None)

        self._system_prompt = system_prompt or "You are a helpful assistant."

        attn_implementation = kwargs.pop("attn_implementation", "sdpa")
        min_pixels = kwargs.pop("min_pixels", 256 * 28 * 28)
        max_pixels = kwargs.pop("max_pixels", 1605632)
        onevision_torch_dtype = kwargs.pop("onevision_torch_dtype", "auto")
        _ov_kw = dict(
            torch_dtype=onevision_torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        if revision is not None:
            _ov_kw["revision"] = revision
        if attn_implementation is not None:
            _ov_kw["attn_implementation"] = attn_implementation
        base_model = AutoModelForCausalLM.from_pretrained(pretrained, **_ov_kw).eval()
        print(
            f"[OnevisionPatchHFLM] Loaded LLaVA-OneVision-1.5 (remote code, AutoModelForCausalLM): "
            f"{pretrained}"
        )

        _proc_kw = dict(
            trust_remote_code=True,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        if revision is not None:
            _proc_kw["revision"] = revision
        _processor = AutoProcessor.from_pretrained(pretrained, **_proc_kw)
        tokenizer = _processor.tokenizer

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
        self.AUTO_MODEL_CLASS = AutoModelForCausalLM
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
    # Model call / generate – route through the full model which handles
    # text-only gracefully (pixel_values defaults to None, skipping vision).
    # ------------------------------------------------------------------

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._model(input_ids=inps, use_cache=False).logits

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
