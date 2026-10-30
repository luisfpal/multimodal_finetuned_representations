"""lm-evaluation-harness HFLM subclass for running Cambrian vision-language models."""

import torch

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
from lm_eval.models.utils import configure_pad_token, stop_sequences_criteria

from utils.cambrian_bridge import load_cambrian_pretrained
from utils.cambrian_utils import (
    get_cambrian_device,
    resolve_cambrian_conv_mode,
)


class CambrianHFLM(HFLM):
    """
    Thin lm-evaluation-harness ``HFLM`` wrapper around Cambrian models.

    No layer-skipping or image-ablation patching is done here. Callers
    MUST apply ``utils.vlm_layer_skipping.patch_cambrian.patch`` on
    ``self._model`` after instantiation, because the vendored
    ``CambrianLlamaModel.forward`` is incompatible with modern
    transformers (it indexes decoder-layer outputs as tuples, but modern
    ``LlamaDecoderLayer`` returns a plain Tensor). Without the patch,
    generation and forward will misbehave.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        max_gen_toks: int = 256,
        attn_implementation: str = "sdpa",
        cache_dir: Optional[str] = None,
        conv_mode: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        base_model, tokenizer = load_cambrian_pretrained(
            pretrained,
            cache_dir=cache_dir,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        print(f"[CambrianPatchHFLM] Loaded Cambrian model: {pretrained}")

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Bypass HFLM's full __init__ (we loaded the model ourselves).
        super(HFLM, self).__init__()

        self._model = base_model
        self.tokenizer = tokenizer
        self._device = (
            get_cambrian_device(base_model)
            if device in (None, "cuda")
            else torch.device(device)
        )
        self.backend = "causal"

        # CambrianConfig subclasses LlamaConfig so the LLM decoder config
        # (including max_position_embeddings) is reachable directly.
        self._max_length = getattr(
            base_model.config,
            "max_position_embeddings",
            getattr(
                getattr(base_model.config, "text_config", None),
                "max_position_embeddings",
                2048,
            ),
        )

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
        # AUTO_MODEL_CLASS is set to the concrete model class so HFLM's
        # introspection checks don't reject the loaded model.
        self.AUTO_MODEL_CLASS = type(base_model)
        self.revision = "main"
        self.pretrained = pretrained
        self.peft = None
        self.delta = None
        self.think_end_token = None
        self.chat_template_args = {}

        self._conv_mode = resolve_cambrian_conv_mode(
            model=base_model,
            model_name_or_path=pretrained,
            explicit_conv_mode=conv_mode,
        )

        from cambrian.conversation import conv_templates

        self._system_prompt = system_prompt or conv_templates[self._conv_mode].system

        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

    # ------------------------------------------------------------------
    # Properties that differ from the parent HFLM defaults
    # ------------------------------------------------------------------

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    # ------------------------------------------------------------------
    # Chat template – use Cambrian's own Conversation format for the
    # selected conv_mode so text-only prompts match how the model was
    # trained. Image tokens are intentionally not injected.
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        from cambrian.conversation import conv_templates

        conv = conv_templates[self._conv_mode].copy()
        # Cambrian's Conversation.get_prompt asserts that the first appended
        # message is user-role and non-empty (LLAMA_2 / LLAMA_3 sep_style).
        # lm-eval can hand us a system-only chat history for multiple_input
        # likelihood tasks like winogrande when num_fewshot=0 and no
        # description is set — substitute a single space so the assert
        # passes; the resulting prefix is still essentially "BOS + system +
        # generation header", which is the right context for likelihood
        # scoring of the doc_to_choice continuations.
        sentinel = " "
        has_user_msg = False
        for msg in chat_history:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                conv.system = content
            elif role == "user":
                conv.append_message(conv.roles[0], content or sentinel)
                has_user_msg = True
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)

        if not has_user_msg:
            conv.append_message(conv.roles[0], sentinel)

        if add_generation_prompt:
            conv.append_message(conv.roles[1], None)

        return conv.get_prompt()

    # ------------------------------------------------------------------
    # Model call / generate
    #
    # After patch_cambrian_image.patch is applied externally,
    # CambrianLlamaForCausalLM.forward handles text-only calls gracefully
    # (no images -> prepare_inputs_labels_for_multimodal returns inputs
    # unchanged), so we route through the outer model and reuse its
    # lm_head.
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

        # After patching, ``model.generate`` is ``generate_cambrian`` which
        # accepts ``inputs`` (not ``input_ids``). Passing images=None drives
        # the text-only branch; LlamaForCausalLM.generate handles the rest.
        return self._model.generate(
            inputs=context,
            images=None,
            image_sizes=None,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **kwargs,
        )
