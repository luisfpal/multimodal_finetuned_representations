import torch
from transformers import AutoModel, AutoTokenizer

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


from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    configure_pad_token,
    stop_sequences_criteria,
)

from utils.internvl2_compat import (
    patch_internvl_model_post_instantiation,
)
from utils.internvl_conversations import get_conv_template


class InternVLHFLM(HFLM):
    """
    Thin lm-evaluation-harness ``HFLM`` wrapper around
    ``InternVLChatModel``.

    Routes evaluation through the language-model backbone
    (``InternLM2ForCausalLM``), bypassing the vision encoder entirely.
    No layer-skipping or image-ablation patching is done here — callers
    that want either should apply it externally via
    ``utils.vlm_layer_skipping.patch_internvl.patch``.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_gen_toks: int = 256,
        **kwargs,
    ):

        base_model = AutoModel.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device,
        ).eval()

        patch_internvl_model_post_instantiation(base_model)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True, use_fast=False
        )

        # Bypass HFLM's full __init__ (we loaded the model ourselves)
        super(HFLM, self).__init__()

        self._model = base_model
        self.tokenizer = tokenizer
        self._device = torch.device(device if isinstance(device, str) else "cuda")
        self.backend = "causal"

        self._max_length = (
            base_model.language_model.model.config.max_position_embeddings
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
        self.add_bos_token = None
        self.custom_prefix_token_id = tokenizer.bos_token_id
        self._max_gen_toks = max_gen_toks if max_gen_toks is not None else 256
        self.AUTO_MODEL_CLASS = AutoModel

        self._system_prompt = base_model.system_message

        self.pretrained = pretrained
        self.revision = "main"
        self.peft = None
        self.delta = None
        self.think_end_token = None

        self._conv_template_name = base_model.template

        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

    # ------------------------------------------------------------------
    # Properties that differ from the parent HFLM defaults
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self):
        template = get_conv_template(self._conv_template_name)
        return self.tokenizer.convert_tokens_to_ids(template.sep.strip())

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    # ------------------------------------------------------------------
    # Chat template – use InternVL's own Conversation format (text-only)
    # so that the prompt is identical to what chat() / batch_chat()
    # produce (minus the image-token injection which is not needed here).
    #
    # tok_encode / tok_batch_encode are intentionally NOT overridden:
    # the parent HFLM implementations do plain tokenization.  The
    # formatting is applied once here; re-applying it inside tok_encode
    # would double-wrap the prompt and break the loglikelihood
    # continuation-offset logic in _encode_pair.
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        template = get_conv_template(self._conv_template_name)
        template.system_message = self._model.system_message

        for msg in chat_history:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                template.system_message = content
            elif role == "user":
                template.append_message(template.roles[0], content)
            elif role == "assistant":
                template.append_message(template.roles[1], content)

        if add_generation_prompt:
            template.append_message(template.roles[1], None)

        result = template.get_prompt()

        if (
            not add_generation_prompt
            and chat_history
            and chat_history[-1]["role"] == "assistant"
            and result.endswith(template.sep)
        ):
            result = result[: -len(template.sep)]

        return result

    # ------------------------------------------------------------------
    # Model call / generate – route through the language model backbone
    # so we bypass InternVLChatModel.forward (which requires pixel_values).
    # ------------------------------------------------------------------

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._model.language_model(
                input_ids=inps,
            ).logits

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ) -> torch.Tensor:
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")

        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        template = get_conv_template(self._conv_template_name)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_kwargs["eos_token_id"] = eos_token_id

        return self._model.language_model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
