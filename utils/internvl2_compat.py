"""
Compatibility patch for InternVL2 models on transformers >= 4.50.

In transformers >= 4.50, PreTrainedModel no longer inherits from GenerationMixin.
The InternVL2 remote code defines InternLM2ForCausalLM(InternLM2PreTrainedModel)
without GenerationMixin, so .generate() is missing at runtime.

Additionally, transformers >= 4.50 uses DynamicCache objects by default, but
the remote InternVL2 code expects legacy tuple-of-tuples cache format.

This module:
  1. Injects GenerationMixin into InternLM2ForCausalLM's class hierarchy
  2. Disables DynamicCache (sets _supports_cache_class = False)
  3. Wraps prepare_inputs_for_generation to convert any stray Cache → legacy format
"""

import functools
import logging
import sys

logger = logging.getLogger(__name__)


def _needs_generation_mixin_patch():
    """Check whether GenerationMixin is missing from PreTrainedModel's MRO."""
    from transformers import GenerationMixin, PreTrainedModel

    return GenerationMixin not in PreTrainedModel.__mro__


def _cache_to_legacy(past_key_values):
    """Convert a Cache object to legacy tuple format, or None if empty."""
    if past_key_values is None:
        return None
    try:
        from transformers.cache_utils import Cache
    except ImportError:
        return past_key_values
    if isinstance(past_key_values, Cache):
        if past_key_values.get_seq_length() == 0:
            return None
        return past_key_values.to_legacy_cache()
    return past_key_values


def _inject_generation_mixin(cls):
    """Inject GenerationMixin into cls's bases and disable new cache machinery."""
    from transformers import GenerationMixin

    patched = False
    if GenerationMixin not in cls.__mro__:
        cls.__bases__ = (GenerationMixin,) + cls.__bases__
        patched = True
        logger.info(
            f"Injected GenerationMixin into {cls.__qualname__} "
            f"(new MRO: {[c.__name__ for c in cls.__mro__[:5]]}...)"
        )

    # Disable DynamicCache — the remote code only handles legacy
    # tuple-of-tuples cache format.
    cls._supports_cache_class = False

    # Wrap prepare_inputs_for_generation as safety net
    _wrap_prepare_inputs_for_generation(cls)

    return patched


def _wrap_prepare_inputs_for_generation(cls):
    """Wrap prepare_inputs_for_generation to convert Cache -> legacy format.

    Safety net: even with _supports_cache_class=False, this ensures any
    stray Cache object is converted before reaching the remote code.
    """
    original = cls.__dict__.get("prepare_inputs_for_generation")
    if original is None or getattr(original, "_compat_patched", False):
        return

    @functools.wraps(original)
    def wrapper(self, input_ids, past_key_values=None, **kwargs):
        past_key_values = _cache_to_legacy(past_key_values)
        return original(self, input_ids, past_key_values=past_key_values, **kwargs)

    wrapper._compat_patched = True
    cls.prepare_inputs_for_generation = wrapper
    logger.info(f"Wrapped prepare_inputs_for_generation on {cls.__qualname__}")


def _find_causal_lm_class(chat_model_cls):
    """Find InternLM2ForCausalLM from the chat model's module."""
    module = sys.modules.get(chat_model_cls.__module__)
    if module is not None:
        cls = getattr(module, "InternLM2ForCausalLM", None)
        if cls is not None:
            return cls

    # Fallback: search all loaded modules
    for _, mod in sys.modules.items():
        if mod is not None and hasattr(mod, "InternLM2ForCausalLM"):
            return mod.InternLM2ForCausalLM
    return None


def _patch_intern_vit_meta_tensor(chat_model_cls):
    """Patch InternVisionEncoder.__init__ to avoid .item() on meta tensors.

    In PyTorch >= 2.8 with transformers' low_cpu_mem_usage, from_pretrained
    initialises on the meta device.  torch.linspace then produces meta tensors,
    and the original code calls .item() on each element, which raises
    ``RuntimeError: Tensor.item() cannot be called on meta tensors``.

    The fix replaces the tensor-based linspace with a pure-Python equivalent.
    """
    vit_mod_name = chat_model_cls.__module__.replace(
        "modeling_internvl_chat", "modeling_intern_vit"
    )
    vit_mod = sys.modules.get(vit_mod_name)
    if vit_mod is None:
        return

    encoder_cls = getattr(vit_mod, "InternVisionEncoder", None)
    if encoder_cls is None:
        return

    original_init = encoder_cls.__init__
    if getattr(original_init, "_meta_tensor_patched", False):
        return

    @functools.wraps(original_init)
    def _patched_init(self, config):
        super(encoder_cls, self).__init__()
        self.config = config
        n = config.num_hidden_layers
        rate = float(config.drop_path_rate)
        dpr = [rate * i / max(n - 1, 1) for i in range(n)]
        self.layers = __import__("torch").nn.ModuleList(
            [
                vit_mod.InternVisionEncoderLayer(config, dpr[idx])
                for idx in range(n)
            ]
        )
        self.gradient_checkpointing = True

    _patched_init._meta_tensor_patched = True
    encoder_cls.__init__ = _patched_init
    logger.info(
        "Patched InternVisionEncoder.__init__ to avoid .item() on meta tensors"
    )


def patch_internvl_classes_pre_instantiation(chat_model_cls):
    """Patch InternLM2ForCausalLM BEFORE from_pretrained() is called.

    Works by accessing the class via the module that chat_model_cls was loaded
    from (modeling_internvl_chat.py imports InternLM2ForCausalLM at module level).

    Args:
        chat_model_cls: The dynamically loaded InternVLChatModel class.
    """
    _patch_intern_vit_meta_tensor(chat_model_cls)

    if not _needs_generation_mixin_patch():
        return

    causal_lm_cls = _find_causal_lm_class(chat_model_cls)
    if causal_lm_cls is None:
        logger.warning(
            "Could not find InternLM2ForCausalLM in loaded modules; "
            "skipping pre-instantiation patch."
        )
        return

    _inject_generation_mixin(causal_lm_cls)


def patch_internvl_model_post_instantiation(model):
    """Patch the language_model's class AFTER the model is instantiated.

    Fallback for cases where we don't control the loading path
    (e.g. AutoModel.from_pretrained).

    Args:
        model: An instantiated InternVL model with a .language_model attribute.
    """
    if not _needs_generation_mixin_patch():
        return

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        logger.warning(
            "Model has no language_model attribute; skipping post-instantiation patch."
        )
        return

    _inject_generation_mixin(type(language_model))

    if getattr(language_model, "generation_config", None) is None:
        from transformers import GenerationConfig
        language_model.generation_config = GenerationConfig()
        logger.info("Assigned default GenerationConfig to language_model")
