"""Layer-skipping primitive.

Replaces a contiguous range of decoder layers with a lightweight identity
module so hidden_states pass through untouched. Works on any nn.ModuleList
of decoder layers, so it can be used against the plain HuggingFace model
implementations (LLaVA, InternVL2, LLaVA-OneVision-1.5, Cambrian) as well
as inside the lm-evaluation-harness HFLM wrappers.

Two layer-return conventions are supported:
  * New (transformers >= 4.48): decoder_layer(...) -> torch.Tensor
  * Legacy: decoder_layer(...) -> tuple[Tensor, ...]

The identity module auto-selects the right one based on the replaced
layer's annotated return type and also threads the cache object through
when the legacy tuple form is in use.
"""

from __future__ import annotations

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Identity decoder layer
# ---------------------------------------------------------------------------


class _IdentityDecoderLayer(nn.Module):
    """Drop-in replacement for a transformer decoder layer that acts as
    identity on hidden_states.

    Handles both the new convention (returns plain Tensor) and the legacy
    convention (returns a tuple) so the surrounding model loop works
    unchanged.  Does NOT write to the KV cache — skipped layers simply
    have no cache entries, which DynamicCache handles transparently.
    """

    def __init__(self, layer_idx: int, returns_tensor: bool):
        super().__init__()
        self._returns_tensor = returns_tensor
        self.layer_idx = layer_idx

    @staticmethod
    def _is_cache_object(obj):
        """Return True if *obj* is a transformers Cache (DynamicCache, etc.)."""
        return hasattr(obj, "update") and hasattr(obj, "get_seq_length")

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        if self._returns_tensor:
            return hidden_states

        output_attentions = kwargs.get("output_attentions", False)
        use_cache = kwargs.get("use_cache", False)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            past_key_value = kwargs.get("past_key_value")
            if past_key_value is None:
                past_key_value = kwargs.get("past_key_values")

            if past_key_value is not None and self._is_cache_object(past_key_value):
                # Shared Cache object (DynamicCache etc.) — just pass it
                # through without writing any dummy entries.  Each real
                # layer uses its own slot; skipped slots stay empty.
                outputs += (past_key_value,)
            else:
                # Legacy per-layer tuple convention — return None so the
                # model loop sees a cache entry exists but carries no data.
                outputs += (None,)
        return outputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_returns_tensor(layer) -> bool:
    """Return True if the layer's forward returns a plain Tensor.

    The new convention (transformers >= 4.48) annotates decoder layers as
    ``-> torch.Tensor`` and the outer model loop does
    ``hidden_states = decoder_layer(...)`` directly.  The legacy convention
    returns a tuple ``(hidden_states, ...)``.
    """
    forward_fn = getattr(type(layer), "forward", None)
    if forward_fn is None:
        return False
    ret = getattr(forward_fn, "__annotations__", {}).get("return")
    return ret is torch.Tensor


def get_decoder_layers(model, model_type: str) -> nn.ModuleList:
    """Return the decoder-layer ``nn.ModuleList`` for *model_type*.

    Supported model types: "llava", "internvl", "onevision", "cambrian",
    "default" (plain LLaMA-like text decoder).
    """
    if model_type == "internvl":
        # InternVLChatModel -> language_model (InternLM2ForCausalLM) -> model
        return model.language_model.model.layers
    if model_type == "onevision":
        # LLaVAOneVision1_5_ForConditionalGeneration.model (inner multimodal)
        # -> language_model (text decoder)
        return model.model.language_model.layers
    if model_type == "llava":
        # LlavaForConditionalGeneration exposes language_model via a property
        # that returns model.model.language_model on transformers >= 4.48.
        # Older layouts had language_model.model.layers.
        lm = model.language_model
        return getattr(lm, "layers", getattr(lm, "model", lm).layers)
    if model_type == "cambrian":
        # CambrianLlamaForCausalLM -> model (CambrianLlamaModel / LlamaModel)
        return model.model.layers
    # default / plain text decoder
    return getattr(model, "layers", getattr(model, "model", model).layers)


# def _resolve_skip_range(
#     num_layers: int,
#     start_layer: int,
#     window_size: Optional[int],
# ) -> list[int]:
#     """Compute the list of layer indices to replace with identity modules.

#     When ``window_size`` is ``None`` we skip the prefix ``[0, start_layer)``.
#     When ``window_size`` is set we skip ``[start_layer, start_layer+window_size)``.
#     Indices past ``num_layers`` are clipped.
#     """
#     if window_size is None:
#         if start_layer <= 0:
#             return []
#         end = min(start_layer, num_layers)
#         return list(range(0, end))

#     if window_size <= 0:
#         return []
#     end = min(start_layer + window_size, num_layers)
#     if start_layer >= end:
#         return []
#     return list(range(start_layer, end))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_ORIGINALS_ATTR = "_apply_layer_skip_originals"


def _originals_for(layers: nn.ModuleList) -> dict:
    """Per-ModuleList snapshot of original (non-identity) layers, lazily filled.

    Stored via ``object.__setattr__`` so the dict is a plain Python attribute
    rather than a registered submodule. The values are the original layer
    objects: still on the right device with their parameters intact, but no
    longer in ``_modules`` once their slot has been overwritten.
    """
    originals = getattr(layers, _ORIGINALS_ATTR, None)
    if originals is None:
        originals = {}
        object.__setattr__(layers, _ORIGINALS_ATTR, originals)
    for i, layer in enumerate(layers):
        if i not in originals and not isinstance(layer, _IdentityDecoderLayer):
            originals[i] = layer
    return originals


def apply_layer_skip(
    model,
    layers_to_skip: list[int],
    model_type: str = "default",
    verbose: bool = True,
) -> list[int]:
    """Set the skipped layer set to exactly ``layers_to_skip``, in place.

    * ``layers_to_skip`` is the explicit list of layer indices to replace
      with identity modules. Out-of-range indices are silently dropped.
    * ``model_type`` picks which attribute path to walk to the layers
      ``nn.ModuleList`` (see :func:`get_decoder_layers`).

    The function snapshots the original layers on first call (per ModuleList)
    and on every subsequent call restores any slot currently holding an
    ``_IdentityDecoderLayer`` that is not in the new ``layers_to_skip`` set.
    This makes repeated calls idempotent w.r.t. the *requested set* — needed
    for in-process sweeps (sliding-window / progressive-ablation) that drive
    multiple iterations against a single model instance.
    """
    layers = get_decoder_layers(model, model_type)
    num_layers = len(layers)
    originals = _originals_for(layers)

    target = sorted({i for i in (layers_to_skip or []) if 0 <= i < num_layers})
    target_set = set(target)

    # Restore any slot currently identity-stubbed that should not be skipped now.
    restored = []
    for idx in range(num_layers):
        if (
            isinstance(layers[idx], _IdentityDecoderLayer)
            and idx not in target_set
            and idx in originals
        ):
            layers[idx] = originals[idx]
            restored.append(idx)

    if not target:
        if verbose:
            msg = f"[apply_layer_skip] No layers skipped (baseline), using all {num_layers} layers."
            if restored:
                msg += f" Restored layers {restored[0]}-{restored[-1]}."
            print(msg)
        return target

    # Detect the return convention from a real layer. After the restore pass
    # any slot not in target_set is real, so prefer those.
    sample_idx = next(
        (i for i in range(num_layers)
         if not isinstance(layers[i], _IdentityDecoderLayer)),
        None,
    )
    returns_tensor = (
        _layer_returns_tensor(layers[sample_idx]) if sample_idx is not None else False
    )

    for idx in target:
        if isinstance(layers[idx], _IdentityDecoderLayer):
            continue
        layers[idx] = _IdentityDecoderLayer(
            layer_idx=idx,
            returns_tensor=returns_tensor,
        )

    if verbose:
        msg = (
            f"[apply_layer_skip] Replacing layers {target[0]}-{target[-1]} "
            f"with identity modules (of {num_layers})."
        )
        if restored:
            msg += f" Restored layers {restored[0]}-{restored[-1]}."
        print(msg)

    # Intentionally do NOT renumber layer_idx on the remaining real layers.
    # Each real layer keeps its original slot so identity and real layers
    # never collide in the same DynamicCache entry. DynamicCache extends its
    # internal lists as needed for non-contiguous indices.
    return target


# # ---------------------------------------------------------------------------
# # Backwards-compatible wrapper
# # ---------------------------------------------------------------------------


# class EarlyLayerSkipWrapper(nn.Module):
#     """
#     Wrapper that skips early (or windowed) decoder layers by replacing them
#     with lightweight identity modules. Kept for backward compatibility —
#     prefer :func:`apply_layer_skip` for new code.

#     Embeddings are passed through identity layers straight to the first real
#     layer, preserving generation since later layers and lm_head are intact.

#     Supports LLaVA, InternVL2, LLaVA-OneVision-1.5 and Cambrian.
#     """

#     def __init__(
#         self,
#         model,
#         start_layer: int,
#         is_llava: bool = False,
#         is_internvl: bool = False,
#         is_llava_onevision: bool = False,
#         is_cambrian: bool = False,
#         window_size: Optional[int] = None,
#     ):
#         super().__init__()

#         self.wrapped_model = model
#         self.start_layer = start_layer
#         self.config = model.config
#         self.is_llava = is_llava
#         self.is_internvl = is_internvl
#         self.is_llava_onevision = is_llava_onevision
#         self.is_cambrian = is_cambrian

#         if is_internvl:
#             model_type = "internvl"
#             self.num_layers = model.language_model.model.config.num_hidden_layers
#         elif is_llava_onevision:
#             model_type = "onevision"
#             self.num_layers = model.config.text_config.num_hidden_layers
#         elif is_llava:
#             model_type = "llava"
#             self.num_layers = model.config.text_config.num_hidden_layers
#         elif is_cambrian:
#             model_type = "cambrian"
#             self.num_layers = model.config.num_hidden_layers
#         else:
#             model_type = "default"
#             try:
#                 self.num_layers = model.config.num_hidden_layers
#             except AttributeError:
#                 self.num_layers = model.config.text_config.num_hidden_layers

#         if start_layer < 0 or start_layer >= self.num_layers:
#             raise ValueError(
#                 f"start_layer ({start_layer}) must be in [0, {self.num_layers - 1}]"
#             )

#         self._model_type = model_type
#         # Store via object.__setattr__ so nn.Module doesn't register it as a
#         # submodule (which would duplicate parameters in the module tree).
#         object.__setattr__(
#             self, "_layers_ref", get_decoder_layers(model, model_type)
#         )

#         self._original_layers: dict[int, nn.Module] = {}
#         if start_layer > 0 or window_size is not None:
#             self._patch_layers(window_size)

#     def _patch_layers(self, window_size: Optional[int]) -> None:
#         layers = self._layers_ref
#         to_skip = _resolve_skip_range(self.num_layers, self.start_layer, window_size)
#         if not to_skip:
#             print(
#                 f"[EarlyLayerSkipWrapper] No layers skipped (baseline), "
#                 f"using all {self.num_layers} layers"
#             )
#             return

#         sample_idx = next(
#             (i for i in to_skip if not isinstance(layers[i], _IdentityDecoderLayer)),
#             to_skip[0],
#         )
#         returns_tensor = _layer_returns_tensor(layers[sample_idx])

#         for idx in to_skip:
#             if isinstance(layers[idx], _IdentityDecoderLayer):
#                 continue
#             self._original_layers[idx] = layers[idx]
#             layers[idx] = _IdentityDecoderLayer(
#                 layer_idx=idx,
#                 returns_tensor=returns_tensor,
#             )

#         if window_size is not None:
#             print(
#                 f"[EarlyLayerSkipWrapper] Sliding window mode: skipping layers "
#                 f"{to_skip[0]} to {to_skip[-1]}"
#             )
#         else:
#             print(
#                 f"[EarlyLayerSkipWrapper] Skipping layers 0-{self.start_layer - 1}, "
#                 f"using layers {self.start_layer}-{self.num_layers - 1}"
#             )

#     def restore_layers(self) -> None:
#         layers = self._layers_ref
#         for idx, original_layer in self._original_layers.items():
#             layers[idx] = original_layer
#         self._original_layers = {}

#     def forward(self, *args, **kwargs):
#         return self.wrapped_model(*args, **kwargs)

#     def generate(self, *args, **kwargs):
#         return self.wrapped_model.generate(*args, **kwargs)

#     def prepare_inputs_for_generation(self, *args, **kwargs):
#         return self.wrapped_model.prepare_inputs_for_generation(*args, **kwargs)

#     @property
#     def dtype(self):
#         return self.wrapped_model.dtype

#     @property
#     def device(self):
#         return self.wrapped_model.device

#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             return getattr(self.wrapped_model, name)

#     def __del__(self):
#         try:
#             self.restore_layers()
#         except Exception:
#             pass
