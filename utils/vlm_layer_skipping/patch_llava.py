"""Monkey-patches LLaVA model forward passes to apply layer skipping during inference."""

from functools import partialmethod
from typing import Unpack

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llava.modeling_llava import (
    LlavaModel,
    LlavaModelOutputWithPast,
)
from transformers.utils import TransformersKwargs

from utils.vlm_layer_skipping.layer_skipping import apply_layer_skip


def llava_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    vision_feature_layer: int | list[int] | None = None,
    vision_feature_select_strategy: str | None = None,
    cache_position: torch.LongTensor | None = None,
    image_sizes: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | LlavaModelOutputWithPast:

    vision_feature_layer = (
        vision_feature_layer
        if vision_feature_layer is not None
        else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=image_sizes,
        )
        image_features = torch.cat(image_features, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        special_image_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # ---- image_limits detection ----
    # OLD (BUGGY): image_limits was only computed when input_ids contained
    # token id 32000. During autoregressive decode HF's generate() passes
    # only the newly sampled token, so input_ids no longer contains the
    # image token and image_limits became None. llama_forward then crashed
    # on `enumerate(image_limits)`. Also 32000 is the llava-1.5-specific
    # image token id; it silently breaks on other Llava variants.
    #
    # if (input_ids == 32000).any():
    #     image_limits = []
    #     for i in range(input_ids.shape[0]):
    #         indices = torch.where(input_ids[i] == 32000)[0]
    #         if indices.numel() > 0:
    #             image_limits.append((indices[0].item(), indices[-1].item()))
    #         else:
    #             image_limits.append(None)
    # else:
    #     image_limits = None

    # Read the image token id from the config so this works across Llava
    # variants (llava-1.5, llava-1.6, ...), with a 32000 fallback.
    image_token_id = getattr(self.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(self.config, "image_token_id", 32000)

    past_len = 0
    if past_key_values is not None:
        try:
            past_len = past_key_values.get_seq_length()
        except Exception:
            past_len = 0

    # Prefill step: clear any stale limits left over from a previous
    # generate() call so a new sample starts clean.
    if past_len == 0:
        self.language_model._cached_image_limits = None

    if input_ids is not None and (input_ids == image_token_id).any():
        image_limits = []
        for i in range(input_ids.shape[0]):
            indices = torch.where(input_ids[i] == image_token_id)[0]
            if indices.numel() > 0:
                image_limits.append((indices[0].item(), indices[-1].item()))
            else:
                image_limits.append(None)
        # Cache on the language model so later decode steps (which no
        # longer see the image token in input_ids) can reuse it.
        self.language_model._cached_image_limits = image_limits
    else:
        image_limits = getattr(self.language_model, "_cached_image_limits", None)

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        image_limits=image_limits,
        **kwargs,
    )

    return LlavaModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


def llama_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    # Pop so image_limits does not leak into decoder_layer(**kwargs).
    image_limits = kwargs.pop("image_limits", None)
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position: torch.Tensor = (
            torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            + past_seen_tokens
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # Pop (not get) so these injected-by-partialmethod kwargs don't leak
    # into decoder_layer(**kwargs), where unknown kwargs can warn or raise
    # depending on the transformers version.
    layers_to_skip = kwargs.pop("layers_to_skip", [])
    ablation_mode = kwargs.pop("ablation_mode", None)

    # With attn_implementation="eager", create_causal_mask always returns a 4D
    # float additive mask (0 / -inf). For flash_attention_2 and some sdpa paths
    # it returns None, in which case our in-place mask edits below would either
    # crash or be silently ignored. patch() enforces eager at patch time; this
    # check catches the case where the impl was switched after patching.
    if causal_mask is None:
        raise RuntimeError(
            "create_causal_mask returned None; the ablation patch requires a "
            "4D additive mask. Load the language model with "
            "attn_implementation='eager'."
        )

    # ---- image attention mask construction ----
    # OLD (BUGGY):
    # (1) Writing `False` is only correct for a BOOL causal_mask. With
    #     attn_implementation="eager" (exposed via --attn_implementation in
    #     lmm_eval.py), create_causal_mask returns a FLOAT additive mask
    #     where -inf = masked and 0.0 = unmasked. `False` casts to 0.0,
    #     which UNMASKS those positions — silently disabling the ablation.
    # (2) The slice `[b, :, end+1:, start:end+1]` assumes the query axis
    #     spans the full sequence. During autoregressive decode the query
    #     axis has size 1, so `end+1:` is an empty slice and nothing gets
    #     masked: newly generated tokens attend to the cached image keys
    #     in the "skipped" layers, partially defeating the ablation.
    # (3) Iterating `enumerate(image_limits)` crashed when image_limits
    #     was None (fixed in llava_forward above).
    #
    # if layers_to_skip:
    #     image_mask = causal_mask.clone()
    #     for b, limit in enumerate(image_limits):
    #         if limit is not None:
    #             start, end = limit
    #             image_mask[b, :, end + 1 :, start : end + 1] = False

    image_mask = None
    if layers_to_skip and image_limits is not None:
        image_mask = causal_mask.clone()
        # Use a dtype-correct "masked" value: False for bool, min-float for
        # additive float masks (eager attention).
        if image_mask.dtype == torch.bool:
            masked_value = False
        else:
            masked_value = torch.finfo(image_mask.dtype).min

        q_len = inputs_embeds.shape[1]
        for b, limit in enumerate(image_limits):
            if limit is None:
                continue
            start, end = limit
            if q_len > 1:
                # Prefill: any post-image query position must not attend to
                # image key positions in the skipped layers.
                image_mask[b, :, end + 1 :, start : end + 1] = masked_value
            else:
                # Decode: the single query sits after every image token, so
                # mask all image KV positions along the key axis for this step.
                image_mask[b, :, :, start : end + 1] = masked_value

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    # During decode q_len == 1 and hidden_states has shape [B, 1, D]; the
    # image token rows only live in the KV cache, so storing/restoring
    # hidden_states at image positions would be out-of-range. Gate the
    # store/restore on the prefill step.
    is_prefill = inputs_embeds.shape[1] > 1
    stored_embeds = None

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

        layer_mask = causal_mask

        # OLD (BUGGY): unconditionally assigned layer_mask = image_mask,
        # but image_mask can be None when image_limits is None (decode of a
        # pure-text prompt, or early generate() step before limits cached).
        # Also stored_embeds was saved on every decode step, and the
        # restore branch indexed hidden_states[b, start:end+1, :] which
        # blows up when hidden_states has shape [B, 1, D].
        #
        # if i in layers_to_skip:
        #     layer_mask = image_mask
        #     if i == layers_to_skip[0] and ablation_mode == "image":
        #         stored_embeds = hidden_states
        # elif (
        #     len(layers_to_skip) > 0
        #     and i == (layers_to_skip[-1] + 1)
        #     and ablation_mode == "image"
        # ):
        #     for b, limit in enumerate(image_limits):
        #         if limit is not None:
        #             start, end = limit
        #             hidden_states[b, start : end + 1, :] = stored_embeds[
        #                 b, start : end + 1, :
        #             ]

        if i in layers_to_skip:
            if image_mask is not None:
                layer_mask = image_mask
            if (
                i == layers_to_skip[0]
                and ablation_mode == "image"
                and is_prefill
            ):
                stored_embeds = hidden_states.clone()

        elif (
            len(layers_to_skip) > 0
            and i == (layers_to_skip[-1] + 1)
            and ablation_mode == "image"
            and is_prefill
            and stored_embeds is not None
            and image_limits is not None
        ):
            # Restore image embeddings to what they were at the start of
            # the ablation block.
            for b, limit in enumerate(image_limits):
                if limit is not None:
                    start, end = limit
                    hidden_states[b, start : end + 1, :] = stored_embeds[
                        b, start : end + 1, :
                    ]

        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


_ABLATION_MODES = {"image", "cross_modal", "layer"}


def patch(model, layers_to_skip, ablation_mode="image"):
    """Apply a layer-level ablation strategy to a LLaVA model.

    ``ablation_mode`` selects the strategy applied to ``layers_to_skip``:
      * ``"image"``: mask cross-modal (text->image) cells AND restore image
        hidden states at the end of the skipped block. Requires eager attn.
      * ``"cross_modal"``: mask cross-modal cells only, leaving image hidden
        states free to evolve. Requires eager attn.
      * ``"layer"``: replace decoder layers with identity modules (full
        layer skip). Works with any attention implementation.
    """
    if ablation_mode not in _ABLATION_MODES:
        raise ValueError(
            f"ablation_mode must be one of {sorted(_ABLATION_MODES)}, "
            f"got {ablation_mode!r}."
        )

    if ablation_mode == "layer":
        apply_layer_skip(model, layers_to_skip, model_type="llava")
        return

    layers_to_skip = list(layers_to_skip or [])

    # The 4D additive mask emitted by create_causal_mask under eager attention
    # is what lets us ablate only the cross-modal (text->image) cells while
    # leaving image->image and image->pre-image-text intact. flash/sdpa may
    # return None or collapse to is_causal, which would make our mask edits
    # a no-op.
    attn_impl = getattr(
        model.language_model.config, "_attn_implementation", None
    )
    if attn_impl != "eager":
        raise ValueError(
            f"patch_llava requires attn_implementation='eager' on the language "
            f"model for ablation_mode={ablation_mode!r}, got {attn_impl!r}. "
            "Reload the model with attn_implementation='eager'."
        )

    # Patch the INNER LlavaModel class. Patching the outer
    # LlavaForConditionalGeneration would bypass lm_head and return a
    # LlavaModelOutputWithPast (no logits), breaking generate().
    model.language_model.__class__.forward = partialmethod(
        llama_forward,
        layers_to_skip=layers_to_skip,
        ablation_mode=ablation_mode,
    )
    LlavaModel.forward = llava_forward
