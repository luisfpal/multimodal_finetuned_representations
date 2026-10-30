"""Patches InternVL model classes with layer-skipping forward and generate methods."""

from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
from transformers import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from utils.vlm_layer_skipping.layer_skipping import apply_layer_skip

logger = logging.get_logger(__name__)

def ilm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:

    # Pop so partialmethod-injected kwargs don't leak into decoder_layer(**kwargs).
    layers_to_skip = kwargs.pop("layers_to_skip", [])
    ablation_mode = kwargs.pop("ablation_mode", None)

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # The ablation edits a 4D additive mask in place. flash_attention_2 uses a
    # 2D padding mask and collapses causal masking into its kernel, so the
    # edits would not apply. Require eager at patch time.
    if self.config.attn_implementation == "flash_attention_2":
        raise RuntimeError(
            "InternVL ablation requires attn_implementation='eager'. Reload "
            "the model with attn_implementation='eager' to expose a 4D "
            "additive mask for cross-modal ablation."
        )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.tok_embeddings(input_ids)

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    causal_mask = self._prepare_decoder_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )
    if causal_mask is None:
        raise RuntimeError(
            "_prepare_decoder_attention_mask returned None; the InternVL "
            "ablation patch requires a 4D additive mask (eager attention)."
        )

    # Build the ablation mask: clone the causal mask and zero-out a slice of
    # key columns (and for "eoi_query", a query row) so the chosen attention
    # cells become inaccessible inside the skipped layers. image_limits stored
    # on self (the InternLM2Model instance) by ivl_generate covers <img> as
    # `start` and </img> (EOI) as `end` via ±1 expansion around IMG_CONTEXT.
    # text_special_positions, also stashed by ivl_generate, lists per-sample
    # post-EOI positions of chat-structure tokens (<|im_end|>, <|im_start|>).
    #
    # Ablation slices:
    #   image / cross_modal           -> keys [start, end] vs queries (end, ...]
    #   eoi / eoi_restore             -> key [end]         vs queries (end, ...]
    #   eoi_query                     -> keys [start, end) vs query  [end]
    #                                    (prefill only)
    #   text_special / text_special_restore
    #                                 -> per ts position p: key [p] vs queries (p, ...]
    image_limits = getattr(self, "image_limits", None)
    text_special_positions = getattr(self, "text_special_positions", None)
    image_mode = ablation_mode in (
        "image", "cross_modal", "eoi", "eoi_restore", "eoi_query"
    )
    text_special_mode = ablation_mode in ("text_special", "text_special_restore")
    all_special_mode = ablation_mode == "all_special"
    image_mask = None
    if layers_to_skip and image_mode and image_limits is not None:
        image_mask = causal_mask.clone()
        masked_value = (
            torch.finfo(image_mask.dtype).min
            if torch.is_floating_point(image_mask)
            else False
        )
        q_len = inputs_embeds.shape[1]
        for b, limit in enumerate(image_limits):
            if limit is None:
                continue
            start, end = limit
            if ablation_mode == "eoi_query":
                # Block the EOI query from attending to image-span keys
                # (start..end-1, i.e. <img> + IMG_CONTEXT, leaving EOI's
                # self-attention intact). Decode is a no-op: by then EOI's
                # KV entry is already fixed from prefill.
                if q_len > 1:
                    image_mask[b, :, end : end + 1, start:end] = masked_value
            else:
                if ablation_mode in ("eoi", "eoi_restore"):
                    k_lo, k_hi = end, end + 1
                else:  # image, cross_modal
                    k_lo, k_hi = start, end + 1
                if q_len > 1:
                    # Prefill: block post-image-text queries from those keys.
                    image_mask[b, :, end + 1 :, k_lo:k_hi] = masked_value
                else:
                    # Decode: single query sits after the image block.
                    image_mask[b, :, :, k_lo:k_hi] = masked_value
    elif layers_to_skip and text_special_mode and text_special_positions is not None:
        image_mask = causal_mask.clone()
        masked_value = (
            torch.finfo(image_mask.dtype).min
            if torch.is_floating_point(image_mask)
            else False
        )
        q_len = inputs_embeds.shape[1]
        for b, positions in enumerate(text_special_positions):
            if not positions:
                continue
            for pos in positions:
                if q_len > 1:
                    image_mask[b, :, pos + 1 :, pos : pos + 1] = masked_value
                else:
                    image_mask[b, :, :, pos : pos + 1] = masked_value
    elif layers_to_skip and all_special_mode and image_limits is not None:
        # Union of {eoi, eoi_query, text_special (span), text_special_query}.
        # Ablation positions A = {EOI} ∪ span(text-special). For each p in A:
        #   (i)  block queries q > p from key [p]                — downstream-block
        #   (ii) block query [p] from image-span keys (excl. p)  — upstream-block
        image_mask = causal_mask.clone()
        masked_value = (
            torch.finfo(image_mask.dtype).min
            if torch.is_floating_point(image_mask)
            else False
        )
        q_len = inputs_embeds.shape[1]
        for b, limit in enumerate(image_limits):
            if limit is None:
                continue
            start, end = limit
            ts_positions = (
                text_special_positions[b] if text_special_positions else None
            ) or []

            # (i) Downstream-block from EOI key.
            if q_len > 1:
                image_mask[b, :, end + 1 :, end : end + 1] = masked_value
            else:
                image_mask[b, :, :, end : end + 1] = masked_value
            # (ii) EOI query → image content keys [start, end-1] (preserves
            # EOI's self-attention to itself).
            if q_len > 1:
                image_mask[b, :, end : end + 1, start:end] = masked_value

            for pos in ts_positions:
                # (i) Downstream-block from text-special key.
                if q_len > 1:
                    image_mask[b, :, pos + 1 :, pos : pos + 1] = masked_value
                else:
                    image_mask[b, :, :, pos : pos + 1] = masked_value
                # (ii) text-special query → full image span [start, end].
                # Safe because pos > end, so pos itself is not in the slice.
                # Decode (q_len == 1) is a no-op here: the query is past pos
                # and pos's KV entry is already fixed from prefill.
                if q_len > 1:
                    image_mask[b, :, pos : pos + 1, start : end + 1] = masked_value

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    is_prefill = inputs_embeds.shape[1] > 1
    stored_embeds = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_mask = causal_mask
        # Restore modes: at the end of the skipped block, write the stored
        # pre-block hidden states back at the ablation positions, so the
        # cross-modal block is the only thing the masked positions saw.
        #   "image"               -> restore image span [start, end]
        #   "eoi_restore"         -> restore [end]
        #   "text_special_restore"-> restore each text-special position
        does_restore = ablation_mode in (
            "image", "eoi_restore", "text_special_restore"
        )
        restore_source_present = (
            (image_mode and image_limits is not None)
            or (text_special_mode and text_special_positions is not None)
        )
        if idx in layers_to_skip:
            if image_mask is not None:
                layer_mask = image_mask
            if (
                idx == layers_to_skip[0]
                and does_restore
                and is_prefill
                and restore_source_present
            ):
                stored_embeds = hidden_states.clone()
        elif (
            len(layers_to_skip) > 0
            and idx == layers_to_skip[-1] + 1
            and does_restore
            and is_prefill
            and stored_embeds is not None
            and restore_source_present
        ):
            if image_mode:
                for b, limit in enumerate(image_limits):
                    if limit is not None:
                        start, end = limit
                        if ablation_mode == "eoi_restore":
                            hidden_states[b, end : end + 1, :] = stored_embeds[
                                b, end : end + 1, :
                            ]
                        else:
                            hidden_states[b, start : end + 1, :] = stored_embeds[
                                b, start : end + 1, :
                            ]
            else:  # text_special_restore
                for b, positions in enumerate(text_special_positions):
                    if not positions:
                        continue
                    for pos in positions:
                        hidden_states[b, pos : pos + 1, :] = stored_embeds[
                            b, pos : pos + 1, :
                        ]

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                layer_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


@torch.no_grad()
def ivl_generate(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_ids: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    visual_features: Optional[torch.FloatTensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    output_hidden_states: Optional[bool] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    assert self.img_context_token_id is not None
    # Clear any image_limits / text_special_positions cached on the inner
    # model by a prior generate() call so a sample without images (or a
    # different image layout) doesn't inherit stale positions.
    self.language_model.model.image_limits = None
    self.language_model.model.text_special_positions = None

    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if (input_ids == self.img_context_token_id).any():
            image_limits = []
            seq_len = input_ids.shape[1]
            for i in range(input_ids.shape[0]):
                indices = torch.where(input_ids[i] == self.img_context_token_id)[0]
                if indices.numel() > 0:
                    # ±1 includes the <img> / </img> delimiters that wrap
                    # the IMG_CONTEXT block in InternVL's chat template.
                    start = max(0, indices[0].item() - 1)
                    end = min(seq_len - 1, indices[-1].item() + 1)
                    image_limits.append((start, end))
                else:
                    image_limits.append(None)

            self.language_model.model.image_limits = image_limits

            # Post-EOI text-special positions (chat-structure tokens like
            # <|im_end|>, <|im_start|> that wrap the user turn / generation
            # prompt). Cached only when text_special_ids was provided at
            # patch time. Mirrors `text_special` pooling in extract_activations.
            text_special_ids = getattr(
                self.language_model.model, "_text_special_ids", None
            )
            if text_special_ids:
                span = bool(getattr(
                    self.language_model.model, "_text_special_span", False
                ))
                text_special_positions = []
                pos_idx = torch.arange(seq_len, device=input_ids.device)
                for i in range(input_ids.shape[0]):
                    ts_mask = torch.zeros(
                        seq_len, dtype=torch.bool, device=input_ids.device
                    )
                    for tid in text_special_ids:
                        ts_mask |= input_ids[i] == tid
                    if image_limits[i] is not None:
                        _, end = image_limits[i]
                        ts_mask &= pos_idx > end
                    positions = torch.where(ts_mask)[0].tolist()
                    # Span mode: extend to the contiguous range from first to
                    # last text-special position so glue tokens between them
                    # (typically the "\n" connecting <|im_end|> and
                    # <|im_start|>) get ablated too.
                    if span and positions:
                        positions = list(range(positions[0], positions[-1] + 1))
                    text_special_positions.append(positions or None)
                self.language_model.model.text_special_positions = (
                    text_special_positions
                )

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id

        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


_ABLATION_MODES = {
    "image",
    "cross_modal",
    "eoi",
    "eoi_restore",
    "eoi_query",
    "text_special",
    "text_special_restore",
    "all_special",
    "layer",
}


def patch(
    model,
    layers_to_skip,
    ablation_mode="image",
    text_special_ids=None,
    text_special_span=False,
):
    """Apply a layer-level ablation strategy to an InternVL model.

    Modes shared with :func:`utils.vlm_layer_skipping.patch_llava.patch`:
      * ``"image"``: mask cross-modal cells [start, end] AND restore image-
        span hidden states at the end of the block.
      * ``"cross_modal"``: same mask as ``"image"`` without the restore.
      * ``"layer"``: replace skipped layers with identity modules.

    EOI-only modes (InternVL/OneVision; image_limits' ``end`` is the EOI
    position, i.e. ``</img>``):
      * ``"eoi"``: block post-image-text queries from attending to the EOI
        key column only. Image content keys are still attended to. Probes
        whether EOI is the bottleneck for image->text information flow.
      * ``"eoi_restore"``: same mask as ``"eoi"`` plus restore EOI's hidden
        state at the end of the block (analog of ``"image"`` for EOI alone).
      * ``"eoi_query"``: block the EOI query position from attending to the
        image span (prefill only). Probes whether EOI summarizes image
        content during the skipped layers.

    Text-special modes (post-EOI chat-structure tokens like ``<|im_end|>``,
    ``<|im_start|>`` that wrap the user turn / generation prompt; same set
    as ``text_special`` pooling in extract_activations):
      * ``"text_special"``: per text-special position p, block downstream
        queries from attending to key column [p]. Probes how much
        downstream generation depends on the chat-structure tokens.
      * ``"text_special_restore"``: same mask plus restore the hidden
        states at the text-special positions at the end of the block.
    Both require ``text_special_ids`` to be a non-empty list of token ids
    (typically ``tokenizer.all_special_ids`` minus image / EOI / start-of-
    image markers).

    ``text_special_span`` (opt-in) extends the ablated positions to the
    contiguous range from the first to the last post-EOI text-special
    token, so non-special "glue" tokens between them (typically the ``\\n``
    connecting ``<|im_end|>`` and ``<|im_start|>``) get ablated too.
    Useful for single-turn prompts; for multi-turn prompts the span may
    cover assistant content between turns — use with care.

    Combined mode:
      * ``"all_special"``: union of ``"eoi"`` + ``"eoi_query"`` +
        ``"text_special"`` (always span) + ``"text_special_query"``
        (block each text-special position as a query from attending to
        image-span keys). For ablation positions A = {EOI} ∪ span(ts):
        downstream queries cannot read any p ∈ A, and queries at p ∈ A
        cannot read the image span (excluding p itself when p == EOI).
        Implies ``text_special_span=True``; requires ``text_special_ids``.
    """
    if ablation_mode not in _ABLATION_MODES:
        raise ValueError(
            f"ablation_mode must be one of {sorted(_ABLATION_MODES)}, "
            f"got {ablation_mode!r}."
        )

    needs_ts = ablation_mode in ("text_special", "text_special_restore", "all_special")
    if needs_ts and not text_special_ids:
        raise ValueError(
            f"ablation_mode={ablation_mode!r} requires non-empty "
            "text_special_ids (post-EOI chat-structure token ids like "
            "<|im_end|>, <|im_start|>)."
        )

    if ablation_mode == "layer":
        apply_layer_skip(model, layers_to_skip, model_type="internvl")
        return

    # `all_special` always treats text-special positions as a span — the user
    # flag is overridden when this mode is selected.
    if ablation_mode == "all_special":
        text_special_span = True

    # Stash on the inner InternLM2Model so ivl_generate can read it during
    # prefill to compute per-sample text_special_positions.
    model.language_model.model._text_special_ids = list(text_special_ids or [])
    model.language_model.model._text_special_span = bool(text_special_span)

    layers_to_skip = list(layers_to_skip or [])

    # InternLM2 uses `config.attn_implementation` (no leading underscore).
    inner_cfg = model.language_model.model.config
    attn_impl = getattr(
        inner_cfg,
        "attn_implementation",
        getattr(inner_cfg, "_attn_implementation", None),
    )
    if attn_impl != "eager":
        raise ValueError(
            f"patch_internvl requires attn_implementation='eager' on the "
            f"InternLM2 language model for ablation_mode={ablation_mode!r}, "
            f"got {attn_impl!r}. Reload the model with "
            "attn_implementation='eager'."
        )

    # Patch the INNER InternLM2Model (inside InternLM2ForCausalLM). The outer
    # InternLM2ForCausalLM.forward is unchanged, so its lm_head (self.output)
    # still runs after decoding. ivl_generate replaces the top-level
    # InternVLChatModel.generate so image_limits is set before each call.
    model.language_model.model.__class__.forward = partialmethod(
        ilm_forward,
        layers_to_skip=layers_to_skip,
        ablation_mode=ablation_mode,
    )
    model.__class__.generate = ivl_generate
