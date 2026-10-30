"""Monkey-patches LLaVA-OneVision 1.5 model forward passes to apply layer skipping."""

from dataclasses import dataclass
from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch

# Compat shim: the 8B OneVision remote code does `from
# transformers.modeling_flash_attention_utils import flash_attn_varlen_func`
# at module load (gated on is_flash_attn_available(), which is True on GPU
# nodes with flash_attn installed). transformers>=4.56 moved that symbol
# into _lazy_imports, so the import fails. We run eager attention and never
# call the function — rebind the symbol so the import just succeeds.
import transformers.modeling_flash_attention_utils as _fa_utils
from transformers.cache_utils import DynamicCache, logger
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.utils import is_torchdynamo_compiling

if not hasattr(_fa_utils, "flash_attn_varlen_func"):
    try:
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
        _fa_utils.flash_attn_varlen_func = _flash_attn_varlen_func
    except ImportError:
        _fa_utils.flash_attn_varlen_func = None

from utils.vlm_layer_skipping.layer_skipping import apply_layer_skip


@dataclass
class LLaVAOneVision1_5_ModelOutputWithPast(ModelOutput):
    """
    Base class for Llava outputs, with hidden states and attentions.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


def onevision_forward(
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
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, LLaVAOneVision1_5_ModelOutputWithPast]:
    r"""
    pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
        The tensors corresponding to the input videos. Pixel values can be obtained using
        [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
        [`Qwen2VLImageProcessor`] for processing videos.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """
    #print("In forward")
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
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
    # Reset stale image_limits / text_special_positions at the start of a
    # fresh generate() call; during decode input_ids no longer contains the
    # image token, so we retrieve the cached values computed on prefill.
    past_len = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    if past_len == 0:
        self.language_model._cached_image_limits = None
        self.language_model._cached_text_special_positions = None

    if input_ids is not None and (input_ids == self.config.image_token_id).any():
        image_limits = []
        seq_len = input_ids.shape[1]
        for i in range(input_ids.shape[0]):
            indices = torch.where(input_ids[i] == self.config.image_token_id)[0]
            if indices.numel() > 0:
                # Expand by ±1 to include <|vision_start|> and <|vision_end|>
                # that wrap the image token block in OneVision. The ablation
                # must treat those delimiters as part of the image region.
                start = max(0, indices[0].item() - 1)
                end = min(seq_len - 1, indices[-1].item() + 1)
                image_limits.append((start, end))
            else:
                image_limits.append(None)
        self.language_model._cached_image_limits = image_limits

        # Post-EOI text-special positions (chat-structure tokens like
        # <|im_end|>, <|im_start|>). Computed only if patch() stashed
        # text_special_ids; mirrors `text_special` pooling in
        # extract_activations.
        text_special_ids = getattr(
            self.language_model, "_text_special_ids", None
        )
        if text_special_ids:
            span = bool(getattr(
                self.language_model, "_text_special_span", False
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
            self.language_model._cached_text_special_positions = (
                text_special_positions
            )
    else:
        image_limits = getattr(self.language_model, "_cached_image_limits", None)
    text_special_positions = getattr(
        self.language_model, "_cached_text_special_positions", None
    )

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        image_limits=image_limits,
        text_special_positions=text_special_positions,
    )

    output = LLaVAOneVision1_5_ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def onevision_llm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # pop so the injected-by-partialmethod kwargs don't leak into
    # decoder_layer(**kwargs), where unknown kwargs may warn or raise.
    image_limits = kwargs.pop("image_limits", None)
    text_special_positions = kwargs.pop("text_special_positions", None)
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

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            3, inputs_embeds.shape[0], -1
        )
    # elif position_ids.dim() == 2: # 这是为了3drope准备的
    #     position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        output_attentions,
    )
    # With attn_implementation="eager", _update_causal_mask returns a 4D float
    # additive mask (0 / -inf). flash_attention_2 / sdpa may return None and
    # our in-place mask edits would silently do nothing or crash. patch()
    # enforces eager at patch time.
    if causal_mask is None:
        raise RuntimeError(
            "_update_causal_mask returned None; the OneVision ablation patch "
            "requires a 4D additive mask. Load the language model with "
            "attn_implementation='eager'."
        )

    # Build the ablation mask: clone the causal mask and zero-out a slice of
    # key columns (and for "eoi_query", a query row) so the chosen attention
    # cells become inaccessible inside the skipped layers. image_limits
    # records `start = <|vision_start|>` and `end = <|vision_end|>` (EOI) via
    # ±1 expansion around <|image_pad|>. text_special_positions lists
    # per-sample post-EOI chat-structure token positions (<|im_end|>, ...).
    #
    # Ablation slices:
    #   image / cross_modal           -> keys [start, end] vs queries (end, ...]
    #   eoi / eoi_restore             -> key [end]         vs queries (end, ...]
    #   eoi_query                     -> keys [start, end) vs query  [end]
    #                                    (prefill only)
    #   text_special / text_special_restore
    #                                 -> per ts position p: key [p] vs queries (p, ...]
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
                # (start..end-1, leaving EOI's self-attention intact). Decode
                # is a no-op: by then EOI's KV entry is already fixed.
                if q_len > 1:
                    image_mask[b, :, end : end + 1, start:end] = masked_value
            else:
                if ablation_mode in ("eoi", "eoi_restore"):
                    k_lo, k_hi = end, end + 1
                else:  # image, cross_modal
                    k_lo, k_hi = start, end + 1
                if q_len > 1:
                    image_mask[b, :, end + 1 :, k_lo:k_hi] = masked_value
                else:
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
            # (ii) EOI query → image content keys [start, end-1].
            if q_len > 1:
                image_mask[b, :, end : end + 1, start:end] = masked_value

            for pos in ts_positions:
                # (i) Downstream-block from text-special key.
                if q_len > 1:
                    image_mask[b, :, pos + 1 :, pos : pos + 1] = masked_value
                else:
                    image_mask[b, :, :, pos : pos + 1] = masked_value
                # (ii) text-special query → full image span [start, end].
                # Decode is a no-op: KV at pos is already fixed from prefill.
                if q_len > 1:
                    image_mask[b, :, pos : pos + 1, start : end + 1] = masked_value

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    is_prefill = inputs_embeds.shape[1] > 1
    stored_embeds = None

    for i, decoder_layer in enumerate(self.layers):
        layer_mask = causal_mask
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Restore modes: at the end of the skipped block, write the stored
        # pre-block hidden states back at the ablation positions.
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
        if i in layers_to_skip:
            if image_mask is not None:
                layer_mask = image_mask
            if (
                i == layers_to_skip[0]
                and does_restore
                and is_prefill
                and restore_source_present
            ):
                stored_embeds = hidden_states.clone()

        elif (
            len(layers_to_skip) > 0
            and i == (layers_to_skip[-1] + 1)
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
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                layer_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
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
    """Apply a layer-level ablation strategy to a OneVision model.

    Modes shared with :func:`utils.vlm_layer_skipping.patch_llava.patch`:
      * ``"image"``: mask cross-modal cells [start, end] AND restore image-
        span hidden states at the end of the block.
      * ``"cross_modal"``: same mask as ``"image"`` without the restore.
      * ``"layer"``: replace skipped layers with identity modules.

    EOI-only modes (InternVL/OneVision; image_limits' ``end`` is the EOI
    position, i.e. ``<|vision_end|>``):
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
    (typically ``tokenizer.all_special_ids`` minus image / EOI / vision-
    start / video-pad markers).

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
        apply_layer_skip(model, layers_to_skip, model_type="onevision")
        return

    # `all_special` always treats text-special positions as a span.
    if ablation_mode == "all_special":
        text_special_span = True

    # Stash on the language_model so onevision_forward (the outer multimodal
    # forward) can read it during prefill to compute text_special_positions.
    model.language_model._text_special_ids = list(text_special_ids or [])
    model.language_model._text_special_span = bool(text_special_span)

    layers_to_skip = list(layers_to_skip or [])

    # Require eager attention so _update_causal_mask returns a 4D float
    # additive mask we can edit to express selective cross-modal ablation.
    attn_impl = getattr(
        model.language_model.config, "_attn_implementation", None
    )
    if attn_impl != "eager":
        raise ValueError(
            f"patch_onevision requires attn_implementation='eager' on the "
            f"language model for ablation_mode={ablation_mode!r}, got "
            f"{attn_impl!r}. Reload the model with attn_implementation='eager'."
        )

    # Hub's remote-code modeling_llavaonevision1_5.py ships a buggy
    # LLaVAOneVision1_5_Attention.__init__ that never assigns self.num_heads,
    # while its eager forward references it (shape check in attn_output). Only
    # eager is affected; sdpa/flash forwards don't touch num_heads. Inject the
    # attribute so the eager path (which this patch requires) doesn't crash.
    num_heads = model.language_model.config.num_attention_heads
    for layer in model.language_model.layers:
        if not hasattr(layer.self_attn, "num_heads"):
            layer.self_attn.num_heads = num_heads

    # Patch the INNER classes:
    #   model.model           -> LLaVAOneVision1_5_Model (base multimodal, no lm_head)
    #   model.language_model  -> LLaVAOneVision1_5_TextModel (decoder backbone)
    # The outer LLaVAOneVision1_5_ForConditionalGeneration.forward is untouched
    # so lm_head still runs after the decoder returns.
    model.language_model.__class__.forward = partialmethod(
        onevision_llm_forward,
        layers_to_skip=layers_to_skip,
        ablation_mode=ablation_mode,
    )
    model.model.__class__.forward = onevision_forward