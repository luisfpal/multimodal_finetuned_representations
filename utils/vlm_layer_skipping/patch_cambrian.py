"""Monkey-patches Cambrian VLM classes with layer-skipping-aware generate, forward, and multimodal input preparation."""

from typing import List, Optional, Tuple, Union

from transformers import LlamaForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import BaseModelOutputWithPast

from utils.cambrian_bridge import _ensure_cambrian_importable

_ensure_cambrian_importable()

from functools import partialmethod

import torch
import torch.nn.functional as F
from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from cambrian.model.cambrian_arch import unpad_image
from cambrian.utils import IS_XLA_AVAILABLE
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.utils import logging

from utils.vlm_layer_skipping.layer_skipping import apply_layer_skip

logger = logging.get_logger(__name__)


@torch.no_grad()
def generate_cambrian(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        (
            inputs,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
            vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list,
            final_vision_feature_size,
            global_context_feature,
        ) = self.prepare_inputs_labels_for_multimodal(
            inputs,
            position_ids,
            attention_mask,
            None,
            None,
            images,
            image_sizes=image_sizes,
        )
        self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
        self.vision_tower_aux_attention_masks_list = (
            vision_tower_aux_attention_masks_list
        )
        self.final_vision_feature_size = final_vision_feature_size
        self.global_context_feature = global_context_feature
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)
    # Clear any state from a prior generate() call so samples don't leak state.
    self.model.image_limits = None
    self.model.text_special_positions = None

    if images is not None:
        inputs = torch.stack(inputs)
        if (inputs == IMAGE_TOKEN_INDEX).any():
            image_limits = []
            seq_len = inputs.shape[1]
            for i in range(inputs.shape[0]):
                indices = torch.where(inputs[i] == IMAGE_TOKEN_INDEX)[0]
                if indices.numel() > 0:
                    # After prepare_inputs_labels_for_multimodal, IMAGE_TOKEN_INDEX
                    # spans the whole expanded image block (image patches plus
                    # per-row image_newline rows are all tagged with
                    # IMAGE_TOKEN_INDEX). Expand by ±1 to also cover the
                    # <im_start>/<im_end> delimiters that wrap the image in the
                    # chat template.
                    start = max(0, indices[0].item() - 1)
                    end = min(seq_len - 1, indices[-1].item() + 1)
                    image_limits.append((start, end))
                else:
                    image_limits.append(None)
            self.model.image_limits = image_limits

            # Post-EOI text-special positions (chat-structure tokens like
            # <|eot_id|>, <|start_header_id|>, <|end_header_id|> in Llama-3's
            # chat template). The post-expansion `inputs` has real tokenizer
            # ids outside the image span (filled with -200), so we can scan
            # for special-id matches with a position > image_limits.end filter.
            text_special_ids = getattr(self.model, "_text_special_ids", None)
            if text_special_ids:
                span = bool(getattr(self.model, "_text_special_span", False))
                text_special_positions = []
                pos_idx = torch.arange(seq_len, device=inputs.device)
                for i in range(inputs.shape[0]):
                    ts_mask = torch.zeros(
                        seq_len, dtype=torch.bool, device=inputs.device
                    )
                    for tid in text_special_ids:
                        ts_mask |= inputs[i] == tid
                    if image_limits[i] is not None:
                        _, end = image_limits[i]
                        ts_mask &= pos_idx > end
                    positions = torch.where(ts_mask)[0].tolist()
                    # Span mode: extend to the contiguous range from first to
                    # last text-special position so glue tokens between them
                    # (e.g. "assistant" between <|start_header_id|> and
                    # <|end_header_id|>) get ablated too.
                    if span and positions:
                        positions = list(range(positions[0], positions[-1] + 1))
                    text_special_positions.append(positions or None)
                self.model.text_special_positions = text_special_positions
    return LlamaForCausalLM.generate(
        self,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )


def llama_forward(
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
    vision_tower_aux_feature_list: Optional[List[torch.FloatTensor]] = None,
    vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
    final_vision_feature_size: Optional[List[tuple]] = None,
    global_context_feature: Optional[torch.Tensor] = None,
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

    # Retrieve input_ids and inputs_embeds.
    #
    # The outer CambrianLlamaForCausalLM.forward (which we don't patch) calls
    # self.model(input_ids=..., inputs_embeds=..., ...) after multimodal prep
    # — it passes BOTH because the patched cmm_prepare_inputs_labels_for_multimodal
    # preserves input_ids (rather than returning None like the original
    # cambrian_arch one does) for generate_cambrian's image_limits /
    # text_special_positions bookkeeping. On modern transformers' Llama
    # decoder this combination is forbidden, so when both arrive we drop
    # input_ids — inputs_embeds already contains the expanded image+text
    # embeddings and is what the rest of llama_forward operates on.
    if input_ids is not None and inputs_embeds is not None:
        input_ids = None
    if inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

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
        inputs_embeds = self.embed_tokens(input_ids)

    # The ablation edits a 4D additive mask in place. flash_attention_2 uses a
    # 2D padding mask and sdpa may collapse to is_causal, neither of which
    # would reflect our edits. Require eager only when image ablation is
    # actually active — text-only paths can run under any attention impl.
    attn_impl = getattr(self.config, "_attn_implementation", None)
    if layers_to_skip and attn_impl != "eager":
        raise RuntimeError(
            f"Cambrian ablation requires attn_implementation='eager', got "
            f"{attn_impl!r}. Reload the model with attn_implementation='eager'."
        )

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length + past_key_values_length),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )
    if causal_mask is None:
        raise RuntimeError(
            "_prepare_4d_causal_attention_mask returned None; the Cambrian "
            "ablation patch requires a 4D additive mask."
        )

    # Build the ablation mask: clone the causal mask and zero-out the chosen
    # key/query slice. image_limits was set on self (CambrianLlamaModel) by
    # generate_cambrian and covers image patches + image_newline rows via
    # ±1 expansion. text_special_positions, also stashed by generate_cambrian,
    # holds per-sample post-EOI chat-structure token positions (Llama-3's
    # <|eot_id|>, <|start_header_id|>, <|end_header_id|>).
    image_limits = getattr(self, "image_limits", None)
    text_special_positions = getattr(self, "text_special_positions", None)
    image_mode = ablation_mode in ("image", "cross_modal")
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
            if q_len > 1:
                # Prefill: block post-image-text queries from image key cols.
                image_mask[b, :, end + 1 :, start : end + 1] = masked_value
            else:
                # Decode: single query sits after the image block.
                image_mask[b, :, :, start : end + 1] = masked_value
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
        # Cambrian has no real EOI marker, so A = span(text-special) only.
        # For each p in A:
        #   (i)  block queries q > p from key [p]                — downstream-block
        #   (ii) block query [p] from image-span keys [start, end] — upstream-block
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

    # embed positions
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    cache_position = position_ids[0] if position_ids is not None else None
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    is_prefill = inputs_embeds.shape[1] > 1
    stored_embeds = None

    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_mask = causal_mask
        # Restore modes: at the end of the skipped block, write the stored
        # pre-block hidden states back at the ablation positions.
        #   "image"               -> restore image span [start, end]
        #   "text_special_restore"-> restore each text-special position
        does_restore = ablation_mode in ("image", "text_special_restore")
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
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        if not self.config.connector_only:
            cross_layers_start_idx = self.config.start_of_vision_sampler_layers
            cross_index_step = self.config.stride_of_vision_sampler_layers
            cross_layers_start_idx_list = [
                cross_layers_start_idx + cross_index * cross_index_step
                for cross_index in range(len(self.vision_sampler_layers))
            ]
            if (
                vision_tower_aux_feature_list is not None
                and i in cross_layers_start_idx_list
                and i not in layers_to_skip
            ):
                latent_query_start_idx = self.config.image_position

                if IS_XLA_AVAILABLE:
                    image_token_len_per_side = int(self.config.image_token_len**0.5)
                    latent_query_newline_num = (
                        self.config.image_token_len + image_token_len_per_side
                    )
                    latent_query_num = self.config.image_token_len
                    latent_query_with_newline = hidden_states[
                        :,
                        latent_query_start_idx : latent_query_start_idx
                        + latent_query_newline_num,
                        :,
                    ].clone()
                    bs = latent_query_with_newline.shape[0]
                    latent_query_with_newline = latent_query_with_newline.view(
                        bs, image_token_len_per_side, image_token_len_per_side + 1, -1
                    )
                    latent_query = latent_query_with_newline[:, :, :-1, :]
                    newline_embd = latent_query_with_newline[:, :, -1:, :]
                    vision_tower_aux_feature_list = [
                        vision_tower_aux_feature.to(latent_query.dtype)
                        for vision_tower_aux_feature in vision_tower_aux_feature_list
                    ]
                    bs = latent_query.shape[0]
                    latent_query = latent_query.view(bs * latent_query_num, 1, -1)
                    if self.gradient_checkpointing and self.training:
                        latent_query = self._gradient_checkpointing_func(
                            self.vision_sampler_layers[
                                (i - cross_layers_start_idx) // cross_index_step
                            ].__call__,
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list,
                        )
                    else:
                        latent_query = self.vision_sampler_layers[
                            (i - cross_layers_start_idx) // cross_index_step
                        ](
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list,
                        )
                    # latent_query = latent_query.view(bs, self.latent_query_num, -1)
                    latent_query = latent_query.view(
                        bs, image_token_len_per_side, image_token_len_per_side, -1
                    )
                    latent_query_with_newline = torch.cat(
                        [latent_query, newline_embd], 2
                    ).flatten(1, 2)
                    hidden_states[
                        :,
                        latent_query_start_idx : latent_query_start_idx
                        + latent_query_newline_num,
                    ] = latent_query_with_newline[:, :, :]
                else:
                    bs = len(final_vision_feature_size)
                    latent_query_num_list = []
                    newline_embd_list = []
                    latent_query_list = []
                    for batch_i in range(bs):
                        cur_h, cur_w = final_vision_feature_size[batch_i]

                        cur_latent_query_num = cur_h * cur_w
                        cur_latent_query_newline_num = cur_h * (cur_w + 1)
                        cur_latent_query_with_newline = hidden_states[
                            batch_i : batch_i + 1,
                            latent_query_start_idx : latent_query_start_idx
                            + cur_latent_query_newline_num,
                            :,
                        ].clone()

                        cur_latent_query_with_newline = (
                            cur_latent_query_with_newline.view(1, cur_h, cur_w + 1, -1)
                        )
                        cur_latent_query = cur_latent_query_with_newline[:, :, :-1, :]
                        cur_newline_embd = cur_latent_query_with_newline[:, :, -1:, :]

                        latent_query_num_list.append(cur_latent_query_num)
                        latent_query_list.append(
                            cur_latent_query.contiguous().view(
                                cur_latent_query_num, 1, -1
                            )
                        )
                        newline_embd_list.append(cur_newline_embd)

                    latent_query = torch.cat(latent_query_list, 0)
                    if self.gradient_checkpointing and self.training:
                        latent_query = self._gradient_checkpointing_func(
                            self.vision_sampler_layers[
                                (i - cross_layers_start_idx) // cross_index_step
                            ].__call__,
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list,
                        )
                    else:
                        latent_query = self.vision_sampler_layers[
                            (i - cross_layers_start_idx) // cross_index_step
                        ](
                            latent_query,
                            global_context_feature,
                            *vision_tower_aux_feature_list,
                            *vision_tower_aux_attention_masks_list,
                        )

                    latent_query = torch.split(latent_query, latent_query_num_list, 0)
                    for batch_i in range(bs):
                        cur_h, cur_w = final_vision_feature_size[batch_i]
                        cur_latent_query = latent_query[batch_i]
                        cur_newline_embd = newline_embd_list[batch_i]
                        cur_latent_query_newline_num = cur_h * (cur_w + 1)
                        cur_latent_query = cur_latent_query.view(1, cur_h, cur_w, -1)
                        cur_latent_query_with_newline = torch.cat(
                            [cur_latent_query, cur_newline_embd], 2
                        ).flatten(1, 2)
                        hidden_states[
                            batch_i : batch_i + 1,
                            latent_query_start_idx : latent_query_start_idx
                            + cur_latent_query_newline_num,
                        ] = cur_latent_query_with_newline[:, :, :]
        if use_cache and isinstance(layer_outputs, tuple):
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions and isinstance(layer_outputs, tuple):
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
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


def cmm_prepare_inputs_labels_for_multimodal(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    image_aux_attention_masks_list=None,
    image_sizes=None,
):
    # vision_tower = self.get_vision_tower()
    vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
    if vision_tower_aux_list is None or images is None or input_ids.shape[1] == 1:
        return (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            None,
            labels,
            None,
            None,
            None,
            None,
        )

    image_aux_list = images

    bs = image_aux_list[0].shape[0]
    dtype = image_aux_list[0].dtype

    image_token_len = self.get_model().config.image_token_len
    query_num_list = self.get_model().config.query_num_list

    final_height = final_width = int(image_token_len**0.5)

    final_image_features_list = []

    # only needed for sva
    vision_tower_aux_feature_list_final = None
    vision_tower_aux_attention_masks_list_final = None
    global_context_feature_final = None

    image_aux_features_list = self.encode_images(image_aux_list)

    if self.get_model().config.mm_projector_type == "sva":
        vision_tower_aux_feature_list = []
        vision_tower_aux_attention_masks_list = []
        # get vision tokens from each vision tower
        for aux_i in range(len(vision_tower_aux_list)):
            image_aux_features = image_aux_features_list[aux_i]

            image_aux_features = getattr(
                self.get_model(), "mm_projector_aux_{}".format(aux_i)
            )(image_aux_features).to(dtype)
            if aux_i == 0:
                global_context_feature = image_aux_features.mean(1).view(bs, 1, 1, -1)

            vision_tower_aux_feature_list.append(image_aux_features)

        # perform vision sampling for each query group
        for query_group_i, query_num in enumerate(query_num_list):
            query_features_i = (
                self.get_model()
                .vision_query[query_group_i, :]
                .view(1, 1, 1, -1)
                .expand(bs, query_num, -1, -1)
            )
            global_context_feature_i = global_context_feature.expand(
                -1, query_num, 1, -1
            ).flatten(0, 1)
            query_side_len = int(query_num**0.5)
            if IS_XLA_AVAILABLE:
                (
                    vision_tower_aux_feature_list_i,
                    vision_tower_aux_attention_masks_list_i,
                ) = self.rearrange_vision_tower_features_train(
                    vision_tower_aux_feature_list,
                    image_aux_attention_masks_list,
                    query_side_len,
                )
            else:
                (
                    vision_tower_aux_feature_list_i,
                    vision_tower_aux_attention_masks_list_i,
                ) = self.rearrange_vision_tower_features_inference(
                    vision_tower_aux_feature_list, query_side_len, image_sizes
                )

            query_features_i = getattr(
                self.get_model(), "vision_sampler_{}".format(query_group_i)
            )(
                query_features_i.flatten(0, 1),
                global_context_feature_i,
                *vision_tower_aux_feature_list_i,
                *vision_tower_aux_attention_masks_list_i,
            )
            query_features_i = query_features_i.view(bs, query_num, -1)
            # interpolate to the final target size
            if query_side_len != final_height:
                query_features_i = (
                    query_features_i.permute(0, 2, 1)
                    .contiguous()
                    .view(bs, -1, query_side_len, query_side_len)
                )
                query_features_i = F.interpolate(
                    query_features_i.float(),
                    size=(final_height, final_width),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=query_features_i.dtype)
                query_features_i = (
                    query_features_i.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                )
            final_image_features_list.append(query_features_i)

        if IS_XLA_AVAILABLE:
            (
                vision_tower_aux_feature_list_final,
                vision_tower_aux_attention_masks_list_final,
            ) = self.rearrange_vision_tower_features_train(
                vision_tower_aux_feature_list,
                image_aux_attention_masks_list,
                final_height,
            )
            global_context_feature_final = global_context_feature.expand(
                -1, final_height * final_width, 1, -1
            ).flatten(0, 1)
    else:
        final_image_features_list = image_aux_features_list

    image_features = torch.cat(final_image_features_list, -1)
    image_features = self.get_model().mm_projector(image_features).to(dtype)

    if IS_XLA_AVAILABLE:
        image_features = image_features.view(
            image_features.shape[0], final_height, final_width, -1
        )
        image_features = torch.cat(
            (
                image_features,
                self.model.image_newline[None, None, None, :].expand(
                    image_features.shape[0], final_height, 1, -1
                ),
            ),
            dim=2,
        )
        image_features = image_features.flatten(1, 2)
        final_size = [(final_height, final_width)] * bs

    else:
        image_features = image_features.view(bs, final_height, final_width, -1)
        image_features_unpadded = []
        final_size = []
        if self.get_model().config.mm_projector_type == "sva":
            (
                vision_tower_aux_feature_list_final,
                vision_tower_aux_attention_masks_list_final,
            ) = self.rearrange_vision_tower_features_inference(
                vision_tower_aux_feature_list, final_height, image_sizes, unpad=True
            )
            global_context_feature_final = []
        for batch_i in range(bs):
            cur_image_feature = image_features[batch_i]
            image_size = image_sizes[batch_i]

            cur_image_feature = unpad_image(cur_image_feature.unsqueeze(0), image_size)

            cur_h, cur_w = cur_image_feature.shape[1:3]
            final_size.append((cur_h, cur_w))
            cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
            cur_image_feature = torch.cat(
                (
                    cur_image_feature,
                    self.model.image_newline.view(1, 1, 1, -1)
                    .expand(1, cur_h, 1, -1)
                    .to(cur_image_feature.device),
                ),
                dim=2,
            )
            cur_image_feature = cur_image_feature.flatten(1, 2)
            image_features_unpadded.append(cur_image_feature.squeeze(0))

            if self.get_model().config.mm_projector_type == "sva":
                cur_global_context_feature = global_context_feature[batch_i].expand(
                    cur_h * cur_w, 1, -1
                )
                global_context_feature_final.append(cur_global_context_feature)
        if self.get_model().config.mm_projector_type == "sva":
            global_context_feature_final = torch.cat(global_context_feature_final, 0)

        image_features = image_features_unpadded

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
        self.config, "mm_use_im_start_end", False
    ):
        raise NotImplementedError

    if IS_XLA_AVAILABLE:
        # embed the input_ids
        new_input_ids_padded_for_emb = torch.where(
            input_ids == IMAGE_TOKEN_INDEX, 0, input_ids
        )
        input_embeds = self.get_model().embed_tokens(new_input_ids_padded_for_emb)
        new_input_embeds = []
        cur_image_idx = 0
        # insert the image embeddings
        for batch_idx, (cur_input_embeds, cur_input_ids) in enumerate(
            zip(input_embeds, input_ids)
        ):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_idx += 1
                new_input_embeds.append(cur_input_embeds)
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )

            cur_input_embeds_im_replaced = []

            prev_image_length = 0
            for i in range(len(image_token_indices) - 1):
                # skip the image tokens (1 indicator + (image_length-1) paddings)
                cur_input_embeds_im_replaced.append(
                    cur_input_embeds[
                        image_token_indices[i]
                        + 1
                        + prev_image_length : image_token_indices[i + 1]
                    ]
                )
                if i < len(image_token_indices) - 2:
                    cur_image_features = image_features[cur_image_idx]
                    prev_image_length = len(cur_image_features) - 1
                    cur_image_idx += 1
                    cur_input_embeds_im_replaced.append(cur_image_features)

            cur_input_embeds_im_replaced = [
                x.to(self.device) for x in cur_input_embeds_im_replaced
            ]
            new_input_embeds.append(torch.cat(cur_input_embeds_im_replaced))

        new_input_embeds = torch.stack(new_input_embeds)
        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            labels,
            vision_tower_aux_feature_list_final,
            vision_tower_aux_attention_masks_list_final,
            final_size,
            global_context_feature_final,
        )

    else:
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
            # assume num_images == 1
            assert (
                num_images == 1
            ), "Only num_images == 1 is supported in the current implementation"
            input_ids[batch_idx] = torch.cat(
                [
                    input_ids[batch_idx][: len(cur_new_input_embeds[0])],
                    torch.tensor(
                        [IMAGE_TOKEN_INDEX] * len(cur_new_input_embeds[1]),
                        device=input_ids[batch_idx].device,
                    ),
                    input_ids[batch_idx][-len(cur_new_input_embeds[-1]) :],
                ]
            )
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            input_ids = [x[:tokenizer_model_max_length] for x in input_ids]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        assert (
            batch_size == 1
        ), "Batch size > 1 is not supported for layer skipping on Cambrian"

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            vision_tower_aux_feature_list_final,
            vision_tower_aux_attention_masks_list_final,
            final_size,
            global_context_feature_final,
        )


_ABLATION_MODES = {
    "image",
    "cross_modal",
    "text_special",
    "text_special_restore",
    "all_special",
    "layer",
}


def install_compat(model):
    """Install the class-level method replacements that make Cambrian work
    with modern transformers. Required on EVERY code path (including the
    no-skip baseline) because the vendored ``CambrianLlamaModel.forward``
    is incompatible with transformers >= 4.45:

      * It calls ``decoder_layer(...)`` without the ``position_embeddings``
        kwarg (modern ``LlamaDecoderLayer`` requires the precomputed
        ``(cos, sin)`` tuple — not computing it raises
        ``TypeError: cannot unpack non-iterable NoneType``).
      * It indexes the decoder-layer output as a tuple, but modern
        ``LlamaDecoderLayer`` returns a plain Tensor.

    This function ONLY installs the compat replacement (forward,
    prepare_inputs_labels_for_multimodal, generate). Layer-skipping
    ablation is set up separately by :func:`configure_layer_skip` — call
    that AFTER this if you want ablation; call only this if you just want
    the model to run on modern transformers.

    Idempotent: calling it twice has the same effect as calling it once.
    """
    # llama_forward reads layers_to_skip / ablation_mode from kwargs (with
    # default []/None) and reads _text_special_ids / _text_special_span
    # from self. Initialise the inner-model attrs so a no-partialmethod
    # install can run without AttributeError.
    inner = model.model
    if not hasattr(inner, "_text_special_ids"):
        inner._text_special_ids = []
    if not hasattr(inner, "_text_special_span"):
        inner._text_special_span = False

    # Patch targets:
    #   model.__class__  -> CambrianLlamaForCausalLM (outer): replace
    #                       prepare_inputs_labels_for_multimodal + generate.
    #                       lm_head stays intact because we don't touch .forward.
    #   model.model      -> CambrianLlamaModel (inner, no lm_head): patched
    #                       forward. No partialmethod here — kwargs.pop in
    #                       llama_forward defaults to []/None for the
    #                       no-ablation case. configure_layer_skip overrides
    #                       this with a partialmethod when ablation is
    #                       requested.
    model.__class__.prepare_inputs_labels_for_multimodal = (
        cmm_prepare_inputs_labels_for_multimodal
    )
    model.model.__class__.forward = llama_forward
    model.__class__.generate = generate_cambrian


def configure_layer_skip(
    model,
    layers_to_skip,
    ablation_mode="image",
    text_special_ids=None,
    text_special_span=False,
):
    """Configure layer-skipping ablation on a Cambrian model.

    Re-installs ``CambrianLlamaModel.forward`` with the given
    ``layers_to_skip`` / ``ablation_mode`` baked in via ``partialmethod``,
    and (in ``"layer"`` mode) swaps the requested layers for identity
    modules. **Assumes :func:`install_compat` has already been called** —
    callers that want both compat + ablation should call install_compat
    first (or use the legacy :func:`patch` wrapper, which does both).

    Modes shared with :func:`utils.vlm_layer_skipping.patch_llava.patch`:
      * ``"image"``: mask cross-modal cells [start, end] AND restore image-
        span hidden states at the end of the block.
      * ``"cross_modal"``: same mask as ``"image"`` without the restore.
      * ``"layer"``: replace skipped layers with identity modules.

    Cambrian has no real EOI token (the image is a -200 sentinel that
    expands inline), so the eoi / eoi_restore / eoi_query modes are not
    available here.

    Text-special modes (post-image chat-structure tokens — Llama-3's
    ``<|eot_id|>``, ``<|start_header_id|>``, ``<|end_header_id|>``):
      * ``"text_special"``: per text-special position p, block downstream
        queries from attending to key column [p].
      * ``"text_special_restore"``: same mask plus restore the hidden
        states at the text-special positions at the end of the block.
    Both require ``text_special_ids`` to be a non-empty list of token ids
    (typically ``tokenizer.all_special_ids``).

    ``text_special_span`` (opt-in) extends the ablated positions to the
    contiguous range from the first to the last post-image text-special
    token. For Cambrian's Llama-3 prompt this typically captures
    ``<|eot_id|> <|start_header_id|> assistant <|end_header_id|>`` as one
    contiguous block (4 positions) instead of just the 3 special tokens.

    Combined mode:
      * ``"all_special"``: union of ``"text_special"`` (always span) and
        a query-side block that prevents each text-special position from
        attending to the image span. Cambrian has no real EOI, so the
        eoi/eoi_query parts of the InternVL/OneVision ``"all_special"``
        do not apply here. Implies ``text_special_span=True``; requires
        ``text_special_ids``.

    In ``"layer"`` mode the patched forward is still installed (compat),
    but the in-forward mask/restore logic is bypassed; ``apply_layer_skip``
    swaps the requested layers for identity modules.
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
            "text_special_ids (post-image chat-structure token ids like "
            "<|eot_id|>, <|start_header_id|>, <|end_header_id|>)."
        )

    # `all_special` always treats text-special positions as a span.
    if ablation_mode == "all_special":
        text_special_span = True

    layers_to_skip = list(layers_to_skip or [])

    if ablation_mode == "layer":
        apply_layer_skip(model, layers_to_skip, model_type="cambrian")
        # Identity layers handle the skip; the patched forward must not also
        # run mask/restore against the same indices.
        forward_layers_to_skip: list[int] = []
    else:
        forward_layers_to_skip = layers_to_skip

    # Eager attention is only required when image/cross_modal ablation is
    # actually active — the patched forward itself also enforces this at
    # call time.
    if forward_layers_to_skip:
        attn_impl = getattr(model.model.config, "_attn_implementation", None)
        if attn_impl != "eager":
            raise ValueError(
                f"patch_cambrian requires attn_implementation='eager' on the "
                f"language model for ablation_mode={ablation_mode!r}, got "
                f"{attn_impl!r}. Reload the model with "
                "attn_implementation='eager'."
            )

    # Stash on the inner CambrianLlamaModel so generate_cambrian (set on
    # the outer class) and llama_forward (set on the inner class) can read
    # them via self / self.model respectively.
    model.model._text_special_ids = list(text_special_ids or [])
    model.model._text_special_span = bool(text_special_span)

    # Re-install the inner forward with ablation kwargs baked in. Overrides
    # the no-ablation install from install_compat.
    model.model.__class__.forward = partialmethod(
        llama_forward,
        layers_to_skip=forward_layers_to_skip,
        ablation_mode=ablation_mode,
    )


def patch(
    model,
    layers_to_skip,
    ablation_mode="image",
    text_special_ids=None,
    text_special_span=False,
):
    """Backward-compat wrapper: install_compat + configure_layer_skip.

    New callers should prefer the split API:
      * :func:`install_compat` once at model load (always required).
      * :func:`configure_layer_skip` per-iteration when running ablations.
    """
    install_compat(model)
    configure_layer_skip(
        model,
        layers_to_skip,
        ablation_mode=ablation_mode,
        text_special_ids=text_special_ids,
        text_special_span=text_special_span,
    )
