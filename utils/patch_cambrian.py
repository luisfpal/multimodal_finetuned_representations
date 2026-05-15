from typing import Optional, Union, List, Tuple
from transformers.generation.utils import GenerateOutput
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

from utils.cambrian_bridge import _ensure_cambrian_importable

_ensure_cambrian_importable()

from cambrian.utils import IS_XLA_AVAILABLE
from cambrian.model.cambrian_arch import unpad_image
from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
import torch
from functools import partialmethod
from transformers.utils import logging

logger = logging.get_logger(__name__)

def forward_cambrian(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_aux_attention_masks_list,
                image_sizes
            )
        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            #self.model.gradient_checkpointing = False
                
            from torch_xla.utils.checkpoint import checkpoint
            # self.model.gradient_checkpointing = True
            self.model._gradient_checkpointing_func = checkpoint

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_limits = None
        inputs = torch.stack(input_ids)
        if (inputs == IMAGE_TOKEN_INDEX).any():
            image_limits = []
            image_attention_mask = torch.ones_like(inputs, dtype=torch.int)
            for i in range(inputs.shape[0]):
                indices = torch.where(inputs[i] == IMAGE_TOKEN_INDEX)[0]
                if indices.numel() > 0:
                    start = indices[0].item()
                    end = indices[-1].item()
                    image_limits.append((start, end))
                    image_attention_mask[i, start : end + 1] = 0
                else:
                    image_limits.append(None)
            self.model.image_limits = image_limits
            self.model.image_attention_mask = image_attention_mask
        # training
        if IS_XLA_AVAILABLE:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision_tower_aux_feature_list=vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list, 
            final_vision_feature_size=final_vision_feature_size,
            global_context_feature=global_context_feature,
        )

        # inference
        else:
            if hasattr(self, "vision_tower_aux_feature_list"):
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                outputs = self.model(
                    # input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    vision_tower_aux_feature_list=vision_tower_aux_feature_list if inputs_embeds is None else self.vision_tower_aux_feature_list,
                    vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list if inputs_embeds is None else self.vision_tower_aux_attention_masks_list, 
                    final_vision_feature_size=final_vision_feature_size if inputs_embeds is None else self.final_vision_feature_size,
                    global_context_feature=global_context_feature if inputs_embeds is None else self.global_context_feature,
                )
            else:
                outputs = self.model(
                    # input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
    image_limits = None
    inputs = torch.stack(inputs)
    if (inputs == IMAGE_TOKEN_INDEX).any():
        image_limits = []
        image_attention_mask = torch.ones_like(inputs, dtype=torch.int)
        for i in range(inputs.shape[0]):
            indices = torch.where(inputs[i] == IMAGE_TOKEN_INDEX)[0]
            if indices.numel() > 0:
                start = indices[0].item()
                end = indices[-1].item()
                image_limits.append((start, end))
                image_attention_mask[i, start : end + 1] = 0
            else:
                image_limits.append(None)
        self.model.image_limits = image_limits
        self.model.image_attention_mask = image_attention_mask
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
    layers_to_skip = kwargs.get("layers_to_skip", [])
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

    # retrieve input_ids and inputs_embeds
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

    # Detect attention implementation (config-based for newer transformers,
    # attribute-based for older versions)
    attn_impl = getattr(self.config, "_attn_implementation", None)
    use_flash = attn_impl == "flash_attention_2" or getattr(
        self, "_use_flash_attention_2", False
    )
    use_sdpa = (
        not use_flash
        and (attn_impl == "sdpa" or getattr(self, "_use_sdpa", True))
        and not output_attentions
    )

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length + past_key_values_length),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    if self.image_attention_mask.shape[-1] < attention_mask.shape[-1]:
        self.image_attention_mask = torch.cat(
            (
                self.image_attention_mask,
                torch.ones(self.image_attention_mask.shape[:-1] + (1,)).to(
                    self.image_attention_mask.device
                ),
            ),
            dim=-1,
        )
    image_mask = torch.logical_and(self.image_attention_mask, attention_mask)
    if use_flash:
        # Flash attention uses 2D masks: 1=attend, 0=ignore.
        # image_mask already has 0s for image positions — use directly.
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif use_sdpa:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
        image_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            image_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # embed positions
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    cache_position = position_ids[0] if position_ids is not None else None
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    # For 4D masks (SDPA/manual), min_dtype is used to mask out positions
    min_dtype = (
        torch.finfo(image_mask.dtype).min
        if torch.is_floating_point(image_mask)
        else False
    )
    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_mask = attention_mask
        if i in layers_to_skip and self.image_limits is not None:
            if i == layers_to_skip[0]:
                stored_embeds = hidden_states
            if use_flash:
                # Flash attention: 2D image_mask already has 0s for image positions
                layer_mask = image_mask
            else:
                # SDPA/manual: 4D mask needs explicit min_dtype masking
                layer_mask = image_mask.clone()
                for b, limit in enumerate(self.image_limits):
                    if limit is not None:
                        start, end = limit
                        if layer_mask.shape[2] > 1:
                            layer_mask[b, :, end + 1 :, start : end + 1] = min_dtype
                        else:
                            layer_mask[b, :, :, start : end + 1] = min_dtype
        elif (
            len(layers_to_skip) > 0
            and i == (layers_to_skip[-1] + 1)
            and self.image_limits is not None
        ):
            for b, limit in enumerate(self.image_limits):
                if limit is not None:
                    start, end = limit
                    hidden_states[b, start : end + 1, :] = stored_embeds[
                        b, start : end + 1, :
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
            assert num_images == 1, (
                "Only num_images == 1 is supported in the current implementation"
            )
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

        assert batch_size == 1, (
            "Batch size > 1 is not supported for layer skipping on Cambrian"
        )

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


def patch(model, layers_to_skip):
    model.__class__.prepare_inputs_labels_for_multimodal = (
        cmm_prepare_inputs_labels_for_multimodal
    )
    model.model.__class__.forward = partialmethod(
        llama_forward, layers_to_skip=layers_to_skip
    )
    model.__class__.generate = generate_cambrian
    model.__class__.forward = forward_cambrian
