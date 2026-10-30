"""Monkey-patches InternVL model classes with custom forward and generate methods."""

import torch
from typing import List, Optional, Tuple, Union
from transformers import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from functools import partialmethod

logger = logging.get_logger(__name__)


def _import_flash_attn():
    global flash_attn_func, flash_attn_varlen_func
    global pad_input, index_first_axis, unpad_input
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis as _index_first_axis
        from flash_attn.bert_padding import pad_input as _pad_input
        from flash_attn.bert_padding import unpad_input as _unpad_input

        flash_attn_func, flash_attn_varlen_func = (
            _flash_attn_func,
            _flash_attn_varlen_func,
        )
        pad_input, index_first_axis, unpad_input = (
            _pad_input,
            _index_first_axis,
            _unpad_input,
        )
    except ImportError:
        raise ImportError("flash_attn is not installed.")


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

    if self.config.attn_implementation == "flash_attention_2":
        _import_flash_attn()

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

    if self.config.attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
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
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
        image_mask = self._prepare_decoder_attention_mask(
            image_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
    min_dtype = (
        torch.finfo(attention_mask.dtype).min
        if torch.is_floating_point(attention_mask)
        else False
    )
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_mask = attention_mask
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        if idx in layers_to_skip and self.image_limits is not None:
            print("skipping layer", idx)
            if idx == layers_to_skip[0]:
                stored_embeds = hidden_states
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
            and idx == (layers_to_skip[-1] + 1)
            and self.image_limits is not None
        ):
            for b, limit in enumerate(self.image_limits):
                if limit is not None:
                    start, end = limit
                    hidden_states[b, start : end + 1, :] = stored_embeds[
                        b, start : end + 1, :
                    ]
        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
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
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        image_limits = None
        if (input_ids == self.img_context_token_id).any():
            image_limits = []
            image_attention_mask = torch.ones_like(input_ids, dtype=torch.int)
            for i in range(input_ids.shape[0]):
                indices = torch.where(input_ids[i] == self.img_context_token_id)[0]
                if indices.numel() > 0:
                    start = indices[0].item() - 1
                    end = indices[-1].item() + 1
                    image_limits.append((start, end))
                    image_attention_mask[i, start : end + 1] = 0
                else:
                    image_limits.append(None)

            self.language_model.model.image_limits = image_limits
            self.language_model.model.image_attention_mask = image_attention_mask
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


def patch(model, layers_to_skip):
    model.language_model.model.__class__.forward = partialmethod(
        ilm_forward, layers_to_skip=layers_to_skip
    )
    model.__class__.generate = ivl_generate
