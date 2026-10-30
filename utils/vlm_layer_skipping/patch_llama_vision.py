"""Monkey-patches a LLaMA vision model's forward pass to skip selected decoder layers."""

from functools import partialmethod
from typing import Optional, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack


def llama_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cross_attention_states: Optional[torch.FloatTensor] = None,
    cross_attention_mask: Optional[torch.Tensor] = None,
    full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    hidden_states = inputs_embeds

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values)

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)



    #########
    # Added
    layers_to_skip = kwargs.get("layers_to_skip", None) or []
    #########


    # decoder layers
    for idx, decoder_layer in enumerate(self.layers):
        # For text-only path we should skip cross attention layers.
        # Let's check if the layer is cross attention layer and if we have cross attention states
        # or cached cross attention states.
        is_cross_attention_layer = idx in self.cross_attention_layers
        is_cross_attention_cache_empty = past_key_values is None or (
            past_key_values is not None and past_key_values.get_seq_length(idx) == 0
        )

        if is_cross_attention_layer and cross_attention_states is None and is_cross_attention_cache_empty:
            continue

        ##############
        # added
        if is_cross_attention_layer and idx in layers_to_skip:
            continue

        ##############

        hidden_states = decoder_layer(
            hidden_states,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            attention_mask=causal_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )
    

def patch(model, layers_to_skip):
    model.model.language_model.__class__.forward = partialmethod(
        llama_forward, layers_to_skip=layers_to_skip
    )






#print(model)
# if ablation_mode == "cross_modal" or ablation_mode == "image":
#     for name, module in model.named_modules():
#         if isinstance(module, MllamaCrossAttentionDecoderLayer):
#             # "model.language_model.layers.23.cross_attn"
#             # print(name)
#             name = name.lstrip("model.language_model.layers.")
#             layer_number = int(name.rstrip(".cross_attn"))
#             # print(layer_number)
#             if layer_number in layers_to_skip:
#                 # print("skipping layer", layer_number)
#                 module.cross_attn_attn_gate = torch.nn.Parameter(
#                     torch.zeros_like(module.cross_attn_attn_gate)
#                 )
#                 module.cross_attn_mlp_gate = torch.nn.Parameter(
#                     torch.zeros_like(module.cross_attn_attn_gate)
#                 )
