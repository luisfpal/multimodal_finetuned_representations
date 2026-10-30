"""Provides the extract_activations class, which registers forward hooks on target layers of a model and collects hidden-state activations over a dataloader, with optional last-token selection and per-token-class pooling."""

import time
from collections import defaultdict

import torch


class extract_activations:
    def __init__(
        self,
        model,
        dataloader,
        target_layers,
        embdims,
        dtypes,
        nsamples=None,
        use_last_token=False,
        print_every=100,
        use_gpu=False,
        only_output=False,
        forward_fn=None,
        special_token_ids=None,
        extract_pools=None,
    ):
        self.model = model

        # embedding size
        self.embdims = embdims

        # number of samples to collect
        self.nsamples = len(dataloader.dataset)
        if nsamples is not None:
            self.nsamples = nsamples

        # whether to compute the id on the last token /class_token
        self.use_last_token = use_last_token
        self.print_every = print_every

        self.tot_tokens = 0
        self.global_batch_size = dataloader.batch_size

        self.nbatches = len(dataloader)

        self.use_gpu = use_gpu
        self.only_output = only_output
        # Optional override for the model forward call. VLMs whose outer
        # wrapper expects pixel_values (InternVL2, Cambrian) need to route
        # text-only batches through model.language_model / model.model so
        # the hooked decoder layers still fire without the multimodal path.
        self.forward_fn = forward_fn

        # Per-token-class pooling. When `extract_pools` is non-empty, the
        # legacy single-pool storage is skipped and per-pool storages are
        # allocated for the requested names. Pool names:
        #   - "text_avg":     mean over real, non-special positions.
        #   - "text_special": mean over real, special positions (chat scaffold).
        #   - "last_text":    rightmost real, non-special position.
        # Specials are identified by membership in `special_token_ids`
        # (typically tokenizer.all_special_ids). For families whose chat
        # template uses no special tokens (Vicuna, LLaVA-1.5-hf), the special
        # pool is empty everywhere → spec_avg is a 0-vector and last_text
        # collapses back to the last real position.
        self.extract_pools = set(extract_pools) if extract_pools else set()
        if special_token_ids is not None and len(special_token_ids) > 0:
            self._special_ids_tensor = torch.tensor(
                list(special_token_ids), dtype=torch.long
            )
        else:
            self._special_ids_tensor = None

        if only_output:
            assert len(self.embdims) == 1
            key = list(self.embdims.keys())[0]
            self.hidden_states = torch.zeros(
                (self.nsamples, self.embdims[key]), dtype=dtypes[key]
            )
        else:
            self.init_hidden_states(target_layers, dtypes=dtypes)
            self.init_hooks(target_layers, use_gpu=use_gpu)

    def init_hidden_states(self, target_layers, dtypes):
        # dict containing the representations extracted in a sigle forward pass
        self.hidden_states_tmp = defaultdict(lambda: None)

        # Per-pool storages. When extract_pools is set, the legacy
        # `hidden_states` is left unallocated to save memory; each pool dict
        # mirrors the per-layer (nsamples, embdim) layout instead.
        if self.extract_pools:
            self.hidden_states = None
            self.hidden_states_text_avg = (
                {
                    name: torch.zeros(
                        (self.nsamples, self.embdims[name]), dtype=dtypes[name]
                    )
                    for name in target_layers
                }
                if "text_avg" in self.extract_pools
                else None
            )
            self.hidden_states_text_special = (
                {
                    name: torch.zeros(
                        (self.nsamples, self.embdims[name]), dtype=dtypes[name]
                    )
                    for name in target_layers
                }
                if "text_special" in self.extract_pools
                else None
            )
            self.hidden_states_last_text = (
                {
                    name: torch.zeros(
                        (self.nsamples, self.embdims[name]), dtype=dtypes[name]
                    )
                    for name in target_layers
                }
                if "last_text" in self.extract_pools
                else None
            )
        else:
            self.hidden_states = {}
            for name in target_layers:
                self.hidden_states[name] = torch.zeros(
                    (self.nsamples, self.embdims[name]), dtype=dtypes[name]
                )
            self.hidden_states_text_avg = None
            self.hidden_states_text_special = None
            self.hidden_states_last_text = None

    def init_hooks(self, target_layers, use_gpu=False):

        for name, module in self.model.named_modules():
            if name in target_layers:
                if use_gpu:
                    module.register_forward_hook(
                        self._get_hook_gpu(name, self.hidden_states_tmp)
                    )
                else:
                    module.register_forward_hook(
                        self._get_hook(name, self.hidden_states_tmp)
                    )

    def _get_hook(self, name, hidden_states):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, dict):  # and "sentence_embedding" in output:
                # Sentence transformers specific. pooling dense and normalize should be here.
                output = output["sentence_embedding"]
            hidden_states[name] = output.cpu()

        return hook_fn

    def _get_hook_gpu(self, name, hidden_states):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, dict):  # and "sentence_embedding" in output:
                # Sentence transformers specific. pooling dense and normalize should be here.
                output = output["sentence_embedding"]
            hidden_states[name] = output

        return hook_fn

    def _update_activation_dict(self, mask, is_last_batch, is_left_padding=False):

        if not self.use_gpu:
            mask = mask.to("cpu")

        last_token_index = torch.sum(mask, dim=1) - 1
        if is_left_padding:
            last_token_index = -torch.ones(
                mask.shape[0], device=mask.device, dtype=torch.int64
            )
        mask = mask.unsqueeze(-1)

        for i, (layer, activations) in enumerate(self.hidden_states_tmp.items()):
            if len(activations.shape) < 3:
                # add sequence dimension (tacitly assume pooling)
                activations = activations.unsqueeze(1)
                # expand sequence dimension to match mask seq_len
                activations = activations.repeat(1, mask.shape[1], 1)

            if self.use_last_token:
                batch_size = mask.shape[0]
                act_tmp = activations[torch.arange(batch_size), last_token_index]
            else:
                denom = torch.sum(mask, dim=1)
                act_tmp = torch.sum(activations * mask, dim=1) / denom

            num_current_tokens = act_tmp.shape[0]

            if self.use_gpu:
                act_tmp = act_tmp.to("cpu")

            if is_last_batch:
                self.hidden_states[layer][self.tot_tokens :] = act_tmp
            else:
                self.hidden_states[layer][
                    self.tot_tokens : self.tot_tokens + num_current_tokens
                ] = act_tmp

        self.tot_tokens += num_current_tokens

    def _store_pool(self, storage, act_tmp, num_current, is_last_batch):
        if is_last_batch:
            storage[self.tot_tokens :] = act_tmp
        else:
            storage[self.tot_tokens : self.tot_tokens + num_current] = act_tmp

    def _update_pool_dicts(self, input_ids, mask, is_last_batch):
        # Hooks store activations on CPU when use_gpu=False. Keep the masks
        # on the same device as the activations to avoid per-layer transfers.
        if not self.use_gpu:
            mask = mask.to("cpu")
            input_ids = input_ids.to("cpu")

        # special_mask: 1 where token id is in special_token_ids. For families
        # whose chat template renders entirely as plain tokens (Vicuna, LLaVA-
        # 1.5-hf with default templates), this is all zeros and the pools
        # collapse to "all real tokens" (text_avg), "none" (text_special), and
        # "last real token" (last_text).
        if self._special_ids_tensor is not None:
            special_mask = torch.isin(input_ids, self._special_ids_tensor).to(mask.dtype)
        else:
            special_mask = torch.zeros_like(mask)

        text_mask = mask * (1 - special_mask)         # real, non-special
        special_real_mask = mask * special_mask       # real, special

        B, L = mask.shape
        positions = (
            torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, L)
        )
        # Rightmost text-token position. (positions * text_mask) zeros out
        # non-text positions, so .max(dim=1) returns the largest position
        # where text_mask==1. If a sample has no text tokens (degenerate),
        # this returns 0 — a pad position under left-padding.
        last_text_idx = (positions * text_mask).max(dim=1).values

        text_denom = text_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        special_denom = special_real_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

        num_current = mask.shape[0]

        for layer, activations in self.hidden_states_tmp.items():
            if len(activations.shape) < 3:
                activations = activations.unsqueeze(1).repeat(1, L, 1)

            if self.hidden_states_text_avg is not None:
                m = text_mask.unsqueeze(-1).to(activations.dtype)
                avg = (activations * m).sum(dim=1) / text_denom.to(activations.dtype)
                self._store_pool(
                    self.hidden_states_text_avg[layer], avg, num_current, is_last_batch
                )

            if self.hidden_states_text_special is not None:
                m = special_real_mask.unsqueeze(-1).to(activations.dtype)
                avg = (activations * m).sum(dim=1) / special_denom.to(activations.dtype)
                self._store_pool(
                    self.hidden_states_text_special[layer], avg, num_current, is_last_batch
                )

            if self.hidden_states_last_text is not None:
                last = activations[torch.arange(B), last_text_idx]
                self._store_pool(
                    self.hidden_states_last_text[layer], last, num_current, is_last_batch
                )

        self.tot_tokens += num_current

    @torch.no_grad()
    def extract(self, dataloader):
        start = time.time()
        is_last_batch = False
        for i, batch in enumerate(dataloader):

            is_last_batch = (i + 1) == self.nbatches
            batch = {k: v.to("cuda") for k, v in batch.items()}

            if self.forward_fn is not None:
                output = self.forward_fn(batch)
            else:
                output = self.model(**batch)

            mask = batch["attention_mask"]
            is_left_padding = mask[:, -1].sum() == mask.shape[0]

            if self.extract_pools:
                self._update_pool_dicts(
                    input_ids=batch["input_ids"],
                    mask=mask,
                    is_last_batch=is_last_batch,
                )
            else:
                self._update_activation_dict(
                    mask, is_last_batch, is_left_padding=is_left_padding
                )

            if (i + 1) % (self.print_every // self.global_batch_size) == 0:
                end = time.time()
                print(
                    f"{(i+1)*self.global_batch_size/1000}k data, \
                    batch {i+1}/{self.nbatches}, \
                    tot_time: {(end-start)/60: .3f}min, "
                )


