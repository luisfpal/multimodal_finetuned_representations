"""Provides utilities to extract and save hidden-state activations from HuggingFace vision-language models via forward hooks on named target layers."""

import os
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


def get_embdims(model, dataloader, target_layers, forward_fn=None):
    embdims = defaultdict(lambda: None)
    dtypes = defaultdict(lambda: None)

    def get_hook(name, embdims):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            embdims[name] = output.shape[-1]
            dtypes[name] = output.dtype

        return hook_fn

    handles = {}
    for name, module in model.named_modules():
        if name in target_layers:
            handles[name] = module.register_forward_hook(get_hook(name, embdims))

    batch = next(iter(dataloader))
    # Pass the full batch (not just input_ids) so VLMs that require
    # image-side inputs in forward() — InternVL2 unconditionally does
    # image_flags.squeeze(-1) — get them on the probe pass too. Matches
    # the call shape used in extract.extract() below. Lists are accepted
    # (Cambrian's `images` is a list of per-aux-tower tensors).
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to("cuda")
        elif isinstance(v, list):
            moved[k] = [
                t.to("cuda") if isinstance(t, torch.Tensor) else t for t in v
            ]
    batch = moved
    # userlast_pos is dataset-side metadata for pooling, not a model input.
    batch.pop("userlast_pos", None)
    if forward_fn is not None:
        _ = forward_fn(batch)
    else:
        _ = model(**batch, use_cache=False)

    for name, module in model.named_modules():
        if name in target_layers:
            handles[name].remove()

    assert len(embdims) == len(target_layers)
    return embdims, dtypes


@torch.inference_mode()
def save_representations(
    model: PreTrainedModel,
    dataloader: DataLoader,
    target_layers: dict,
    nsamples,
    dirpath=".",
    filename="",
    use_last_token=False,
    print_every=100,
    collect_predictions=False,
    image_token_id=None,
    eoi_token_id=None,
    pool_info_provider=None,
    special_token_ids=None,
    forward_fn=None,
    track_user_last=False,
    image_newline_token_id=None,
):
    """Extract per-layer hidden states.

    When ``image_token_id`` or ``eoi_token_id`` is provided, additionally
    returns an ``extras`` dict with the corresponding per-sample pooled
    activations:
      - ``image_avg``: mean over positions where input_ids == image_token_id
        (the IMG_CONTEXT positions for InternVL2).
      - ``image_avg_with_newline``: mean over image_token_id positions PLUS
        ``image_newline_token_id`` positions. Allocated only when both ids
        are given (Cambrian-only — its cmm appends one learned image_newline
        embedding per row of image features, and we want both an image-only
        pool and a pool that includes those LM-domain newline embeddings).
      - ``eoi``: hidden state at the </img> position (first match per sample).
      - ``image_eoi_avg``: mean over image positions PLUS the EOI position.
        Allocated only when both ``image_token_id`` and ``eoi_token_id`` are
        set (skipped for LLaVA-1.5 which has no EOI marker).
      - ``last``: hidden state at the last non-pad position (final prompt token).
      - ``text_avg``: mean over post-image non-pad positions excluding image,
        EOI, and any tokenizer-special token ids — i.e. "post-image content".
        The post-image boundary is the EOI position when ``eoi_token_id`` is
        set; otherwise it's the last image-token position (LLaVA-1.5 has no
        EOI marker — the processor expands one ``<image>`` placeholder into
        ``image_seq_length`` copies of the same id).
      - ``text_special``: mean over post-image non-pad positions whose token
        id IS in ``special_token_ids`` (with image/EOI ids removed). Captures
        structural tokens like <|im_end|>, <|eot_id|>, EOS in the post-image
        region. Zero when a sample has no such tokens. Only computed when
        ``special_token_ids`` is provided.
      - ``user_last``: hidden state at the last token of the user content
        (immediately before the assistant role header begins). Allocated when
        ``track_user_last=True`` and the dataloader emits a per-sample
        ``userlast_pos`` field in each batch (text-only mode).

    ``forward_fn`` overrides the default ``model(**batch, use_cache=False)``
    call. Required for VLMs whose wrapper forward demands image inputs
    (InternVL2, Cambrian) when running in text-only mode — pass a closure
    that routes through ``model.language_model`` / ``model.model`` so the
    same hooked decoder layers still fire.

    The return shape becomes ``(hidden_states, extras)`` or
    ``(hidden_states, predicted_token_ids, extras)`` when extras are
    requested. Without those kwargs the original return signature is preserved.
    """
    dirpath = str(dirpath)
    os.makedirs(dirpath, exist_ok=True)

    target_layer_names = list(target_layers.values())
    target_layer_labels = list(target_layers.keys())

    start = time.time()
    print("layer_to_extract: ", target_layer_labels)
    embdims, dtypes = get_embdims(
        model, dataloader, target_layer_names, forward_fn=forward_fn
    )

    extr_act = extract_activations(
        model=model,
        dataloader=dataloader,
        target_layers=target_layer_names,
        embdims=embdims,
        dtypes=dtypes,
        nsamples=nsamples,
        use_last_token=use_last_token,
        print_every=print_every,
        collect_predictions=collect_predictions,
        image_token_id=image_token_id,
        eoi_token_id=eoi_token_id,
        pool_info_provider=pool_info_provider,
        special_token_ids=special_token_ids,
        forward_fn=forward_fn,
        track_user_last=track_user_last,
        image_newline_token_id=image_newline_token_id,
    )

    extr_act.extract(dataloader)
    print(f"num_tokens: {extr_act.tot_tokens/10**3}k")
    print((time.time() - start) / 3600, "hours")

    extract_all = (
        image_token_id is not None or eoi_token_id is not None or track_user_last
    )
    if extract_all:
        extras = {
            "image_avg": extr_act.hidden_states_image_avg,
            "image_avg_with_newline": extr_act.hidden_states_image_avg_with_newline,
            "eoi": extr_act.hidden_states_eoi,
            "image_eoi_avg": extr_act.hidden_states_image_eoi_avg,
            "last": extr_act.hidden_states_last,
            "text_avg": extr_act.hidden_states_text_avg,
            "text_special": extr_act.hidden_states_text_special,
            "user_last": extr_act.hidden_states_user_last,
        }
        if collect_predictions:
            return extr_act.hidden_states, extr_act.predicted_token_ids, extras
        return extr_act.hidden_states, extras

    if collect_predictions:
        return extr_act.hidden_states, extr_act.predicted_token_ids
    return extr_act.hidden_states


class extract_activations:
    def __init__(
        self,
        model,
        dataloader,
        target_layers,
        embdims,
        dtypes,
        nsamples,
        use_last_token=False,
        print_every=100,
        collect_predictions=False,
        image_token_id=None,
        eoi_token_id=None,
        pool_info_provider=None,
        special_token_ids=None,
        forward_fn=None,
        track_user_last=False,
        image_newline_token_id=None,
    ):
        self.model = model
        # embedding size
        self.embdims = embdims
        # number of samples to collect
        self.nsamples = nsamples
        # whether to compute the id on the last token /class_token
        self.use_last_token = use_last_token
        self.print_every = print_every
        self.collect_predictions = collect_predictions
        self.predicted_token_ids: list = []

        # Optional per-sample extras pooled at specific token positions.
        # Storage is allocated only when extras are requested.
        self.image_token_id = image_token_id
        self.eoi_token_id = eoi_token_id
        # Optional second image-region id covering positions that should be
        # excluded from the default _imgavg pool but included in
        # _imgavg_with_newline. Cambrian-only: marks per-row image_newline
        # positions inside the image span.
        self.image_newline_token_id = image_newline_token_id
        # Tokenizer special-token ids excluding image / EOI (used to
        # define what counts as "structural" vs "content" in the post-EOI
        # region for text_avg / text_special pooling).
        self.special_token_ids = (
            list(special_token_ids) if special_token_ids is not None else None
        )
        self.hidden_states_image_avg = None
        self.hidden_states_image_avg_with_newline = None
        self.hidden_states_eoi = None
        self.hidden_states_image_eoi_avg = None
        self.hidden_states_last = None
        self.hidden_states_text_avg = None
        self.hidden_states_text_special = None
        # Last token of the user content, immediately before the assistant
        # role header. Computed dataset-side and supplied as
        # batch["userlast_pos"] so the model-specific chat scaffold is
        # accounted for correctly (the boundary is not derivable from
        # special_token_ids alone — e.g. LLaVA-1.5 renders "ASSISTANT:" as
        # plain text, and chatml templates emit non-special "assistant\n"
        # after <|im_start|>).
        self.track_user_last = track_user_last
        self.hidden_states_user_last = None
        # Optional forward override (e.g. model.language_model for InternVL2
        # text-only); see save_representations docstring.
        self.forward_fn = forward_fn

        # Optional callable(batch) -> {"mask", "input_ids"} that overrides
        # the default attention_mask / input_ids used for pooling. Required
        # for models whose hidden-state length differs from input_ids length
        # (e.g. Cambrian, where the model expands a -200 sentinel into 576
        # image embeddings inside its forward).
        self.pool_info_provider = pool_info_provider

        self.tot_tokens = 0
        self.global_batch_size = dataloader.batch_size
        self.nbatches = len(dataloader)

        self.init_hidden_states(target_layers, dtypes=dtypes)
        self.init_hooks(target_layers)

    def init_hidden_states(self, target_layers, dtypes):
        # dict containing the representations extracted in a sigle forward pass
        self.hidden_states_tmp = defaultdict(lambda: None)

        # dict storing the all the representations
        self.hidden_states = {}
        for name in target_layers:
            self.hidden_states[name] = torch.zeros(
                (self.nsamples, self.embdims[name]), dtype=dtypes[name]
            )

        if self.image_token_id is not None:
            self.hidden_states_image_avg = {
                name: torch.zeros((self.nsamples, self.embdims[name]), dtype=dtypes[name])
                for name in target_layers
            }
        # _imgavg_with_newline: image-region pool that ALSO includes the
        # learned image_newline embeddings (Cambrian only). Allocated when
        # both ids are set so the caller opts in explicitly.
        if (
            self.image_token_id is not None
            and self.image_newline_token_id is not None
        ):
            self.hidden_states_image_avg_with_newline = {
                name: torch.zeros((self.nsamples, self.embdims[name]), dtype=dtypes[name])
                for name in target_layers
            }
        if self.eoi_token_id is not None:
            self.hidden_states_eoi = {
                name: torch.zeros((self.nsamples, self.embdims[name]), dtype=dtypes[name])
                for name in target_layers
            }
        # Image+EOI joint average. Allocated only when both ids are set.
        # Skipped for LLaVA-1.5 which has no EOI marker.
        if self.image_token_id is not None and self.eoi_token_id is not None:
            self.hidden_states_image_eoi_avg = {
                name: torch.zeros((self.nsamples, self.embdims[name]), dtype=dtypes[name])
                for name in target_layers
            }

        # The last/text_avg/text_special pools all key off the same EOI
        # position info, so allocate them whenever extras are requested
        # (image_token_id or eoi_token_id is set, OR text-only mode opted
        # in via track_user_last). text_special further requires
        # special_token_ids to identify which positions count. In text-only
        # mode there's no image/EOI boundary, so text_avg's pooling falls
        # through to "all non-pad, non-special positions" which equals the
        # description tokens — exactly the desired "average description
        # token" pool.
        extras_requested = (
            self.image_token_id is not None
            or self.eoi_token_id is not None
            or self.track_user_last
        )
        if extras_requested:
            self.hidden_states_last = {
                name: torch.zeros((self.nsamples, self.embdims[name]), dtype=dtypes[name])
                for name in target_layers
            }
            self.hidden_states_text_avg = {
                name: torch.zeros((self.nsamples, self.embdims[name]), dtype=dtypes[name])
                for name in target_layers
            }
            if self.special_token_ids is not None and len(self.special_token_ids) > 0:
                self.hidden_states_text_special = {
                    name: torch.zeros((self.nsamples, self.embdims[name]), dtype=dtypes[name])
                    for name in target_layers
                }

        # user_last is independent of image/eoi extras (text-only mode has
        # neither) — gate it on the explicit track_user_last flag instead.
        if self.track_user_last:
            self.hidden_states_user_last = {
                name: torch.zeros((self.nsamples, self.embdims[name]), dtype=dtypes[name])
                for name in target_layers
            }

    def init_hooks(self, target_layers):
        for name, module in self.model.named_modules():
            if name in target_layers:
                module.register_forward_hook(
                    self._get_hook(name, self.hidden_states_tmp)
                )

    def _get_hook(self, name, hidden_states):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            hidden_states[name] = output.cpu()

        return hook_fn

    def _update_activation_dict(
        self, mask, is_last_batch, input_ids=None, user_last_pos=None
    ):
        seq_len = torch.sum(mask, dim=1)
        mask_2d = mask
        mask = mask.unsqueeze(-1)

        # Token-position selectors for the optional extras. Independent of
        # layer, so compute once per batch.
        img_mask = None
        eoi_pos = None
        post_eoi_content_mask = None
        post_eoi_special_mask = None
        last_pos = None
        if self.image_token_id is not None:
            assert input_ids is not None, "input_ids required for image_avg pooling"
            img_mask = (input_ids == self.image_token_id) & mask_2d.bool()
        # Per-row image_newline positions (Cambrian). Kept separate from
        # img_mask so the default _imgavg excludes them; combined into
        # img_with_newline_mask below for the joint pool, and added to
        # the post-image boundary detection so the trailing newline
        # doesn't get counted as "post-image text".
        img_newline_mask = None
        img_with_newline_mask = None
        if self.image_newline_token_id is not None:
            assert input_ids is not None, (
                "input_ids required for image_newline pooling"
            )
            img_newline_mask = (
                (input_ids == self.image_newline_token_id) & mask_2d.bool()
            )
            if img_mask is not None:
                img_with_newline_mask = img_mask | img_newline_mask
        if self.eoi_token_id is not None:
            assert input_ids is not None, "input_ids required for eoi pooling"
            eoi_mask = (input_ids == self.eoi_token_id) & mask_2d.bool()
            assert eoi_mask.any(dim=1).all(), "Some samples have no eoi token in input_ids"
            # First EOI position per sample (one image per sample at bs=1).
            eoi_pos = eoi_mask.int().argmax(dim=1)

        # Last-token, text_avg, text_special: all key off mask + eoi_pos.
        # Allocated together (see init_hidden_states), so a single guard
        # using hidden_states_last as the witness covers all three.
        if self.hidden_states_last is not None:
            assert input_ids is not None, (
                "input_ids required for last/text_avg/text_special pooling"
            )
            B, T = input_ids.shape
            # Last non-pad position. If a sample is fully padded (shouldn't
            # happen with our datasets), seq_len-1 = -1 → guard with clamp.
            last_pos = (seq_len.long() - 1).clamp_min(0)

            # Post-image boundary, strict >, so the boundary token itself
            # is excluded from text_avg / text_special.
            #   - eoi_token_id given (InternVL/OneVision/Cambrian): boundary
            #     = EOI position.
            #   - else image_token_id given (LLaVA-1.5: no EOI marker, the
            #     processor expands <image> into image_seq_length copies of
            #     the same id): boundary = last image-token position.
            #   - neither: pool over the entire non-pad span.
            position_idx = torch.arange(T, device=input_ids.device).unsqueeze(0)
            if eoi_pos is not None:
                boundary_pos = eoi_pos
            elif img_mask is not None:
                assert img_mask.any(dim=1).all(), (
                    "extract_all with image_token_id but no eoi_token_id "
                    "requires every sample to contain at least one image token"
                )
                # Last image-region position per sample. Use the joint
                # (image OR newline) mask when newline ids are tracked, so
                # the trailing per-row newline that follows the last image
                # column is NOT classified as post-image text.
                boundary_mask = (
                    img_with_newline_mask
                    if img_with_newline_mask is not None
                    else img_mask
                )
                boundary_pos = (
                    (T - 1) - boundary_mask.flip(dims=[1]).int().argmax(dim=1)
                )
            else:
                boundary_pos = None
            if boundary_pos is None:
                post_eoi_mask = mask_2d.bool()
            else:
                post_eoi_mask = (position_idx > boundary_pos.unsqueeze(1)) & mask_2d.bool()

            # Build excluded-id mask: image, image_newline (Cambrian), EOI,
            # and tokenizer specials. Newline must be excluded explicitly
            # because it sits past the boundary_pos for non-square images
            # and is otherwise indistinguishable from real text by id.
            excluded_ids = []
            if self.image_token_id is not None:
                excluded_ids.append(self.image_token_id)
            if self.image_newline_token_id is not None:
                excluded_ids.append(self.image_newline_token_id)
            if self.eoi_token_id is not None:
                excluded_ids.append(self.eoi_token_id)
            if self.special_token_ids is not None:
                excluded_ids.extend(self.special_token_ids)
            is_excluded = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in excluded_ids:
                is_excluded |= (input_ids == tid)
            post_eoi_content_mask = post_eoi_mask & ~is_excluded

            # text_special positions: post-EOI AND in special_token_ids
            # (image and EOI ids are NOT in this set).
            if (
                self.hidden_states_text_special is not None
                and self.special_token_ids is not None
            ):
                is_special = torch.zeros_like(input_ids, dtype=torch.bool)
                for tid in self.special_token_ids:
                    is_special |= (input_ids == tid)
                post_eoi_special_mask = post_eoi_mask & is_special

        for _, (layer, activations) in enumerate(self.hidden_states_tmp.items()):
            if self.use_last_token:
                batch_size = seq_len.shape[0]
                act_tmp = activations[
                    torch.arange(batch_size), torch.tensor(seq_len) - 1
                ]
            else:
                denom = torch.sum(mask, dim=1)  # batch x 1
                # act_tmp -> batch x seq_len x embed
                # mask -> batch x seq_len x 1
                # denom -> batch x 1
                act_tmp = torch.sum(activations * mask, dim=1) / denom

            num_current_tokens = act_tmp.shape[0]

            def _store(target, value):
                if is_last_batch:
                    target[layer][self.tot_tokens:] = value
                else:
                    target[layer][
                        self.tot_tokens : self.tot_tokens + num_current_tokens
                    ] = value

            _store(self.hidden_states, act_tmp)

            if self.hidden_states_image_avg is not None:
                weights = img_mask.unsqueeze(-1).to(activations.dtype)
                denom_img = weights.sum(dim=1).clamp_min(1)
                act_img = (activations * weights).sum(dim=1) / denom_img
                _store(self.hidden_states_image_avg, act_img)
            if self.hidden_states_image_avg_with_newline is not None:
                weights = img_with_newline_mask.unsqueeze(-1).to(activations.dtype)
                denom = weights.sum(dim=1).clamp_min(1)
                act_joint = (activations * weights).sum(dim=1) / denom
                _store(self.hidden_states_image_avg_with_newline, act_joint)
            if self.hidden_states_eoi is not None:
                batch_idx = torch.arange(activations.shape[0])
                act_eoi = activations[batch_idx, eoi_pos]
                _store(self.hidden_states_eoi, act_eoi)
            if self.hidden_states_image_eoi_avg is not None:
                # img_mask and eoi_pos are guaranteed populated when this
                # storage is allocated (both ids were given). Build a joint
                # boolean mask: image positions OR the single EOI position.
                joint = img_mask.clone()
                joint[torch.arange(joint.shape[0]), eoi_pos] = True
                weights = joint.unsqueeze(-1).to(activations.dtype)
                denom = weights.sum(dim=1).clamp_min(1)
                act_joint = (activations * weights).sum(dim=1) / denom
                _store(self.hidden_states_image_eoi_avg, act_joint)
            if self.hidden_states_last is not None:
                batch_idx = torch.arange(activations.shape[0])
                act_last = activations[batch_idx, last_pos]
                _store(self.hidden_states_last, act_last)
            if self.hidden_states_text_avg is not None:
                w = post_eoi_content_mask.unsqueeze(-1).to(activations.dtype)
                denom = w.sum(dim=1).clamp_min(1)
                act_text = (activations * w).sum(dim=1) / denom
                _store(self.hidden_states_text_avg, act_text)
            if self.hidden_states_text_special is not None:
                # post_eoi_special_mask is None when special_token_ids
                # wasn't provided (storage wouldn't have been allocated
                # in that case, so we wouldn't be in this branch).
                w = post_eoi_special_mask.unsqueeze(-1).to(activations.dtype)
                denom = w.sum(dim=1).clamp_min(1)
                act_special = (activations * w).sum(dim=1) / denom
                _store(self.hidden_states_text_special, act_special)
            if self.hidden_states_user_last is not None:
                assert user_last_pos is not None, (
                    "track_user_last=True but batch did not supply userlast_pos"
                )
                batch_idx = torch.arange(activations.shape[0])
                act_userlast = activations[batch_idx, user_last_pos]
                _store(self.hidden_states_user_last, act_userlast)

        self.tot_tokens += num_current_tokens

    @torch.no_grad()
    def extract(self, dataloader):
        start = time.time()
        is_last_batch = False
        for i, batch in enumerate(dataloader):

            if (i + 1) == self.nbatches:
                is_last_batch = True

            # Move tensors (and lists of tensors — Cambrian's `images` is a
            # list of per-aux-tower tensors with mismatched H/W, so it can't
            # be stacked into a single tensor) to CUDA. Non-tensor values
            # like image_sizes are passed through.
            for key in list(batch.keys()):
                v = batch[key]
                if isinstance(v, torch.Tensor):
                    batch[key] = v.to("cuda")
                elif isinstance(v, list):
                    batch[key] = [
                        t.to("cuda") if isinstance(t, torch.Tensor) else t
                        for t in v
                    ]

            # userlast_pos is dataset-side metadata for pooling, not a model
            # input — strip before forward so VLM kwargs don't see it.
            user_last_pos = batch.pop("userlast_pos", None)
            if user_last_pos is not None:
                user_last_pos = user_last_pos.cpu()

            if self.forward_fn is not None:
                out = self.forward_fn(batch)
            else:
                out = self.model(**batch, use_cache=False)

            # Pool info: shape must match activation length. Default
            # path uses the input attention_mask / input_ids; Cambrian
            # supplies a provider that returns the post-expansion mask
            # and a synthetic input_ids with sentinel image / eoi positions.
            if self.pool_info_provider is not None:
                pool_info = self.pool_info_provider(batch)
                mask = pool_info["mask"]
                if mask.is_cuda:
                    mask = mask.cpu()
                pool_input_ids = pool_info.get("input_ids")
                input_ids_cpu = (
                    pool_input_ids.cpu()
                    if (
                        pool_input_ids is not None
                        and (
                            self.image_token_id is not None
                            or self.eoi_token_id is not None
                        )
                    )
                    else None
                )
            else:
                mask = (batch["attention_mask"] != 0).cpu()
                # text_avg / text_special also need input_ids (for masking
                # specials and finding the post-image boundary). Allocated
                # whenever extras_requested fires — the witness here is
                # hidden_states_last, which is set under the same gate.
                input_ids_cpu = (
                    batch["input_ids"].cpu()
                    if (
                        self.image_token_id is not None
                        or self.eoi_token_id is not None
                        or self.hidden_states_last is not None
                    )
                    else None
                )
            self._update_activation_dict(
                mask,
                is_last_batch,
                input_ids=input_ids_cpu,
                user_last_pos=user_last_pos,
            )

            if self.collect_predictions:
                if not hasattr(out, "logits") or out.logits is None:
                    raise RuntimeError(
                        "Model output has no logits field; cannot collect predictions."
                    )
                seq_len = batch["attention_mask"].sum(dim=1)
                for b in range(out.logits.shape[0]):
                    last_pos = int(seq_len[b].item()) - 1
                    token_id = int(out.logits[b, last_pos].argmax().item())
                    self.predicted_token_ids.append(token_id)

            if (i + 1) % (self.print_every // self.global_batch_size) == 0:
                end = time.time()
                print(
                    f"{(i+1)*self.global_batch_size/1000}k data, \
                    batch {i+1}/{self.nbatches}, \
                    tot_time: {(end-start)/60: .3f}min, "
                )
