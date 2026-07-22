from collections import defaultdict

import torch


def _detect_lm_layer_layout(model, n_layer):
    """Auto-detect the named_modules path prefix for the LM decoder layers.

    Different VLM families place the LM decoder under different attribute
    paths (and HF restructures the wrappers between transformers versions).
    Returns ``(layer_prefix, norm_path_or_None)`` such that
    ``f"{layer_prefix}.{i}"`` is a valid named_modules entry for every
    ``i in [0, n_layer)``.

    Patterns tried (most specific first):
      * ``model.language_model.layers``  — OneVision (Qwen2.5-VL family).
      * ``language_model.layers``        — modern LLaVA-1.5-hf.
      * ``language_model.model.layers``  — InternVL2 (CausalLM with nested .model).
      * ``model.layers``                 — Cambrian, plain LlamaModel.
    """
    names = {name for name, _ in model.named_modules()}
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.layers",
    ]
    for cand in candidates:
        if f"{cand}.0" in names and f"{cand}.{n_layer - 1}" in names:
            norm_candidate = cand[: -len(".layers")] + ".norm"
            return cand, (norm_candidate if norm_candidate in names else None)
    raise RuntimeError(
        f"Could not auto-detect LM layer prefix for n_layer={n_layer}. "
        f"Tried: {candidates}. Top-level module names: "
        f"{sorted({n.split('.', 1)[0] for n in names if n})}"
    )


def get_target_layers(
    model,
    n_layer,
    option="norm1",
    every=1,
    only_output=False,
    last_k=0,
    is_internvl=False,
    layer_prefix=None,
    norm_path=None,
):

    if only_output:
        every = 1

    if is_internvl:
        # InternVL2's InternLM2 backbone uses .attention_norm / .ffn_norm
        # instead of the LLaMA-style .input_layernorm / .post_attention_layernorm.
        map_names = dict(
            norm1=".attention_norm",
            norm2=".ffn_norm",
            res2="",
        )
    else:
        map_names = dict(
            norm1=".input_layernorm",
            norm2=".post_attention_layernorm",
            res2="",
        )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    if layer_prefix is None:
        if is_internvl:
            layer_prefix = "language_model.model.layers"
            norm_path = "language_model.model.norm"
        else:
            layer_prefix, norm_path = _detect_lm_layer_layout(model, n_layer)
    elif norm_path is None:
        assert layer_prefix.endswith(
            ".layers"
        ), f"Cannot derive norm_path from {layer_prefix!r}; pass it explicitly."
        norm_path = layer_prefix[: -len(".layers")] + ".norm"

    target_layers = {
        i: f"{layer_prefix}.{i}{suffix}" for i in range(0, n_layer - last_k, every)
    }
    for i in range(n_layer - last_k, n_layer):
        target_layers[i] = f"{layer_prefix}.{i}{suffix}"

    if option == "norm1" or option == "norm2":
        if norm_path is None:
            raise RuntimeError(
                f"target_layer={option!r} requires a norm sibling next to "
                f"{layer_prefix!r}, but it's absent from named_modules()."
            )
        target_layers[n_layer] = norm_path

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers


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
    batch = {
        k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }

    if forward_fn is not None:
        _ = forward_fn(batch)
    else:
        _ = model(batch["input_ids"].to("cuda"))

    for name, module in model.named_modules():
        if name in target_layers:
            handles[name].remove()

    assert len(embdims) == len(target_layers)
    return embdims, dtypes
