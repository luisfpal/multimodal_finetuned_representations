from __future__ import annotations

from typing import Any, Optional

import torch
from PIL import Image
from transformers import BatchEncoding, PreTrainedTokenizer

from .cambrian_bridge import _ensure_cambrian_importable


def is_cambrian_model(
    model: Any,
    model_name_or_path: Optional[str] = None,
) -> bool:
    if model_name_or_path and "cambrian" in model_name_or_path.lower():
        return True
    if model is None:
        return False
    return "cambrian" in model.__class__.__name__.lower()


def resolve_cambrian_conv_mode(
    model: Any,
    model_name_or_path: Optional[str] = None,
    explicit_conv_mode: Optional[str] = None,
) -> str:
    if explicit_conv_mode:
        return explicit_conv_mode

    if model_name_or_path:
        model_name = model_name_or_path.lower()
    else:
        model_name = str(
            getattr(getattr(model, "config", None), "_name_or_path", "")
        ).lower()
    known_map = {
        "cambrian-phi3-3b": "phi3",
        "cambrian-8b": "llama_3",
        "cambrian-13b": "vicuna_v1",
        "cambrian-34b": "chatml_direct",
    }
    for marker, conv_mode in known_map.items():
        if marker in model_name:
            return conv_mode

    if "llama-2" in model_name:
        return "cambrian_llama_2"
    if "mpt" in model_name:
        return "mpt"
    if "v1" in model_name or "vicuna" in model_name:
        return "cambrian_v1"

    raise ValueError(
        "Could not infer Cambrian conversation mode for model "
        f"'{model_name or model.__class__.__name__}'. "
        "Pass an explicit conversation mode."
    )


def get_cambrian_device(model: Any) -> torch.device:
    """Get the primary compute device using a hierarchical lookup."""
    # 1. Check the device map (common in multi-GPU/HF models)
    device_map = getattr(model, "hf_device_map", {})
    for dev in device_map.values():
        if str(dev) not in {"cpu", "disk"}:
            # torch.device handles ints (0), strings ("cuda:0"), and device objects
            return torch.device(dev if isinstance(dev, (int, str)) else dev)

    # 2. Fallback to model.device or the first parameter's device
    device = getattr(model, "device", None)
    if device:
        return torch.device(device)
        
    return next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")


def apply_chat_template(
    model: Any,
    text: str | list[str],
    has_image: bool = False,
    guide_text: str = "",
    conv_mode: Optional[str] = None,
    assistant_prefix: str = "",
) -> list[str]:
    """Apply the Cambrian chat template to questions, returning formatted prompts.

    Mirrors the ``apply_chat_template`` function in ``internvl_utils.py``.
    Conv mode is resolved internally from ``model.config._name_or_path``.
    Use ``resolve_cambrian_conv_mode`` directly if you need the mode string.

    When ``assistant_prefix`` is non-empty, seed the assistant turn with this
    partial response so the model continues from that point (continuation mode).
    """
    _ensure_cambrian_importable()
    from cambrian.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
    )
    from cambrian.conversation import conv_templates

    questions = [text] if isinstance(text, str) else text

    selected_mode = resolve_cambrian_conv_mode(
        model=model,
        explicit_conv_mode=conv_mode,
    )
    use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))

    prompts: list[str] = []
    for question in questions:
        q = f"{question}\n{guide_text}\n" if guide_text else question
        q = q.strip()
        if has_image:
            if use_im_start_end:
                image_prefix = (
                    f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
                )
            else:
                image_prefix = DEFAULT_IMAGE_TOKEN
            q = f"{image_prefix}\n{q}"

        conv = conv_templates[selected_mode].copy()
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if assistant_prefix:
            # Passing None for the assistant role gives just the role header with no
            # turn-closing separator.  We then manually append the partial content.
            # Vicuna-style templates (TWO/SINGLE) end with "ROLE:" → add a space.
            # LLAMA_3 / CHATML / MPT end with a header token or "\n" → append directly.
            sep = " " if prompt.rstrip("\n").endswith(":") else ""
            prompt = prompt + sep + assistant_prefix
        prompts.append(prompt)

    return prompts


def tokenize_cambrian_prompts(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    conv_mode: str,
    device: torch.device,
) -> BatchEncoding:
    _ensure_cambrian_importable()
    from cambrian.constants import IMAGE_TOKEN_INDEX
    from cambrian.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3

    tokenize_fn = (
        tokenizer_image_token_llama3
        if conv_mode in {"llama_3", "llama_v3"}
        else tokenizer_image_token
    )
    ids_per_sample = [
        tokenize_fn(
            prompt,
            tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        for prompt in prompts
    ]

    if tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        pad_token_id = tokenizer.eos_token_id
    elif tokenizer.bos_token_id is not None:
        pad_token_id = tokenizer.bos_token_id
    else:
        pad_token_id = 0

    max_len = max(sample_ids.shape[0] for sample_ids in ids_per_sample)
    batch_size = len(ids_per_sample)

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for idx, sample_ids in enumerate(ids_per_sample):
        seq_len = sample_ids.shape[0]
        input_ids[idx, -seq_len:] = sample_ids.to(device)
        attention_mask[idx, -seq_len:] = 1

    return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


def preprocess_cambrian_images(
    images: list[Image.Image],
    image_processor: Any,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    _ensure_cambrian_importable()
    from cambrian.mm_utils import process_images

    # Ensure all images are RGB — Cambrian's expand2square assumes RGB
    # for background_color (3-tuple from image_mean). Some HF dataset
    # images may arrive in L/P/RGBA mode despite upstream conversion.
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    image_tensors = process_images(images, image_processor, model.config)
    if isinstance(image_tensors, list):
        return [tensor.to(device=device, dtype=dtype) for tensor in image_tensors]
    return image_tensors.to(device=device, dtype=dtype)


def prepare_cambrian_batch_inputs(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    images: Optional[list[Image.Image]] = None,
    guide_text: str = "",
    conv_mode: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    assistant_prefix: str = "",
) -> dict[str, Any]:
    """Prepare tokenized inputs, prompts, and image tensors for Cambrian.

    Returns a dict with keys: ``tokenized``, ``prompts``, ``image_tensors``,
    ``image_sizes``, ``conv_mode``.
    """
    device = get_cambrian_device(model)
    has_image = images is not None
    selected_mode = resolve_cambrian_conv_mode(
        model=model,
        model_name_or_path=model_name_or_path,
        explicit_conv_mode=conv_mode,
    )
    prompts = apply_chat_template(
        model=model,
        text=questions,
        has_image=has_image,
        guide_text=guide_text,
        conv_mode=selected_mode,
        assistant_prefix=assistant_prefix,
    )
    tokenized = tokenize_cambrian_prompts(
        prompts=prompts,
        tokenizer=tokenizer,
        conv_mode=selected_mode,
        device=device,
    )

    result: dict[str, Any] = {
        "tokenized": tokenized,
        "prompts": prompts,
        "image_tensors": None,
        "image_sizes": None,
        "conv_mode": selected_mode,
    }

    if not has_image:
        return result

    if any(image is None for image in images):
        raise ValueError("Cambrian multimodal batch cannot contain None images.")

    result["image_sizes"] = [image.size for image in images]
    result["image_tensors"] = preprocess_cambrian_images(
        images=images,
        image_processor=tokenizer.image_processor,
        model=model,
        device=device,
        dtype=torch.float16,
    )
    return result


def generate_cambrian_batch(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    images: Optional[list[Image.Image]],
    max_new_tokens: int,
    guide_text: str = "",
    conv_mode: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    temperature: float = 0.0,
) -> list[str]:
    """Generate responses one sample at a time.

    Cambrian cannot true-batch due to variable image token counts per sample.
    """
    if images is not None and len(images) != len(questions):
        raise ValueError("Cambrian generation expects len(images) == len(questions).")

    responses: list[str] = []

    for idx, question in enumerate(questions):
        sample_images = None if images is None else [images[idx]]
        prepared = prepare_cambrian_batch_inputs(
            model=model,
            tokenizer=tokenizer,
            questions=[question],
            images=sample_images,
            guide_text=guide_text,
            conv_mode=conv_mode,
            model_name_or_path=model_name_or_path,
        )

        tokenized = prepared["tokenized"]
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        prompt_length = int(attention_mask.sum(dim=1).item())

        generate_kwargs: dict[str, Any] = {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "num_beams": 1,
            "use_cache": False,
            "min_new_tokens": 1,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        if prepared["image_tensors"] is not None:
            generate_kwargs["images"] = prepared["image_tensors"]
            generate_kwargs["image_sizes"] = prepared["image_sizes"]

        with torch.inference_mode():
            output_ids = model.generate(**generate_kwargs)

        if output_ids.shape[1] > prompt_length:
            generated_ids = output_ids[0, prompt_length:]
        else:
            generated_ids = output_ids[0]
        responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())

    return responses