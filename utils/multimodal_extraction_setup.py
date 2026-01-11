import torch
from typing import Optional, Tuple, Union
from .model_utils import (
    replace_multimodal_projector,
    replace_multimodal_lm,
    load_hf_model_and_processor_or_tokenizer,
    ModelType,
    ProcessorType,
)


def setup_multimodal_model(
    multimodal_model_name_or_path: str,
    model_cache_dir: str,
    language_model_name_or_path: Optional[str] = None,
    pretrained_projector_name_or_path: Optional[str] = None,
    model_dtype: torch.dtype = torch.float16,
    attn_implementation: str = "flash_attention_2",
    replace_language_model: bool = False,
    replace_pretrained_projector: bool = False,
    device_map: Union[str, dict] = "cuda:0",
    skip_processor: bool = False,
) -> Union[Tuple[ModelType, ProcessorType], ModelType]:
    """
    Sets up a HuggingFace-based multimodal model with optional projector and LM replacement.

    Args:
        multimodal_model_name_or_path: Path or name of the base multimodal model.
        model_cache_dir: Directory for caching model weights.
        language_model_name_or_path: Optional new language model path for replacement.
        pretrained_projector_name_or_path: Optional path to pretrained projector to use.
        model_dtype: Torch dtype to load the model with (default: torch.float16).
        attn_implementation: Attention implementation type (default: "flash_attention_2").
        replace_language_model: Whether to replace the internal LM.
        replace_pretrained_projector: Whether to replace the projector module.
        device_map: Device mapping (e.g. "cuda:0" or {"model": "cuda:0"}).
        skip_processor: Whether to skip loading the processor/tokenizer.

    Returns:
        Tuple[ModelType, ProcessorType]: If skip_processor is False.
        ModelType: If skip_processor is True.
    """

    # Validate input logic
    if replace_language_model and not language_model_name_or_path:
        raise ValueError(
            "`language_model_name_or_path` must be provided if `replace_language_model` is True."
        )

    if replace_pretrained_projector and not pretrained_projector_name_or_path:
        raise ValueError(
            "`pretrained_projector_name_or_path` must be provided if `replace_pretrained_projector` is True."
        )

    # Build argument set
    loader_args = dict(
        model_name_or_path=multimodal_model_name_or_path,
        cache_dir=model_cache_dir,
        device_map=device_map,
        dtype=model_dtype,
        attn_implementation=attn_implementation,
    )

    if skip_processor:
        loader_args["skip_processor"] = True

    model_output = load_hf_model_and_processor_or_tokenizer(**loader_args)

    if skip_processor:
        model = model_output
        processor = None
    else:
        model, processor = model_output

    if replace_pretrained_projector:
        model = replace_multimodal_projector(
            multimodal_model=model,
            pretrained_projector_model_name_or_path=pretrained_projector_name_or_path,
            cache_dir=model_cache_dir,
        )

    if replace_language_model:
        model = replace_multimodal_lm(
            multimodal_model=model,
            replacement_lm_name_or_path=language_model_name_or_path,
            cache_dir=model_cache_dir,
            dtype=model_dtype,
        )

    model.to(device_map)
    return model if skip_processor else (model, processor)
