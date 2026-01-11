import torch
import os
import gc
from typing import Tuple, Type, Union, List
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPProcessor,
    CLIPModel,
    # LlamaForCausalLM,
    # LlamaTokenizerFast
)
from huggingface_hub import snapshot_download
from .constants import LLAVA_CHAT_TEMPLATE

# Define supported vision-language model types and their corresponding classes
SUPPORTED_VL_MODELS = {
    "llava-1.5": {
        "processor": LlavaProcessor,
        "model": LlavaForConditionalGeneration,
    },
    "llava-v1.6": {
        "processor": LlavaNextProcessor,
        "model": LlavaNextForConditionalGeneration,
    },
    "paligemma2": {
        "processor": PaliGemmaProcessor,
        "model": PaliGemmaForConditionalGeneration,
    },
}

# Define types for clarity
ProcessorType = Union[
    LlavaProcessor, LlavaNextProcessor, PaliGemmaProcessor, CLIPProcessor
]
ModelType = Union[
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    AutoModelForCausalLM,
    CLIPModel,
]


def _get_vl_model_classes(
    model_name_or_path: str,
) -> Tuple[Type[ProcessorType], Type[ModelType]]:
    """
    Helper function to determine the processor and model classes based on the model name.

    Args:
        model_name_or_path (str): The name or path of the model.

    Returns:
        Tuple[Type[ProcessorType], Type[ModelType]]: The processor and model classes.

    Raises:
        ValueError: If the model type is not supported.
    """
    model_name_lower = model_name_or_path.lower()
    for key, classes in SUPPORTED_VL_MODELS.items():
        if key in model_name_lower:
            return classes["processor"], classes["model"]
    if "llava" in model_name_lower:
        return LlavaProcessor, LlavaForConditionalGeneration
    raise ValueError(
        f"Model type for '{model_name_or_path}' is not supported. "
        f"Supported types: {', '.join(SUPPORTED_VL_MODELS.keys())}"
    )


def load_hf_model_and_processor_or_tokenizer(
    model_name_or_path: str,
    cache_dir: str,
    text_model: bool = False,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
    attn_implementation: str = "flash_attention_2",
    low_cpu_mem_usage: bool = True,
    use_fast: bool = True,
    skip_model: bool = False,
    skip_processor: bool = False,
) -> Union[
    ModelType,
    ProcessorType,
    Tuple[ModelType, Union[ProcessorType, AutoTokenizer]],
]:
    """
    Load a Hugging Face model and its processor or tokenizer.

    Args:
        model_name_or_path (str): Model identifier or local path.
        cache_dir (str): Directory for caching the model and processor.
        text_model (bool): Whether to load a text-only model.
        device_map (str): Device map strategy for model loading.
        dtype (torch.dtype): Data type for model parameters.
        attn_implementation (str): Attention implementation type.
        low_cpu_mem_usage (bool): Optimize for low CPU memory usage.
        use_fast (bool): Use the fast version of the processor if available.
        skip_model (bool): Whether to not load the model.
        skip_processor (bool): Whether to not load the processor.

    Returns:
        The loaded model and processor/tokenizer or just the model or the processor.

    Raises:
        ValueError: If the model type is unsupported.
        RuntimeError: If loading the model or processor fails.
    """
    cache_dir = os.path.expanduser(cache_dir)

    if skip_model and skip_processor:
        raise ValueError("Cannot not load both the model and the processor.")

    try:
        if not text_model:
            ProcessorClass, ModelClass = _get_vl_model_classes(model_name_or_path)
        else:
            ProcessorClass = AutoTokenizer
            ModelClass = AutoModelForCausalLM
    except ValueError as e:
        if "clip" in model_name_or_path.lower():
            ProcessorClass = CLIPProcessor
            ModelClass = CLIPModel
        else:
            raise e

    processor_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "cache_dir": cache_dir,
    }

    model_kwargs = {
        **processor_kwargs,
        "device_map": device_map,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "attn_implementation": attn_implementation,
        "torch_dtype": dtype,
    }

    if not skip_processor:
        try:
            print(f"\nLoading {model_name_or_path} processor...")
            processor = ProcessorClass.from_pretrained(
                **processor_kwargs, use_fast=use_fast
            )

            if "vicuna" in model_name_or_path.lower():
                processor.chat_template = LLAVA_CHAT_TEMPLATE
            elif "lbasile/llava" in model_name_or_path.lower():
                processor.chat_template = LLAVA_CHAT_TEMPLATE

        except Exception as e:
            raise RuntimeError(
                "Failed to load "
                f"{('processor' if not text_model else 'tokenizer')} "
                f"for {model_name_or_path} from cache {cache_dir}. Error: {e}"
            )
    if not skip_model:
        try:
            print(f"\nLoading {model_name_or_path} model...")
            model = ModelClass.from_pretrained(**model_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model for '{model_name_or_path}' from cache '{cache_dir}'. Error: {e}"
            )

    if skip_model:
        return processor
    elif skip_processor:
        return model

    if "clip" in model_name_or_path.lower():
        return model, processor

    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
        if processor.pad_token is None:
            processor.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(processor))
    elif hasattr(processor, "tokenizer") and hasattr(model, "language_model"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.language_model.resize_token_embeddings(len(processor.tokenizer))
    else:
        raise ValueError(
            f"Processor/tokenizer {processor} has no padding_side attribute and model {model} has no language_model attribute."
        )

    return model, processor


def replace_multimodal_lm(
    multimodal_model: ModelType,
    replacement_lm_name_or_path: str,
    cache_dir: str,
    dtype=torch.float16,
) -> ModelType:
    """
    Replace the Multimodal language model with the Vicuna model.

    Args:
        model (ModelType): The original Multimodal model.
        replacement_lm_name_or_path (str): The name or path of the language model to replace with (e.g., Vicuna).
        cache_dir (str): Directory for caching the model.

    Returns:
        ModelType: The updated Multimodal model.
    """
    # Original Multimodal model device
    multimodal_model_device = next(multimodal_model.parameters()).device

    print(
        f"\nLoading {replacement_lm_name_or_path} model for language model replacement..."
    )
    # Load the replacement language model (e.g., Vicuna)
    replacement_lm = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=replacement_lm_name_or_path,
        cache_dir=cache_dir,
        text_model=True,
        device_map="cpu",
        dtype=dtype,
        attn_implementation="sdpa",
        skip_processor=True,
    )

    # Check if the type of the replacement model is compatible
    if type(replacement_lm.model) is not type(multimodal_model.language_model.model):
        raise TypeError(
            f"\nReplacement model type {type(replacement_lm.model)} is not compatible "
            f"with Multimodal model type {type(multimodal_model.language_model.model)}."
        )
    print(f"\nSuccessfully loaded {replacement_lm_name_or_path} mode")

    # Move Multimodal model to CPU
    multimodal_model.to("cpu")

    print("Replacing Multimodal language model...")
    min_vocab_size = min(
        multimodal_model.language_model.model.embed_tokens.weight.size(0),
        replacement_lm.model.embed_tokens.weight.size(0),
    )

    with torch.no_grad():
        multimodal_model.language_model.model.embed_tokens.weight[:min_vocab_size] = (
            replacement_lm.model.embed_tokens.weight[:min_vocab_size].clone()
        )
        multimodal_model.language_model.model.layers.load_state_dict(
            replacement_lm.model.layers.state_dict()
        )
        multimodal_model.language_model.model.norm.load_state_dict(
            replacement_lm.model.norm.state_dict()
        )
        multimodal_model.language_model.lm_head.weight[:min_vocab_size] = (
            replacement_lm.lm_head.weight[:min_vocab_size].clone()
        )

    # Explanation for LLaVA:
    # - `multimodal_model.language_model.model` is the LLaVA language model.
    # - `replacement_lm` contains a `model` attribute that is the language model
    # - The LLaVA language model includes additional parameters, that act as placeholders
    #  for image embeddings, in the embed_tokens and lm_head layers.
    # !These additional parameters do not influence the language model's behavior but
    # !are necessary for the model to function correctly.
    # !These parameters are changed for image embeddings when there is an image as context.
    # !For more details check the modelling_llava.py of the transformers library.
    # !Specifically, lines 413-430 under the class `LlavaForConditionalGeneration`
    # !in transformers/models/llava/modeling_llava.py, version 4.50.1.

    # Remove the vicuna model and its tokenizer from memory
    del replacement_lm
    torch.cuda.empty_cache()
    gc.collect()

    # Move the updated llava model back to its original device
    multimodal_model.to(multimodal_model_device)

    print("Successfully replaced multimodal language model.\n")
    return multimodal_model


def download_mm_projector_bin(
    model_id: str,
    cache_dir: str,
):
    """
    Download the mm_projector.bin file from the model_id.
    """
    cache_dir = os.path.expanduser(cache_dir)

    snapshot_download(repo_id=model_id, cache_dir=cache_dir)


def find_mm_projector_bin(
    model_id: str,
    cache_dir: str,
):
    cache_dir = os.path.expanduser(cache_dir)

    # Convert the model_id into the format huggingface_hub uses
    model_dir = model_id.replace("/", "--")
    full_model_dir = os.path.join(cache_dir, f"models--{model_dir}")

    if not os.path.isdir(full_model_dir):
        print(f"Model directory not found: {full_model_dir}")
        print(f"Downloading model {model_id}...")
        download_mm_projector_bin(model_id, cache_dir)
        print(f"Model {model_id} downloaded successfully.")

    # Find the snapshot directory inside that model directory
    snapshots_dir = os.path.join(full_model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(f"Snapshots directory not found: {snapshots_dir}")

    snapshot_subdirs = os.listdir(snapshots_dir)
    if not snapshot_subdirs:
        raise FileNotFoundError(f"No snapshot found in: {snapshots_dir}")

    snapshot_path = os.path.join(snapshots_dir, snapshot_subdirs[0])
    projector_path = os.path.join(snapshot_path, "mm_projector.bin")

    if not os.path.isfile(projector_path):
        raise FileNotFoundError(f"mm_projector.bin not found in: {snapshot_path}")

    return projector_path


def replace_multimodal_projector(
    multimodal_model: ModelType,
    pretrained_projector_model_name_or_path: str,
    cache_dir: str,
) -> ModelType:
    """
    Replace the multi-modal projector in a multimodal model with weights from mm_projector.bin.

    Args:
        multimodal_model (ModelType): The multimodal model.
        pretrained_projector_model_name_or_path (str): HF repo name with the mm_projector.bin file.
        cache_dir (str): Hugging Face cache directory.

    Returns:
        ModelType: Updated multimodal model with projector weights replaced.
    """
    print(
        f"\nReplacing multi-modal projector weights for {pretrained_projector_model_name_or_path}..."
    )
    device = next(multimodal_model.parameters()).device
    projector_path = find_mm_projector_bin(
        pretrained_projector_model_name_or_path, cache_dir
    )
    print(f"Found projector weights at: {projector_path}")
    mm_state_dict = torch.load(
        projector_path,
        map_location="cpu",
        weights_only=True,
    )

    if not hasattr(multimodal_model, "multi_modal_projector"):
        raise AttributeError("Target model has no `multi_modal_projector` attribute.")

    # Mapping from file keys to model attributes
    projector_mapping = {
        "model.mm_projector.0.weight": "linear_1.weight",
        "model.mm_projector.0.bias": "linear_1.bias",
        "model.mm_projector.2.weight": "linear_2.weight",
        "model.mm_projector.2.bias": "linear_2.bias",
    }

    projector = multimodal_model.multi_modal_projector
    multimodal_model.to("cpu")  # Reduce memory usage during assignment

    print("Replacing multi-modal projector weights...")

    with torch.no_grad():
        for old_key, new_key in projector_mapping.items():
            if not hasattr(projector, new_key.split(".")[0]):
                raise AttributeError(f"Projector has no attribute `{new_key}`")

            target_param = getattr(projector, new_key.split(".")[0])
            param_tensor = mm_state_dict[old_key]

            if "weight" in new_key:
                if target_param.weight.shape != param_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch: {new_key} — expected {target_param.weight.shape}, got {param_tensor.shape}"
                    )
                target_param.weight.copy_(param_tensor)
            elif "bias" in new_key:
                if target_param.bias.shape != param_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch: {new_key} — expected {target_param.bias.shape}, got {param_tensor.shape}"
                    )
                target_param.bias.copy_(param_tensor)

        # or simply
        # projector_state_dict = {
        #     projector_mapping[k]: v
        #     for k, v in mm_state_dict.items()
        #     if k in projector_mapping
        # }

        # # Load the weights into the model projector
        # multimodal_model.multi_modal_projector.load_state_dict(projector_state_dict)

    print("Multi-modal projector successfully updated with pre-trained weights.\n")
    multimodal_model.to(device)
    return multimodal_model
