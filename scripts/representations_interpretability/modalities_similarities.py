import torch
import argparse
import os
import gc
import time
from safetensors.torch import save_file
from typing import List
from src.residual_stream_tracer import residual_stream_tracer
from utils import (
    setup_multimodal_model,
    get_dataloader,
    get_dataloader_for_captioning,
    seed_all,
    ModelType,
    compute_layers_cosine_similarity,
    compute_layers_homogeneity_score,
)

# Use float16 for relatively light weight memory usage and fast extraction
MODEL_DTYPE = torch.float16
ATTN_IMPLEMENTATION = "flash_attention_2"
# !must be changed to "sdpa" for torch.float32

# Valid residual stream extraction types
RESIDUAL_STREAM_TYPE: set[str] = {
    "output_layer",
    "post_mlp",
}

DATASETS_TYPES = {
    "cocoqa_img": {
        "images_qa": True,
        "guide_text": "Answer the question using a single word or phrase.\n",
        "downsample_size": 2500,
        "chat_mode": True,
    },
    "coco_captioning": {
        "downsample_size": 2500,
        # use chat mode and captioning prompt
        # for extracting the residual stream
        "chat_mode": True,
        "concatenate_captions_and_add_prompt": True,
        # use plain text and concatenate captions
        # for extracting the residual stream
        "concatenate_captions": False,
    },
}

SUFFIX = "chat"

MODELS = {
    "multimodal_model": {
        "multimodal_model_name_or_path": "llava-hf/llava-1.5-7b-hf",
    },
    "multimodal_model_pretrained_connector": {
        "multimodal_model_name_or_path": "llava-hf/llava-1.5-7b-hf",
        "replace_pretrained_projector": True,
        "pretrained_projector_name_or_path": "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        "replace_language_model": True,
        "language_model_name_or_path": "lmsys/vicuna-7b-v1.5",
    },
}


def parse_list_argument(arg_value: str, valid_values: set) -> List[str]:
    """
    Parse comma-separated string argument and validate against valid values.

    Args:
        arg_value: Comma-separated string of values
        valid_values: Set of valid values to validate against

    Returns:
        List of validated values

    Raises:
        ValueError: If any value is not in valid_values
    """
    if not arg_value:
        return list(valid_values)

    values = [v.strip() for v in arg_value.split(",")]
    invalid_values = set(values) - valid_values

    if invalid_values:
        raise ValueError(
            f"Invalid values: {invalid_values}. Valid values are: {valid_values}"
        )

    return values


def get_model_identifier(model_args: dict) -> str:
    """
    Generate a standardized model identifier from model arguments.

    Args:
        model_args: Dictionary containing model configuration

    Returns:
        String identifier for the model
    """
    mm_model_name = model_args["multimodal_model_name_or_path"].split("/")[-1]

    language_model_path = model_args.get("language_model_name_or_path")
    pretrained_projector_path = model_args.get("pretrained_projector_name_or_path")
    if language_model_path and pretrained_projector_path:
        pp_model_name = pretrained_projector_path.split("/")[-1]
        return pp_model_name
    else:
        return mm_model_name


def get_embeddings_sampling_args(model: ModelType) -> dict:
    """Extract embedding sampling arguments from model config."""
    if hasattr(model, "config"):
        image_token_id = model.config.image_token_index
        pad_token_id = model.config.pad_token_id
        image_seq_length = model.config.image_seq_length
    else:
        raise ValueError("Model does not have a config")

    return {
        "image_token_id": image_token_id,
        "pad_token_id": pad_token_id,
        "image_seq_length": image_seq_length,
        "subtract_padding_per_sample": True,
    }


def main():
    """
    Extract residual streams from multimodal models and compute modalities similarities
    between text and image modalities in the residual stream.

    Results are organized in the following directory structure:
    results/
    ├── dataset_name/
    │   ├── model_name/
    │   │   ├── modalities_similarity/
    │   │   │   └── <stream_type>_sample_cosine_similarity.safetensors
    │   │   │   └── <stream_type>_sample_homogeneity_score_<metric>.safetensors

    ! Comment: 🤦‍♂️ a better structure would have been:
    results/
    ├── modalities_similarities/
    │   ├── model_name/
    │   │   ├── dataset_name/
    │   │   │   └── <stream_type>_sample_cosine_similarity.safetensors
    │   │   │   └── <stream_type>_sample_homogeneity_score_<metric>.safetensors
    ! ✅ I corrected this in the code for plotting purposes

    Command-line arguments allow selection of specific residual stream types to control which analyses are performed.
    """
    parser = argparse.ArgumentParser(
        description="Extract and analyze residual streams from multimodal models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--dataset-path-or-name", type=str, required=True)
    parser.add_argument("--model-cache-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="coco_captioning",
        help=f"Comma-separated list of types of dataset to process. "
        f"Valid options: {', '.join(sorted(DATASETS_TYPES.keys()))}. "
        f"If not specified, the default dataset type will be processed.",
    )
    parser.add_argument("--maxk", type=int, default=128)
    parser.add_argument("--range-max", type=int, default=128)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--Z", type=float, default=1.65)
    parser.add_argument("--bootstrap", type=bool, default=False)
    parser.add_argument("--n-boot", type=int, default=1000)

    # Analysis configuration
    parser.add_argument(
        "--residual-stream-types",
        type=str,
        default=None,
        help=f"Comma-separated list of residual stream types to extract. "
        f"Valid options: {', '.join(sorted(RESIDUAL_STREAM_TYPE))}. "
        f"If not specified, all types will be processed.",
    )

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate and parse arguments
    try:
        residual_stream_types = parse_list_argument(
            args.residual_stream_types, RESIDUAL_STREAM_TYPE
        )
        dataset_types = parse_list_argument(args.dataset_type, DATASETS_TYPES.keys())
        dataset_configs = [
            DATASETS_TYPES[dataset_type] for dataset_type in dataset_types
        ]
    except ValueError as e:
        parser.error(f"Argument parsing error: {e}")

    # Initialize the embeddings sampling args
    embeddings_sampling_args = None

    # Set random seed for reproducibility
    seed_all(args.seed)
    results_dir = os.path.expanduser(args.results_dir)

    print(
        f"🔧 Processing {len(residual_stream_types)} residual stream types: {residual_stream_types}"
    )

    count = 0
    num_combinations = len(residual_stream_types) * len(dataset_types) * len(MODELS)

    # Process each residual stream type
    for residual_stream_type in residual_stream_types:
        print(f"\n{'=' * 80}")
        print(f"🔍 Processing: {residual_stream_type} residual stream type")
        print(f"{'=' * 80}")

        start_time = time.time()

        # Process each dataset type
        for dataset_type, dataset_config in zip(dataset_types, dataset_configs):
            print(f"\n{'*' * 60}")
            print(f"📊 Processing dataset: {dataset_type}")
            print(f"{'*' * 60}")

            # Set up output directories
            common_dir = os.path.join(results_dir, dataset_type)

            # Process each model configuration
            for model_key, model_args in MODELS.items():
                print(f"\n{'+' * 40}")
                print(f"🤖 Processing model: {model_key}")
                print(f"{'+' * 40}")

                # Generate model identifier and create output directory
                model_identifier = get_model_identifier(model_args)
                modalities_similarity_dir = os.path.join(
                    common_dir,
                    model_identifier,
                    "modalities_similarity",
                )
                os.makedirs(modalities_similarity_dir, exist_ok=True)

                # Load model and processor
                model, processor = setup_multimodal_model(
                    **{
                        **model_args,
                        "model_cache_dir": args.model_cache_dir,
                    }
                )

                # Prepare dataloader arguments
                dataloader_kwargs = {
                    "dataset_path_or_name": args.dataset_path_or_name,
                    "seed": args.seed,
                    "processor": processor,
                    "batch_size": args.batch_size,
                }

                # Select appropriate dataloader function
                if dataset_type == "coco_captioning":
                    get_dataloader_func = get_dataloader_for_captioning
                elif dataset_type == "cocoqa_img":
                    # !not so neat but it works for now
                    # todo: redesign the interface if used in the future
                    get_dataloader_func = get_dataloader

                if embeddings_sampling_args is None:
                    embeddings_sampling_args = get_embeddings_sampling_args(model)

                residual_stream_text_embeddings = None
                residual_stream_image_embeddings = None

                chat_mode = dataset_config["chat_mode"]
                for binary_flag in [True, False]:
                    if binary_flag:
                        print(
                            "\n*****🔤 Residual stream with random text positions*****"
                        )
                    else:
                        print(
                            "\n*****🖼️ Residual stream with random image positions*****"
                        )

                    if chat_mode:
                        dataloader_kwargs["chat_template_exists"] = binary_flag
                    else:
                        dataloader_kwargs["image_first"] = binary_flag

                    processed_dataloader = get_dataloader_func(
                        **dataset_config,
                        **dataloader_kwargs,
                    )

                    # Create generator on the correct device
                    generator = torch.Generator(device=model.device).manual_seed(
                        args.seed
                    )

                    embeddings_sampling_args = {
                        **embeddings_sampling_args,
                        "text_direction": "right",  # only effective for chat_mode=True
                        "skip_image_pos": binary_flag,
                        "skip_text_pos": not binary_flag,
                        "generator": generator,
                        "chat_mode": chat_mode,
                    }

                    # Extract residual stream
                    residual_stream = residual_stream_tracer(
                        model=model,
                        processed_dataloader=processed_dataloader,
                        residual_stream_type=residual_stream_type,
                        tokens_pooling_method="sample",
                        return_deepcopy=True,
                        embeddings_sampling_args=embeddings_sampling_args,
                    )
                    if binary_flag:
                        residual_stream_text_embeddings = residual_stream
                    else:
                        residual_stream_image_embeddings = residual_stream

                    del processed_dataloader
                    gc.collect()
                    torch.cuda.empty_cache()

                print(
                    "Computing cosine similarity and homogeneity score between text and image embeddings in the residual stream..."
                )
                layers_cosine_similarity = compute_layers_cosine_similarity(
                    residual_stream_text_embeddings,
                    residual_stream_image_embeddings,
                    bootstrap=args.bootstrap,
                    n_boot=args.n_boot,
                )
                save_file(
                    {"layers_cosine_similarity": layers_cosine_similarity},
                    os.path.join(
                        modalities_similarity_dir,
                        f"{residual_stream_type}_sample_cosine_similarity_{SUFFIX}.safetensors",
                    ),
                )
                for metric in ["cosine"]:
                    layers_homogeneity_scores = compute_layers_homogeneity_score(
                        residual_stream_text_embeddings,
                        residual_stream_image_embeddings,
                        maxk=args.maxk,
                        range_max=args.range_max,
                        k=args.k,
                        Z=args.Z,
                        metric=metric,
                    )
                    save_file(
                        {"layers_homogeneity_scores": layers_homogeneity_scores},
                        os.path.join(
                            modalities_similarity_dir,
                            f"{residual_stream_type}_sample_homogeneity_score_{metric}_{SUFFIX}.safetensors",
                        ),
                    )
                # Clean up model from memory
                del (
                    model,
                    processor,
                    residual_stream_text_embeddings,
                    residual_stream_image_embeddings,
                    layers_cosine_similarity,
                )
                gc.collect()
                torch.cuda.empty_cache()

                print(f"\n{'+-' * 40}")
                count += 1
                print(f"Progress: {count}/{num_combinations} combinations completed ✅")
                print(f"{'+-' * 40}\n")

            gc.collect()
            torch.cuda.empty_cache()

        end_time = time.time()
        print(f"\n{'+-' * 60}")
        print(
            f"Combination {residual_stream_type} completed in ⌛ {(end_time - start_time) / 60:.2f} minutes"
        )
        print(f"{'+-' * 60}\n")

    print(f"\n{'=' * 80}")
    print("Analysis completed successfully! 🎉")
    print(f"Results saved to: {results_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
