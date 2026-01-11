import torch
import argparse
import os
import gc
import time
from safetensors.torch import save_file
from typing import Union, List
from src.residual_stream_tracer import residual_stream_tracer
from utils import (
    setup_multimodal_model,
    get_dataloader,
    get_dataloader_for_captioning,
    seed_all,
    compute_layers_residual_stream_similarities,
    compute_heads_projection_residual_stream_similarities,
    compute_layers_intrinsic_dimension,
    compute_layers_residual_stream_entropy,
)

# Use float16 for relatively light weight memory usage and fast extraction
MODEL_DTYPE = torch.float16
ATTN_IMPLEMENTATION = "flash_attention_2"
# !must be changed to "sdpa" for torch.float32

# Valid residual stream extraction types
RESIDUAL_STREAM_TYPE: set[str] = {
    "output_layer",
    "post_mlp",
    "heads_projection",
}

# Valid token pooling methods
TOKENS_POOLING_METHOD: set[Union[str]] = {
    "mean",
    "last",
    "none",  # Note: "none" gets converted to None internally
}

# Valid similarity measures
SIMILARITY_MEASURES: set[Union[str]] = {
    "neighborhood_overlap",
    "linear_cka",
    "svcca",
}

COCOQA_DATASET_ARGS = {
    "guide_text": "Answer the question using a single word or phrase.\n",
    "downsample_size": 2500,
}

DATASETS = {
    "cocoqa_txt": {
        "texts_qa": True,
        **COCOQA_DATASET_ARGS,
        "chat_mode": True,
    },
    "cocoqa_img": {
        "images_qa": True,
        **COCOQA_DATASET_ARGS,
        "chat_mode": True,  #! check this
    },
    "coco_captioning": {
        "downsample_size": 2500,
        # use chat mode and captioning prompt
        # for extracting the residual stream
        "chat_mode": True,  #! check this
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

MODELS_SIMILARITIES_NAME = "llava-1.5-7b-hf_vs_vicuna-7b-v1.5"


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


def check_empty_argument(arg_value: str) -> bool:
    """
    Check if an argument is empty.
    """
    return arg_value is None or arg_value == ""


def validate_combination(residual_stream_type: str, tokens_pooling_method: str) -> bool:
    """
    Validate if a combination of residual stream type and tokens pooling method is valid.

    Args:
        residual_stream_type: Type of residual stream extraction
        tokens_pooling_method: Method for pooling tokens

    Returns:
        True if combination is valid, False otherwise
    """
    # These combinations are not supported due to memory constraints
    if residual_stream_type == "heads_projection" and tokens_pooling_method == "none":
        return False
    return True


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


def main():
    """
    Extract residual streams from multimodal models and computes similarity
    measures and data measures.

    This script performs analysis of neural network representations by:
    1. Extracting residual streams at different model locations (output_layer, post_mlp, heads_projection)
    2. Computing intrinsic dimension and entropy measures
    3. Calculating similarity measures between different model configurations

    Results are organized in the following directory structure:
    results/
    ├── dataset_name/
    │   ├── similarities/
    │   │   └── model1_vs_model2_<pooling>_<stream_type>.safetensors
    │   └── models_data_measures/
    │       └── model_name/
    │           ├── dataset_entropy_<pooling>_<stream_type>.safetensors
    │           ├── id_<rank>_<range>_<pooling>_<stream_type>.safetensors
    │           └── prompt_entropy_<pooling>_<stream_type>.safetensors

    Command-line arguments allow selection of specific residual stream types and
    token pooling methods to control which analyses are performed.
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

    # Analysis configuration
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="cocoqa_txt,cocoqa_img",
        help=f"Comma-separated list of dataset types to process. "
        f"Valid options: {', '.join(sorted(DATASETS.keys()))}. "
        f"If not specified, cocoqa_txt and cocoqa_img will be processed.",
    )
    parser.add_argument(
        "--residual-stream-types",
        type=str,
        default=None,
        help=f"Comma-separated list of residual stream types to extract. "
        f"Valid options: {', '.join(sorted(RESIDUAL_STREAM_TYPE))}. "
        f"If not specified, all types will be processed.",
    )
    parser.add_argument(
        "--tokens-pooling-methods",
        type=str,
        default=None,
        help=f"Comma-separated list of token pooling methods. "
        f"Valid options: {', '.join(sorted(TOKENS_POOLING_METHOD))}. "
        f"If not specified, all methods will be processed.",
    )
    parser.add_argument(
        "--similarity-measures",
        type=str,
        default=None,
        help=f"Comma-separated list of similarity measures. "
        f"Valid options: {', '.join(sorted(SIMILARITY_MEASURES))}. "
        f"If not specified, all measures will be processed.",
    )

    # Similarity measure parameters
    parser.add_argument("--maxk", type=int, default=30)
    parser.add_argument("--accept-rate", type=float, default=0.95)

    # Intrinsic dimension parameters
    parser.add_argument("--id-nn-rank", type=int, default=16)
    parser.add_argument("--id-nn-range-max", type=int, default=100)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate and parse arguments
    try:
        dataset_types = parse_list_argument(args.dataset_type, DATASETS.keys())
        residual_stream_types = parse_list_argument(
            args.residual_stream_types, RESIDUAL_STREAM_TYPE
        )
        tokens_pooling_methods = parse_list_argument(
            args.tokens_pooling_methods, TOKENS_POOLING_METHOD
        )
    except ValueError as e:
        parser.error(f"Argument parsing error: {e}")

    # Validate argument ranges
    if not (0.0 < args.accept_rate <= 1.0):
        parser.error("accept-rate must be between 0 and 1")
    if args.maxk <= 0:
        parser.error("maxk must be positive")
    if args.id_nn_rank <= 0:
        parser.error("id-nn-rank must be positive")
    if args.id_nn_range_max <= 0:
        parser.error("id-nn-range-max must be positive")

    # Get similarity measures
    if not check_empty_argument(args.similarity_measures):
        similarity_measures = parse_list_argument(
            args.similarity_measures, SIMILARITY_MEASURES
        )
    else:
        similarity_measures = []

    # Set random seed for reproducibility
    seed_all(args.seed)
    results_dir = os.path.expanduser(args.results_dir)

    print(f"🔧 Processing {len(dataset_types)} dataset types: {dataset_types}")
    print(
        f"🔧 Processing {len(residual_stream_types)} residual stream types: {residual_stream_types}"
    )
    print(
        f"🔧 Processing {len(tokens_pooling_methods)} token pooling methods: {tokens_pooling_methods}"
    )

    count = 0
    num_combinations = (
        len(residual_stream_types) * len(tokens_pooling_methods) * len(dataset_types)
    )

    # Process each combination of residual stream type and token pooling method
    for residual_stream_type in residual_stream_types:
        for tokens_pooling_method in tokens_pooling_methods:
            # Validate combination feasibility
            if not validate_combination(residual_stream_type, tokens_pooling_method):
                print(
                    f"⏭️ Skipping invalid combination: {residual_stream_type} + {tokens_pooling_method}"
                )
                continue

            print(f"\n{'=' * 80}")
            print(
                f"🔍 Processing: {residual_stream_type} with {tokens_pooling_method} pooling"
            )
            print(f"{'=' * 80}")

            start_time = time.time()

            # Determine if we need deep copy (only False for "none" pooling to save memory)
            return_deepcopy = tokens_pooling_method != "none"

            for dataset_name in dataset_types:
                dataset_args = DATASETS[dataset_name]
                print(f"\n{'*' * 60}")
                print(f"📊 Processing dataset: {dataset_name}")
                print(f"{'*' * 60}")

                # Initialize storage for model residual streams
                residual_stream_multimodal_model = None
                residual_stream_multimodal_model_pretrained_connector = None

                # Set up output directories
                common_dir = os.path.join(results_dir, dataset_name)
                similarities_dir = os.path.join(common_dir, "similarities")
                models_data_measures_dir = os.path.join(
                    common_dir, "models_data_measures"
                )

                os.makedirs(similarities_dir, exist_ok=True)
                os.makedirs(models_data_measures_dir, exist_ok=True)

                # Process each model configuration
                for model_key, model_args in MODELS.items():
                    print(f"\n{'+' * 60}")
                    print(f"🤖 Processing model: {model_key}")
                    print(f"{'+' * 60}")

                    # Generate model identifier and create output directory
                    model_identifier = get_model_identifier(model_args)
                    model_dir = os.path.join(models_data_measures_dir, model_identifier)
                    os.makedirs(model_dir, exist_ok=True)

                    # Load model and processor
                    model, processor = setup_multimodal_model(
                        **{
                            **model_args,
                            "model_cache_dir": args.model_cache_dir,
                        }
                    )

                    # ! Don't touch this comments
                    # Get dataloader for current dataset
                    # !this is the same dataloader for the models in the loop
                    # this is a minimal overhead since here it doesn't consume much memory nor time
                    # !not so neat but it works for now
                    # todo: redesign the interface if used in the future

                    # Select appropriate dataloader function based on dataset type
                    if dataset_name == "coco_captioning":
                        get_dataloader_func = get_dataloader_for_captioning
                    else:
                        get_dataloader_func = get_dataloader

                    processed_dataloader = get_dataloader_func(
                        **{
                            **dataset_args,
                            "dataset_path_or_name": args.dataset_path_or_name,
                            "seed": args.seed,
                            "processor": processor,
                            "batch_size": args.batch_size,
                        }
                    )

                    # Extract residual stream
                    residual_stream = residual_stream_tracer(
                        model=model,
                        processed_dataloader=processed_dataloader,
                        residual_stream_type=residual_stream_type,
                        tokens_pooling_method=tokens_pooling_method,
                        return_deepcopy=return_deepcopy,
                    )

                    # Store residual streams for similarity computation (only when pooling method is not None)
                    if tokens_pooling_method != "none":
                        if model_key == "multimodal_model":
                            residual_stream_multimodal_model = residual_stream
                        elif model_key == "multimodal_model_pretrained_connector":
                            residual_stream_multimodal_model_pretrained_connector = (
                                residual_stream
                            )

                    # Compute intrinsic dimension and entropy measures for layer-wise extractions
                    if residual_stream_type in ["output_layer", "post_mlp"]:
                        if tokens_pooling_method != "none":
                            # Compute intrinsic dimension (only for pooled data)
                            print("🧮 Computing intrinsic dimension...")
                            layers_intrinsic_dimension = (
                                compute_layers_intrinsic_dimension(
                                    residual_stream,
                                    algorithm="scaling_gride",
                                    k=args.id_nn_rank,
                                    range_max=args.id_nn_range_max,
                                )
                            )
                            save_file(
                                {
                                    "layers_intrinsic_dimension": layers_intrinsic_dimension
                                },
                                os.path.join(
                                    model_dir,
                                    f"id_{args.id_nn_rank}_{args.id_nn_range_max}_{residual_stream_type}_{tokens_pooling_method}.safetensors",
                                ),
                            )

                            # Compute dataset entropy (only for pooled data)
                            print("📊 Computing dataset entropy...")
                            layers_dataset_entropy = (
                                compute_layers_residual_stream_entropy(
                                    residual_stream,
                                    entropy_type="dataset-entropy",
                                )
                            )
                            save_file(
                                {"layers_dataset_entropy": layers_dataset_entropy},
                                os.path.join(
                                    model_dir,
                                    f"dataset_entropy_{residual_stream_type}_{tokens_pooling_method}_{SUFFIX}.safetensors",
                                ),
                            )
                        else:
                            # Compute prompt entropy (only for non-pooled data)
                            print("📊 Computing prompt entropy...")
                            layers_prompt_entropy = (
                                compute_layers_residual_stream_entropy(
                                    residual_stream,
                                    entropy_type="prompt-entropy",
                                )
                            )
                            save_file(
                                {"layers_prompt_entropy": layers_prompt_entropy},
                                os.path.join(
                                    model_dir,
                                    f"prompt_entropy_{residual_stream_type}_{tokens_pooling_method}_{SUFFIX}.safetensors",
                                ),
                            )

                    if tokens_pooling_method == "none":
                        del residual_stream

                    # Clean up model from memory
                    del model, processor, processed_dataloader
                    gc.collect()
                    torch.cuda.empty_cache()

                # Skip similarity computation when pooling method is None
                if len(similarity_measures) > 0:
                    if tokens_pooling_method != "none":
                        # Compute similarity measures between models
                        print("🤝 Computing similarity measures between models...")
                        if residual_stream_type == "heads_projection":
                            compute_residual_stream_similarities = (
                                compute_heads_projection_residual_stream_similarities
                            )
                        else:
                            compute_residual_stream_similarities = (
                                compute_layers_residual_stream_similarities
                            )

                        residual_stream_similarities = (
                            compute_residual_stream_similarities(
                                residual_stream_multimodal_model,
                                residual_stream_multimodal_model_pretrained_connector,
                                similarity_measures=similarity_measures,
                                maxk=args.maxk,
                                accept_rate=args.accept_rate,
                            )
                        )

                        # Save similarity results
                        save_file(
                            residual_stream_similarities,
                            os.path.join(
                                similarities_dir,
                                f"{MODELS_SIMILARITIES_NAME}_{residual_stream_type}_{tokens_pooling_method}.safetensors",
                            ),
                        )

                # Clean up residual streams from memory (only if they were stored)
                if tokens_pooling_method != "none":
                    del (
                        residual_stream_multimodal_model,
                        residual_stream_multimodal_model_pretrained_connector,
                    )
                gc.collect()
                torch.cuda.empty_cache()

                count += 1
                print(f"\n{'+-' * 30}")
                print(f"✅ Progress: {count}/{num_combinations} combinations completed")
                print(f"{'+-' * 30}")

            end_time = time.time()
            print(f"\n{'+-' * 30}")
            print(
                f"✅ Combination {residual_stream_type} + {tokens_pooling_method} completed in ⌛ {(end_time - start_time) / 60:.2f} minutes"
            )
            print(f"{'+-' * 30}")

    print(f"\n{'=' * 80}")
    print("🎉 Analysis completed successfully!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
