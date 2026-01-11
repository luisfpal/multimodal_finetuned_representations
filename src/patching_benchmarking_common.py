import gc
import os
import argparse
from typing import Dict, List, Tuple, Callable, Any
import time
import pandas as pd
import torch
from utils import setup_multimodal_model, patch_layers_weights
from utils.operations_utils import seed_all

# Patching methods configuration - will be set dynamically based on args
PATCHING_WEIGHTS_METHODS = {}


# todo:
# if used in the future add support to pass list of layers to patch
# for more flexible experimentation
def get_patching_layers(method: str, num_layers: int, stride: int) -> List[int]:
    """Get the list of layers to process for patching."""
    base_layers = [i * stride for i in range(num_layers // stride + 1)]
    if method == "cumulative" or method == "two_parts":  # Accept both names
        return [layer_idx - 1 for layer_idx in base_layers][1:]
    return base_layers[:-1]


# todo:
# add the layers_list method for the case of passing a list of layers to patch
# for more flexible experimentation
def get_patching_params(method: str, layer_value: int, method_args: Dict) -> Dict:
    """Get the parameters for patch_layers_weights based on method."""
    if method == "sliding_window":
        return {
            "start_layer": layer_value,
            "window_size": method_args["window_size"],
        }
    else:  # cumulative (or two_parts for backward compatibility)
        return {
            "split_layer": layer_value,
            "reverse_cumulative_layers": method_args.get(
                "reverse_cumulative_layers", False
            ),
        }


def run_patching_experiments(
    method: str,
    method_args: Dict,
    source_model,
    target_model,
    num_layers: int,
    processed_dataloader,
    processor,
    max_new_tokens: int,
    benchmark_func: Callable,
) -> Dict[str, Dict[str, float]]:
    """Run patching experiments for the specified method."""
    stride = method_args["stride"]
    layers = get_patching_layers(method, num_layers, stride)
    benchmarking_results = {}

    print(f"\n{'+-' * 40}")
    print(f"Running {method} patching... 🔄")
    print(f"Number of layers patching operations: {len(layers)}")
    print(f"{'+-' * 40}\n")

    for idx, layer_value in enumerate(layers):
        progress = (idx + 1) / len(layers)
        print(
            f"\nProcessing layer {layer_value}: [{idx + 1}/{len(layers)}] ({progress:.2%})"
        )

        patch_params = get_patching_params(method, layer_value, method_args)

        model_with_patched_layers = patch_layers_weights(
            source_model=source_model,
            target_model=target_model,
            method=method,
            **patch_params,
            in_place_patching=False,
        )

        results = benchmark_func(
            model_with_patched_layers,
            processed_dataloader,
            processor,
            max_new_tokens,
        )

        key = (
            f"{'start' if method == 'sliding_window' else 'split'}_layer_{layer_value}"
        )
        benchmarking_results[key] = results

        del model_with_patched_layers
        gc.collect()
        torch.cuda.empty_cache()

    return benchmarking_results


def get_output_filename(
    model_identifiers_list: List[str], method: str = None, method_args: Dict = None
) -> str:
    """Generate output filename based on patching method or baseline mode."""
    combined_model_id = "_".join(model_identifiers_list)

    # Baseline-only mode (no patching)
    if method is None or method_args is None:
        return f"{combined_model_id}_baseline.csv"

    # Patching mode
    stride = method_args["stride"]

    if method == "sliding_window":
        window_size = method_args["window_size"]
        return f"{combined_model_id}_sliding_window_ws{window_size}_s{stride}.csv"
    else:  # cumulative (accept two_parts for backward compatibility)
        reverse_suffix = (
            "_reverse_cumulative"
            if method_args.get("reverse_cumulative_layers", False)
            else ""
        )
        # Generate filename with cumulative naming (but accept two_parts as method)
        method_name = "cumulative" if method == "cumulative" else "cumulative"
        return f"{combined_model_id}_{method_name}_s{stride}{reverse_suffix}.csv"


def setup_models(args) -> Tuple[object, object, object]:
    """Setup and return the multimodal models and processor."""
    multimodal_model, processor = setup_multimodal_model(
        multimodal_model_name_or_path=args.multimodal_model_name_or_path,
        model_cache_dir=args.model_cache_dir,
        device_map="cpu",
        attn_implementation="flash_attention_2",
        model_dtype=torch.float16,
    )

    multimodal_model_pretrained_connector = setup_multimodal_model(
        multimodal_model_name_or_path=args.multimodal_model_name_or_path,
        model_cache_dir=args.model_cache_dir,
        replace_pretrained_projector=True,
        pretrained_projector_name_or_path=args.pretrained_projector_name_or_path,
        replace_language_model=True,
        language_model_name_or_path=args.language_model_name_or_path,
        skip_processor=True,
        device_map="cpu",
        attn_implementation="flash_attention_2",
        model_dtype=torch.float16,
    )

    return multimodal_model, multimodal_model_pretrained_connector, processor


def benchmark_baseline_models(
    multimodal_model,
    multimodal_model_pretrained_connector,
    processed_dataloader,
    processor,
    max_new_tokens: int,
    benchmark_func: Callable,
    results_dir: str,
    model_identifiers_list: List[str],
    save_baseline_outputs: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Benchmark both baseline models and return their results."""
    print("\nBenchmarking baseline multimodal model... 📊")

    # Prepare save path for baseline multimodal model outputs
    baseline_model_save_path = None
    if save_baseline_outputs:
        multimodal_model_id = model_identifiers_list[0]  # Multimodal model identifier
        baseline_model_save_path = os.path.join(
            results_dir, f"{multimodal_model_id}_outputs.json"
        )
        # Skip if file already exists
        if os.path.exists(baseline_model_save_path):
            print(
                f"Skipping baseline model outputs save - file already exists: {baseline_model_save_path}"
            )
            baseline_model_save_path = None

    baseline_model_results = benchmark_func(
        multimodal_model,
        processed_dataloader,
        processor,
        max_new_tokens,
        dtype=torch.float16,
        save_outputs_path=baseline_model_save_path,
    )

    print("\nBenchmarking multimodal model with pretrained connector... 📊")

    # Prepare save path for pretrained connector model outputs
    pretrained_connector_save_path = None
    if save_baseline_outputs:
        pretrained_connector_model_id = model_identifiers_list[
            1
        ]  # Pretrained connector model identifier
        pretrained_connector_save_path = os.path.join(
            results_dir, f"{pretrained_connector_model_id}_outputs.json"
        )
        # Skip if file already exists
        if os.path.exists(pretrained_connector_save_path):
            print(
                f"Skipping pretrained connector outputs save - file already exists: {pretrained_connector_save_path}"
            )
            pretrained_connector_save_path = None

    pretrained_connector_results = benchmark_func(
        multimodal_model_pretrained_connector,
        processed_dataloader,
        processor,
        max_new_tokens,
        dtype=torch.float16,
        save_outputs_path=pretrained_connector_save_path,
    )

    return baseline_model_results, pretrained_connector_results


def create_benchmark_model_wrapper(
    benchmark_func: Callable, args: argparse.Namespace
) -> Callable:
    """Create a wrapper for the benchmark function with GPU device management."""

    def benchmark_model(
        model,
        processed_dataloader,
        processor,
        max_new_tokens: int,
        dtype: torch.dtype = torch.float16,
        save_outputs_path: str = None,
    ) -> Dict[str, float]:
        """Benchmark model with automatic GPU device management."""
        model.to(device="cuda:0", dtype=dtype)
        benchmark_func_kwargs = {
            "model": model,
            "processed_dataloader": processed_dataloader,
            "processor": processor,
            "max_new_tokens": max_new_tokens,
        }

        # Add CLIP parameters for captioning tasks
        if "captioning" in benchmark_func.__name__:
            benchmark_func_kwargs["save_captions_path"] = save_outputs_path
            benchmark_func_kwargs["clip_model_name_or_path"] = getattr(
                args, "clip_model_name_or_path", "openai/clip-vit-large-patch14-336"
            )
            benchmark_func_kwargs["clip_cache_dir"] = getattr(
                args, "clip_cache_dir", "~/scratch/huggingface/hub"
            )
            benchmark_func_kwargs["clip_weight"] = getattr(args, "clip_weight", 2.5)
            benchmark_func_kwargs["clip_batch_size"] = getattr(
                args, "clip_batch_size", 16
            )
        elif "vqa" in benchmark_func.__name__:
            benchmark_func_kwargs["save_answers_path"] = save_outputs_path

        results = benchmark_func(**benchmark_func_kwargs)

        model.to("cpu")
        return results

    return benchmark_model


def run_patching_benchmarking(
    args: argparse.Namespace,
    dataset_config: Dict[str, Any],
    get_dataloader_func: Callable,
    benchmark_func: Callable,
    results_subdir: str,
) -> None:
    """
    Run patching benchmarking experiments.

    Args:
        args: Command line arguments
        dataset_config: Configuration for the dataset
        get_dataloader_func: Function to get the dataloader
        benchmark_func: Function to benchmark the model
        results_subdir: Subdirectory name for results
    """
    seed_all(args.seed)

    # Setup models
    multimodal_model, multimodal_model_pretrained_connector, processor = setup_models(
        args
    )
    model_num_layers = multimodal_model.language_model.config.num_hidden_layers

    # Create descriptive model identifiers for file naming
    multimodal_model_id = args.multimodal_model_name_or_path.split("/")[-1]
    language_model_id = args.language_model_name_or_path.split("/")[-1]
    model_identifiers_list = [
        multimodal_model_id,
        f"{multimodal_model_id}_{language_model_id}_pretrained_connector",
    ]

    # Setup output directories
    results_dir = os.path.expanduser(args.results_dir)
    benchmarking_results_dir = os.path.join(results_dir, results_subdir)
    os.makedirs(benchmarking_results_dir, exist_ok=True)

    # Create benchmark model wrapper
    benchmark_model = create_benchmark_model_wrapper(benchmark_func, args)

    # Process each dataset
    for dataset_name, dataset_args in dataset_config.items():
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset_name} 📊")
        print(f"{'=' * 80}\n")

        # Prepare dataloader arguments
        dataloader_kwargs = {
            **dataset_args,
            "dataset_path_or_name": args.dataset_path_or_name,
            "processor": processor,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "cache_dir": getattr(args, "dataset_cache_dir", None),
        }

        processed_dataloader = get_dataloader_func(**dataloader_kwargs)

        dataset_dir = os.path.join(benchmarking_results_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Initialize results dictionary
        all_results = {}

        # Benchmark baseline models if requested
        benchmark_mode = getattr(args, "benchmark_mode", "patching")
        if benchmark_mode in ["baseline", "both"]:
            baseline_model_results, pretrained_connector_results = (
                benchmark_baseline_models(
                    multimodal_model,
                    multimodal_model_pretrained_connector,
                    processed_dataloader,
                    processor,
                    args.max_new_tokens,
                    benchmark_model,
                    dataset_dir,
                    model_identifiers_list,
                    getattr(args, "save_baseline_outputs", False),
                )
            )
            all_results.update(
                {
                    "mm_model": baseline_model_results,
                    "mm_pretrained_connector": pretrained_connector_results,
                }
            )

        # Run patching experiments if requested
        patching_method_config = None
        if benchmark_mode in ["patching", "transplantation", "both"]:
            patching_method_config = PATCHING_WEIGHTS_METHODS[args.patching_method]

            start_time = time.time()
            patching_results = run_patching_experiments(
                method=args.patching_method,
                method_args=patching_method_config,
                source_model=multimodal_model,
                target_model=multimodal_model_pretrained_connector,
                num_layers=model_num_layers,
                processed_dataloader=processed_dataloader,
                processor=processor,
                max_new_tokens=args.max_new_tokens,
                benchmark_func=benchmark_model,
            )
            end_time = time.time()

            print(
                f"Time taken for {args.patching_method} patching with {dataset_name} dataset: {(end_time - start_time) / 60:.2f} minutes 🕒"
            )

            all_results.update(patching_results)

        # Save results to CSV
        if benchmark_mode == "baseline":
            # Baseline-only mode: no patching parameters
            output_filename = get_output_filename(model_identifiers_list)
        else:
            # Patching or both mode: use patching parameters
            output_filename = get_output_filename(
                model_identifiers_list,
                args.patching_method,
                patching_method_config,
            )
        results_df = pd.DataFrame.from_dict(all_results, orient="index")
        output_path = os.path.join(dataset_dir, output_filename)
        results_df.to_csv(output_path, index=True)
        print(f"Results saved to: {output_path} 📝")
        print(f"{'=' * 80}\n")


def create_common_parser() -> argparse.ArgumentParser:
    """Create a common argument parser for patching benchmarking."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal_model_name_or_path", type=str, required=True)
    parser.add_argument("--model_cache_dir", type=str, required=True)
    parser.add_argument("--pretrained_projector_name_or_path", type=str, required=True)
    parser.add_argument("--language_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--patching_method",
        type=str,
        required=True,
        choices=[
            "sliding_window",
            "cumulative",
            "two_parts",
        ],  # Accept both for backward compatibility
    )
    parser.add_argument(
        "--stride", type=int, default=2, help="Stride value for patching methods"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=2,
        help="Window size for sliding_window method (ignored for cumulative)",
    )
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--dataset_path_or_name", type=str, required=True)
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets (required for benchmark datasets)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_baseline_outputs",
        action="store_true",
        help="Save baseline model outputs (captions/answers) to JSON files",
    )
    parser.add_argument(
        "--reverse_cumulative_layers",
        action="store_true",
        help="Patch layers cumulatively from end to start (cumulative method only)",
    )
    parser.add_argument(
        "--benchmark_mode",
        type=str,
        default="patching",
        choices=[
            "baseline",
            "patching",
            "transplantation",
            "both",
        ],  # Accept transplantation for backward compatibility
        help="Which models to benchmark: 'baseline' (only baseline models), 'patching' (only patching experiments), or 'both' (baseline + patching)",
    )

    # CLIP evaluation parameters
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP model name or path for evaluation",
    )
    parser.add_argument(
        "--clip_cache_dir",
        type=str,
        default="~/scratch/huggingface/hub",
        help="Cache directory for CLIP model",
    )
    parser.add_argument(
        "--clip_weight",
        type=float,
        default=2.5,
        help="Weight for CLIP score computation",
    )
    parser.add_argument(
        "--clip_batch_size", type=int, default=16, help="Batch size for CLIP evaluation"
    )

    return parser


def setup_patching_methods(args: argparse.Namespace) -> None:
    """Setup patching methods configuration based on command line arguments."""
    global PATCHING_WEIGHTS_METHODS

    # Backward compatibility: handle both old and new argument names
    if not hasattr(args, "patching_method") and hasattr(args, "transplantation_method"):
        args.patching_method = args.transplantation_method

    PATCHING_WEIGHTS_METHODS = {
        "sliding_window": {
            "stride": args.stride,
            "window_size": args.window_size,
        },
        "cumulative": {
            "stride": args.stride,
            "reverse_cumulative_layers": args.reverse_cumulative_layers,
        },
        "two_parts": {  # Backward compatibility
            "stride": args.stride,
            "reverse_cumulative_layers": args.reverse_cumulative_layers,
        },
    }
