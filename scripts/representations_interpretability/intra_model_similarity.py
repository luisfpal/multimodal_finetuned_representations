import argparse
import gc
import os
from typing import Dict

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file

from src.residual_stream_tracer import residual_stream_tracer
from utils import (
    compute_cross_layers_residual_stream_similarities,
    get_dataloader,
    load_hf_model_and_processor_or_tokenizer,
    seed_all,
)

# Load environment variables
load_dotenv()

# Configuration
MODEL_DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "flash_attention_2"
RESIDUAL_STREAM_TYPE = "output_layer"
TOKENS_POOLING_METHODS_DEFAULT = ["last"]
SIMILARITY_MEASURES_DEFAULT = [
    "neighborhood_overlap",
    "linear_cka",
    "distance_correlation",
    "id_correlation",
    "information_imbalance",
    "information_imbalance_reversed",
    "information_imbalance_mean",
]


def extract_representation(
    model_path: str,
    dataset_config: Dict,
    dataset_name: str,
    cache_dir: str,
    datasets_cache_dir: str,
    results_dir: str,
    pooling_method: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    model_id = model_path.split("/")[-1]
    # Use configurable representations directory
    repr_base_dir = os.path.expanduser(args.representations_dir)
    repr_dir = os.path.join(
        repr_base_dir,
        "representations",
        model_id,
        dataset_name,
        RESIDUAL_STREAM_TYPE,
        pooling_method,
    )
    os.makedirs(repr_dir, exist_ok=True)
    repr_path = os.path.join(repr_dir, "representations.safetensors")

    # Optimization: Load if exists
    if os.path.exists(repr_path):
        print(f"  💾 Found cached representation at: {repr_path}")
        try:
            loaded = load_file(repr_path)
            if "residual_stream" in loaded:
                tensor = loaded["residual_stream"]
                # Validate tensor shape is reasonable (has samples and layers)
                if tensor.ndim >= 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                    print(f"  ✅ Loaded successfully. Shape: {tensor.shape}")
                    return tensor[:]
                else:
                    print(f"  ⚠️ Invalid tensor shape: {tensor.shape}. Re-extracting...")
            else:
                print("  ⚠️ Key 'residual_stream' not found. Re-extracting...")
        except Exception as e:
            print(f"  ⚠️ Error loading cache: {type(e).__name__}: {e}. Re-extracting...")

    print(f"\n  🤖 Loading model: {model_path}")
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=model_path,
        cache_dir=cache_dir,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="cuda:0",
    )

    print(f"  📚 Loading dataset: {dataset_name}")
    dl_args = {
        "dataset_path_or_name": args.dataset_path_or_name,
        "seed": args.seed,
        "processor": processor,
        "batch_size": args.batch_size,
        "cache_dir": datasets_cache_dir,
    }
    dataloader = get_dataloader(**dataset_config, **dl_args)

    metadata = {
        "model": model_id,
        "dataset": dataset_name,
        "pooling_method": pooling_method,
        "type": RESIDUAL_STREAM_TYPE,
    }

    print(
        f"  👉 Extracting Representations ({RESIDUAL_STREAM_TYPE} | {pooling_method})..."
    )
    residual_stream = residual_stream_tracer(
        model=model,
        processed_dataloader=dataloader,
        residual_stream_type=RESIDUAL_STREAM_TYPE,
        tokens_pooling_method=pooling_method,
        return_deepcopy=True,
        save_path=repr_path,
        save_representations=args.save_representations,
        save_metadata=metadata,
    )

    # Cleanup
    del model, processor, dataloader
    gc.collect()
    torch.cuda.empty_cache()

    return residual_stream


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze intra-model residual streams similarity (LxL matrix)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument(
        "--representations-dir",
        type=str,
        required=True,
        help="Base directory for storing representations (e.g., $HOME/scratch)",
    )
    parser.add_argument("--dataset-path-or-name", type=str, required=True, dest="dataset_path_or_name")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of models to analyze (one by one)",
    )

    # Optional arguments
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--guide-text",
        type=str,
        default="Answer the question using a single word or phrase.\n",
        help="Guide text for the QA task",
    )
    parser.add_argument(
        "--downsample-size",
        type=int,
        default=10000,
        help="Number of samples to use from the dataset",
    )
    parser.add_argument(
        "--pooling-methods",
        nargs="+",
        default=TOKENS_POOLING_METHODS_DEFAULT,
        help="List of pooling methods to use (e.g., last mean_image)"
    )
    parser.add_argument(
        "--similarity-measures",
        nargs="+",
        default=SIMILARITY_MEASURES_DEFAULT,
        help="List of similarity measures to use.",
    )
    parser.add_argument(
        "--save-representations",
        action="store_true",
        help="Whether to save the extracted representations to disk (default: False)",
    )
    parser.add_argument(
        "--lower-triangular-only",
        action="store_true",
        help="Compute only the lower triangular part of the similarity matrix (i > j). Useful for symmetric intra-model analysis.",
    )
    parser.add_argument(
        "--dataset-splits",
        nargs="+",
        default=["images_qa", "texts_qa"],
        choices=["images_qa", "texts_qa"],
        help="List of dataset splits to process (e.g. images_qa texts_qa)"
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Define sub-directory
    analysis_dir = os.path.join(results_dir, "model_comparison", "intra_model_similarities")
    os.makedirs(analysis_dir, exist_ok=True)

    # Environment variables for cache
    cache_dir = os.environ.get("HF_HUB_CACHE")
    datasets_cache_dir = os.environ.get("HF_DATASETS_CACHE")
    
    # Dataset slug derivation (aligned with models_similarity.py)
    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace(".", "_").replace("-", "_").lower()

    seed_all(args.seed)

    for pooling_method in args.pooling_methods:
        print(f"\n🏊 Pooling Method: {pooling_method}")

        for model_path in args.models:
            model_id = model_path.split("/")[-1]
            print(f"  🌈 Processing Model: {model_id}")

            # Iterate over requested splits
            for split_name in args.dataset_splits:
                # Construct dataset name and config based on split
                if split_name == "images_qa":
                    current_dataset_name = f"{dataset_slug}_img"
                    dataset_config = {
                        "images_qa": True,
                        "guide_text": args.guide_text,
                        "downsample_size": args.downsample_size,
                    }
                elif split_name == "texts_qa":
                    current_dataset_name = f"{dataset_slug}_txt"
                    dataset_config = {
                        "texts_qa": True,
                        "guide_text": args.guide_text,
                        "downsample_size": args.downsample_size,
                    }
                else:
                    continue

                print(f"    🧩 Processing Split: {split_name} -> {current_dataset_name}")

                # Check for existing results first
                missing_measures = []
                for measure_name in args.similarity_measures:
                    filename = f"{model_id}_{current_dataset_name}_{pooling_method}_{measure_name}_intra_similarity.safetensors"
                    save_path = os.path.join(analysis_dir, filename)
                    if not os.path.exists(save_path):
                        missing_measures.append(measure_name)
                    else:
                        print(f"      ✅ Found existing: {filename}")

                if not missing_measures:
                    print("      ⏩ All measures computed for this setting. Skipping...")
                    continue

                # 1. Extract (or Load) Representation
                residual_stream = extract_representation(
                    model_path=model_path,
                    dataset_config=dataset_config,
                    dataset_name=current_dataset_name,
                    cache_dir=cache_dir,
                    datasets_cache_dir=datasets_cache_dir,
                    results_dir=results_dir,
                    pooling_method=pooling_method,
                    args=args
                )

                # 2. Compute Similarity (Model vs Itself = LxL Matrix)
                print("      🧮 Computing Self-Similarity Matrix (LxL)...")
                
                # Verify shape
                print(f"      Tensor shape: {residual_stream.shape}")

                metadata = {
                    "model": model_id,
                    "dataset": current_dataset_name,
                    "pooling_method": pooling_method,
                    "type": RESIDUAL_STREAM_TYPE,
                    "lower_triangular_only": str(args.lower_triangular_only),
                }

                # Loop over measures to save incrementally
                for measure_name in missing_measures:
                    print(f"        👉  Computing: {measure_name}")
                    try:
                        # Compute single measure
                        # kwargs passed to the utils function
                        sim_results = compute_cross_layers_residual_stream_similarities(
                            residual_stream_network1=residual_stream,
                            residual_stream_network2=residual_stream,  # Same tensor for intra-model
                            similarity_measures=[measure_name],
                            lower_triangular_only=args.lower_triangular_only,
                            maxk=30,
                        )

                        measure_val = sim_results[measure_name]

                        # 3. Save Results
                        fname = f"{model_id}_{current_dataset_name}_{pooling_method}_{measure_name}_intra_similarity.safetensors"
                        save_path = os.path.join(analysis_dir, fname)

                        print(f"      💾 Saving similarities to: {save_path}")
                        save_file({f"{measure_name}": measure_val}, save_path, metadata=metadata)
                    except Exception as e:
                        print(f"        ❌ Error computing {measure_name}: {type(e).__name__}: {e}")
                        print(f"        ⚠️  Skipping {measure_name} and continuing with next measure...")

                # Cleanup
                del residual_stream
                gc.collect()
                torch.cuda.empty_cache()

    print("\n✅ Intra-model similarity analysis completed!")


if __name__ == "__main__":
    main()
