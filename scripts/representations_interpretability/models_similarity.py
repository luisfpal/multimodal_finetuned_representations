"""Compute cross-layer residual-stream similarity measures between model representations.

Reads environment variables from .env (load_dotenv). Results are saved as safetensors under the results directory.
"""

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
    # "distance_correlation",
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
    """
    Loads model/processor, loads dataloader, extracts representation, cleans up.
    Returns the residual stream tensor.
    """
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
            # residual_stream_tracer saves with key "residual_stream" or acts as dict for 'none' pooling
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
        "split": "train",
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
        description="Extract and analyze residual streams similarity between two models (Pretrained vs Finetuned)",
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
    parser.add_argument(
        "--dataset-path-or-name", type=str, required=True, dest="dataset_path_or_name"
    )
    parser.add_argument(
        "--model-a-name-or-path",
        type=str,
        required=True,
        help="Path/Name of Model A (rows)",
    )
    parser.add_argument(
        "--model-b-name-or-path",
        type=str,
        required=True,
        help="Path/Name of Model B (columns)",
    )

    # Optional arguments
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pooling-methods",
        nargs="+",
        default=TOKENS_POOLING_METHODS_DEFAULT,
        help="List of pooling methods to use (e.g., last mean_image)",
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
        "--guide-text",
        type=str,
        default="Answer the question using a single word or phrase.\n",
        help="Guide text for the dataset",
    )
    parser.add_argument(
        "--downsample-size",
        type=int,
        default=100,
        help="Downsample size for the dataset",
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Define sub-directory for this specific analysis
    analysis_dir = os.path.join(results_dir, "model_comparison")
    similarities_dir = os.path.join(analysis_dir, "similarities")
    os.makedirs(similarities_dir, exist_ok=True)

    # Set seed
    seed_all(args.seed)

    # Environment Paths
    model_cache_dir = os.getenv(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE")
    print(f"📂 Model Cache: {model_cache_dir}")

    # Model identifiers
    model_a_id = args.model_a_name_or_path.split("/")[-1]
    model_b_id = args.model_b_name_or_path.split("/")[-1]

    # Dataset Setup - automatically detect prefix from dataset name
    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    prefix = dataset_slug

    current_datasets_config = {
        f"{prefix}_txt": {
            "texts_qa": True,
            "guide_text": args.guide_text,
            "downsample_size": args.downsample_size,
        },
        f"{prefix}_img": {
            "images_qa": True,
            "guide_text": args.guide_text,
            "downsample_size": args.downsample_size,
        },
    }

    # Iterate over pooling methods
    for pooling_method in args.pooling_methods:
        # Iterate over datasets
        for dataset_name, dataset_config in current_datasets_config.items():
            print(f"\n{'-' * 80}")
            print(f"🌍 Dataset: {dataset_name} | Pooling: {pooling_method}")
            print(f"{'-' * 80}")

            # Check for existing results
            missing_measures = []
            for measure_name in args.similarity_measures:
                filename = f"{dataset_name}_{model_a_id}_vs_{model_b_id}_{pooling_method}_{measure_name}.safetensors"
                save_path = os.path.join(similarities_dir, filename)
                if not os.path.exists(save_path):
                    missing_measures.append(measure_name)
                else:
                    print(f"    ✅ Found existing: {filename}")

            if not missing_measures:
                print("    ⏩ All measures computed for this setting. Skipping...")
                continue

            print(f"    ⚠️  Missing measures: {missing_measures}")

            # Phase 1: Model A
            print(f"\n[Phase 1] Processing Model A: {model_a_id}")
            res_a = extract_representation(
                model_path=args.model_a_name_or_path,
                dataset_config=dataset_config,
                dataset_name=dataset_name,
                cache_dir=model_cache_dir,
                datasets_cache_dir=datasets_cache_dir,
                results_dir=results_dir,
                pooling_method=pooling_method,
                args=args,
            )

            # Phase 2: Model B
            print(f"\n[Phase 2] Processing Model B: {model_b_id}")
            res_b = extract_representation(
                model_path=args.model_b_name_or_path,
                dataset_config=dataset_config,
                dataset_name=dataset_name,
                cache_dir=model_cache_dir,
                datasets_cache_dir=datasets_cache_dir,
                results_dir=results_dir,
                pooling_method=pooling_method,
                args=args,
            )

            # Phase 3: Compute Similarity
            print(f"\n[Phase 3] Computing Similarities: {model_a_id} vs {model_b_id}")

            # Save Results Metadata
            metadata = {
                "row_model": model_a_id,
                "col_model": model_b_id,
                "dataset": dataset_name,
                "pooling_method": pooling_method,
                "description": f"Rows: {model_a_id}, Cols: {model_b_id}, Pooling: {pooling_method}",
            }
            print(f"    ℹ️  Matrix orientation: Rows={model_a_id}, Columns={model_b_id}")

            # Compute one by one
            for measure_name in missing_measures:
                print(f"    👉  Computing: {measure_name}")
                try:
                    sim_results = compute_cross_layers_residual_stream_similarities(
                        res_a,
                        res_b,
                        similarity_measures=[measure_name],
                        maxk=30,
                    )

                    measure_val = sim_results[measure_name]

                    # Filename: {dataset}_{model_a}_vs_{model_b}_{pooling}_{measure}.safetensors
                    filename = f"{dataset_name}_{model_a_id}_vs_{model_b_id}_{pooling_method}_{measure_name}.safetensors"
                    save_path = os.path.join(similarities_dir, filename)

                    save_file(
                        {f"{measure_name}": measure_val}, save_path, metadata=metadata
                    )
                    print(f"    💾 Saved: {save_path}")
                except Exception as e:
                    print(f"    ❌ Error computing {measure_name}: {type(e).__name__}: {e}")
                    print(f"    ⚠️  Skipping {measure_name} and continuing with next measure...")

            # Cleanup representations
            del res_a, res_b
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n✅ All analyses complete. Results saved to {similarities_dir}")


if __name__ == "__main__":
    main()
