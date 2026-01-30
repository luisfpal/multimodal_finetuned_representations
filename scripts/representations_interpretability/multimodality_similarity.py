import argparse
import gc
import os

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


def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument(
        "--representations-dir",
        type=str,
        required=True,
        help="Base directory for storing representations (e.g., $HOME/scratch)",
    )
    parser.add_argument("--dataset-path-or-name", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)

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
        default=100
    )
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
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Define sub-directory for this specific analysis
    # Dataset Setup - automatically detect prefix from dataset name
    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    dataset_prefix = dataset_slug

    analysis_dir = os.path.join(results_dir, "multimodality_comparison", f"{dataset_prefix}_img_vs_txt")
    similarities_dir = os.path.join(analysis_dir, "similarities")
    os.makedirs(similarities_dir, exist_ok=True)

    # Set seed
    seed_all(args.seed)

    # Load Model
    print(f"\n🤖 Loading model: {args.model_name_or_path}")

    # Get cache dir from env, default to ~/.cache/huggingface/hub if not set
    model_cache_dir = os.getenv(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    print(f"📂 Using model cache directory: {model_cache_dir}")

    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
        cache_dir=model_cache_dir,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="cuda:0",  # Explicitly set device map
    )

    # Identify model name for filename
    model_name = args.model_name_or_path.split("/")[-1]

    # Prepare Dataloaders
    print("\n📚 Preparing dataloaders...")

    # Get datasets cache dir from env
    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE")

    # Common args for dataloader
    dl_args = {
        "dataset_path_or_name": args.dataset_path_or_name,
        "seed": args.seed,
        "processor": processor,
        "batch_size": args.batch_size,
        "cache_dir": datasets_cache_dir,
        "guide_text": args.guide_text,
        "downsample_size": args.downsample_size,
        "split": "train",
    }
    
    if "intern" in model.__class__.__name__.lower():
        dl_args['model'] = model

    # Text Dataloader
    print(f"  - Loading {dataset_prefix}_txt...")
    dl_txt = get_dataloader(texts_qa=True, **dl_args)

    # Image Dataloader
    print(f"  - Loading {dataset_prefix}_img...")
    dl_img = get_dataloader(images_qa=True, **dl_args)

    for pooling_method in args.pooling_methods:
        print(f"\n{'-' * 80}")
        print(f"🔍 Analysis: {RESIDUAL_STREAM_TYPE} | {pooling_method}")
        print(f"{'-' * 80}")

        # Check for existing results first to avoid unnecessary loading
        missing_measures = []
        for measure_name in args.similarity_measures:
            filename = f"{model_name}_{pooling_method}_{measure_name}.safetensors"
            save_path = os.path.join(similarities_dir, filename)
            if not os.path.exists(save_path):
                missing_measures.append(measure_name)
            else:
                print(f"    ✅ Found existing: {filename}")

        if not missing_measures:
            print(f"    ⏩ All measures computed for {pooling_method}. Skipping...")
            continue

        print(f"    ⚠️  Missing measures: {missing_measures}")

        # helper for path construction
        def get_repr_path(dataset_n):
            # Use configurable representations directory
            repr_base_dir = os.path.expanduser(args.representations_dir)

            repr_dir = os.path.join(
                repr_base_dir,
                "representations",
                model_name,
                dataset_n,
                RESIDUAL_STREAM_TYPE,
                pooling_method,
            )
            os.makedirs(repr_dir, exist_ok=True)
            return os.path.join(repr_dir, "representations.safetensors")

        # 1. Extract Text Representations
        print("  👉 Extracting Text Representations...")
        txt_metadata = {
            "model": model_name,
            "dataset": f"{dataset_prefix}_txt",
            "pooling_method": pooling_method,
            "type": RESIDUAL_STREAM_TYPE,
        }
        repr_path_txt = get_repr_path(f"{dataset_prefix}_txt")

        # Try loading cached representation first
        res_txt = None
        if os.path.exists(repr_path_txt):
            print(f"  💾 Found cached text representation at: {repr_path_txt}")
            try:
                loaded = load_file(repr_path_txt)
                if "residual_stream" in loaded:
                    tensor = loaded["residual_stream"]
                    if tensor.ndim >= 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                        print(f"  ✅ Loaded successfully. Shape: {tensor.shape}")
                        res_txt = tensor[:]
                    else:
                        print(
                            f"  ⚠️ Invalid tensor shape: {tensor.shape}. Re-extracting..."
                        )
                else:
                    print("  ⚠️ Key 'residual_stream' not found. Re-extracting...")
            except Exception as e:
                print(
                    f"  ⚠️ Error loading cache: {type(e).__name__}: {e}. Re-extracting..."
                )

        if res_txt is None:
            res_txt = residual_stream_tracer(
                model=model,
                processed_dataloader=dl_txt,
                residual_stream_type=RESIDUAL_STREAM_TYPE,
                tokens_pooling_method=pooling_method,
                return_deepcopy=True,
                save_path=repr_path_txt,
                save_representations=args.save_representations,
                save_metadata=txt_metadata,
            )

        # 2. Extract Image Representations
        print("  👉 Extracting Image Representations...")
        img_metadata = {
            "model": model_name,
            "dataset": f"{dataset_prefix}_img",
            "pooling_method": pooling_method,
            "type": RESIDUAL_STREAM_TYPE,
        }
        repr_path_img = get_repr_path(f"{dataset_prefix}_img")

        # Try loading cached representation first
        res_img = None
        if os.path.exists(repr_path_img):
            print(f"  💾 Found cached image representation at: {repr_path_img}")
            try:
                loaded = load_file(repr_path_img)
                if "residual_stream" in loaded:
                    tensor = loaded["residual_stream"]
                    if tensor.ndim >= 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                        print(f"  ✅ Loaded successfully. Shape: {tensor.shape}")
                        res_img = tensor[:]
                    else:
                        print(
                            f"  ⚠️ Invalid tensor shape: {tensor.shape}. Re-extracting..."
                        )
                else:
                    print("  ⚠️ Key 'residual_stream' not found. Re-extracting...")
            except Exception as e:
                print(
                    f"  ⚠️ Error loading cache: {type(e).__name__}: {e}. Re-extracting..."
                )

        if res_img is None:
            res_img = residual_stream_tracer(
                model=model,
                processed_dataloader=dl_img,
                residual_stream_type=RESIDUAL_STREAM_TYPE,
                tokens_pooling_method=pooling_method,
                return_deepcopy=True,
                save_path=repr_path_img,
                save_representations=args.save_representations,
                save_metadata=img_metadata,
            )

        # 3. Compute Similarities
        print("  👉 Computing Similarities...")

        # Save each measure
        metadata = {
            "row_modality": "text",
            "col_modality": "image",
            "description": "Rows represent text layers, Columns represent image layers",
            "pooling_method": pooling_method,
        }
        print("    ℹ️  Matrix orientation: Rows=Text, Columns=Image")

        for measure_name in missing_measures:
            print(f"    👉  Computing: {measure_name}")
            try:
                sim_results = compute_cross_layers_residual_stream_similarities(
                    res_txt,
                    res_img,
                    similarity_measures=[measure_name],
                    maxk=30,  # Default
                )
                measure_val = sim_results[measure_name]

                # Filename: {model_name}_{pooling_method}_{measure_name}.safetensors
                save_path = os.path.join(
                    similarities_dir,
                    f"{model_name}_{pooling_method}_{measure_name}.safetensors",
                )
                save_file(
                    {f"{measure_name}": measure_val}, save_path, metadata=metadata
                )
                print(f"    💾 Saved: {save_path}")
            except Exception as e:
                print(f"    ❌ Error computing {measure_name}: {type(e).__name__}: {e}")
                print(
                    f"    ⚠️  Skipping {measure_name} and continuing with next measure..."
                )

        # Cleanup
        del res_txt, res_img
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ All analyses complete. Results saved to {similarities_dir}")


if __name__ == "__main__":
    main()
