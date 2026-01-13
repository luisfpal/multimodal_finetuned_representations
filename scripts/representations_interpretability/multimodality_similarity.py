import argparse
import gc
import os
from typing import List

import torch
from dotenv import load_dotenv
from safetensors.torch import save_file

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
MODEL_DTYPE = torch.float16
ATTN_IMPLEMENTATION = "flash_attention_2"
RESIDUAL_STREAM_TYPE = "output_layer"
TOKENS_POOLING_METHOD = "last"
SIMILARITY_MEASURES = ["neighborhood_overlap", "linear_cka", "distance_correlation"]

COCOQA_DATASET_ARGS = {
    "guide_text": "Answer the question using a single word or phrase.\n",
    "downsample_size": 2500,
}

DATASETS_CONFIG = {
    "cocoqa_txt": {
        "texts_qa": True,
        **COCOQA_DATASET_ARGS,
    },
    "cocoqa_img": {
        "images_qa": True,
        **COCOQA_DATASET_ARGS,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze residual streams for multimodality similarity (Text vs Image)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--dataset-path-or-name", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)

    # Optional arguments
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Create results directory
    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Define sub-directory for this specific analysis
    analysis_dir = os.path.join(results_dir, "cocoqa_img_vs_txt")
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
    }

    # Text Dataloader
    print("  - Loading cocoqa_txt...")
    dl_txt = get_dataloader(**DATASETS_CONFIG["cocoqa_txt"], **dl_args)

    # Image Dataloader
    print("  - Loading cocoqa_img...")
    dl_img = get_dataloader(**DATASETS_CONFIG["cocoqa_img"], **dl_args)

    print(f"\n{'-' * 80}")
    print(f"🔍 Analysis: {RESIDUAL_STREAM_TYPE} | {TOKENS_POOLING_METHOD}")
    print(f"{'-' * 80}")

    # 1. Extract Text Representations
    print("  👉 Extracting Text Representations...")
    res_txt = residual_stream_tracer(
        model=model,
        processed_dataloader=dl_txt,
        residual_stream_type=RESIDUAL_STREAM_TYPE,
        tokens_pooling_method=TOKENS_POOLING_METHOD,
        return_deepcopy=True,
    )

    # 2. Extract Image Representations
    print("  👉 Extracting Image Representations...")
    res_img = residual_stream_tracer(
        model=model,
        processed_dataloader=dl_img,
        residual_stream_type=RESIDUAL_STREAM_TYPE,
        tokens_pooling_method=TOKENS_POOLING_METHOD,
        return_deepcopy=True,
    )

    # 3. Compute Similarities
    print("  👉 Computing Similarities...")
    print(f"    - Measures: {SIMILARITY_MEASURES}")

    sim_results = compute_cross_layers_residual_stream_similarities(
        res_txt,
        res_img,
        similarity_measures=SIMILARITY_MEASURES,
        maxk=30,  # Default
    )

    # Save each measure
    metadata = {
        "row_modality": "text",
        "col_modality": "image",
        "description": "Rows represent text layers, Columns represent image layers",
    }
    print("    ℹ️  Matrix orientation: Rows=Text, Columns=Image")

    for measure_name, measure_val in sim_results.items():
        # Simplified filename: {model_name}_{measure_name}.safetensors
        save_path = os.path.join(
            similarities_dir,
            f"{model_name}_{measure_name}.safetensors",
        )
        save_file({f"{measure_name}": measure_val}, save_path, metadata=metadata)
        print(f"    💾 Saved: {save_path}")

    # Cleanup
    del res_txt, res_img
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n✅ All analyses complete. Results saved to {similarities_dir}")


if __name__ == "__main__":
    main()
