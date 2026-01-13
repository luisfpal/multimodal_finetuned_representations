import argparse
import gc
import os
import time
from typing import Dict, List

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
SIMILARITY_MEASURES = ["neighborhood_overlap", "linear_cka"]

COCOQA_DATASET_ARGS = {
    "guide_text": "Answer the question using a single word or phrase.\n",
    "downsample_size": 2500,
}

DATASETS_CONFIG = {
    "cocoqa_txt": {
        "texts_qa": True,
        **COCOQA_DATASET_ARGS,
        "chat_mode": True,
    },
    "cocoqa_img": {
        "images_qa": True,
        **COCOQA_DATASET_ARGS,
        "chat_mode": True,
    },
}


def extract_representation(
    model_path: str,
    dataset_config: Dict,
    dataset_name: str,
    cache_dir: str,
    datasets_cache_dir: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    """
    Loads model/processor, loads dataloader, extracts representation, cleans up.
    Returns the residual stream tensor on CPU.
    """
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

    print(
        f"  👉 Extracting Representations ({RESIDUAL_STREAM_TYPE} | {TOKENS_POOLING_METHOD})..."
    )
    residual_stream = residual_stream_tracer(
        model=model,
        processed_dataloader=dataloader,
        residual_stream_type=RESIDUAL_STREAM_TYPE,
        tokens_pooling_method=TOKENS_POOLING_METHOD,
        return_deepcopy=True,
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
    parser.add_argument("--dataset-path-or-name", type=str, required=True)
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

    # Iterate over datasets
    for dataset_name, dataset_config in DATASETS_CONFIG.items():
        print(f"\n{'-' * 80}")
        print(f"🌍 Dataset: {dataset_name}")
        print(f"{'-' * 80}")

        # Phase 1: Model A
        print(f"\n[Phase 1] Processing Model A: {model_a_id}")
        res_a = extract_representation(
            model_path=args.model_a_name_or_path,
            dataset_config=dataset_config,
            dataset_name=dataset_name,
            cache_dir=model_cache_dir,
            datasets_cache_dir=datasets_cache_dir,
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
            args=args,
        )

        # Phase 3: Compute Similarity
        print(f"\n[Phase 3] Computing Similarities: {model_a_id} vs {model_b_id}")
        print(f"    - Measures: {SIMILARITY_MEASURES}")

        sim_results = compute_cross_layers_residual_stream_similarities(
            res_a,
            res_b,
            similarity_measures=SIMILARITY_MEASURES,
            maxk=30,
        )

        # Save Results
        metadata = {
            "row_model": model_a_id,
            "col_model": model_b_id,
            "dataset": dataset_name,
            "description": f"Rows: {model_a_id}, Cols: {model_b_id}",
        }
        print(f"    ℹ️  Matrix orientation: Rows={model_a_id}, Columns={model_b_id}")

        for measure_name, measure_val in sim_results.items():
            # Filename: {dataset}_{model_a}_vs_{model_b}_{measure}.safetensors
            filename = f"{dataset_name}_{model_a_id}_vs_{model_b_id}_{measure_name}.safetensors"
            save_path = os.path.join(similarities_dir, filename)

            save_file({f"{measure_name}": measure_val}, save_path, metadata=metadata)
            print(f"    💾 Saved: {save_path}")

        # Cleanup representations
        del res_a, res_b
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ All analyses complete. Results saved to {similarities_dir}")


if __name__ == "__main__":
    main()
