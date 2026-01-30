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
SIMILARITY_MEASURES_DEFAULT = [
    "neighborhood_overlap",
    "linear_cka",
    "distance_correlation",
    "id_correlation",
    "information_imbalance",
    "information_imbalance_reversed",
    "information_imbalance_mean",
]

# Specific pooling configuration requested
# texts_qa -> mean
# images_qa -> mean_image
POOLING_CONFIG = {
    "texts_qa": "mean",
    "images_qa": "mean_image"
}


def extract_representation(
    model,
    processor,
    dataset_split: str,
    dataset_name: str,
    pooling_method: str,
    args: argparse.Namespace,
    datasets_cache_dir: str,
    results_dir: str,
    model_name: str,
) -> torch.Tensor:
    
    # Use configurable representations directory
    repr_base_dir = os.path.expanduser(args.representations_dir)
    repr_dir = os.path.join(
        repr_base_dir,
        "representations",
        model_name,
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
                if tensor.ndim >= 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                    print(f"  ✅ Loaded successfully. Shape: {tensor.shape}")
                    return tensor[:]
                else:
                    print(f"  ⚠️ Invalid tensor shape: {tensor.shape}. Re-extracting...")
            else:
                print("  ⚠️ Key 'residual_stream' not found. Re-extracting...")
        except Exception as e:
            print(f"  ⚠️ Error loading cache: {type(e).__name__}: {e}. Re-extracting...")

    # Prepare dataloader
    print(f"  📚 Loading dataset split: {dataset_split} for {dataset_name}")
    dl_args = {
        "dataset_path_or_name": args.dataset_path_or_name,
        "seed": args.seed,
        "processor": processor,
        "batch_size": args.batch_size,
        "cache_dir": datasets_cache_dir,
        "guide_text": args.guide_text,
        "downsample_size": args.downsample_size,
    }
    
    # Configure dataloader based on split
    if dataset_split == "texts_qa":
        dataloader = get_dataloader(texts_qa=True, **dl_args)
    elif dataset_split == "images_qa":
        dataloader = get_dataloader(images_qa=True, **dl_args)
    else:
        raise ValueError(f"Unknown dataset split: {dataset_split}")

    metadata = {
        "model": model_name,
        "dataset": dataset_name,
        "pooling_method": pooling_method,
        "type": RESIDUAL_STREAM_TYPE,
        "split": dataset_split
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
    
    del dataloader
    return residual_stream


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze residual streams similarity with specific pooling per split",
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
        default=100,
        help="Number of samples to use from the dataset",
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
        help="Whether to save the extracted representations to disk",
    )
    parser.add_argument(
        "--dataset-splits",
        nargs="+",
        default=["texts_qa", "images_qa"],
        choices=["texts_qa", "images_qa"],
        help="List of dataset splits to process (required for determining pooling methods)"
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Dataset slug derivation
    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace(".", "_").replace("-", "_").lower()
    dataset_prefix = dataset_slug

    analysis_dir = os.path.join(results_dir, f"{dataset_prefix}_img_vs_txt_specific_pooling")
    similarities_dir = os.path.join(analysis_dir, "similarities")
    os.makedirs(similarities_dir, exist_ok=True)

    # Set seed
    seed_all(args.seed)

    # Load Model
    print(f"\n🤖 Loading model: {args.model_name_or_path}")
    model_cache_dir = os.getenv(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
        cache_dir=model_cache_dir,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="cuda:0",
    )
    
    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE")
    model_name = args.model_name_or_path.split("/")[-1]

    # Verify we have both splits if we want to calculate similarity
    # If user provides only one, we can't compute cross-modality similarity
    # But sticking to the "multimodality_similarity.py" logic, we assume we need both.
    
    required_splits = ["texts_qa", "images_qa"]
    for split in required_splits:
        if split not in args.dataset_splits:
            print(f"⚠️ Warning: Split '{split}' missing from arguments. Cross-modality similarity requires both texts_qa and images_qa.")

    # 1. Extract Text Representation (if requested)
    res_txt = None
    txt_pooling = POOLING_CONFIG["texts_qa"]
    if "texts_qa" in args.dataset_splits:
        print(f"\nProcessing Text Split (Pooling: {txt_pooling})")
        res_txt = extract_representation(
            model=model,
            processor=processor,
            dataset_split="texts_qa",
            dataset_name=f"{dataset_prefix}_txt",
            pooling_method=txt_pooling,
            args=args,
            datasets_cache_dir=datasets_cache_dir,
            results_dir=results_dir,
            model_name=model_name
        )

    # 2. Extract Image Representation (if requested)
    res_img = None
    img_pooling = POOLING_CONFIG["images_qa"]
    if "images_qa" in args.dataset_splits:
        print(f"\nProcessing Image Split (Pooling: {img_pooling})")
        res_img = extract_representation(
            model=model,
            processor=processor,
            dataset_split="images_qa",
            dataset_name=f"{dataset_prefix}_img",
            pooling_method=img_pooling,
            args=args,
            datasets_cache_dir=datasets_cache_dir,
            results_dir=results_dir,
            model_name=model_name
        )

    # 3. Compute Similarities if both exist
    if res_txt is not None and res_img is not None:
        print(f"\n{'-' * 80}")
        print(f"🔍 Analysis: Text ({txt_pooling}) vs Image ({img_pooling})")
        print(f"{'-' * 80}")

        missing_measures = []
        filename_base = f"{model_name}_txt-{txt_pooling}_vs_img-{img_pooling}"
        
        for measure_name in args.similarity_measures:
            filename = f"{filename_base}_{measure_name}.safetensors"
            save_path = os.path.join(similarities_dir, filename)
            if not os.path.exists(save_path):
                missing_measures.append(measure_name)
            else:
                print(f"    ✅ Found existing: {filename}")

        if not missing_measures:
            print("    ⏩ All measures computed. Skipping...")
        else:
            print(f"    ⚠️  Missing measures: {missing_measures}")
            print("    👉 Computing Similarities...")
            
            metadata = {
                "row_modality": "text",
                "col_modality": "image",
                "row_pooling": txt_pooling,
                "col_pooling": img_pooling,
                "description": f"Text({txt_pooling}) vs Image({img_pooling})",
            }
            
            for measure_name in missing_measures:
                print(f"    👉  Computing: {measure_name}")
                try:
                    sim_results = compute_cross_layers_residual_stream_similarities(
                        res_txt,
                        res_img,
                        similarity_measures=[measure_name],
                        maxk=30,
                    )
                    measure_val = sim_results[measure_name]

                    save_path = os.path.join(similarities_dir, f"{filename_base}_{measure_name}.safetensors")
                    save_file({f"{measure_name}": measure_val}, save_path, metadata=metadata)
                    print(f"    💾 Saved: {save_path}")
                except Exception as e:
                    print(f"    ❌ Error computing {measure_name}: {type(e).__name__}: {e}")

    # Cleanup
    if res_txt is not None: del res_txt
    if res_img is not None: del res_img
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n✅ All analyses complete. Results saved to {similarities_dir}")


if __name__ == "__main__":
    main()
