import argparse
from ast import arg
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
from utils.dataset import OpenVQADataset
from utils.semantic_filter_utils import (
    JinaCaptionSemanticScorer,
    compute_semantic_keep_image_ids,
)

# Load environment variables
load_dotenv()

# Configuration
MODEL_DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "flash_attention_2"
RESIDUAL_STREAM_TYPE = "output_layer"
TOKENS_POOLING_METHODS_DEFAULT = ["last", "mean_context"]
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
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--representations-dir",
        type=str,
        default=None,
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
        default="Answer the question using a single word or phrase.",
        help="Guide text for the QA task",
    )
    parser.add_argument("--downsample-size", type=int, default=100)
    parser.add_argument(
        "--pooling-methods",
        nargs="+",
        default=TOKENS_POOLING_METHODS_DEFAULT,
        help="List of pooling methods to use (last or mean_context)",
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
    parser.add_argument(
        "--caption-source",
        type=str,
        default="synthetic",
        choices=["human", "synthetic"],
        help=(
            "Caption source: 'human' for datasets with human captions (COCOQA, VQAv2-ICQ), "
            "'synthetic' for datasets with only synthetic captions (MME, MMBench, GQA, etc.)."
        ),
    )
    parser.add_argument(
        "--enable-semantic-filter",
        action="store_true",
        help=(
            "Enable optional semantic filtering with Jina on CPU. "
            "Keeps samples with similarity >= mean - std."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force recomputation: skip all 'already exists' checks and overwrite "
            "both cached representations and similarity results. Unlike --dry-run, "
            "results ARE saved. Useful when you want to rerun with updated code."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip ALL caching: no load, no save, no skip existing. For debugging.",
    )

    args = parser.parse_args()

    # Validate dirs: required unless --dry-run
    if args.dry_run:
        print("\n🧪 DRY-RUN MODE — no caching, no saving, fresh computation only")
        if args.results_dir is None:
            args.results_dir = "/tmp/debug_run"
        if args.representations_dir is None:
            args.representations_dir = "/tmp/debug_run"
    else:
        if args.results_dir is None or args.representations_dir is None:
            parser.error(
                "--results-dir and --representations-dir are required "
                "(unless --dry-run is set)"
            )

    if args.force:
        print("\n🔄 FORCE MODE — recomputing everything, existing files will be overwritten")

    # Create results directory
    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Dataset slug: lowercase, dashes→underscores
    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    # caption_source is always set (default="synthetic"), always part of dataset_prefix
    dataset_prefix = f"{dataset_slug}_{args.caption_source}"

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
        "caption_source": args.caption_source,
    }

    # Optional one-time semantic filtering shared by both dataloaders.
    # NOTE: semantic filter affects WHICH SAMPLES enter the model, so it must be
    # reflected in both the representation paths and the results paths.
    allowed_image_ids = None
    sem_tag = ""
    if args.enable_semantic_filter:
        print("\n🧹 Computing semantic keep-set with Jina (CPU)...")
        semantic_dataset = OpenVQADataset(
            dataset_path_or_name=args.dataset_path_or_name,
            cache_dir=datasets_cache_dir,
            split="train",
            downsample_size=args.downsample_size,
            seed=args.seed,
            images_qa=False,
            texts_qa=False,
            caption_source=args.caption_source,
        )
        scorer = JinaCaptionSemanticScorer(device="cpu")
        allowed_image_ids, semantic_stats = compute_semantic_keep_image_ids(
            dataset=semantic_dataset.dataset,
            caption_source=args.caption_source,
            scorer=scorer,
            threshold_std_multiplier=1.0,
        )
        sem_tag = "sf-jina_cpu_m1sd"
        print(
            "  ✅ Semantic filter stats: "
            f"kept={semantic_stats['num_kept']}/{semantic_stats['num_candidates']} | "
            f"mean={semantic_stats['mean_score']:.4f} | "
            f"std={semantic_stats['std_score']:.4f} | "
            f"threshold={semantic_stats['threshold']:.4f}"
        )

        dl_args["allowed_image_ids"] = allowed_image_ids

    if "intern" in model.__class__.__name__.lower():
        dl_args["model"] = model

    # Text Dataloader
    print(f"  - Loading {dataset_prefix}_txt...")
    dl_txt = get_dataloader(texts_qa=True, **dl_args)

    # Image Dataloader
    print(f"  - Loading {dataset_slug}_img...")
    dl_img = get_dataloader(images_qa=True, **dl_args)

    for pooling_method in args.pooling_methods:
        # ── Results directory ──────────────────────────────────────────────────
        # Structure: results/multimodality_comparison/{dataset}_{caption_source}/
        #              {pooling_method}/[sf-{tag}/]similarities/
        analysis_dir = os.path.join(
            results_dir,
            "multimodality_comparison",
            dataset_prefix,        # e.g. cocoqa_restval_icqa_human
            pooling_method,        # e.g. last | mean_context
            sem_tag,               # "" (no subdir) or "sf-jina_cpu_m1sd"
        )
        # os.path.join ignores empty string components correctly
        similarities_dir = os.path.join(analysis_dir, "similarities")
        os.makedirs(similarities_dir, exist_ok=True)

        pooling_method_txt = (
            "mean_text_context" if pooling_method == "mean_context" else pooling_method
        )
        pooling_method_img = (
            "mean_image" if pooling_method == "mean_context" else pooling_method
        )

        print(f"\n{'-' * 80}")
        print(f"🔍 Analysis: {RESIDUAL_STREAM_TYPE} | {pooling_method}")
        print(f"{'-' * 80}")

        # Check for existing results first to avoid unnecessary loading
        if args.dry_run or args.force:
            missing_measures = list(args.similarity_measures)
            if args.force:
                print(f"    🔄 Force mode: recomputing all {len(missing_measures)} measures")
        else:
            missing_measures = []
            for measure_name in args.similarity_measures:
                filename = f"{model_name}_{measure_name}.safetensors"
                save_path = os.path.join(similarities_dir, filename)
                if not os.path.exists(save_path):
                    missing_measures.append(measure_name)
                else:
                    print(f"    ✅ Found existing: {filename}")

            if not missing_measures:
                print(f"    ⏩ All measures computed for {pooling_method}. Skipping...")
                continue

        print(f"    ⚠️  Missing/forced measures: {missing_measures}")

        # helper for representation path construction
        # Structure: repr_base/representations/{model}/{dataset_name}/output_layer/
        #              {pooling_method}/[sf-{tag}/]representations.safetensors
        def get_repr_path(dataset_n):
            repr_base_dir = os.path.expanduser(args.representations_dir)
            repr_dir = os.path.join(
                repr_base_dir,
                "representations",
                model_name,
                dataset_n,
                RESIDUAL_STREAM_TYPE,
                pooling_method,
                sem_tag,           # "" (no subdir) or "sf-jina_cpu_m1sd"
            )
            os.makedirs(repr_dir, exist_ok=True)
            return os.path.join(repr_dir, "representations.safetensors")

        # ── Text Representations ───────────────────────────────────────────────
        print("  👉 Extracting Text Representations...")
        txt_metadata = {
            "model": model_name,
            "dataset": f"{dataset_prefix}_txt",
            "pooling_method": pooling_method_txt,
            "pooling_method_requested": pooling_method,
            "caption_source": args.caption_source,
            "semantic_filter": sem_tag,
            "type": RESIDUAL_STREAM_TYPE,
        }
        # Text repr: uses dataset_prefix (includes caption_source)
        repr_path_txt = get_repr_path(f"{dataset_prefix}_txt")

        # Try loading cached representation first (unless force/dry-run)
        res_txt = None
        if not args.dry_run and not args.force and os.path.exists(repr_path_txt):
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
            save_repr = args.save_representations and not args.dry_run
            res_txt = residual_stream_tracer(
                model=model,
                processed_dataloader=dl_txt,
                residual_stream_type=RESIDUAL_STREAM_TYPE,
                tokens_pooling_method=pooling_method_txt,
                return_deepcopy=True,
                save_path=repr_path_txt,
                save_representations=save_repr,
                save_metadata=txt_metadata,
                processor=processor,
            )

        # ── Image Representations ──────────────────────────────────────────────
        print("  👉 Extracting Image Representations...")
        img_metadata = {
            "model": model_name,
            "dataset": f"{dataset_slug}_img",
            "pooling_method": pooling_method_img,
            "pooling_method_requested": pooling_method,
            "caption_source": "n/a",   # images are caption-source-agnostic
            "semantic_filter": sem_tag,
            "type": RESIDUAL_STREAM_TYPE,
        }
        # Image repr: uses raw dataset_slug (no caption_source — images are caption-agnostic)
        repr_path_img = get_repr_path(f"{dataset_slug}_img")

        # Try loading cached representation first (unless force/dry-run)
        res_img = None
        if not args.dry_run and not args.force and os.path.exists(repr_path_img):
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
            save_repr = args.save_representations and not args.dry_run
            res_img = residual_stream_tracer(
                model=model,
                processed_dataloader=dl_img,
                residual_stream_type=RESIDUAL_STREAM_TYPE,
                tokens_pooling_method=pooling_method_img,
                return_deepcopy=True,
                save_path=repr_path_img,
                save_representations=save_repr,
                save_metadata=img_metadata,
                processor=processor,
            )

        # ── Compute Similarities ───────────────────────────────────────────────
        print("  👉 Computing Similarities...")

        metadata = {
            "row_modality": "text",
            "col_modality": "image",
            "description": "Rows represent text layers, Columns represent image layers",
            "pooling_method": pooling_method,
            "pooling_method_text": pooling_method_txt,
            "pooling_method_image": pooling_method_img,
            "caption_source": args.caption_source,
            "semantic_filter": sem_tag,
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
                print(f"    ✅ Shape: {measure_val.shape}")

                if not args.dry_run:
                    # Filename: {model}_{metric}.safetensors
                    # (pooling and semantic_filter captured by directory structure)
                    save_path = os.path.join(
                        similarities_dir,
                        f"{model_name}_{measure_name}.safetensors",
                    )
                    save_file(
                        {f"{measure_name}": measure_val},
                        save_path,
                        metadata=metadata,
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
