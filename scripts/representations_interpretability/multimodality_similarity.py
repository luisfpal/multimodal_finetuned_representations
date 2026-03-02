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
from utils.dataset import MULTIPLE_CHOICE_BENCHMARKS, get_hf_dataset_split
from utils.metrics_utils import preprocess_data
from utils.semantic_filter_utils import (
    CaptionSemanticScorer,
    compute_semantic_keep_indices,
    load_semantic_filter_cache,
    save_semantic_filter_cache,
)

load_dotenv()

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


def resolve_semantic_filter(
    args: argparse.Namespace,
    results_dir: str,
    dataset_slug: str,
    datasets_cache_dir: str | None,
) -> tuple[set[int] | None, str]:
    """Compute or load cached semantic filter indices.

    Returns (allowed_indices, sem_tag) where sem_tag is "" when filtering
    is disabled, or a path component like "sf-jina_m1sd" when active.
    """
    if not args.enable_semantic_filter:
        return None, ""

    sem_tag = "sf-jina_m1sd"
    cache_path = os.path.join(
        results_dir, "semantic_filter", dataset_slug, f"{sem_tag}_indices.json"
    )

    cached = None
    if not args.force and not args.dry_run:
        cached = load_semantic_filter_cache(cache_path)

    if cached is not None:
        allowed_indices, semantic_stats = cached
        print(f"\n💾 Loaded cached semantic filter: {cache_path}")
    else:
        print("\n🧹 Computing semantic keep-set (GPU)...")
        raw_dataset = get_hf_dataset_split(
            dataset_path_or_name=args.dataset_path_or_name,
            cache_dir=datasets_cache_dir,
            split="train",
        )
        scorer = CaptionSemanticScorer(device="cuda")
        allowed_indices, semantic_stats = compute_semantic_keep_indices(
            dataset=raw_dataset,
            scorer=scorer,
            threshold_std_multiplier=1.0,
        )

        if not args.dry_run:
            save_semantic_filter_cache(
                cache_path=cache_path,
                indices=allowed_indices,
                stats=semantic_stats,
                dataset_name=args.dataset_path_or_name,
                threshold_std_multiplier=1.0,
                num_dataset_rows=len(raw_dataset),
            )
            print(f"    💾 Cached indices to: {cache_path}")

        del scorer, raw_dataset
        gc.collect()
        torch.cuda.empty_cache()

    print(
        "  ✅ Semantic filter stats: "
        f"kept={semantic_stats['num_kept']}/{semantic_stats['num_candidates']} | "
        f"mean={semantic_stats['mean_score']:.4f} | "
        f"std={semantic_stats['std_score']:.4f} | "
        f"threshold={semantic_stats['threshold']:.4f}"
    )

    return allowed_indices, sem_tag


def load_or_extract_representation(
    path: str,
    dl,
    model,
    processor,
    pooling_method: str,
    dry_run: bool,
    force: bool,
) -> torch.Tensor:
    """Load representation from cache, or extract fresh if unavailable."""
    if not dry_run and not force and os.path.exists(path):
        print(f"  💾 Found cached representation at: {path}")
        try:
            loaded = load_file(path)
            if "residual_stream" in loaded:
                tensor = loaded["residual_stream"]
                if tensor.ndim >= 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                    print(f"  ✅ Loaded successfully. Shape: {tensor.shape}")
                    return tensor[:]
                print(f"  ⚠️ Invalid tensor shape: {tensor.shape}. Re-extracting...")
            else:
                print("  ⚠️ Key 'residual_stream' not found. Re-extracting...")
        except Exception as e:
            print(f"  ⚠️ Error loading cache: {type(e).__name__}: {e}. Re-extracting...")

    return residual_stream_tracer(
        model=model,
        processed_dataloader=dl,
        residual_stream_type=RESIDUAL_STREAM_TYPE,
        tokens_pooling_method=pooling_method,
        return_deepcopy=True,
        save_representations=False,
        processor=processor,
    )


def find_missing_measures(
    similarity_measures: list[str],
    similarities_dir: str,
    model_name: str,
    dry_run: bool,
    force: bool,
) -> list[str]:
    """Return measures not yet computed, or all measures if dry_run/force."""
    if dry_run or force:
        if force:
            print(
                f"    🔄 Force mode: recomputing all {len(similarity_measures)} measures"
            )
        return list(similarity_measures)

    missing = []
    for measure_name in similarity_measures:
        filename = f"{model_name}_{measure_name}.safetensors"
        if not os.path.exists(os.path.join(similarities_dir, filename)):
            missing.append(measure_name)
        else:
            print(f"    ✅ Found existing: {filename}")
    return missing


def preprocess_and_save_pair(
    res_txt: torch.Tensor,
    res_img: torch.Tensor,
    repr_path_txt: str,
    repr_path_img: str,
    txt_metadata: dict,
    img_metadata: dict,
    save: bool,
    dry_run: bool,
    force: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess both representations per layer (float32 + dedup), then save."""
    print("  👉 Preprocessing representations...")
    proc_txt, proc_img = [], []
    for l in range(res_txt.shape[0]):
        t, i = preprocess_data(
            res_txt[l], res_img[l], output_format="tensor", remove_duplicates=True
        )
        proc_txt.append(t.cpu())
        proc_img.append(i.cpu())
    res_txt = torch.stack(proc_txt, dim=0)
    res_img = torch.stack(proc_img, dim=0)
    print(f"  ✅ Shape after preprocessing: {res_txt.shape}")

    if save and not dry_run:
        for tensor, path, metadata in [
            (res_txt, repr_path_txt, txt_metadata),
            (res_img, repr_path_img, img_metadata),
        ]:
            if force or not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                valid_meta = {k: str(v) for k, v in metadata.items()}
                try:
                    save_file({"residual_stream": tensor}, path, metadata=valid_meta)
                    print(f"  💾 Saved: {path}")
                except Exception as e:
                    print(f"  ❌ Error saving {path}: {type(e).__name__}: {e}")

    return res_txt, res_img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--representations-dir",
        type=str,
        default=None,
        help="Base directory for storing representations (e.g., $HOME/scratch)",
    )
    parser.add_argument("--dataset-path-or-name", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--guide-text",
        type=str,
        default="Answer the question using a single word or phrase.",
        help="Guide text for the QA task",
    )
    parser.add_argument("--downsample-size", type=int, default=None)
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
    parser.add_argument("--save-representations", action="store_true")
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
    parser.add_argument(
        "--repeated-images",
        action="store_true",
        help=(
            "Run MC datasets (MMBench, MME, POPE, SEED) WITHOUT deduplication, "
            "keeping all samples including repeated images. Results land under "
            "repeated_images/. Default (without this flag) deduplicates MC datasets "
            "to unique images. Ignored for non-MC datasets (COCOQA, VQAv2, GQA) "
            "which are naturally unique."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        help=(
            "Name of a precomputed split (e.g. 'split01') for OpenVQA datasets. "
            "Mutates the dataset slug so results land in a separate directory."
        ),
    )

    args = parser.parse_args()

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
        print(
            "\n🔄 FORCE MODE — recomputing everything, existing files will be overwritten"
        )

    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    is_mc = any(b in dataset_slug for b in MULTIPLE_CHOICE_BENCHMARKS)

    if args.dataset_split:
        if is_mc:
            parser.error("--dataset-split is only valid for OpenVQA datasets, not MC.")
        if args.enable_semantic_filter:
            parser.error(
                "--dataset-split and --enable-semantic-filter are mutually exclusive "
                "(split indices reference the full dataset)."
            )
        dataset_slug = f"{dataset_slug}_{args.dataset_split}"

    sample_set_tag = "repeated_images" if (is_mc and args.repeated_images) else "unique_images"
    dataset_prefix = f"{dataset_slug}_{args.caption_source}"

    seed_all(args.seed)

    print(f"\n🤖 Loading model: {args.model_name_or_path}")
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
        device_map="cuda:0",
    )
    model_name = args.model_name_or_path.split("/")[-1]

    print("\n📚 Preparing dataloaders...")
    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE")
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

    allowed_indices, _ = resolve_semantic_filter(
        args, results_dir, dataset_slug, datasets_cache_dir
    )
    if allowed_indices is not None:
        dl_args["allowed_indices"] = allowed_indices
    if is_mc and not args.repeated_images:
        dl_args["unique_images"] = True
    if args.dataset_split:
        dl_args["dataset_split"] = args.dataset_split
    if "intern" in model.__class__.__name__.lower():
        dl_args["model"] = model

    print(f"  - Loading {dataset_prefix}_txt...")
    dl_txt = get_dataloader(texts_qa=True, **dl_args)
    print(f"  - Loading {dataset_prefix}_img...")
    dl_img = get_dataloader(images_qa=True, **dl_args)

    for pooling_method in args.pooling_methods:
        analysis_dir = os.path.join(
            results_dir,
            "multimodality_comparison",
            dataset_prefix,
            pooling_method,
            sample_set_tag,
        )
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

        missing_measures = find_missing_measures(
            args.similarity_measures,
            similarities_dir,
            model_name,
            args.dry_run,
            args.force,
        )
        if not missing_measures:
            print(f"    ⏩ All measures computed for {pooling_method}. Skipping...")
            continue
        print(f"    ⚠️  Missing/forced measures: {missing_measures}")

        # Closure: captures pooling_method, sample_set_tag from loop scope
        def get_repr_path(dataset_n: str) -> str:
            repr_dir = os.path.join(
                os.path.expanduser(args.representations_dir),
                "representations",
                model_name,
                dataset_n,
                RESIDUAL_STREAM_TYPE,
                pooling_method,
                sample_set_tag,
            )
            os.makedirs(repr_dir, exist_ok=True)
            return os.path.join(repr_dir, "representations.safetensors")

        repr_path_txt = get_repr_path(f"{dataset_prefix}_txt")
        repr_path_img = get_repr_path(f"{dataset_prefix}_img")

        print("  👉 Extracting Text Representations...")
        res_txt = load_or_extract_representation(
            repr_path_txt,
            dl_txt,
            model,
            processor,
            pooling_method_txt,
            args.dry_run,
            args.force,
        )

        print("  👉 Extracting Image Representations...")
        res_img = load_or_extract_representation(
            repr_path_img,
            dl_img,
            model,
            processor,
            pooling_method_img,
            args.dry_run,
            args.force,
        )

        txt_metadata = {
            "model": model_name,
            "dataset": f"{dataset_prefix}_txt",
            "pooling_method": pooling_method_txt,
            "pooling_method_requested": pooling_method,
            "caption_source": args.caption_source,
            "type": RESIDUAL_STREAM_TYPE,
        }
        img_metadata = {
            "model": model_name,
            "dataset": f"{dataset_prefix}_img",
            "pooling_method": pooling_method_img,
            "pooling_method_requested": pooling_method,
            "caption_source": "n/a",
            "type": RESIDUAL_STREAM_TYPE,
        }

        res_txt, res_img = preprocess_and_save_pair(
            res_txt,
            res_img,
            repr_path_txt,
            repr_path_img,
            txt_metadata,
            img_metadata,
            args.save_representations,
            args.dry_run,
            args.force,
        )

        # ── Compute Similarities ───────────────────────────────────────────────
        print("  👉 Computing Similarities...")
        sim_metadata = {
            "row_modality": "text",
            "col_modality": "image",
            "description": "Rows represent text layers, Columns represent image layers",
            "pooling_method": pooling_method,
            "pooling_method_text": pooling_method_txt,
            "pooling_method_image": pooling_method_img,
            "caption_source": args.caption_source,
        }
        print("    ℹ️  Matrix orientation: Rows=Text, Columns=Image")

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
                print(f"    ✅ Shape: {measure_val.shape}")
                if not args.dry_run:
                    save_path = os.path.join(
                        similarities_dir, f"{model_name}_{measure_name}.safetensors"
                    )
                    save_file(
                        {f"{measure_name}": measure_val},
                        save_path,
                        metadata=sim_metadata,
                    )
                    print(f"    💾 Saved: {save_path}")
            except Exception as e:
                print(f"    ❌ Error computing {measure_name}: {type(e).__name__}: {e}")
                print(
                    f"    ⚠️  Skipping {measure_name} and continuing with next measure..."
                )

        del res_txt, res_img
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ All analyses complete. Results saved to {similarities_dir}")


if __name__ == "__main__":
    main()
