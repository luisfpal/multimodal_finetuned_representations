"""Compare text-only and multimodal residual-stream representations layer by layer (CKA, neighborhood overlap, information imbalance), optionally against CLIP embeddings.

Inputs: --dataset-path-or-name, --model-name-or-path, --batch-size, --seed, --downsample-size,
--save-representations, pooling and semantic-filter options.
Outputs: similarity results under --results-dir; optionally cached representations.
"""

import argparse
import gc
import os

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file

from src.residual_stream_tracer import VALID_TOKENS_POOLING_METHODS, residual_stream_tracer
from utils.cambrian_bridge import (
    build_cambrian_vision_variant_name,
    cambrian_vision_selection_metadata,
    normalize_cambrian_vision_encoder_selection,
)
from utils.cambrian_utils import is_cambrian_model
from utils import (
    compute_cross_layers_residual_stream_similarities,
    get_dataloader,
    get_dataloader_for_captioning,
    load_hf_model_and_processor_or_tokenizer,
    seed_all,
)
from utils.dataset import MULTIPLE_CHOICE_BENCHMARKS, get_hf_dataset_split
from utils.clip_utils import CLIP_MODEL_ID, extract_clip_embeddings
from utils.metrics_utils import (
    compute_clip_residual_stream_similarities_per_layer,
    preprocess_data,
)
from utils.semantic_filter_utils import (
    CaptionSemanticScorer,
    compute_semantic_keep_indices,
    load_semantic_filter_cache,
    save_semantic_filter_cache,
)

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "flash_attention_2" if DEVICE.type == "cuda" else "sdpa"
# ATTN_IMPLEMENTATION = "eager"
RESIDUAL_STREAM_TYPE = "output_layer"

# ── Pooling spec format ────────────────────────────────────────────────────
# Pooling methods are specified as "img_method|txt_method" (or a bare method
# name to use the same for both modalities).  Each individual method must be
# one of VALID_TOKENS_POOLING_METHODS from residual_stream_tracer.py.
# Use parse_pooling_spec() to split a spec into (img_method, txt_method).
TOKENS_POOLING_METHODS_DEFAULT: list[str] = ["last|last", "mean_image|mean_text_context"]
# ──────────────────────────────────────────────────────────────────────────

SIMILARITY_MEASURES_DEFAULT = [
    "neighborhood_overlap",
    "linear_cka",
    "distance_correlation",
    "id_correlation",
    "information_imbalance",
    "information_imbalance_reversed",
    "information_imbalance_mean",
    "linear_regression_r2",
    "linear_regression_r2_reversed",
]

REGRESSION_SIMILARITY_MEASURES = {
    "linear_regression_r2",
    "linear_regression_r2_reversed",
}
SEMANTIC_FILTER_TAG = "sf-jina_m1sd"


def build_similarity_filename(
    model_name: str, measure_name: str, regression_type: str
) -> str:
    """Build similarity filename with backend suffix for regression metrics."""
    if measure_name in REGRESSION_SIMILARITY_MEASURES:
        return f"{model_name}_{measure_name}_reg-{regression_type}.safetensors"
    return f"{model_name}_{measure_name}.safetensors"


def parse_pooling_spec(spec: str) -> tuple[str, str]:
    """Parse a pooling spec into (img_method, txt_method).

    Accepts "img_method|txt_method" or a bare method name (applied to both
    modalities).  Validates each method against VALID_TOKENS_POOLING_METHODS.
    """
    if "|" in spec:
        img_m, txt_m = spec.split("|", 1)
    else:
        img_m = txt_m = spec
    for m in (img_m, txt_m):
        if m not in VALID_TOKENS_POOLING_METHODS:
            raise ValueError(
                f"Unknown pooling method '{m}' in spec '{spec}'. "
                f"Valid methods: {VALID_TOKENS_POOLING_METHODS}"
            )
    return img_m, txt_m


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

    sem_tag = SEMANTIC_FILTER_TAG
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


def load_representation_from_cache(
    path: str,
    dry_run: bool,
    force: bool,
) -> torch.Tensor | None:
    """Load representation from cache, returning None if cache is unavailable."""
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
        except (OSError, IOError, ValueError, KeyError) as e:
            # File not found, read error, malformed safetensors, or missing key
            print(f"  ⚠️ Error loading cache: {type(e).__name__}: {e}. Re-extracting...")
        except Exception as e:
            # Unexpected error — likely a bug, not a cache issue
            print(f"  ❌ Unexpected error loading cache: {type(e).__name__}: {e}")
            raise

    return None


def extract_representation(
    dl,
    model,
    processor,
    pooling_method: str,
) -> torch.Tensor:
    """Extract representation from model and dataloader."""
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
    regression_type: str,
    dry_run: bool,
    force_recompute: bool,
) -> list[str]:
    """Return measures not yet computed, or all measures if dry_run/force."""
    if dry_run or force_recompute:
        if force_recompute:
            print(
                f"    🔄 Force mode: recomputing all {len(similarity_measures)} measures"
            )
        return list(similarity_measures)

    missing = []
    for measure_name in similarity_measures:
        filename = build_similarity_filename(model_name, measure_name, regression_type)
        if not os.path.exists(os.path.join(similarities_dir, filename)):
            missing.append(measure_name)
        else:
            print(f"    ✅ Found existing: {filename}")
    return missing


def save_raw_representation(
    tensor: torch.Tensor,
    path: str,
    metadata: dict,
    dry_run: bool,
    force: bool,
) -> None:
    """Save a raw (unprocessed) representation tensor to a safetensors file.

    Skipped when dry_run is set or the file already exists and force is False.
    The tensor is saved as-is (CPU copy); preprocessing happens at metric-compute
    time inside the individual measure functions (which each call preprocess_data).
    """
    if dry_run or (not force and os.path.exists(path)):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    valid_meta = {k: str(v) for k, v in metadata.items()}
    try:
        save_file({"residual_stream": tensor.cpu()}, path, metadata=valid_meta)
        print(f"  💾 Saved raw: {path}")
    except Exception as e:
        print(f"  ❌ Error saving {path}: {type(e).__name__}: {e}")


def run_pair_analysis(
    res_a: torch.Tensor,
    res_b: torch.Tensor,
    modality_a: str,
    modality_b: str,
    pooling_spec: str,
    pooling_method_a: str,
    pooling_method_b: str,
    similarities_dir: str,
    model_variant_name: str,
    missing_measures: list[str],
    args: argparse.Namespace,
    extra_metadata: dict,
) -> None:
    """Compute and save all similarity measures for a (A, B) representation pair.

    Shared by the standard text-vs-image path and the img-img path.
    ``modality_a`` / ``modality_b`` are free-form strings stored in metadata
    (e.g. "text", "image").
    """
    pooling_dir_name = pooling_spec.replace("|", "--")
    sim_metadata = {
        "task": args.task,
        "row_modality": modality_a,
        "col_modality": modality_b,
        "description": (
            f"Rows represent {modality_a} layers, Columns represent {modality_b} layers"
        ),
        "pooling_method": pooling_dir_name,
        "pooling_method_a": pooling_method_a,
        "pooling_method_b": pooling_method_b,
        "model_variant_name": model_variant_name,
    }
    if args.task != "captioning":
        sim_metadata["caption_source"] = args.caption_source
    # extra_metadata may contain "semantic_filter" (sem_tag), cambrian metadata, etc.
    sim_metadata.update(extra_metadata)
    print(
        f"    ℹ️  Matrix orientation: Rows={modality_a.capitalize()}, "
        f"Columns={modality_b.capitalize()}"
    )

    for measure_name in missing_measures:
        print(f"    👉  Computing: {measure_name}")
        try:
            sim_results = compute_cross_layers_residual_stream_similarities(
                res_a,
                res_b,
                similarity_measures=[measure_name],
                maxk=30,
                regression_type=args.regression_type,
            )
            measure_val = sim_results[measure_name]
            print(f"    ✅ Shape: {measure_val.shape}")
            if not args.dry_run:
                measure_metadata = dict(sim_metadata)
                if measure_name in REGRESSION_SIMILARITY_MEASURES:
                    measure_metadata["regression_type"] = args.regression_type
                save_path = os.path.join(
                    similarities_dir,
                    build_similarity_filename(
                        model_variant_name,
                        measure_name,
                        args.regression_type,
                    ),
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_file(
                    {f"{measure_name}": measure_val},
                    save_path,
                    metadata={k: str(v) for k, v in measure_metadata.items()},
                )
                print(f"    💾 Saved: {save_path}")
        except Exception as e:
            print(f"    ❌ Error computing {measure_name}: {type(e).__name__}: {e}")
            print(
                f"    ⚠️  Skipping {measure_name} and continuing with next measure..."
            )


# ── CLIP measure configuration ─────────────────────────────────────────────
# These three measures are computed for EACH of the two modalities (img, txt),
# giving 6 files in total per pooling-spec run.
_CLIP_MEASURES = [
    "neighborhood_overlap",
    "information_imbalance",          # repr[layer] → clip
    "information_imbalance_reversed",  # clip → repr[layer]
]

# (modality_tag, measure_name) → file-key / filename suffix
_CLIP_RESULT_KEYS: dict[tuple[str, str], str] = {
    ("img", "neighborhood_overlap"):            "img_clip_neighborhood_overlap",
    ("txt", "neighborhood_overlap"):            "txt_clip_neighborhood_overlap",
    ("img", "information_imbalance"):           "img_to_clip_information_imbalance",
    ("txt", "information_imbalance"):           "txt_to_clip_information_imbalance",
    ("img", "information_imbalance_reversed"):  "clip_to_img_information_imbalance",
    ("txt", "information_imbalance_reversed"):  "clip_to_txt_information_imbalance",
}
# ───────────────────────────────────────────────────────────────────────────


def run_clip_analysis(
    res_txt: torch.Tensor,
    res_img: torch.Tensor,
    clip_embeddings: torch.Tensor,
    clip_results_dir: str,
    model_variant_name: str,
    args: argparse.Namespace,
    pooling_spec: str,
    pooling_method_img: str,
    pooling_method_txt: str,
    extra_metadata: dict,
) -> None:
    """Compute per-layer CLIP similarity metrics for both modalities and save.

    For each of (img, txt) and each of the three measures in ``_CLIP_MEASURES``,
    calls ``compute_clip_residual_stream_similarities_per_layer`` and saves a
    ``(num_layers,)`` tensor to a dedicated safetensors file.

    Args:
        res_txt: Preprocessed text representations, shape (num_layers, N, D).
        res_img: Preprocessed image representations, shape (num_layers, N, D).
        clip_embeddings: CLIP features filtered by valid_indices, shape (N, clip_dim).
        clip_results_dir: Directory for result files (created if needed).
        model_variant_name: Used as filename prefix.
        args: Script arguments (for dry_run, task, regression_type).
        pooling_spec: Raw spec string e.g. "mean_image|last_caption".
        pooling_method_img: Concrete image pooling method.
        pooling_method_txt: Concrete text pooling method.
        extra_metadata: Additional fields merged into each file's metadata
            (e.g. semantic_filter tag, cambrian variant info).
    """
    os.makedirs(clip_results_dir, exist_ok=True)

    base_meta = {
        "clip_model": CLIP_MODEL_ID,
        "pooling_method": pooling_spec.replace("|", "--"),
        "pooling_method_img": pooling_method_img,
        "pooling_method_txt": pooling_method_txt,
        "model_variant_name": model_variant_name,
        "task": args.task,
    }
    if args.task != "captioning":
        base_meta["caption_source"] = args.caption_source
    base_meta.update(extra_metadata)

    for modality_tag, representations in [("img", res_img), ("txt", res_txt)]:
        try:
            results = compute_clip_residual_stream_similarities_per_layer(
                representations,
                clip_embeddings,
                similarity_measures=_CLIP_MEASURES,
            )
        except Exception as e:
            print(
                f"    ❌ Error computing CLIP metrics for {modality_tag}: "
                f"{type(e).__name__}: {e}"
            )
            print(f"    ⚠️  Skipping CLIP metrics for {modality_tag}...")
            continue

        for measure, values in results.items():
            key = _CLIP_RESULT_KEYS[(modality_tag, measure)]
            if args.dry_run:
                print(f"    [dry-run] CLIP {key}: shape={tuple(values.shape)}")
                continue
            save_path = os.path.join(
                clip_results_dir,
                f"{model_variant_name}_{key}.safetensors",
            )
            meta = {
                **base_meta,
                "modality": modality_tag,
                "measure": measure,
                "direction": key,
            }
            save_file(
                {key: values},
                save_path,
                metadata={k: str(v) for k, v in meta.items()},
            )
            print(f"    💾 Saved CLIP: {save_path}")


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
    parser.add_argument(
        "--task",
        type=str,
        default="vqa",
        choices=["vqa", "captioning"],
        help="Experiment task. 'vqa' preserves the existing QA path; 'captioning' uses ImageCaptioningDataset.",
    )
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
        help=(
            "Pooling specs in 'img_method|txt_method' format (or a bare method "
            "name to use the same for both modalities).  "
            f"Valid individual methods: {VALID_TOKENS_POOLING_METHODS}.  "
            f"Default: {TOKENS_POOLING_METHODS_DEFAULT}"
        ),
    )
    parser.add_argument(
        "--similarity-measures",
        nargs="+",
        default=SIMILARITY_MEASURES_DEFAULT,
        help="List of similarity measures to use.",
    )
    parser.add_argument(
        "--regression-type",
        type=str.lower,
        default="ridge",
        choices=["ols", "ridge", "lasso"],
        help=(
            "Regression backend for linear_regression_r2* measures. "
            "Choices: 'ols', 'ridge', or 'lasso'."
        ),
    )
    parser.add_argument("--save-representations", action="store_true")
    parser.add_argument(
        "--img-img-pair",
        action="store_true",
        help=(
            "Run image-vs-image comparison instead of the default text-vs-image. "
            "Both representations are extracted from the image dataloader using the "
            "img_method and txt_method from each pooling spec respectively.  "
            "Results are saved under 'img_img_comparison/' to avoid collision with "
            "the default 'multimodality_comparison/' directory."
        ),
    )
    parser.add_argument(
        "--enable-clip-metrics",
        action="store_true",
        help=(
            "Compute per-layer CLIP similarity metrics (neighborhood_overlap and "
            "information_imbalance in both directions) for the text and image "
            "representations against CLIP image embeddings.  Results are saved "
            "under 'clip_multimodality_comparison/' alongside the standard "
            "'multimodality_comparison/' directory.  Only applies to the standard "
            "text-vs-image path (not --img-img-pair)."
        ),
    )
    parser.add_argument(
        "--caption-source",
        type=str,
        default="synthetic",
        choices=["human", "synthetic"],
        help=(
            "Caption source: 'human' for datasets with human captions (COCOQA, VQAv2-ICQ), "
            "or 'synthetic' for datasets with only synthetic captions (MME, MMBench, GQA, etc.). "
            "Ignored for --task captioning."
        ),
    )
    parser.add_argument(
        "--captioning-image-dir",
        type=str,
        default=None,
        help=(
            "Directory containing COCO train2014 images for LLaVA-Instruct captioning. "
            "Defaults to COCO_TRAIN2014_DIR or the project ORFEO path."
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
        "--force-metrics-only",
        action="store_true",
        help=(
            "Force similarity recomputation only: skip 'already exists' checks for "
            "similarity outputs while preserving normal representation cache behavior. "
            "Use this to refresh metrics without forcing representation overwrite."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip ALL caching: no load, no save, no skip existing. For debugging.",
    )
    parser.add_argument(
        "--representations-only",
        action="store_true",
        help=(
            "Extract and save representations only — skip similarity computation. "
            "Use this for a two-phase workflow: run with --representations-only first "
            "across all datasets/models, then rerun without it to compute similarities "
            "from the cached representations."
        ),
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
        "--open-ended",
        action="store_true",
        help=(
            "Format MC datasets (MMBench, SEED) as open-ended VQA: skip the choices "
            "list and 'There are several options:' header. Adds 'open_ended' to the "
            "dataset prefix so results land in a separate directory. Ignored for "
            "non-MC datasets and for MME/POPE which are already Yes/No."
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
    parser.add_argument(
        "--cambrian-vision-encoders",
        nargs="+",
        default=None,
        help=(
            "Optional Cambrian-only encoder selection. Omit to use the original "
            "full Cambrian vision encoder set. Valid aliases: siglip, clip, "
            "dinov2, convnext. Examples: --cambrian-vision-encoders siglip "
            "or --cambrian-vision-encoders clip dinov2."
        ),
    )

    args = parser.parse_args()

    if args.force and args.force_metrics_only:
        parser.error("--force and --force-metrics-only are mutually exclusive.")

    # Validate and pre-parse all pooling specs so errors surface early.
    try:
        parsed_specs: list[tuple[str, str]] = [
            parse_pooling_spec(spec) for spec in args.pooling_methods
        ]
    except ValueError as exc:
        parser.error(str(exc))

    is_cambrian_requested = is_cambrian_model(
        model=None,
        model_name_or_path=args.model_name_or_path,
    )
    try:
        args.cambrian_vision_encoders = normalize_cambrian_vision_encoder_selection(
            args.cambrian_vision_encoders
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.cambrian_vision_encoders is not None and not is_cambrian_requested:
        parser.error("--cambrian-vision-encoders is only valid for Cambrian models.")

    if args.dry_run:
        print("\n🧪 DRY-RUN MODE — no caching, no saving, fresh computation only")
        if args.results_dir is None:
            args.results_dir = "/tmp/debug_run"
            print(f"    (results dir defaulted to: {args.results_dir})")
        if args.representations_dir is None:
            args.representations_dir = "/tmp/debug_run"
            print(f"    (representations dir defaulted to: {args.representations_dir})")
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
    elif args.force_metrics_only:
        print(
            "\n🔄 FORCE METRICS-ONLY MODE — recomputing similarities while reusing "
            "representation caches"
        )

    if args.representations_only and args.force_metrics_only:
        print(
            "\nℹ️  --force-metrics-only has no effect with --representations-only "
            "(similarity computation is skipped)."
        )

    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    is_mc = any(b in dataset_slug for b in MULTIPLE_CHOICE_BENCHMARKS)

    if args.task == "captioning":
        if args.dataset_split:
            parser.error("--dataset-split is only valid for the VQA task.")
        if args.enable_semantic_filter:
            parser.error("--enable-semantic-filter is not implemented for captioning.")
        if args.open_ended:
            parser.error("--open-ended is only valid for multiple-choice VQA datasets.")
        if args.repeated_images:
            parser.error("--repeated-images is only valid for multiple-choice VQA datasets.")

    if args.dataset_split:
        if is_mc:
            parser.error("--dataset-split is only valid for OpenVQA datasets, not MC.")
        if args.enable_semantic_filter:
            parser.error(
                "--dataset-split and --enable-semantic-filter are mutually exclusive "
                "(split indices reference the full dataset)."
            )
        dataset_slug = f"{dataset_slug}_{args.dataset_split}"

    effective_downsample_size = args.downsample_size
    if args.dataset_split and args.downsample_size is not None:
        print(
            "ℹ️  --downsample-size ignored because --dataset-split is active "
            "(split selection takes precedence)."
        )
        effective_downsample_size = None

    if args.task == "captioning":
        sample_set_tag = (
            f"top{effective_downsample_size}_responses"
            if effective_downsample_size
            else "all_instances"
        )
    else:
        sample_set_tag = (
            "repeated_images" if (is_mc and args.repeated_images) else "unique_images"
        )
    open_ended_tag = "open_ended" if (is_mc and args.open_ended) else ""

    # Warn when flags are inapplicable (shell already guards these, but be explicit)
    if args.open_ended and not is_mc:
        print(
            f"⚠️  --open-ended ignored: '{args.dataset_path_or_name}' is not a "
            "multiple-choice benchmark."
        )
    if args.repeated_images and not is_mc:
        print(
            f"⚠️  --repeated-images ignored: '{args.dataset_path_or_name}' is not a "
            "multiple-choice benchmark."
        )

    if args.task == "captioning":
        dataset_prefix = f"{dataset_slug}_detail_23k"
    else:
        dataset_prefix = (
            f"{dataset_slug}_{open_ended_tag}_{args.caption_source}"
            if open_ended_tag
            else f"{dataset_slug}_{args.caption_source}"
        )
    sem_tag = SEMANTIC_FILTER_TAG if args.enable_semantic_filter else ""

    seed_all(args.seed)

    model_name = args.model_name_or_path.split("/")[-1]
    model_variant_name = build_cambrian_vision_variant_name(
        model_name,
        args.cambrian_vision_encoders,
    )
    if (
        args.cambrian_vision_encoders is not None
        and model_variant_name == model_name
    ):
        parser.error(
            "Cambrian vision encoder selection did not create a distinct model "
            "variant name; refusing to risk cache/result collision."
        )
    selected_cambrian_metadata = cambrian_vision_selection_metadata(
        args.cambrian_vision_encoders
    )
    if args.cambrian_vision_encoders is not None:
        print(
            "\n🧩 Cambrian vision encoders selected: "
            f"{'+'.join(args.cambrian_vision_encoders)}"
        )
        print(f"    Results/cache model variant: {model_variant_name}")

    model_cache_dir = os.getenv(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE")

    model = None
    processor = None
    dl_txt = None
    dl_img = None
    extraction_runtime_initialized = False

    def ensure_extraction_runtime():
        nonlocal model, processor, dl_txt, dl_img, extraction_runtime_initialized

        if extraction_runtime_initialized:
            return model, processor, dl_txt, dl_img

        print(f"\n🤖 Loading model: {args.model_name_or_path}")
        print(f"📂 Using model cache directory: {model_cache_dir}")
        model, processor = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=args.model_name_or_path,
            cache_dir=model_cache_dir,
            dtype=MODEL_DTYPE,
            attn_implementation=ATTN_IMPLEMENTATION,
            device_map=DEVICE,
            cambrian_vision_encoders=args.cambrian_vision_encoders,
        )

        print("\n📚 Preparing dataloaders...")
        if args.task == "captioning":
            dl_args = {
                "dataset_path_or_name": args.dataset_path_or_name,
                "processor": processor,
                "batch_size": args.batch_size,
                "downsample_size": effective_downsample_size,
                "image_dir": args.captioning_image_dir,
            }
            if "intern" in model.__class__.__name__.lower() or is_cambrian_model(
                model=model
            ):
                dl_args["model"] = model
            if is_cambrian_model(model=model):
                dl_args["num_workers"] = 0

            if not args.img_img_pair:
                print(f"  - Loading {dataset_prefix}_txt...")
                dl_txt = get_dataloader_for_captioning(texts_qa=True, **dl_args)
            print(f"  - Loading {dataset_prefix}_img...")
            dl_img = get_dataloader_for_captioning(images_qa=True, **dl_args)
        else:
            dl_args = {
                "dataset_path_or_name": args.dataset_path_or_name,
                "seed": args.seed,
                "processor": processor,
                "batch_size": args.batch_size,
                "cache_dir": datasets_cache_dir,
                "guide_text": args.guide_text,
                "downsample_size": effective_downsample_size,
                "split": "train",
                "caption_source": args.caption_source,
            }

            if args.enable_semantic_filter:
                allowed_indices, _ = resolve_semantic_filter(
                    args, results_dir, dataset_slug, datasets_cache_dir
                )
                if allowed_indices is not None:
                    dl_args["allowed_indices"] = allowed_indices

            if is_mc and not args.repeated_images:
                dl_args["unique_images"] = True
            if open_ended_tag:
                dl_args["open_ended"] = True
            if args.dataset_split:
                dl_args["dataset_split"] = args.dataset_split
            if "intern" in model.__class__.__name__.lower() or is_cambrian_model(
                model=model
            ):
                dl_args["model"] = model
            if is_cambrian_model(model=model):
                dl_args["num_workers"] = 0

            if not args.img_img_pair:
                print(f"  - Loading {dataset_prefix}_txt...")
                dl_txt = get_dataloader(texts_qa=True, **dl_args)
            print(f"  - Loading {dataset_prefix}_img...")
            dl_img = get_dataloader(images_qa=True, **dl_args)

        extraction_runtime_initialized = True
        return model, processor, dl_txt, dl_img

    def get_repr_path(dataset_n: str, method: str) -> str:
        """Build and create the cache path for a representation."""
        repr_dir = os.path.join(
            os.path.expanduser(args.representations_dir),
            "representations",
            model_variant_name,
            dataset_n,
            RESIDUAL_STREAM_TYPE,
            method,
            sample_set_tag,
        )
        os.makedirs(repr_dir, exist_ok=True)
        return os.path.join(repr_dir, "representations.safetensors")

    last_similarities_dir = None

    for pooling_spec, (pooling_method_img, pooling_method_txt) in zip(
        args.pooling_methods, parsed_specs
    ):
        pooling_dir_name = pooling_spec.replace("|", "--")
        comparison_root = "img_img_comparison" if args.img_img_pair else "multimodality_comparison"
        analysis_dir = os.path.join(
            results_dir,
            comparison_root,
            dataset_prefix,
            pooling_dir_name,
            sem_tag,
            sample_set_tag,
        )
        similarities_dir = os.path.join(analysis_dir, "similarities")
        os.makedirs(similarities_dir, exist_ok=True)
        last_similarities_dir = similarities_dir

        print(f"\n{'-' * 80}")
        print(f"🔍 Analysis: {RESIDUAL_STREAM_TYPE} | {pooling_spec}")
        if args.img_img_pair:
            print(f"    Mode: image-vs-image ({pooling_method_img} | {pooling_method_txt})")
        else:
            print(f"    img={pooling_method_img}, txt={pooling_method_txt}")
        print(f"{'-' * 80}")

        missing_measures = find_missing_measures(
            args.similarity_measures,
            similarities_dir,
            model_variant_name,
            args.regression_type,
            args.dry_run,
            args.force or args.force_metrics_only,
        )
        if not missing_measures:
            print(f"    ⏩ All measures computed for {pooling_spec}. Skipping...")
            continue
        print(f"    ⚠️  Missing/forced measures: {missing_measures}")

        if args.img_img_pair:
            # ── Image-vs-image path ─────────────────────────────────────────
            repr_path_a = get_repr_path(f"{dataset_prefix}_img", pooling_method_img)
            repr_path_b = get_repr_path(f"{dataset_prefix}_img", pooling_method_txt)

            res_a = load_representation_from_cache(repr_path_a, args.dry_run, args.force)
            res_b = load_representation_from_cache(repr_path_b, args.dry_run, args.force)
            a_from_cache = res_a is not None
            b_from_cache = res_b is not None

            if not a_from_cache or not b_from_cache:
                model, processor, dl_txt, dl_img = ensure_extraction_runtime()

            if not a_from_cache:
                print(f"  👉 Extracting Image Representations (A: {pooling_method_img})...")
                res_a = extract_representation(dl_img, model, processor, pooling_method_img)
            if not b_from_cache:
                print(f"  👉 Extracting Image Representations (B: {pooling_method_txt})...")
                res_b = extract_representation(dl_img, model, processor, pooling_method_txt)

            base_meta = {
                "model": model_variant_name,
                "model_variant_name": model_variant_name,
                "task": args.task,
                "semantic_filter": sem_tag,
                "type": RESIDUAL_STREAM_TYPE,
                "open_ended": str(bool(open_ended_tag)),
                "repeated_images": str(sample_set_tag == "repeated_images"),
            }
            if args.task != "captioning":
                base_meta["caption_source"] = args.caption_source
            meta_a = {**base_meta, "dataset": f"{dataset_prefix}_img", "pooling_method": pooling_method_img}
            meta_b = {**base_meta, "dataset": f"{dataset_prefix}_img", "pooling_method": pooling_method_txt}
            meta_a.update(selected_cambrian_metadata)
            meta_b.update(selected_cambrian_metadata)

            if not a_from_cache and args.save_representations:
                save_raw_representation(res_a, repr_path_a, meta_a, args.dry_run, args.force)
            if not b_from_cache and args.save_representations:
                save_raw_representation(res_b, repr_path_b, meta_b, args.dry_run, args.force)

            if args.representations_only:
                print("  ⏭️  --representations-only: skipping similarity computation.")
                del res_a, res_b
                gc.collect()
                torch.cuda.empty_cache()
                continue

            print("  👉 Computing Similarities (image-vs-image)...")
            run_pair_analysis(
                res_a, res_b,
                modality_a="image", modality_b="image",
                pooling_spec=pooling_spec,
                pooling_method_a=pooling_method_img,
                pooling_method_b=pooling_method_txt,
                similarities_dir=similarities_dir,
                model_variant_name=model_variant_name,
                missing_measures=missing_measures,
                args=args,
                extra_metadata={**selected_cambrian_metadata, "semantic_filter": sem_tag},
            )

        else:
            # ── Standard text-vs-image path ─────────────────────────────────
            repr_path_txt = get_repr_path(f"{dataset_prefix}_txt", pooling_method_txt)
            repr_path_img = get_repr_path(f"{dataset_prefix}_img", pooling_method_img)

            res_txt = load_representation_from_cache(repr_path_txt, args.dry_run, args.force)
            res_img = load_representation_from_cache(repr_path_img, args.dry_run, args.force)
            txt_from_cache = res_txt is not None
            img_from_cache = res_img is not None

            if not txt_from_cache or not img_from_cache:
                model, processor, dl_txt, dl_img = ensure_extraction_runtime()

            if not txt_from_cache:
                print("  👉 Extracting Text Representations...")
                res_txt = extract_representation(dl_txt, model, processor, pooling_method_txt)
            if not img_from_cache:
                print("  👉 Extracting Image Representations...")
                res_img = extract_representation(dl_img, model, processor, pooling_method_img)

            base_meta = {
                "model": model_variant_name,
                "model_variant_name": model_variant_name,
                "task": args.task,
                "semantic_filter": sem_tag,
                "type": RESIDUAL_STREAM_TYPE,
                "open_ended": str(bool(open_ended_tag)),
                "repeated_images": str(sample_set_tag == "repeated_images"),
            }
            if args.task != "captioning":
                base_meta["caption_source"] = args.caption_source
            txt_metadata = {**base_meta, "dataset": f"{dataset_prefix}_txt", "pooling_method": pooling_method_txt}
            img_metadata = {**base_meta, "dataset": f"{dataset_prefix}_img", "pooling_method": pooling_method_img}
            txt_metadata.update(selected_cambrian_metadata)
            img_metadata.update(selected_cambrian_metadata)

            if not txt_from_cache and args.save_representations:
                save_raw_representation(res_txt, repr_path_txt, txt_metadata, args.dry_run, args.force)
            if not img_from_cache and args.save_representations:
                save_raw_representation(res_img, repr_path_img, img_metadata, args.dry_run, args.force)

            if args.representations_only:
                print("  ⏭️  --representations-only: skipping similarity computation.")
                del res_txt, res_img
                gc.collect()
                torch.cuda.empty_cache()
                continue

            print("  👉 Computing Similarities...")
            run_pair_analysis(
                res_txt, res_img,
                modality_a="text", modality_b="image",
                pooling_spec=pooling_spec,
                pooling_method_a=pooling_method_txt,
                pooling_method_b=pooling_method_img,
                similarities_dir=similarities_dir,
                model_variant_name=model_variant_name,
                missing_measures=missing_measures,
                args=args,
                extra_metadata={**selected_cambrian_metadata, "semantic_filter": sem_tag},
            )

            if args.enable_clip_metrics:
                # Compute the sample indices that survive txt-img deduplication.
                # Using layer 0 is sufficient: all layers share the same valid mask.
                _, _, valid_indices = preprocess_data(
                    res_txt[0], res_img[0],
                    output_format="tensor",
                    remove_duplicates=True,
                    return_indices=True,
                )
                print("  👉 Computing CLIP Similarities...")
                model, processor, dl_txt, dl_img = ensure_extraction_runtime()
                clip_raw = extract_clip_embeddings(dl_img, DEVICE)
                clip_filtered = clip_raw[valid_indices]
                clip_results_dir = os.path.join(
                    results_dir,
                    "clip_multimodality_comparison",
                    dataset_prefix,
                    pooling_dir_name,
                    sem_tag,
                    sample_set_tag,
                    "similarities",
                )
                run_clip_analysis(
                    res_txt[:, valid_indices, :],
                    res_img[:, valid_indices, :],
                    clip_filtered,
                    clip_results_dir=clip_results_dir,
                    model_variant_name=model_variant_name,
                    args=args,
                    pooling_spec=pooling_spec,
                    pooling_method_img=pooling_method_img,
                    pooling_method_txt=pooling_method_txt,
                    extra_metadata={**selected_cambrian_metadata, "semantic_filter": sem_tag},
                )

            res_txt, res_img = None, None  # let GC collect

        gc.collect()
        torch.cuda.empty_cache()

    if extraction_runtime_initialized:
        del model, processor, dl_txt, dl_img
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = last_similarities_dir or results_dir
    print(f"\n✅ All analyses complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
