import argparse
import gc
import os

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file

from src.residual_stream_tracer import residual_stream_tracer
from utils import (
    compute_layers_intrinsic_dimension,
    get_dataloader,
    load_hf_model_and_processor_or_tokenizer,
    seed_all,
)
from utils.dataset import MULTIPLE_CHOICE_BENCHMARKS
from utils.metrics_utils import preprocess_data

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "flash_attention_2" if DEVICE.type == "cuda" else "sdpa"
RESIDUAL_STREAM_TYPE = "output_layer"

VIRTUAL_POOLING_DISPATCH: dict[str, tuple[str, str]] = {
    "mean_context": ("mean_text_context", "mean_image"),
    "mean_image_last_caption": ("last_caption", "mean_image"),
    "last_image_last_caption": ("last_caption", "last_image"),
}
_PASS_THROUGH_POOLING_METHODS: list[str] = ["last", "mean", "none"]
VALID_POOLING_METHODS: list[str] = _PASS_THROUGH_POOLING_METHODS + list(
    VIRTUAL_POOLING_DISPATCH.keys()
)
TOKENS_POOLING_METHODS_DEFAULT: list[str] = ["last", "mean_context"]


def load_representation_from_cache(
    path: str,
    dry_run: bool,
    force: bool,
) -> torch.Tensor | None:
    """Load representation cache if available and valid."""
    if not dry_run and not force and os.path.exists(path):
        print(f"  💾 Found cached representation at: {path}")
        try:
            loaded = load_file(path)
            if "residual_stream" not in loaded:
                print("  ⚠️ Key 'residual_stream' not found. Re-extracting...")
                return None

            tensor = loaded["residual_stream"]
            if tensor.ndim >= 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                print(f"  ✅ Loaded successfully. Shape: {tensor.shape}")
                return tensor[:]

            print(f"  ⚠️ Invalid tensor shape: {tensor.shape}. Re-extracting...")
        except (OSError, IOError, ValueError, KeyError) as error:
            print(
                f"  ⚠️ Error loading cache: {type(error).__name__}: {error}. "
                "Re-extracting..."
            )
        except Exception as error:
            print(f"  ❌ Unexpected cache error: {type(error).__name__}: {error}")
            raise

    return None


def extract_representation(
    dl,
    model,
    processor,
    pooling_method: str,
) -> torch.Tensor:
    """Extract representations from one dataloader."""
    return residual_stream_tracer(
        model=model,
        processed_dataloader=dl,
        residual_stream_type=RESIDUAL_STREAM_TYPE,
        tokens_pooling_method=pooling_method,
        return_deepcopy=True,
        save_representations=False,
        processor=processor,
    )


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
    """Preprocess paired representations and optionally save to cache."""
    print("  👉 Preprocessing paired representations...")
    proc_txt: list[torch.Tensor] = []
    proc_img: list[torch.Tensor] = []

    for layer_idx in range(res_txt.shape[0]):
        txt_layer, img_layer = preprocess_data(
            res_txt[layer_idx],
            res_img[layer_idx],
            output_format="tensor",
            remove_duplicates=True,
        )
        proc_txt.append(txt_layer.cpu())
        proc_img.append(img_layer.cpu())

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
                valid_meta = {key: str(value) for key, value in metadata.items()}
                save_file(
                    {"residual_stream": tensor},
                    path,
                    metadata=valid_meta,
                )
                print(f"  💾 Saved: {path}")

    return res_txt, res_img


def build_id_filename(
    model_name: str,
    dataset_name: str,
    pooling_method: str,
    modality: str,
    algorithm: str,
) -> str:
    return (
        f"{model_name}_{dataset_name}_{pooling_method}_{modality}_{algorithm}_"
        "intrinsic_dimension.safetensors"
    )


def build_algorithm_kwargs(args: argparse.Namespace) -> dict:
    if args.id_algorithm == "scaling_gride":
        return {
            "k": args.id_nn_rank,
            "range_max": args.id_nn_range_max,
        }
    if args.id_algorithm == "MLE":
        return {
            "k": args.id_nn_rank,
        }
    if args.id_algorithm == "twoNN":
        return {
            "fraction": args.id_fraction,
        }
    raise ValueError(f"Unsupported intrinsic-dimension algorithm: {args.id_algorithm}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute intrinsic dimension for text/image representations using "
            "the same cache identity as multimodality_similarity.py"
        )
    )

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
        help="Guide text used by dataset collate templates.",
    )
    parser.add_argument("--downsample-size", type=int, default=None)
    parser.add_argument(
        "--pooling-methods",
        nargs="+",
        default=TOKENS_POOLING_METHODS_DEFAULT,
        help=(
            f"List of pooling methods to use. "
            f"Valid: {VALID_POOLING_METHODS}. "
            f"Default: {TOKENS_POOLING_METHODS_DEFAULT}"
        ),
    )
    parser.add_argument(
        "--caption-source",
        type=str,
        default="synthetic",
        choices=["human", "synthetic"],
        help="Caption source used for dataloader construction.",
    )
    parser.add_argument("--save-representations", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation and overwrite intrinsic-dimension outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable load/save side effects for debugging.",
    )
    parser.add_argument(
        "--repeated-images",
        action="store_true",
        help=(
            "Use repeated-image sample set for MC datasets. "
            "Default is unique-images mode."
        ),
    )
    parser.add_argument(
        "--open-ended",
        action="store_true",
        help="Use open-ended formatting where supported (MMBench/SEED).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        help="Optional split tag for OpenVQA datasets (e.g. split01).",
    )

    parser.add_argument(
        "--id-algorithm",
        type=str,
        default="scaling_gride",
        choices=["scaling_gride", "MLE", "twoNN"],
        help="Intrinsic-dimension algorithm.",
    )
    parser.add_argument(
        "--id-nn-rank",
        type=int,
        default=16,
        help="Neighbor rank k for scaling_gride/MLE.",
    )
    parser.add_argument(
        "--id-nn-range-max",
        type=int,
        default=100,
        help="Maximum neighbor range for scaling_gride.",
    )
    parser.add_argument(
        "--id-fraction",
        type=float,
        default=0.9,
        help="Fraction used by twoNN algorithm.",
    )

    args = parser.parse_args()

    invalid_pooling = [
        method for method in args.pooling_methods if method not in VALID_POOLING_METHODS
    ]
    if invalid_pooling:
        parser.error(
            f"Invalid --pooling-methods: {invalid_pooling}. "
            f"Valid options: {VALID_POOLING_METHODS}"
        )

    if args.dry_run:
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
        print("\n🔄 FORCE MODE — recomputing intrinsic dimensions and caches")

    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    is_mc = any(benchmark in dataset_slug for benchmark in MULTIPLE_CHOICE_BENCHMARKS)

    if args.dataset_split:
        if is_mc:
            parser.error("--dataset-split is only valid for OpenVQA datasets, not MC.")
        dataset_slug = f"{dataset_slug}_{args.dataset_split}"

    sample_set_tag = (
        "repeated_images" if (is_mc and args.repeated_images) else "unique_images"
    )
    open_ended_tag = "open_ended" if (is_mc and args.open_ended) else ""

    if args.open_ended and not is_mc:
        print(
            f"⚠️  --open-ended ignored: '{args.dataset_path_or_name}' is not "
            "a multiple-choice benchmark."
        )
    if args.repeated_images and not is_mc:
        print(
            f"⚠️  --repeated-images ignored: '{args.dataset_path_or_name}' is not "
            "a multiple-choice benchmark."
        )

    dataset_prefix = (
        f"{dataset_slug}_{open_ended_tag}_{args.caption_source}"
        if open_ended_tag
        else f"{dataset_slug}_{args.caption_source}"
    )

    seed_all(args.seed)

    model_name = args.model_name_or_path.split("/")[-1]
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
        )

        print("\n📚 Preparing paired dataloaders...")
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

        if is_mc and not args.repeated_images:
            dl_args["unique_images"] = True
        if open_ended_tag:
            dl_args["open_ended"] = True
        if args.dataset_split:
            dl_args["dataset_split"] = args.dataset_split
        if "intern" in model.__class__.__name__.lower():
            dl_args["model"] = model

        print(f"  - Loading {dataset_prefix}_txt...")
        dl_txt = get_dataloader(texts_qa=True, **dl_args)
        print(f"  - Loading {dataset_prefix}_img...")
        dl_img = get_dataloader(images_qa=True, **dl_args)

        extraction_runtime_initialized = True
        return model, processor, dl_txt, dl_img

    for pooling_method in args.pooling_methods:
        pooling_method_txt, pooling_method_img = VIRTUAL_POOLING_DISPATCH.get(
            pooling_method, (pooling_method, pooling_method)
        )

        print(f"\n{'-' * 80}")
        print(f"📐 Intrinsic Dimension | {RESIDUAL_STREAM_TYPE} | {pooling_method}")
        print(f"{'-' * 80}")

        def get_repr_path(dataset_name: str) -> str:
            repr_dir = os.path.join(
                os.path.expanduser(args.representations_dir),
                "representations",
                model_name,
                dataset_name,
                RESIDUAL_STREAM_TYPE,
                pooling_method,
                sample_set_tag,
            )
            os.makedirs(repr_dir, exist_ok=True)
            return os.path.join(repr_dir, "representations.safetensors")

        repr_path_txt = get_repr_path(f"{dataset_prefix}_txt")
        repr_path_img = get_repr_path(f"{dataset_prefix}_img")

        res_txt = load_representation_from_cache(
            repr_path_txt,
            args.dry_run,
            args.force,
        )
        res_img = load_representation_from_cache(
            repr_path_img,
            args.dry_run,
            args.force,
        )

        txt_from_cache = res_txt is not None
        img_from_cache = res_img is not None

        if not txt_from_cache or not img_from_cache:
            model, processor, dl_txt, dl_img = ensure_extraction_runtime()

        if not txt_from_cache:
            print("  👉 Extracting text representations...")
            res_txt = extract_representation(
                dl_txt,
                model,
                processor,
                pooling_method_txt,
            )

        if not img_from_cache:
            print("  👉 Extracting image representations...")
            res_img = extract_representation(
                dl_img,
                model,
                processor,
                pooling_method_img,
            )

        txt_metadata = {
            "model": model_name,
            "dataset": f"{dataset_prefix}_txt",
            "pooling_method": pooling_method_txt,
            "caption_source": args.caption_source,
            "type": RESIDUAL_STREAM_TYPE,
            "open_ended": str(bool(open_ended_tag)),
            "repeated_images": str(sample_set_tag == "repeated_images"),
            "task": "intrinsic_dimension",
        }
        img_metadata = {
            "model": model_name,
            "dataset": f"{dataset_prefix}_img",
            "pooling_method": pooling_method_img,
            "caption_source": args.caption_source,
            "type": RESIDUAL_STREAM_TYPE,
            "open_ended": str(bool(open_ended_tag)),
            "repeated_images": str(sample_set_tag == "repeated_images"),
            "task": "intrinsic_dimension",
        }

        if txt_from_cache and img_from_cache:
            print("  ⏩ Pair loaded from cache. Skipping preprocessing.")
        else:
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

        algorithm_kwargs = build_algorithm_kwargs(args)

        print(
            f"  👉 Computing intrinsic dimension with {args.id_algorithm} "
            f"(kwargs={algorithm_kwargs})..."
        )
        intrinsic_txt = compute_layers_intrinsic_dimension(
            residual_stream=res_txt,
            algorithm=args.id_algorithm,
            **algorithm_kwargs,
        )
        intrinsic_img = compute_layers_intrinsic_dimension(
            residual_stream=res_img,
            algorithm=args.id_algorithm,
            **algorithm_kwargs,
        )

        analysis_dir = os.path.join(
            results_dir,
            "intrinsic_dimension",
            dataset_prefix,
            pooling_method,
            sample_set_tag,
            model_name,
            RESIDUAL_STREAM_TYPE,
        )
        os.makedirs(analysis_dir, exist_ok=True)

        txt_save_path = os.path.join(
            analysis_dir,
            build_id_filename(
                model_name=model_name,
                dataset_name=dataset_prefix,
                pooling_method=pooling_method,
                modality="txt",
                algorithm=args.id_algorithm,
            ),
        )
        img_save_path = os.path.join(
            analysis_dir,
            build_id_filename(
                model_name=model_name,
                dataset_name=dataset_prefix,
                pooling_method=pooling_method,
                modality="img",
                algorithm=args.id_algorithm,
            ),
        )

        if not args.dry_run:
            metadata_common = {
                "model": model_name,
                "dataset": dataset_prefix,
                "pooling_method": pooling_method,
                "pooling_method_text": pooling_method_txt,
                "pooling_method_image": pooling_method_img,
                "sample_set": sample_set_tag,
                "caption_source": args.caption_source,
                "type": RESIDUAL_STREAM_TYPE,
                "id_algorithm": args.id_algorithm,
                "id_nn_rank": str(args.id_nn_rank),
                "id_nn_range_max": str(args.id_nn_range_max),
                "id_fraction": str(args.id_fraction),
            }

            save_file(
                {"intrinsic_dimension": intrinsic_txt},
                txt_save_path,
                metadata={**metadata_common, "modality": "text"},
            )
            save_file(
                {"intrinsic_dimension": intrinsic_img},
                img_save_path,
                metadata={**metadata_common, "modality": "image"},
            )
            print(f"  💾 Saved text intrinsic dimension: {txt_save_path}")
            print(f"  💾 Saved image intrinsic dimension: {img_save_path}")

        del res_txt, res_img, intrinsic_txt, intrinsic_img
        gc.collect()
        torch.cuda.empty_cache()

    if extraction_runtime_initialized:
        del model, processor, dl_txt, dl_img
        gc.collect()
        torch.cuda.empty_cache()

    print("\n✅ Intrinsic dimension analysis completed.")


if __name__ == "__main__":
    main()
