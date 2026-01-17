import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# --- Configuration ---

# Define model collections. The `{}` is a placeholder for the checkpoint number.
MODELS_COLLECTION = {
    "Llava1.5-7B": "l11p/llava-v1.5-7b-lbasile-full-step-{}",
    "Llava1.5-7B-frozen": "l11p/llava-v1.5-7b-lbasile-frozen-llm-step-{}",
    "Llava1.5-7B-0-16": "l11p/llava-v1.5-7b-lbasile-layers-0-16-step-{}",
    "Llava1.5-7B-16-32": "l11p/llava-v1.5-7b-lbasile-layers-16-32-step-{}",
    # Raw lbasile collections
    "Llava1.5-7B-lbasile-raw": "lbasile/llava-{}",
    "Llava1.5-7B-frozen-lbasile-raw": "lbasile/llava_frozen-{}",
    "Llava1.5-7B-0-16-lbasile-raw": "lbasile/llava_0_16-{}",
    "Llava1.5-7B-16-32-lbasile-raw": "lbasile/llava_16_32-{}",
}

CHECKPOINTS = list(range(1000, 10001, 1000))


def download_model(model_name: str, cache_dir: str) -> bool:
    """
    Downloads a model to the Hugging Face cache using snapshot_download.

    Args:
        model_name: The name of the model on the Hub (e.g., 'llava-hf/llava-1.5-7b-hf').
        cache_dir: The directory to cache the model in.

    Returns:
        True if download was successful, False otherwise.
    """
    print(f"\nAttempting to download: {model_name}")
    try:
        # Use snapshot_download to efficiently download all model files
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),  # Use HF_TOKEN for private models
        )
        print(f"  \033[92m✓ Successfully downloaded {model_name}\033[0m")
        return True
    except Exception as e:
        print(f"  \033[91m✗ Failed to download {model_name}\033[0m")
        print(f"    Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models or collections of models from the Hugging Face Hub.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        nargs="+",
        help="One or more specific model IDs to download.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        nargs="+",
        help="One or more collection names to download.\nAvailable: "
        + ", ".join(MODELS_COLLECTION.keys()),
    )
    parser.add_argument(
        "--all-collections",
        action="store_true",
        help="Download all predefined model collections.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Hugging Face cache directory.",
    )
    args = parser.parse_args()

    models_to_download = set()

    # 1. Add single models from --model-id
    if args.model_id:
        for model in args.model_id:
            models_to_download.add(model)

    # 2. Determine which collections to process
    collections_to_process = []
    if args.all_collections:
        collections_to_process = MODELS_COLLECTION.keys()
    elif args.collection:
        collections_to_process = args.collection

    # 3. Add models from the selected collections
    for collection_name in collections_to_process:
        if collection_name not in MODELS_COLLECTION:
            print(
                f"\033[93mWarning: Collection '{collection_name}' not found. Skipping.\033[0m"
            )
            continue

        template = MODELS_COLLECTION[collection_name]
        for checkpoint in CHECKPOINTS:
            models_to_download.add(template.format(checkpoint))

    if not models_to_download:
        print(
            "\033[93mNo models specified. Use --model-id, --collection, or --all-collections.\033[0m"
        )
        parser.print_help()
        return 1

    print(f"--- Starting Download of {len(models_to_download)} Models ---")

    successful_downloads = 0
    failed_downloads = []

    for model_name in sorted(list(models_to_download)):
        if download_model(model_name, args.cache_dir):
            successful_downloads += 1
        else:
            failed_downloads.append(model_name)

    print("\n" + "=" * 80)
    print("--- Download Summary ---")
    print(
        f"\033[92m✓ Successful: {successful_downloads}/{len(models_to_download)}\033[0m"
    )
    if failed_downloads:
        print(
            f"\033[91m✗ Failed: {len(failed_downloads)}/{len(models_to_download)}\033[0m"
        )
        print("  Failed models:")
        for model in failed_downloads:
            print(f"    - {model}")
        return 1

    print("\033[92mAll specified models are cached locally.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
