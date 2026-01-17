import argparse
import gc
import io
import os
import sys
import torch
from huggingface_hub import HfApi, create_repo

# Add project root to PYTHONPATH so `utils` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.model_utils import load_hf_model_and_processor_or_tokenizer


# Checkpoints to process for each collection.
CHECKPOINTS = list(range(1000, 10001, 1000))

# Maps a short collection name to a source model name template.
# The `{}` will be replaced by the checkpoint number.
SOURCE_MODEL_TEMPLATES = {
    "lbasile-7b-raw": "lbasile/llava-{}",
    "lbasile-7b-frozen-raw": "lbasile/llava_frozen-{}",
    "lbasile-7b-0-16-raw": "lbasile/llava_0_16-{}",
    "lbasile-7b-16-32-raw": "lbasile/llava_16_32-{}",
}

# Maps a short collection name to a target repository name template.
# This defines the naming scheme for the new models on the Hub.
TARGET_REPO_TEMPLATES = {
    "lbasile-7b-raw": "llava-v1.5-7b-lbasile-full-step-{}",
    "lbasile-7b-frozen-raw": "llava-v1.5-7b-lbasile-frozen-llm-step-{}",
    "lbasile-7b-0-16-raw": "llava-v1.5-7b-lbasile-layers-0-16-step-{}",
    "lbasile-7b-16-32-raw": "llava-v1.5-7b-lbasile-layers-16-32-step-{}",
}


def port_weights_and_push(
    source_model_id: str,
    target_repo_id: str,
    reference_model_id: str,
    cache_dir: str,
    reference_processor,
):
    """
    Loads a source model, transfers its weights to a reference model architecture,
    and pushes the result to a new repository on the Hugging Face Hub.

    This is useful for "fixing" models that have incorrect configurations or are
    missing necessary files (e.g., processor_config.json) by porting their
    weights to a known-good model structure.
    """
    print(f"\n--- Starting port for {source_model_id} ---")
    print(f"  -> Target: {target_repo_id}")
    try:
        # 1. Load the source model (weights only). We skip the processor because
        # it might be broken or missing, which is the reason for this porting.
        print(f"  Loading source model weights: {source_model_id}...")
        source_model = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=source_model_id,
            cache_dir=cache_dir,
            device_map="cpu",
            dtype=torch.float16,
            attn_implementation="sdpa",
            skip_processor=True,
        )

        # 2. Create a fresh instance of the reference model. This will be our
        # "template" with the correct architecture and configuration files.
        print(f"  Creating target model structure from {reference_model_id}...")
        ported_model = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=reference_model_id,
            cache_dir=cache_dir,
            device_map="cpu",
            dtype=torch.float16,
            attn_implementation="sdpa",
            skip_processor=True,
        )

        # 3. Transfer the weights from the source model to our new template.
        # This assumes the sub-module names are consistent across LLaVA 1.5 models.
        print("  Transferring weights...")
        ported_model.vision_tower.load_state_dict(
            source_model.vision_tower.state_dict()
        )
        ported_model.multi_modal_projector.load_state_dict(
            source_model.multi_modal_projector.state_dict()
        )
        ported_model.language_model.load_state_dict(
            source_model.language_model.state_dict()
        )

        # Clean up memory
        del source_model
        gc.collect()
        torch.cuda.empty_cache()

        # 4. Push the ported model and the correct processor to the Hub.
        print(f"  Pushing to Hub: {target_repo_id}...")
        create_repo(target_repo_id, exist_ok=True, private=False)

        readme_content = f"""---
tags:
- image-text-to-text
- llava
- multimodal
- vision-language
---

# {target_repo_id.split('/')[-1]}

Model weights from `{source_model_id}` ported to the `{reference_model_id}` architecture to provide a standardized chat template and processor configuration.
"""

        ported_model.push_to_hub(target_repo_id, commit_message="Port weights and add standard config")
        reference_processor.push_to_hub(target_repo_id)

        # Upload the generated README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=target_repo_id,
            repo_type="model",
        )

        del ported_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  \033[92m✓ Successfully ported and pushed {target_repo_id}\033[0m")
        return True

    except Exception as e:
        print(f"  \033[91m✗ Error during porting: {e}\033[0m")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Port weights from source models to a reference architecture and push to the Hub.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Group for collection-based porting
    collection_group = parser.add_argument_group('Collection Porting')
    collection_group.add_argument(
        "--collection",
        type=str,
        nargs="+",
        help="One or more collection names to process.\nAvailable: "
        + ", ".join(SOURCE_MODEL_TEMPLATES.keys()),
    )
    collection_group.add_argument(
        "--all-collections", action="store_true", help="Process all defined collections."
    )

    # Group for single model porting
    single_model_group = parser.add_argument_group('Specified Models Porting')
    single_model_group.add_argument(
        "--source-model-id",
        type=str,
        nargs="+",
        help="One or more source model IDs to port."
    )
    single_model_group.add_argument(
        "--target-repo-id",
        type=str,
        nargs="+",
        help="The corresponding target repository IDs on the Hub."
    )

    # General arguments
    parser.add_argument(
        "--reference-model",
        type=str,
        required=True,
        help="The reference model ID for architecture and processor.",
    )
    parser.add_argument(
        "--target-org",
        type=str,
        required=True,
        help="The target organization on the Hub (for collections).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Hugging Face cache directory.",
    )
    args = parser.parse_args()

    # --- Argument validation ---
    is_collection_mode = args.collection or args.all_collections
    is_specified_models_mode = args.source_model_id or args.target_repo_id

    if is_specified_models_mode and is_collection_mode:
        parser.error("Cannot use Specified Models Porting arguments (--source-model-id, --target-repo-id) with Collection Porting arguments (--collection, --all-collections).")

    if is_specified_models_mode:
        if not args.source_model_id or not args.target_repo_id:
            parser.error("Both --source-model-id and --target-repo-id must be provided for specified models porting.")
        if len(args.source_model_id) != len(args.target_repo_id):
            parser.error("The number of --source-model-id and --target-repo-id arguments must be the same.")

    if not is_specified_models_mode and not is_collection_mode:
        print("\033[93mNo operation specified. Use arguments for either specified models or collection porting.\033[0m")
        parser.print_help()
        return 1

    # Load the reference processor once to be reused for all models
    print(f"Loading reference processor from {args.reference_model}...")
    try:
        reference_processor = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=args.reference_model,
            cache_dir=args.cache_dir,
            skip_model=True,
        )
        print("\033[92m✓ Reference processor loaded.\033[0m")
    except Exception as e:
        print(f"\033[91m✗ Failed to load reference processor: {e}\033[0m")
        return 1

    # --- Execute porting logic ---
    overall_success = []
    overall_failed = []

    if is_specified_models_mode:
        # --- SPECIFIED MODELS PORTING ---
        print(f"\nProcessing {len(args.source_model_id)} specified models...")
        for source_id, target_id in zip(args.source_model_id, args.target_repo_id):
            success = port_weights_and_push(
                source_model_id=source_id,
                target_repo_id=target_id,
                reference_model_id=args.reference_model,
                cache_dir=args.cache_dir,
                reference_processor=reference_processor,
            )
            if success:
                overall_success.append(target_id)
            else:
                overall_failed.append(source_id)

    else:
        # --- COLLECTION PORTING ---
        collections_to_process = []
        if args.all_collections:
            collections_to_process = SOURCE_MODEL_TEMPLATES.keys()
        elif args.collection:
            collections_to_process = args.collection

        print(f"\nProcessing {len(collections_to_process)} collections...")

        for name in collections_to_process:
            if name not in SOURCE_MODEL_TEMPLATES:
                print(f"\033[93mWarning: Collection '{name}' not found. Skipping.\033[0m")
                continue

            print("\n" + "=" * 80)
            print(f"--- Collection: {name} ---")

            for checkpoint in CHECKPOINTS:
                source_id = SOURCE_MODEL_TEMPLATES[name].format(checkpoint)
                target_name = TARGET_REPO_TEMPLATES[name].format(checkpoint)
                target_id = f"{args.target_org}/{target_name}"

                success = port_weights_and_push(
                    source_model_id=source_id,
                    target_repo_id=target_id,
                    reference_model_id=args.reference_model,
                    cache_dir=args.cache_dir,
                    reference_processor=reference_processor,
                )
                if success:
                    overall_success.append(target_id)
                else:
                    overall_failed.append(source_id)

    # --- Final Summary ---
    total_models = len(overall_success) + len(overall_failed)
    print("\n" + "=" * 80)
    print("--- Overall Summary ---")
    print(f"\033[92m✓ Successful ports: {len(overall_success)}/{total_models}\033[0m")
    if overall_failed:
        print(f"\033[91m✗ Failed ports: {len(overall_failed)}/{total_models}\033[0m")
        print("  Failed source models:")
        for model in overall_failed:
            print(f"    - {model}")
        return 1

    print("\033[92mAll specified models ported successfully.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
