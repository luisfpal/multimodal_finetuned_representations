import os
import sys

CACHE_DIR = os.path.expanduser("~/scratch/huggingface/hub/")
os.environ["HF_HUB_CACHE"] = CACHE_DIR
import gc
import io
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import torch
from huggingface_hub import create_repo, HfApi, whoami
from utils import load_hf_model_and_processor_or_tokenizer

# Model collections to fix
MODELS_COLLECTION = {
    # "Llava1.5-7B": "lbasile/llava-",
    # "Llava1.5-7B-0-16": "lbasile/llava_0_16-",
    # "Llava1.5-7B-16-32": "lbasile/llava_16_32-",
    "Llava1.5-7B-frozen": "lbasile/llava_frozen-",
}

# Reference model with proper processor config
REFERENCE_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

# Target organization for fixed models
TARGET_ORG = "l11p"


def get_collection_description(collection_name):
    """Get human-readable description for collection."""
    descriptions = {
        "Llava1.5-7B": "full model",
        "Llava1.5-7B-0-16": "layers 0-16",
        "Llava1.5-7B-16-32": "layers 16-32",
        "Llava1.5-7B-frozen": "frozen layers",
    }
    return descriptions.get(collection_name, collection_name)


def fix_single_model(source_model_name, target_repo_id, reference_model, reference_processor):
    """
    Fix a single model by transferring weights to reference architecture.
    
    Args:
        source_model_name: HF identifier of source lbasile model
        target_repo_id: Target repository ID for fixed model
        reference_model: Pre-loaded reference model (to avoid reloading)
        reference_processor: Pre-loaded reference processor
    """
    try:
        # Load source model weights
        print(f"  [cyan]Loading {source_model_name}...[/cyan]")
        source_model = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=source_model_name,
            cache_dir=CACHE_DIR,
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation="sdpa",
            skip_processor=True,
        )

        # Create a fresh copy of reference model for this iteration
        print(f"  [cyan]Transferring weights...[/cyan]")
        fixed_model = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=REFERENCE_MODEL_NAME,
            cache_dir=CACHE_DIR,
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation="sdpa",
            skip_processor=True,
        )

        # Transfer weights
        fixed_model.vision_tower.load_state_dict(source_model.vision_tower.state_dict())
        fixed_model.multi_modal_projector.load_state_dict(source_model.multi_modal_projector.state_dict())
        fixed_model.language_model.load_state_dict(source_model.language_model.state_dict())

        del source_model
        gc.collect()
        torch.cuda.empty_cache()

        # Push to Hub
        print(f"  [cyan]Pushing to {target_repo_id}...[/cyan]")
        create_repo(target_repo_id, exist_ok=True)
        
        # Create README
        readme_content = f"""---
tags:
- image-text-to-text
- llava
- multimodal
- vision-language
---

# {target_repo_id.split('/')[-1]}

`{source_model_name}` with proper chat template and processor config added.
"""
        
        fixed_model.push_to_hub(target_repo_id, commit_message="Add proper processor config")
        reference_processor.push_to_hub(target_repo_id)
        
        # Upload README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=target_repo_id,
            repo_type="model",
        )
        
        del fixed_model
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"  [green]✓ Successfully pushed {target_repo_id}[/green]")
        return True

    except Exception as e:
        print(f"  [red]✗ Error: {e}[/red]")
        return False


def main():
    # Check authentication
    print("[cyan]Checking HuggingFace authentication...[/cyan]")
    try:
        user_info = whoami()
        print(f"[green]✓ Authenticated as: {user_info['name']}[/green]")
    except Exception as e:
        print(f"[red]✗ Authentication error: {e}[/red]")
        print("[yellow]Please login with: huggingface-cli login[/yellow]")
        sys.exit(1)

    # Load reference processor once (reuse for all models)
    print(f"\n[cyan]Loading reference processor from {REFERENCE_MODEL_NAME}...[/cyan]")
    _, reference_processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=REFERENCE_MODEL_NAME,
        cache_dir=CACHE_DIR,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    print("[green]✓ Reference processor loaded[/green]")

    # Process each collection
    checkpoints = list(range(1000, 10001, 1000))
    total_models = len(MODELS_COLLECTION) * len(checkpoints)
    
    print(f"\n[bold cyan]Processing {total_models} models across {len(MODELS_COLLECTION)} collections[/bold cyan]\n")
    
    results = {"success": [], "failed": []}
    
    for collection_name, model_prefix in MODELS_COLLECTION.items():
        collection_desc = get_collection_description(collection_name)
        print(f"\n[bold yellow]{'='*80}[/bold yellow]")
        print(f"[bold yellow]Collection: {collection_name} ({collection_desc})[/bold yellow]")
        print(f"[bold yellow]{'='*80}[/bold yellow]\n")
        
        for checkpoint in checkpoints:
            source_model_name = f"{model_prefix}{checkpoint}"
            
            # Generate target repo ID matching pattern: l11p/llava-v1.5-7b-lbasile-layers-0-16-step-1000
            if collection_name == "Llava1.5-7B":
                target_repo_name = f"llava-v1.5-7b-lbasile-full-step-{checkpoint}"
            elif collection_name == "Llava1.5-7B-0-16":
                target_repo_name = f"llava-v1.5-7b-lbasile-layers-0-16-step-{checkpoint}"
            elif collection_name == "Llava1.5-7B-16-32":
                target_repo_name = f"llava-v1.5-7b-lbasile-layers-16-32-step-{checkpoint}"
            elif collection_name == "Llava1.5-7B-frozen":
                target_repo_name = f"llava-v1.5-7b-lbasile-frozen-llm-step-{checkpoint}"
            else:
                # Fallback for any new collections
                collection_suffix = collection_name.replace("Llava1.5-7B-", "").lower()
                target_repo_name = f"llava-v1.5-7b-lbasile-{collection_suffix}-step-{checkpoint}"
            
            target_repo_id = f"{TARGET_ORG}/{target_repo_name}"
            
            print(f"[bold]Checkpoint {checkpoint}:[/bold]")
            print(f"  Source: {source_model_name}")
            print(f"  Target: {target_repo_id}")
            
            success = fix_single_model(
                source_model_name=source_model_name,
                target_repo_id=target_repo_id,
                reference_model=None,  # Load fresh each time
                reference_processor=reference_processor
            )
            
            if success:
                results["success"].append(target_repo_id)
            else:
                results["failed"].append(source_model_name)
            
            print()  # Empty line for readability
    
    # Print summary
    print(f"\n[bold yellow]{'='*80}[/bold yellow]")
    print(f"[bold yellow]Summary[/bold yellow]")
    print(f"[bold yellow]{'='*80}[/bold yellow]\n")
    print(f"[green]✓ Success: {len(results['success'])}/{total_models}[/green]")
    print(f"[red]✗ Failed: {len(results['failed'])}/{total_models}[/red]")
    
    if results["failed"]:
        print("\n[red]Failed models:[/red]")
        for model in results["failed"]:
            print(f"  - {model}")


if __name__ == "__main__":
    main()
