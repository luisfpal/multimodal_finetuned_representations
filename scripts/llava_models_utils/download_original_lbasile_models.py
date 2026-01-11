import os
import sys
import traceback
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_hf_model_and_processor_or_tokenizer,
    setup_multimodal_model,
)


# ===== Configuration =====
MODEL_CACHE_DIR = os.path.expanduser("~/scratch/huggingface/hub")

MODELS_COLLECTION = {
    # "Llava1.5-7B": "lbasile/llava-",
    # "Llava1.5-7B-frozen": "lbasile/llava_frozen-",
    # "Llava1.5-7B-0-16": "lbasile/llava_0_16-",
    # "Llava1.5-7B-16-32": "lbasile/llava_16_32-",
    # "Llava1.5-7B": "l11p/llava-v1.5-7b-lbasile-full-step-",
    # "Llava1.5-7B-frozen": "l11p/llava-v1.5-7b-lbasile-frozen-llm-step-",
    "Llava1.5-7B-0-16": "l11p/llava-v1.5-7b-lbasile-layers-0-16-step-",
    # "Llava1.5-7B-16-32": "l11p/llava-v1.5-7b-lbasile-layers-16-32-step-",
}

console = Console()


def load_pretrained_model():
    """Load the pretrained LLava-0 model."""
    model, processor = setup_multimodal_model(
        multimodal_model_name_or_path="llava-hf/llava-1.5-7b-hf",
        model_cache_dir=MODEL_CACHE_DIR,
        device_map="cpu",
        attn_implementation="flash_attention_2",
        model_dtype=torch.float16,
    )
    return model, processor


def check_model_loading(model_name, is_pretrained=False):
    """
    Try to load a model and return success status.

    Args:
        model_name: Name of the model to load
        is_pretrained: Whether to use pretrained loading path

    Returns:
        Tuple of (success, error_message, model, processor)
    """
    try:
        console.print(f"\n[cyan]Loading {model_name}...[/cyan]")

        if is_pretrained:
            model, processor = load_pretrained_model()
        else:
            model, processor = load_hf_model_and_processor_or_tokenizer(
                model_name,
                MODEL_CACHE_DIR,
                dtype=torch.float32,
                device_map="cpu",
                attn_implementation="sdpa",
            )

        # Basic validation
        if model is None:
            return False, "Model is None", None, None
        if processor is None:
            return False, "Processor is None", None, None

        console.print(f"[green]✓ Successfully loaded {model_name}[/green]")

        # Clean up to free memory
        del model
        del processor
        torch.cuda.empty_cache()

        return True, None, None, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        console.print(f"[red]✗ Failed to load {model_name}[/red]")
        console.print(f"[red]Error: {error_msg}[/red]")

        # Print full traceback for debugging
        console.print("\n[yellow]Full traceback:[/yellow]")
        traceback.print_exc()

        return False, error_msg, None, None


def check_collection(collection_name):
    """Check all models in a collection."""
    console.print(
        f"\n[bold cyan]Checking models from {collection_name} collection[/bold cyan]\n"
    )

    # Get model names
    collection_prefix = MODELS_COLLECTION[collection_name]
    model_names = [f"{collection_prefix}{i}" for i in range(1000, 10001, 1000)]

    # Track results
    successful_models = []
    failed_models = []

    # Check each model
    for model_name in model_names:
        is_pretrained = "-0" in model_name
        success, error_msg, _, _ = check_model_loading(model_name, is_pretrained)

        if success:
            successful_models.append(model_name)
        else:
            failed_models.append({"name": model_name, "error": error_msg})

    # Print summary
    console.print("\n" + "=" * 80)
    console.print(f"\n[bold]Summary for {collection_name}:[/bold]\n")
    console.print(
        f"[green]✓ Successful: {len(successful_models)}/{len(model_names)}[/green]"
    )
    console.print(f"[red]✗ Failed: {len(failed_models)}/{len(model_names)}[/red]")

    # Show detailed results in table
    if successful_models:
        console.print("\n[bold green]Successfully loaded models:[/bold green]")
        for model in successful_models:
            console.print(f"  ✓ {model}")

    if failed_models:
        console.print("\n[bold red]Failed models:[/bold red]")
        table = Table(show_header=True, header_style="bold red")
        table.add_column("Model Name", style="cyan")
        table.add_column("Error", style="red")

        for failed in failed_models:
            table.add_row(failed["name"], failed["error"])

        console.print(table)

    console.print("\n" + "=" * 80 + "\n")

    return len(failed_models)


def main():
    """Check all model collections."""
    console.print("\n[bold magenta]Checking all model collections[/bold magenta]\n")

    total_failed = 0

    # Check each collection
    for collection_name in MODELS_COLLECTION.keys():
        failed_count = check_collection(collection_name)
        total_failed += failed_count

    # Final summary
    console.print("\n" + "=" * 80)
    console.print("\n[bold]Overall Summary:[/bold]\n")
    console.print(f"Collections checked: {len(MODELS_COLLECTION)}")

    if total_failed == 0:
        console.print("[bold green]All models loaded successfully! ✓[/bold green]")
        return 0
    else:
        console.print(
            f"[bold red]Total failed models across all collections: {total_failed}[/bold red]"
        )
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
