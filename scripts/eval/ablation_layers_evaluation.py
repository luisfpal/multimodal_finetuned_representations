import argparse
import json
import os
import sys
from pathlib import Path

from lmms_eval.evaluator import simple_evaluate

# Try specific imports, handling environment differences
try:
    from lmms_eval.models.chat.llava_hf import LlavaHf
except ImportError:
    LlavaHf = None

# Add project root to path for local imports if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from scripts.models_utils.internvl2 import InternVL2Custom
except ImportError:
    InternVL2Custom = None

from utils import SYSTEM_INSTRUCTION, INTERNVL_SYSTEM_INSTRUCTION

# =============================================================================
# CONSTANTS
# =============================================================================

# Reproducibility (all seeds = 42)
RANDOM_SEED = NUMPY_RANDOM_SEED = TORCH_RANDOM_SEED = FEWSHOT_RANDOM_SEED = 42

TASK_LIMITS = {
    "mmbench_en_dev": 2000,
    "gqa": 2000,
    "vqav2_val": 2000,
    "vizwiz_vqa_val": 2000,
}

# Tasks requiring LLM judge (need API key)
LLM_JUDGE_TASKS = {"mmbench_en_dev"}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_cli_args(output_path: str) -> argparse.Namespace:
    """Create cli_args for tasks that generate submission files."""
    return argparse.Namespace(output_path=output_path, process_with_media=True)


def validate_task_requirements(task_name: str) -> None:
    """Validate environment variables for tasks requiring LLM judge."""
    if task_name in LLM_JUDGE_TASKS:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
        model_version = os.environ.get("MODEL_VERSION", "gpt-4o-2024-11-20")

        if not api_key:
            print("[bold red]✗ ERROR: LLM judge API key missing![/bold red]")
            print(f"  Task '{task_name}' requires OPENAI_API_KEY\n")
            sys.exit(1)
        print(f"[green]✓ LLM Judge: {model_version} @ {api_url}[/green]")

    if task_name == "seedbench":
        print(
            "[yellow]⚠ Note: seedbench writes to ./seed_submission.json (hardcoded)[/yellow]"
        )


def save_model_responses(
    output_file: str,
    samples: dict,
    task_name: str,
) -> None:
    """Save model responses to JSON file."""
    task_keys = [k for k in samples if k.startswith(task_name)]
    if not task_keys:
        print(f"[yellow]⚠ Warning: No samples found for task '{task_name}'[/yellow]")
        return

    output = (
        samples[task_keys[0]]
        if len(task_keys) == 1
        else {k: samples[k] for k in task_keys}
    )

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation for LLaVA models")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Path or HF ID of the model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llava_hf",
        choices=["llava_hf", "internvl2"],
        help="Type of model to evaluate",
    )
    parser.add_argument("--task", type=str, required=True, help="Task to evaluate")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples"
    )
    parser.add_argument(
        "--layers-to-skip", type=int, default=0, help="Number of layers to skip"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/ablations", help="Output directory"
    )

    args = parser.parse_args()

    # Determine limit
    limit = args.limit if args.limit is not None else TASK_LIMITS.get(args.task, None)

    # Prepare output
    model_name_safe = args.model_name_or_path.replace("/", "_")
    output_dir = Path(args.output_dir) / model_name_safe / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / f"{args.task}.jsonl"

    # Responses output path
    responses_file = os.path.join(
        output_dir, f"{args.layers_to_skip}-ablated_layers.json"
    )

    # Avoid recomputing results
    if os.path.exists(responses_file):
        print(f"    ✅ Found existing: {responses_file}")
        print("    ⏩ Skipping...")
    else:
        # Patching
        if args.layers_to_skip > 0:
            try:
                from utils.patch_skipping import patch

                patch(layers_to_skip=args.layers_to_skip)
                print(f"Applied patch: skipping first {args.layers_to_skip} layers")
            except ImportError:
                print(
                    "Error: Could not import 'src.patch_skipping'. Ensure you are running from the project root or src is in PYTHONPATH."
                )
                sys.exit(1)

        # Load Model
        try:
            if args.model_type == "llava_hf":
                if LlavaHf is None:
                    raise ImportError(
                        "LlavaHf class not available. Check your environment."
                    )
                model = LlavaHf(
                    pretrained=args.model_name_or_path,
                    device_map="auto",
                    batch_size=1,  # required by the lmms_eval library
                )
            elif args.model_type == "internvl2":
                if InternVL2Custom is None:
                    raise ImportError(
                        "InternVL2Custom class not available. Check imports."
                    )

                # Prepare cachedir if possible (not passed in args but can use env)
                cache_dir = os.environ.get("HF_HUB_CACHE")

                model = InternVL2Custom(
                    pretrained=args.model_name_or_path,
                    device_map="auto",
                    batch_size="1",
                    cache_dir=cache_dir,
                )
            else:
                raise ValueError(f"Unknown model type: {args.model_type}")

        except Exception as e:
            print(f"Failed to load model {args.model_name_or_path}: {e}")
            sys.exit(1)

        # Select system instruction
        system_instruction = (
            INTERNVL_SYSTEM_INSTRUCTION
            if args.model_type == "internvl2"
            else SYSTEM_INSTRUCTION
        )

        # Evaluate
        results = simple_evaluate(
            model=model,
            tasks=[args.task],
            limit=limit,
            batch_size=1,
            log_samples=True,
            system_instruction=system_instruction,
            cli_args=create_cli_args(str(output_dir)),
            random_seed=RANDOM_SEED,
            numpy_random_seed=NUMPY_RANDOM_SEED,
            torch_random_seed=TORCH_RANDOM_SEED,
            fewshot_random_seed=FEWSHOT_RANDOM_SEED,
        )

        # Extract metrics
        if args.task in results["results"]:
            metrics = results["results"][args.task]

            entry = {"layers_ablated": args.layers_to_skip, **metrics}

            # Append to JSONL
            # Use str(metrics_file) to avoid any Path object issues
            with open(str(metrics_file), "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"Results saved to {metrics_file}")
        else:
            print(f"[red]✗ ERROR: No results found for task '{args.task}'[/red]")

        if results and "samples" in results:
            save_model_responses(
                output_file=str(responses_file),
                samples=results["samples"],
                task_name=args.task,
            )
        else:
            print(
                "[yellow]⚠ Warning: No samples found, skipping responses save[/yellow]"
            )


if __name__ == "__main__":
    main()
