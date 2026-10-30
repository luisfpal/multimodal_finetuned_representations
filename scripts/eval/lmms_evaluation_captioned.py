"""
Captioned evaluation script for lmms-eval.

Extends the standard evaluation pipeline to support BOTH:
  - Original visual evaluation (images as context) — any built-in task
  - Captioned evaluation (text descriptions replace images) — custom tasks

Uses custom task definitions via lmms-eval's `include_path` mechanism.
No library modifications required. Existing scripts are untouched.

Available captioned tasks:
  mme_captioned              — MME with captions replacing images
  mmbench_en_dev_captioned   — MMBench EN dev with captions
    mmbench_en_dev_open_ended  — MMBench EN dev visual open-ended
    mmbench_en_dev_captioned_open_ended — MMBench EN dev captioned open-ended
    seedbench_open_ended       — SEED-Bench visual open-ended
    seedbench_captioned_open_ended — SEED-Bench captioned open-ended
  vqav2_val_captioned_synthetic — VQAv2 val with synthetic captions
  vqav2_val_captioned_human    — VQAv2 val with human captions
  gqa_val_balanced           — GQA val-balanced WITH images (new dataset)
  gqa_val_balanced_captioned — GQA val-balanced with captions

Original tasks also work (mme, mmbench_en_dev, vqav2_val, gqa, etc.)
because the TaskManager loads both built-in and custom tasks.

Usage:
    python scripts/eval/lmms_evaluation_captioned.py \\
        --task-name mme_captioned \\
        --model-name-or-path OpenGVLab/InternVL2-8B \\
        --lmms-eval-model internvl2 \\
        --model-cache-dir ~/scratch/huggingface/hub \\
        --output-dir evals/InternVL2-8B/lmms_eval/mme_captioned
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from rich import print
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.tasks import TaskManager

from utils import (
    LLaVA1_5_SYSTEM_INSTRUCTION,
    INTERNVL2_SYSTEM_INSTRUCTION,
    ensure_llm_judge_env,
    infer_lmms_eval_backend,
)

try:
    from lmms_eval.models.chat.llava_hf import LlavaHf
except ImportError:
    LlavaHf = None

try:
    from scripts.models_utils.internvl2 import InternVL2Custom
except ImportError:
    InternVL2Custom = None


# =============================================================================
# CONSTANTS
# =============================================================================

# Project root and custom tasks path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CUSTOM_TASKS_DIR = str(PROJECT_ROOT / "tasks" / "captioned")

# Reproducibility (all seeds = 42)
RANDOM_SEED = NUMPY_RANDOM_SEED = TORCH_RANDOM_SEED = FEWSHOT_RANDOM_SEED = 42

# Generation token limits
DEFAULT_MAX_NEW_TOKENS = 128
LONG_ANSWER_MAX_NEW_TOKENS = 1024

# Task-specific sample limits (covers both original and captioned tasks)
TASK_LIMITS = {
    # Original tasks
    # "mme": 26,
    "pope": 5000,
    "gqa": 5000,
    "seedbench": 5000,
    "mmbench_en_dev": 5000,
    "mmbench_cn_dev": 5000,
    "vizwiz_vqa_val": 5000,
    "vqav2_val": 5000,
    "ai2d": 5000,
    "flickr30k": 5000,
    "coco2017_cap_val": 5000,
    # Captioned variants (same limits as originals)
    # "mme_captioned": 26,
    "mmbench_en_dev_captioned": 5000,
    # "mmbench_en_dev_open_ended": 5000,
    # "mmbench_en_dev_captioned_open_ended": 5000,
    # "seedbench_open_ended": 5000,
    # "seedbench_captioned_open_ended": 5000,
    "vqav2_val_captioned_synthetic": 5000,
    "vqav2_val_captioned_human": 5000,
    "gqa_val_balanced": 5000,
    "gqa_val_balanced_captioned": 5000,
}

# Tasks requiring longer outputs
LONG_ANSWER_TASKS = {
    "llava_in_the_wild",
    "mmvet",
    "mmbench_en_dev",
    "mmbench_cn_dev",
    "scienceqa_img",
    "mmbench_en_dev_captioned",
}

# Tasks requiring LLM judge
LLM_JUDGE_TASKS = {
    "mmvet",
    "mmbench_en_dev",
    "mmbench_cn_dev",
    "llava_in_the_wild",
    "mmbench_en_dev_captioned",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_max_new_tokens(task_name: str, override: Optional[int] = None) -> int:
    """Get max_new_tokens for a task (128 default, 1024 for long-answer tasks)."""
    if override is not None:
        return override
    return (
        LONG_ANSWER_MAX_NEW_TOKENS
        if task_name in LONG_ANSWER_TASKS
        else DEFAULT_MAX_NEW_TOKENS
    )


def build_gen_kwargs(
    task_name: str, max_new_tokens_override: Optional[int] = None
) -> str:
    """Build generation kwargs string for simple_evaluate."""
    return f"max_new_tokens={get_max_new_tokens(task_name, max_new_tokens_override)}"


def create_cli_args(output_path: str) -> argparse.Namespace:
    """Create cli_args for tasks that generate submission files."""
    return argparse.Namespace(output_path=output_path, process_with_media=True)


def validate_task_requirements(task_name: str) -> None:
    """Validate environment variables for tasks requiring LLM judge.

    Automatically loads .env and acquires an Orfeo JWT if OPENAI_API_KEY
    is not already set in the environment.
    """
    if task_name in LLM_JUDGE_TASKS:
        if not ensure_llm_judge_env():
            print("[bold red]✗ ERROR: LLM judge API key missing![/bold red]")
            print(f"  Task '{task_name}' requires OPENAI_API_KEY")
            print("  Set it directly, or configure Orfeo credentials in .env\n")
            sys.exit(1)

        api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
        model_version = os.environ.get("MODEL_VERSION", "gpt-4o-2024-11-20")
        print(f"[green]✓ LLM Judge: {model_version} @ {api_url}[/green]")


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time as Xh Ym Zs."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m > 0 else f"{seconds:.1f}s"


def create_task_manager() -> TaskManager:
    """Create a TaskManager that includes both built-in and custom captioned tasks."""
    return TaskManager(include_path=[CUSTOM_TASKS_DIR])


def save_and_update_models_metrics(
    model_identifier: str,
    dataset_dir: str,
    raw_results: dict,
    prefix: str = "",
    suffix: str = "",
    num_fewshot: Optional[int] = None,
    elapsed_time_sec: Optional[float] = None,
) -> None:
    """Append evaluation metrics to JSONL results file."""
    dataset_dir_name = os.path.basename(dataset_dir)
    filename = f"{dataset_dir_name}_results"
    if num_fewshot is not None:
        filename += f"_fewshot{num_fewshot}"

    full_identifier = f"{model_identifier}_{suffix}" if suffix else model_identifier
    results = {"experiment_name": f"{prefix}/{full_identifier}", **raw_results}
    if elapsed_time_sec is not None:
        results["elapsed_time"] = format_elapsed_time(elapsed_time_sec)

    with open(os.path.join(dataset_dir, f"{filename}.jsonl"), "a") as f:
        f.write(json.dumps(results) + "\n")


def save_model_responses(
    model_identifier: str,
    output_dir: str,
    samples: dict,
    task_name: str,
    num_fewshot: Optional[int] = None,
) -> None:
    """Save model responses to JSON file."""
    filename = f"{model_identifier}_responses"
    if num_fewshot is not None:
        filename += f"_fewshot{num_fewshot}"

    task_keys = [k for k in samples if k.startswith(task_name)]
    output = (
        samples[task_keys[0]]
        if len(task_keys) == 1
        else {k: samples[k] for k in task_keys}
    )

    with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
        json.dump(output, f, indent=2)


# =============================================================================
# MODEL INSTANTIATION
# =============================================================================


def _patch_llava_for_text_only(model_instance):
    """Patch LlavaHf to handle empty visuals (text-only captioned mode).

    LlavaHf's generate_until crashes when doc_to_visual returns [] because the
    processor's image_processing pipeline can't handle images=[]. This wraps the
    processor to convert empty image lists to None, enabling text-only generation.

    For visual tasks (non-empty images), the wrapper passes through unchanged.
    """
    original_processor = model_instance._image_processor

    class _TextOnlyCompatProcessor:
        """Thin wrapper: converts images=[] → images=None for text-only mode."""

        def __init__(self, processor):
            self._processor = processor

        def __call__(self, *args, **kwargs):
            if (
                "images" in kwargs
                and isinstance(kwargs["images"], list)
                and len(kwargs["images"]) == 0
            ):
                kwargs["images"] = None
            return self._processor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._processor, name)

    model_instance._image_processor = _TextOnlyCompatProcessor(original_processor)
    return model_instance


def instantiate_model(
    model_name: str,
    args: argparse.Namespace,
    model_args: str,
    backend: str,
):
    """Instantiate the appropriate model based on the resolved backend.

    Returns:
        The model instance (object or string for registry lookup).
    """
    if backend == "llava_hf":
        if LlavaHf is None:
            print("[yellow]⚠ LlavaHf not importable, falling back to registry[/yellow]")
            return backend
        print(f"[cyan]Instantiating LlavaHf for {model_name}...[/cyan]")
        instance = LlavaHf(
            pretrained=model_name,
            device_map=(
                model_args.split("device_map=")[1]
                if "device_map=" in model_args
                else "auto"
            ),
            batch_size=args.batch_size if args.batch_size else 1,
        )
        return _patch_llava_for_text_only(instance)

    elif backend == "internvl2":
        if InternVL2Custom is None:
            print(
                "[yellow]⚠ InternVL2Custom not importable, falling back to registry[/yellow]"
            )
            return backend
        print(f"[cyan]Instantiating InternVL2Custom for {model_name}...[/cyan]")
        return InternVL2Custom(
            pretrained=model_name,
            device=(
                f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
                if torch.cuda.is_available()
                else "cpu"
            ),
            device_map=(
                model_args.split("device_map=")[1]
                if "device_map=" in model_args
                else "auto"
            ),
            batch_size=str(args.batch_size) if args.batch_size else "1",
            cache_dir=args.model_cache_dir,
        )

    return backend  # Fallback: string for registry


def get_system_instruction(lmms_eval_model: str) -> str:
    """Select the appropriate system instruction for the model."""
    if lmms_eval_model == "internvl2":
        return INTERNVL2_SYSTEM_INSTRUCTION
    return LLaVA1_5_SYSTEM_INSTRUCTION


# =============================================================================
# EVALUATION
# =============================================================================


def run_evaluation(
    model_name: str,
    output_dir: str,
    args: argparse.Namespace,
    gen_kwargs: str,
    task_manager: TaskManager,
    model_label: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
) -> float:
    """
    Run a single model evaluation and save results.

    Returns elapsed time in seconds.
    """
    model_identifier = model_name.replace("/", "_")
    label = model_label or model_name

    print(f"[cyan]>>> {label}[/cyan]")
    start_time = time.time()

    # --- Limit Logic ---
    limit = args.limit
    if limit is None:
        limit = TASK_LIMITS.get(args.task_name)
    if limit:
        print(f"[yellow]Applying sample limit: {limit}[/yellow]")

    # Device mapping (NUM_GPUS > 1 = accelerate distributed)
    num_gpus = int(os.environ.get("NUM_GPUS", 1))
    if num_gpus > 1:
        model_args = f"pretrained={model_name},device_map="
    else:
        model_args = f"pretrained={model_name},device_map=auto"

    # Resolve backend (auto-infer from model name if not explicitly provided)
    backend = args.lmms_eval_model or infer_lmms_eval_backend(model_name)

    # Instantiate model
    model_instance = instantiate_model(model_name, args, model_args, backend)
    system_instruction = get_system_instruction(backend)

    # === Core evaluation call with custom task_manager ===
    results = simple_evaluate(
        model=model_instance,
        model_args=model_args if isinstance(model_instance, str) else None,
        tasks=[args.task_name],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        log_samples=True,
        limit=limit,
        system_instruction=system_instruction,
        cli_args=create_cli_args(output_dir),
        gen_kwargs=gen_kwargs,
        task_manager=task_manager,
        random_seed=RANDOM_SEED,
        numpy_random_seed=NUMPY_RANDOM_SEED,
        torch_random_seed=TORCH_RANDOM_SEED,
        fewshot_random_seed=FEWSHOT_RANDOM_SEED,
    )

    elapsed_sec = time.time() - start_time
    print(f"[green]✓ {label}: {format_elapsed_time(elapsed_sec)}[/green]")

    # --- Save results (rank 0 only in distributed mode) ---
    is_main_process = True
    if num_gpus > 1:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                is_main_process = dist.get_rank() == 0
        except Exception:
            pass

    if is_main_process:
        if results and "results" in results:
            found_results = False
            for res_key, res_metrics in results["results"].items():
                if not res_metrics:
                    continue
                if len(res_metrics) == 1 and " " in res_metrics:
                    continue
                metric_keys = [
                    k
                    for k in res_metrics.keys()
                    if k != "alias" and k != " " and not k.endswith("_stderr,none")
                ]
                if not metric_keys:
                    continue

                found_results = True
                current_suffix = suffix
                if res_key != args.task_name:
                    current_suffix = f"{suffix}_{res_key}" if suffix else res_key

                save_and_update_models_metrics(
                    dataset_dir=args.output_dir,
                    model_identifier=model_identifier,
                    prefix=prefix,
                    suffix=current_suffix,
                    raw_results=res_metrics,
                    num_fewshot=args.num_fewshot,
                    elapsed_time_sec=elapsed_sec,
                )

            if not found_results:
                print(f"[yellow]⚠ No valid metrics found for {args.task_name}[/yellow]")
        else:
            print(f"[yellow]⚠ Incomplete results for {args.task_name}[/yellow]")

        response_identifier = (
            f"{model_identifier}_{suffix}" if suffix else model_identifier
        )
        if results and "samples" in results:
            save_model_responses(
                model_identifier=response_identifier,
                output_dir=output_dir,
                samples=results["samples"],
                task_name=args.task_name,
                num_fewshot=args.num_fewshot,
            )
        else:
            print("[yellow]⚠ No samples found, skipping responses save[/yellow]")

    # Memory cleanup
    del results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    gc.collect()

    return elapsed_sec


# =============================================================================
# EVALUATION MODES
# =============================================================================


def eval_single(args: argparse.Namespace) -> None:
    """Evaluate a single model."""
    print("[bold magenta]=== Single Model ===[/bold magenta]")
    print(f"[cyan]{args.model_name_or_path}[/cyan]")
    validate_task_requirements(args.task_name)

    os.makedirs(args.output_dir, exist_ok=True)
    gen_kwargs = build_gen_kwargs(args.task_name, args.max_new_tokens)
    task_manager = create_task_manager()

    print(f"[cyan]Task: {args.task_name} | Generation: {gen_kwargs}[/cyan]")
    print(f"[dim]Custom tasks dir: {CUSTOM_TASKS_DIR}[/dim]")
    print()

    run_evaluation(
        model_name=args.model_name_or_path,
        output_dir=args.output_dir,
        args=args,
        gen_kwargs=gen_kwargs,
        task_manager=task_manager,
        prefix="single",
    )

    print()
    print("[green]✓ Complete[/green]")
    print(f"[cyan]  Saved to: {args.output_dir}[/cyan]")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs with lmms-eval (supports captioned + visual tasks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Captioned tasks (images replaced by text captions):
  mme_captioned, mmbench_en_dev_captioned,
    mmbench_en_dev_open_ended, mmbench_en_dev_captioned_open_ended,
    seedbench_open_ended, seedbench_captioned_open_ended,
  vqav2_val_captioned_synthetic, vqav2_val_captioned_human,
  gqa_val_balanced_captioned

Visual tasks using new datasets:
  gqa_val_balanced (GQA val-balanced with images, no-yesno)

Original tasks also work:
  mme, mmbench_en_dev, vqav2_val, gqa, etc.
""",
    )

    parser.add_argument(
        "--lmms-eval-model",
        type=str,
        default=None,
        help="Model backend (auto-inferred if not set): llava_hf, internvl2",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--model-cache-dir", type=str, required=True, help="Directory for model cache"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for evaluation outputs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--task-name", type=str, required=True, help="Task name (original or captioned)"
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=None, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max samples (overrides TASK_LIMITS)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max generation tokens (auto if not set)",
    )

    args = parser.parse_args()
    eval_single(args)


if __name__ == "__main__":
    main()
