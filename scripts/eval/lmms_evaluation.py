# Standard library
import argparse
import gc
import json
import os
import sys
import time
from typing import Optional

# Third-party
import torch
from rich import print

# Local
from lmms_eval.evaluator import simple_evaluate
from utils import SYSTEM_INSTRUCTION


# =============================================================================
# CONSTANTS
# =============================================================================

# Reproducibility (all seeds = 42)
RANDOM_SEED = NUMPY_RANDOM_SEED = TORCH_RANDOM_SEED = FEWSHOT_RANDOM_SEED = 42

# Generation token limits
DEFAULT_MAX_NEW_TOKENS = 128
LONG_ANSWER_MAX_NEW_TOKENS = 1024

# Task-specific sample limits (default for quick evaluation/testing; full dataset sizes in inline comments)
TASK_LIMITS = {
    # "scienceqa_img": 25,  # 2018 samples
    # "mme": 25,  # 2374 samples
    "textvqa_val": 2000,  # 5000 samples
    "pope": 2000,  # 9000 samples
    "gqa": 2000,  # 12578 samples
    "seedbench": 2000,  # 17990 samples
    "mmbench_en_dev": 2000,  # 4330 samples
    "mmbench_cn_dev": 2000,  # 4330 samples
    "mmvet": 20,  # 218 samples (API judge required)
    "vizwiz_vqa_val": 2000,  # 4320 samples
    "vqav2_val": 2000,  # 214000 samples
    # "llava_in_the_wild": 15,  # 60 samples, heavy post processing
}

# Tasks requiring longer outputs (reasoning, descriptions)
LONG_ANSWER_TASKS = {
    "llava_in_the_wild",
    "mmvet",
    "mmbench_en_dev",
    "mmbench_en_test",
    "mmbench_cn_dev",
    "mmbench_cn_test",
    "scienceqa_img",
}


# Tasks requiring LLM judge (need API key)
LLM_JUDGE_TASKS = {"mmvet", "mmbench_en_dev", "mmbench_cn_dev", "llava_in_the_wild"}

# Tasks with standard evaluation (no external API needed)
SAFE_TASKS = {
    "scienceqa_img",
    "mme",
    "pope",
    "gqa",
    "textvqa_val",
    "vizwiz_vqa_test",
    "vqav2_test",
    "seedbench",
    "mmbench_en_test",
    "mmbench_cn_test",  # Test variants: submission only
}

# Tasks that generate submission files for offline evaluation
SUBMISSION_ONLY_TASKS = {
    "mmbench_en_test",
    "mmbench_cn_test",
    "vqav2_test",
    "vizwiz_vqa_test",
    "textvqa_val",
}

MODELS_COLLECTION = {
    "Llava1.5-7B": "l11p/llava-v1.5-7b-lbasile-full-step-",
    "Llava1.5-7B-frozen": "l11p/llava-v1.5-7b-lbasile-frozen-llm-step-",
    "Llava1.5-7B-0-16": "l11p/llava-v1.5-7b-lbasile-layers-0-16-step-",
    "Llava1.5-7B-16-32": "l11p/llava-v1.5-7b-lbasile-layers-16-32-step-",
    "Llava1.5-7B-0-8": "l11p/llava-v1.5-7b-lbasile-layers-0-8-step-",
    "Llava1.5-7B-8-16": "l11p/llava-v1.5-7b-lbasile-layers-8-16-step-",
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
    """Validate environment variables for tasks requiring LLM judge."""
    if task_name in LLM_JUDGE_TASKS:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
        model_version = os.environ.get("MODEL_VERSION", "gpt-4o-2024-11-20")

        if not api_key:
            print("[bold red]✗ ERROR: LLM judge API key missing![/bold red]")
            print(f"  Task '{task_name}' requires OPENAI_API_KEY\n")
            print("[bold cyan]Supported APIs:[/bold cyan]")
            print("  • Groq (FREE):      OPENAI_API_URL=https://api.groq.com/openai/v1")
            print("  • Together.ai:      OPENAI_API_URL=https://api.together.xyz/v1")
            print("  • Local vLLM:       OPENAI_API_URL=http://localhost:8000/v1")
            print("  • OpenAI (paid):    OPENAI_API_URL=https://api.openai.com/v1\n")
            print(f"[yellow]Safe tasks (no judge):[/yellow] {', '.join(sorted(SAFE_TASKS))}")
            sys.exit(1)
        print(f"[green]✓ LLM Judge: {model_version} @ {api_url}[/green]")

    if task_name == "seedbench":
        print("[yellow]⚠ Note: seedbench writes to ./seed_submission.json (hardcoded)[/yellow]")


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time as Xh Ym Zs."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m > 0 else f"{seconds:.1f}s"


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
# EVALUATION FUNCTIONS
# =============================================================================


def run_evaluation(
    model_name: str,
    output_dir: str,
    args: argparse.Namespace,
    gen_kwargs: str,
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
    # 1. User-provided --limit takes highest priority.
    # 2. If no override, check for a task-specific default limit.
    # 3. Otherwise, no limit.
    limit = args.limit
    if limit is None:
        limit = TASK_LIMITS.get(args.task_name)

    if limit:
        print(f"[yellow]Applying sample limit: {limit}[/yellow]")

    # Device mapping logic (inferred from NUM_GPUS):
    # - NUM_GPUS>1: device_map="" (empty) → accelerate distributed mode
    # - NUM_GPUS=1: device_map=auto → plain python with HF auto-sharding
    num_gpus = int(os.environ.get("NUM_GPUS", 1))

    if num_gpus > 1:
        # Empty string signals to llava_hf that we're using accelerate distributed
        # The model will use accelerator.device for device placement
        model_args = f"pretrained={model_name},device_map="
    else:
        # Single GPU: use HF auto device mapping
        model_args = f"pretrained={model_name},device_map=auto"

    results = simple_evaluate(
        model=args.lmms_eval_model,
        model_args=model_args,
        tasks=[args.task_name],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        log_samples=True,
        limit=limit,
        system_instruction=SYSTEM_INSTRUCTION,
        apply_chat_template="chat" in args.lmms_eval_model,  # Auto-enable for chat models
        cli_args=create_cli_args(output_dir),
        gen_kwargs=gen_kwargs,
        random_seed=RANDOM_SEED,
        numpy_random_seed=NUMPY_RANDOM_SEED,
        torch_random_seed=TORCH_RANDOM_SEED,
        fewshot_random_seed=FEWSHOT_RANDOM_SEED,
    )
    elapsed_sec = time.time() - start_time
    print(f"[green]✓ {label}: {format_elapsed_time(elapsed_sec)}[/green]")

    # In distributed mode, only rank 0 should save results
    # This prevents race conditions and duplicate saves
    is_main_process = True
    if num_gpus > 1:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                is_main_process = dist.get_rank() == 0
                if not is_main_process:
                    print("[dim]  (Rank {}: skipping save - main process handles it)[/dim]".format(dist.get_rank()))
        except Exception:
            pass  # Not in distributed mode

    # Only main process saves results
    if is_main_process:
        # Save metrics (with defensive checks for NoneType)
        if results and "results" in results and args.task_name in results["results"]:
            save_and_update_models_metrics(
                dataset_dir=args.output_dir,
                model_identifier=model_identifier,
                prefix=prefix,
                suffix=suffix,
                raw_results=results["results"][args.task_name],
                num_fewshot=args.num_fewshot,
                elapsed_time_sec=elapsed_sec,
            )
        else:
            print(f"[yellow]⚠ Warning: Incomplete results for {args.task_name}, skipping metrics save[/yellow]")

        # Save responses (with defensive check)
        response_identifier = f"{model_identifier}_{suffix}" if suffix else model_identifier
        if results and "samples" in results:
            save_model_responses(
                model_identifier=response_identifier,
                output_dir=output_dir,
                samples=results["samples"],
                task_name=args.task_name,
                num_fewshot=args.num_fewshot,
            )
        else:
            print("[yellow]⚠ Warning: No samples found, skipping responses save[/yellow]")

    # Memory cleanup WITHOUT barriers (barriers cause deadlock if one process errors)
    del results
    gc.collect()

    # Clear CUDA cache - no synchronization needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Per-device cleanup
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    gc.collect()

    return elapsed_sec


def eval_finetuned_pretrained(args) -> None:
    """Evaluate both finetuned and pretrained models."""
    print("[bold magenta]=== Finetuned + Pretrained ===[/bold magenta]")
    validate_task_requirements(args.task_name)

    os.makedirs(args.output_dir, exist_ok=True)
    effective_output_dir = os.path.join(args.output_dir, "finetuned_pretrained")
    os.makedirs(effective_output_dir, exist_ok=True)

    gen_kwargs = build_gen_kwargs(args.task_name, args.max_new_tokens)
    print(f"[cyan]Generation: {gen_kwargs}[/cyan]   ")
    print()

    eval_start_time = time.time()

    # Evaluate finetuned model
    run_evaluation(
        model_name=args.fine_tuned_model_name,
        output_dir=effective_output_dir,
        args=args,
        gen_kwargs=gen_kwargs,
        model_label="finetuned model",
        prefix="finetuned_pretrained",
        suffix="finetuned",
    )

    # Evaluate pretrained model
    print()
    run_evaluation(
        model_name=args.pretrained_model_name,
        output_dir=effective_output_dir,
        args=args,
        gen_kwargs=gen_kwargs,
        model_label="pretrained model",
        prefix="finetuned_pretrained",
        suffix="pretrained",
    )

    print()
    print(f"[green]✓ Complete in {format_elapsed_time(time.time() - eval_start_time)}[/green]")
    print(f"[cyan]  Saved to: {effective_output_dir}[/cyan]")


def eval_single(args) -> None:
    """Evaluate a single model."""
    print("[bold magenta]=== Single Model ===[/bold magenta]")
    print(f"[cyan]{args.model_name_or_path}[/cyan]")
    validate_task_requirements(args.task_name)

    os.makedirs(args.output_dir, exist_ok=True)
    gen_kwargs = build_gen_kwargs(args.task_name, args.max_new_tokens)
    print(f"[cyan]Generation: {gen_kwargs}[/cyan]")
    print()

    run_evaluation(
        model_name=args.model_name_or_path,
        output_dir=args.output_dir,
        args=args,
        gen_kwargs=gen_kwargs,
        prefix="single",
        suffix="",
    )

    print()
    print("[green]✓ Complete[/green]")
    print(f"[cyan]  Saved to: {args.output_dir}[/cyan]")


def eval_collection(args) -> None:
    """Evaluate a collection of models across checkpoints."""
    print("[bold magenta]=== Collection ===[/bold magenta]")
    print(f"[cyan]{args.collection_name}[/cyan]")
    validate_task_requirements(args.task_name)

    if args.collection_name not in MODELS_COLLECTION:
        print(f"[bold red]✗ Unknown collection: {args.collection_name}[/bold red]")
        print(f"  Available: {list(MODELS_COLLECTION.keys())}")
        return

    model_prefix = MODELS_COLLECTION[args.collection_name]
    checkpoints = list(range(4000, 10001, 1000))

    os.makedirs(args.output_dir, exist_ok=True)
    effective_output_dir = os.path.join(args.output_dir, args.collection_name)
    os.makedirs(effective_output_dir, exist_ok=True)

    gen_kwargs = build_gen_kwargs(args.task_name, args.max_new_tokens)
    print(f"[cyan]Generation: {gen_kwargs}[/cyan]")
    print()

    collection_start_time = time.time()
    for i, checkpoint in enumerate(checkpoints):
        model_name = f"{model_prefix}{checkpoint}"
        print(f"[cyan]Checkpoint {checkpoint} ({i+1}/{len(checkpoints)})[/cyan]")

        try:
            run_evaluation(
                model_name=model_name,
                output_dir=effective_output_dir,
                args=args,
                gen_kwargs=gen_kwargs,
                model_label=f"Checkpoint {checkpoint}",
                prefix=args.collection_name,
                suffix="",
            )

            # Memory cleanup after checkpoint (no barriers - prevents deadlock)
            print("[dim]→ Cleaning up...[/dim]")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            gc.collect()

            # NO barrier here - causes deadlock if one process errors
            # Memory cleanup doesn't need perfect synchronization

        except Exception as e:
            print(f"[bold red]✗ Error: {model_name}[/bold red]")
            print(f"  {str(e)[:200]}")  # Truncate long errors

            # Cleanup on error (no barriers to prevent deadlock)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            continue
        print()

    print(f"[green]✓ Complete in {format_elapsed_time(time.time() - collection_start_time)}[/green]")
    print(f"[cyan]  Saved to: {effective_output_dir}[/cyan]")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLaVA models using lmms-eval"
    )

    parser.add_argument("--lmms-eval-model", type=str, default="llava_hf_chat")
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--model-cache-dir", type=str, required=True)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--pretrained-model-name", type=str, default=None)
    parser.add_argument("--fine-tuned-model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--task-name", type=str, default="vqav2_test")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)

    args = parser.parse_args()

    # Run appropriate evaluation mode
    if args.collection_name:
        eval_collection(args)
    elif args.pretrained_model_name and args.fine_tuned_model_name:
        eval_finetuned_pretrained(args)
    elif args.model_name_or_path:
        eval_single(args)
    else:
        parser.error(
            "Provide --collection-name, (--pretrained-model-name + --fine-tuned-model-name), or --model-name-or-path"
        )


if __name__ == "__main__":
    main()
