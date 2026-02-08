# Standard library
import argparse
import gc
import json
import os
import time
from typing import Optional

# Third-party
import torch
from rich import print
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# Local
from utils import (
    load_hf_model_and_processor_or_tokenizer,
    LLaVA1_5_SYSTEM_INSTRUCTION,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Reproducibility
RANDOM_SEED = 42

MODELS_COLLECTION = {
    "Llava1.5-7B": "lbasile/llava-",
    "Llava1.5-7B-2-12": "lbasile/llava_2_12-",
    "Llava1.5-7B-2-18": "lbasile/llava_2_18-",
    "Llava1.5-7B-18-32": "lbasile/llava_18_32-",
    "Llava1.5-7B-frozen-0-32": "lbasile/llava_frozen_0_32-",
    "Llava1.5-7B-frozen-2-18": "lbasile/llava_frozen_2_18-",
    "Llava1.5-7B-frozen": "lbasile/llava_frozen-",
    "Llava1.5-7B-2-2": "lbasile/llava_2_2-",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time as Xh Ym Zs."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m > 0 else f"{seconds:.1f}s"


def create_llava_wrapper_for_lm_eval(model, processor) -> HFLM:
    """Wrap LLaVA's language model for lm-eval-harness."""
    language_model = model.language_model
    tokenizer = processor.tokenizer

    # HFLM wrapper treats it as a standard text model
    lm = HFLM(
        pretrained=language_model,
        tokenizer=tokenizer,
        device=str(language_model.device),
        batch_size=1, # Overridden by simple_evaluate args usually, but good fallback
    )
    return lm


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

    output_file = os.path.join(output_dir, f"{filename}.json")
    
    task_keys = [k for k in samples if k.startswith(task_name)]
    if len(task_keys) == 1:
        output_data = samples[task_keys[0]]
    else:
        output_data = {key: samples[key] for key in task_keys}

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def run_evaluation(
    model_name: str,
    output_dir: str,
    args: argparse.Namespace,
    model_label: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
) -> float:
    """
    Run a single model evaluation and save results.
    Returns elapsed time in seconds.
    """
    model_identifier = model_name.split("/")[-1]
    label = model_label or model_identifier

    print(f"[cyan]>>> {label}[/cyan]")
    start_time = time.time()

    # 1. Load Model
    print(f"[dim]Loading {model_name}...[/dim]")
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=model_name,
        cache_dir=args.model_cache_dir,
        device_map="cuda:0",
        dtype=torch.float16,
    )

    lm = create_llava_wrapper_for_lm_eval(
        model=model,
        processor=processor,
    )

    # 2. Evaluate
    results = simple_evaluate(
        model=lm,
        tasks=[args.task_name],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        log_samples=True,
        limit=args.limit,
        device=str(model.language_model.device),
        system_instruction=LLaVA1_5_SYSTEM_INSTRUCTION,
    )
    
    elapsed_sec = time.time() - start_time
    print(f"[green]✓ {label}: {format_elapsed_time(elapsed_sec)}[/green]")

    # 3. Save Results
    # Extract results for the specific task
    # simple_evaluate returns dict with 'results' key which contains task keys
    task_results = results.get("results", {}).get(args.task_name)
    
    if task_results:
        save_and_update_models_metrics(
            dataset_dir=args.output_dir,
            model_identifier=model_identifier,
            prefix=prefix,
            suffix=suffix,
            raw_results=task_results,
            num_fewshot=args.num_fewshot,
            elapsed_time_sec=elapsed_sec,
        )
    else:
        print(f"[yellow]⚠ Warning: No results found for task {args.task_name}[/yellow]")

    if results.get("samples"):
        save_model_responses(
            model_identifier=f"{model_identifier}_{suffix}" if suffix else model_identifier,
            output_dir=output_dir,
            samples=results["samples"],
            task_name=args.task_name,
            num_fewshot=args.num_fewshot,
        )

    # 4. Cleanup
    del model, lm, processor, results
    gc.collect()
    torch.cuda.empty_cache()
    
    return elapsed_sec


def eval_finetuned_pretrained(args) -> None:
    """Evaluate both finetuned and pretrained models."""
    print("[bold magenta]=== Finetuned + Pretrained ===[/bold magenta]")
    os.makedirs(args.output_dir, exist_ok=True)
    
    eval_start_time = time.time()

    # Evaluate finetuned model
    run_evaluation(
        model_name=args.fine_tuned_model_name,
        output_dir=args.output_dir,
        args=args,
        model_label="finetuned model",
        prefix="finetuned_pretrained",
        suffix="finetuned",
    )

    print()

    # Evaluate pretrained model
    run_evaluation(
        model_name=args.pretrained_model_name,
        output_dir=args.output_dir,
        args=args,
        model_label="pretrained model",
        prefix="finetuned_pretrained",
        suffix="pretrained",
    )

    print()
    print(f"[green]✓ Complete in {format_elapsed_time(time.time() - eval_start_time)}[/green]")


def eval_collection(args) -> None:
    """Evaluate a collection of models"""
    if args.collection_name not in MODELS_COLLECTION:
        print(f"[bold red]✗ Unknown collection: {args.collection_name}[/bold red]")
        print(f"  Available: {list(MODELS_COLLECTION.keys())}")
        return

    print(f"[bold magenta]=== Collection: {args.collection_name} ===[/bold magenta]")
    
    models_collection_prefix = MODELS_COLLECTION[args.collection_name]
    # Assuming checkpoints 1000 to 10000 with step 1000
    checkpoints = list(range(1000, 10001, 1000))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    collection_start_time = time.time()

    for i, checkpoint in enumerate(checkpoints):
        model_name = f"{models_collection_prefix}{checkpoint}"
        print(f"[cyan]Checkpoint {checkpoint} ({i+1}/{len(checkpoints)})[/cyan]")

        try:
            run_evaluation(
                model_name=model_name,
                output_dir=args.output_dir,
                args=args,
                model_label=f"Checkpoint {checkpoint}",
                prefix=args.collection_name,
                suffix="", # Checkpoint is part of model name/identifier
            )
        except Exception as e:
            print(f"[bold red]✗ Error evaluating {model_name}[/bold red]")
            print(f"  {e}")
            # Try to cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        
        print()

    print(f"[green]✓ Collection evaluation complete in {format_elapsed_time(time.time() - collection_start_time)}[/green]")


def eval_single(args) -> None:
    """Evaluate a single model"""
    print("[bold magenta]=== Single Model ===[/bold magenta]")
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_evaluation(
        model_name=args.model_name_or_path,
        output_dir=args.output_dir,
        args=args,
        prefix="single",
        suffix="",
    )
    
    print("[green]✓ Single evaluation complete[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models using lm-eval-harness library"
    )

    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--model-cache-dir", type=str, required=True)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--pretrained-model-name", type=str, default=None)
    parser.add_argument("--fine-tuned-model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--task-name", type=str, default="gsm8k_cot")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # Validate arguments
    if args.collection_name:
        eval_collection(args)
    elif args.pretrained_model_name and args.fine_tuned_model_name:
        eval_finetuned_pretrained(args)
    elif args.model_name_or_path:
        eval_single(args)
    else:
        parser.error(
            "Must provide either --collection-name, "
            "(--pretrained-model-name and --fine-tuned-model-name), "
            "or --model-name-or-path"
        )

if __name__ == "__main__":
    main()
