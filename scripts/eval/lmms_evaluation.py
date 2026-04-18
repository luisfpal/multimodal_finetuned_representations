import argparse
import gc
import json
import os
import sys
import time
from typing import Optional

import torch
from rich import print
from lmms_eval.evaluator import simple_evaluate

from utils.patch_llava import patch as patch_llava
from utils.patch_internvl import patch as patch_internvl
from utils.patch_onevision import patch as patch_onevision
from utils.patch_cambrian import patch as patch_cambrian

try:
    from lmms_eval.models.chat.llava_hf import LlavaHf
except ImportError:
    LlavaHf = None

try:
    from scripts.models_utils.internvl2 import InternVL2Custom
except ImportError:
    InternVL2Custom = None

try:
    from scripts.models_utils.internvl3_5 import InternVL3_5Custom
except ImportError:
    InternVL3_5Custom = None

try:
    from scripts.models_utils.llava_onevision1_5 import LlavaOneVision1_5Custom
except ImportError:
    LlavaOneVision1_5Custom = None

try:
    from scripts.models_utils.cambrian import CambrianCustom
except ImportError:
    CambrianCustom = None

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = NUMPY_RANDOM_SEED = TORCH_RANDOM_SEED = FEWSHOT_RANDOM_SEED = 42

# Task-specific sample limits (default for quick evaluation/testing; full dataset sizes in inline comments)
TASK_LIMITS = {
    "scienceqa_img": 25,  # 2018 samples
    "mme": 5000,  # 2374 samples
    # "textvqa_val": 5000,  # 5000 samples
    "pope": 5000,  # 9000 samples
    "gqa": 5000,  # 12578 samples
    "seedbench": 5000,  # 17990 samples
    "mmbench_en_dev": 5000,  # 4330 samples
    "mmbench_cn_dev": 5000,  # 4330 samples
    # "mmvet": 20,  # 218 samples (Strong API judge required or evals are incorrect!)
    "vizwiz_vqa_val": 5000,  # 4320 samples
    "vqav2_val": 5000,  # 214000 samples
    "ai2d": 5000,  # 500 samples
    # "cv_bench": 5000,  # 1000 samples
    # "flickr30k": 5000,  # 1000 samples (test)
    # "coco2017_cap_val": 5000,  # 5000 samples
    # "llava_in_the_wild": 15,  # 60 samples, heavy post processing
    "flickr30k": 50,  # 1000 samples (test)
    # "coco2017_cap_val": 25,  # 5000 samples
    # "mmmu_val": 5000,
}

LLM_JUDGE_TASKS = {"mmvet", "mmbench_en_dev", "mmbench_cn_dev", "llava_in_the_wild"}

# Model name → backend mapping (order matters: more specific patterns first)
MODEL_BACKEND_REGISTRY = [
    ("internvl3", "internvl3.5"),
    ("internvl", "internvl2"),
    ("onevision", "llava_onevision1_5"),
    ("cambrian", "cambrian"),
    ("llava", "llava_hf"),
]

def infer_backend(model_name: str) -> str:
    """Infer the lmms-eval backend from model name."""
    name_lower = model_name.lower()
    for pattern, backend in MODEL_BACKEND_REGISTRY:
        if pattern in name_lower:
            return backend
    raise ValueError(
        f"Cannot infer backend for '{model_name}'. "
        f"Known patterns: {[p for p, _ in MODEL_BACKEND_REGISTRY]}"
    )


# =============================================================================
# HELPERS
# =============================================================================


def get_num_layers(model_instance, backend: str) -> int:
    """Get the number of transformer layers from a model instance."""
    if backend == "llava_hf":
        return model_instance.model.language_model.config.num_hidden_layers
    elif backend in ("internvl2", "internvl3.5"):
        return model_instance._model.language_model.config.num_hidden_layers
    elif backend == "cambrian":
        return model_instance.model.config.num_hidden_layers
    elif backend == "llava_onevision1_5":
        return model_instance.model.language_model.config.num_hidden_layers
    raise ValueError(f"Unknown backend: {backend}")


def create_cli_args(output_path: str) -> argparse.Namespace:
    return argparse.Namespace(output_path=output_path, process_with_media=True)


def validate_task_requirements(task_name: str) -> None:
    if task_name in LLM_JUDGE_TASKS:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[bold red]✗ ERROR: LLM judge API key missing![/bold red]")
            print(f"  Task '{task_name}' requires OPENAI_API_KEY\n")
            sys.exit(1)

        api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
        model_version = os.environ.get("MODEL_VERSION", "gpt-4o-2024-11-20")
        print(f"[green]✓ LLM Judge: {model_version} @ {api_url}[/green]")


def format_elapsed_time(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m > 0 else f"{seconds:.1f}s"


def save_metrics(
    model_identifier: str,
    output_dir: str,
    raw_results: dict,
    num_fewshot: Optional[int] = None,
    elapsed_time_sec: Optional[float] = None,
) -> None:
    """Append evaluation metrics to JSONL results file."""
    output_dir_name = os.path.basename(output_dir)
    filename = f"{output_dir_name}_results"
    if num_fewshot is not None:
        filename += f"_fewshot{num_fewshot}"

    results = {"model": model_identifier, **raw_results}
    if elapsed_time_sec is not None:
        results["elapsed_time"] = format_elapsed_time(elapsed_time_sec)

    with open(os.path.join(output_dir, f"{filename}.jsonl"), "a") as f:
        f.write(json.dumps(results) + "\n")


def save_responses(
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
# EVALUATION
# =============================================================================


def run_evaluation(args: argparse.Namespace) -> float:
    """Run evaluation for a single model on a single task. Returns elapsed seconds."""
    model_name = args.model_name_or_path
    backend = args.backend
    model_identifier = model_name.replace("/", "_")

    validate_task_requirements(args.task_name)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[cyan]>>> {model_name} (backend: {backend})[/cyan]")
    start_time = time.time()

    # Limit logic
    limit = args.limit
    if limit is None:
        limit = TASK_LIMITS.get(args.task_name)
    if limit:
        print(f"[yellow]Sample limit: {limit}[/yellow]")

    # Device mapping
    num_gpus = int(os.environ.get("NUM_GPUS", 1))
    device_map_value = "" if num_gpus > 1 else "auto"
    model_args = f"pretrained={model_name},device_map={device_map_value}"
    cuda_device = (
        f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )
    batch_size = args.batch_size if args.batch_size else 1

    # Instantiate model
    model_instance: object = backend  # fallback: string for registry lookup

    if backend == "llava_hf":
        if LlavaHf is None:
            print("[yellow]LlavaHf not importable, falling back to registry[/yellow]")
        else:
            print(f"[cyan]Instantiating LlavaHf for {model_name}...[/cyan]")
            model_instance = LlavaHf(
                pretrained=model_name,
                device_map=device_map_value,
                batch_size=batch_size,
                trust_remote_code=args.trust_remote_code,
            )

    elif backend == "internvl2":
        if InternVL2Custom is None:
            print("[yellow]InternVL2Custom not importable, falling back to registry[/yellow]")
        else:
            print(f"[cyan]Instantiating InternVL2Custom for {model_name}...[/cyan]")
            model_instance = InternVL2Custom(
                pretrained=model_name,
                device=cuda_device,
                device_map=device_map_value,
                batch_size=str(batch_size),
                cache_dir=args.model_cache_dir,
            )

    elif backend == "internvl3.5":
        if InternVL3_5Custom is None:
            print("[yellow]InternVL3_5Custom not importable, falling back to registry[/yellow]")
        else:
            print(f"[cyan]Instantiating InternVL3_5Custom for {model_name}...[/cyan]")
            model_instance = InternVL3_5Custom(
                pretrained=model_name,
                device=cuda_device,
                device_map=device_map_value,
                batch_size=str(batch_size),
                cache_dir=args.model_cache_dir,
            )

    elif backend == "llava_onevision1_5":
        if LlavaOneVision1_5Custom is None:
            print("[yellow]LlavaOneVision1_5Custom not importable[/yellow]")
        else:
            print(f"[cyan]Instantiating LlavaOneVision1_5Custom for {model_name}...[/cyan]")
            model_instance = LlavaOneVision1_5Custom(
                pretrained=model_name,
                device=cuda_device,
                device_map=device_map_value,
                batch_size=batch_size,
                cache_dir=args.model_cache_dir,
            )

    elif backend == "cambrian":
        if CambrianCustom is None:
            print("[yellow]CambrianCustom not importable, falling back to registry[/yellow]")
        else:
            print(f"[cyan]Instantiating CambrianCustom for {model_name}...[/cyan]")
            model_instance = CambrianCustom(
                pretrained=model_name,
                device=cuda_device,
                device_map=device_map_value,
                batch_size=batch_size,
                cache_dir=args.model_cache_dir,
            )

    # No-image patching (skip vision layers for text-only evaluation)
    if args.no_image:
        num_layers = get_num_layers(model_instance, backend)
        if backend == "llava_onevision1_5":
            patch_onevision(model_instance.model, layers_to_skip=range(num_layers))
        elif backend == "llava_hf":
            patch_llava(layers_to_skip=range(num_layers))
        elif backend == "internvl2":
            patch_internvl(model_instance._model, layers_to_skip=range(num_layers))
        elif backend == "cambrian":
            patch_cambrian(model_instance._model, layers_to_skip=range(num_layers))

    results = simple_evaluate(
        model=model_instance,
        model_args=model_args if isinstance(model_instance, str) else None,
        tasks=[args.task_name],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        log_samples=True,
        limit=limit,
        cli_args=create_cli_args(args.output_dir),
        random_seed=RANDOM_SEED,
        numpy_random_seed=NUMPY_RANDOM_SEED,
        torch_random_seed=TORCH_RANDOM_SEED,
        fewshot_random_seed=FEWSHOT_RANDOM_SEED,
    )
    elapsed_sec = time.time() - start_time
    print(f"[green]✓ {model_name}: {format_elapsed_time(elapsed_sec)}[/green]")

    # In distributed mode, only rank 0 saves
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
            for res_key, res_metrics in results["results"].items():
                if not res_metrics:
                    continue
                if len(res_metrics) == 1 and " " in res_metrics:
                    continue
                metric_keys = [
                    k
                    for k in res_metrics
                    if k != "alias" and k != " " and not k.endswith("_stderr,none")
                ]
                if not metric_keys:
                    continue

                save_metrics(
                    output_dir=args.output_dir,
                    model_identifier=model_identifier,
                    raw_results=res_metrics,
                    num_fewshot=args.num_fewshot,
                    elapsed_time_sec=elapsed_sec,
                )
        else:
            print(f"[yellow]⚠ No results for {args.task_name}[/yellow]")

        if results and "samples" in results:
            save_responses(
                model_identifier=model_identifier,
                output_dir=args.output_dir,
                samples=results["samples"],
                task_name=args.task_name,
                num_fewshot=args.num_fewshot,
            )

    # Cleanup
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
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate vision-language models using lmms-eval"
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--model-cache-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--task-name", type=str, default="scienceqa_img")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()
    args.backend = infer_backend(args.model_name_or_path)
    print(f"[cyan]Backend: {args.backend}[/cyan]")

    run_evaluation(args)

    print()
    print("[green]✓ Complete[/green]")
    print(f"[cyan]  Results: {args.output_dir}[/cyan]")


if __name__ == "__main__":
    main()
