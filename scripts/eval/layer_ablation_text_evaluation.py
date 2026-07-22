#!/usr/bin/env python3
"""
LM Evaluation Harness with early layer skipping support.

This module allows evaluating a model by skipping early layers (0 to k-1),
using only layers k to N-1. This preserves generation capability since
the final layers and lm_head remain intact.
"""

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from typing import List, Optional

import torch
from lm_eval import evaluator

from utils.vlm_layer_skipping.lm_harness_cambrian import CambrianHFLM
from utils.vlm_layer_skipping.lm_harness_internvl import InternVLHFLM
from utils.vlm_layer_skipping.lm_harness_llava import LlavaHFLM
from utils.vlm_layer_skipping.lm_harness_onevision import OnevisionHFLM
from utils.vlm_layer_skipping.patch_cambrian import (
    configure_layer_skip as configure_cambrian_layer_skip,
)
from utils.vlm_layer_skipping.patch_cambrian import (
    install_compat as install_cambrian_compat,
)
from utils.vlm_layer_skipping.patch_internvl import patch as patch_internvl
from utils.vlm_layer_skipping.patch_llava import patch as patch_llava
from utils.vlm_layer_skipping.patch_onevision import patch as patch_onevision

warnings.filterwarnings("ignore")

# Task-specific num_fewshot settings
TASK_NUM_FEWSHOT = {
    "mmlu": 0,
    "gsm8k": 5,
    "triviaqa": 1,
    "bbh": 3,
    "hendrycks_math": 4,
    "nq_open": 5,
}

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# When running triviaqa without global --limit, evaluate only the first N examples
# (lm-eval limit is deterministic: first N in dataset order). Use --triviaqa_limit 0
# for the full split.
_DEFAULT_TRIVIAQA_LIMIT = 5000


def _layers_from_window(start_layer: int, window_size: Optional[int]) -> List[int]:
    """Translate the (start_layer, window_size) CLI args into an explicit list.

    * ``window_size is None``: skip the prefix ``[0, start_layer)``.
    * ``window_size > 0``: skip ``[start_layer, start_layer + window_size)``.
    * Otherwise: no layers skipped.
    """
    if window_size is None:
        return list(range(0, start_layer)) if start_layer > 0 else []
    if window_size <= 0:
        return []
    return list(range(start_layer, start_layer + window_size))


def evaluate_with_skipped_layers(
    model_path: str,
    start_layer: int,
    tasks: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    window_size: Optional[int] = None,
    limit: Optional[int] = None,
    sliding_window: bool = False,
    max_gen_toks: Optional[int] = None,
    triviaqa_limit: Optional[int] = None,
    evaluate_full_sliding_window: bool = False,
    evaluate_full_progressive_ablation: bool = False,
    interval: int = 1,
    restart_layer: int = 0,
):
    """
    Evaluate a model with early layers skipped.

    Args:
        model_path: HuggingFace model path
        start_layer: First layer to skip (0-indexed)
        tasks: List of task names to evaluate
        batch_size: Batch size for evaluation
        device: Device to use
        output_dir: Base directory for results
        system_prompt: Optional system prompt to prepend
        window_size: Size of the window to skip starting at start_layer
        limit: Limit number of samples per task (for testing)
        triviaqa_limit: Cap for triviaqa when ``limit`` is unset (0 = full split; default 1000)
        max_gen_toks: Cap on new tokens for ``generate_until`` (lm-eval name; default 256)
        evaluate_full_sliding_window: Sweep a fixed-size window across all
            decoder layers. Requires ``window_size > 0``.
        evaluate_full_progressive_ablation: Sweep the progressive prefix
            ``[0:w]`` for ``w`` in ``range(restart_layer, num_layers + 1, interval)``.
        interval: Step size for the sweep modes.
        restart_layer: Starting offset for the sweep modes (lets you resume
            an interrupted sweep without re-running prior iterations).

    Returns:
        Dictionary of evaluation results from the final iteration
    """
    if evaluate_full_sliding_window and evaluate_full_progressive_ablation:
        raise ValueError(
            "Pass at most one of --evaluate_full_sliding_window / "
            "--evaluate_full_progressive_ablation."
        )

    if "internvl" in model_path.lower():
        lm = InternVLHFLM(
            pretrained=model_path,
            device=device,
            batch_size=batch_size,
            max_gen_toks=max_gen_toks,
        )
        num_layers = lm._model.language_model.model.config.num_hidden_layers

        def configure_skip(layers):
            patch_internvl(lm._model, layers_to_skip=layers, ablation_mode="layer")
    elif "onevision" in model_path.lower():
        lm = OnevisionHFLM(
            pretrained=model_path,
            device=device,
            batch_size=batch_size,
            max_gen_toks=max_gen_toks,
        )
        num_layers = lm._model.config.text_config.num_hidden_layers

        def configure_skip(layers):
            patch_onevision(lm._model, layers_to_skip=layers, ablation_mode="layer")
    elif "cambrian" in model_path.lower():
        lm = CambrianHFLM(
            pretrained=model_path,
            device=device,
            batch_size=batch_size,
            max_gen_toks=max_gen_toks,
        )
        # Cambrian requires the class-level forward patch even at baseline
        # (its vendored forward is incompatible with modern transformers).
        # Install compat once here so re-running configure_layer_skip per
        # sweep iteration doesn't reinstall it each time.
        install_cambrian_compat(lm._model)
        num_layers = lm._model.config.num_hidden_layers

        def configure_skip(layers):
            configure_cambrian_layer_skip(
                model=lm._model, layers_to_skip=layers, ablation_mode="layer"
            )
    else:
        lm = LlavaHFLM(
            pretrained=model_path,
            device=device,
            batch_size=batch_size,
            system_prompt=system_prompt,
            max_gen_toks=max_gen_toks,
        )
        num_layers = lm._model.config.text_config.num_hidden_layers

        def configure_skip(layers):
            patch_llava(lm._model, layers_to_skip=layers, ablation_mode="layer")

    print(f"Detected {num_layers} LLM decoder layers for {model_path}")

    sweep_mode = evaluate_full_sliding_window or evaluate_full_progressive_ablation

    layers_to_skip_list: List[List[int]] = []
    if evaluate_full_sliding_window:
        if window_size is None or window_size <= 0:
            raise ValueError(
                "--window_size > 0 is required for --evaluate_full_sliding_window"
            )
        for s in range(restart_layer, num_layers - window_size + 2, interval):
            layers_to_skip_list.append(list(range(s, s + window_size)))
    elif evaluate_full_progressive_ablation:
        # Include the no-skip baseline (w=0) and all progressive windows up to num_layers.
        for w in range(restart_layer, num_layers + 1, interval):
            layers_to_skip_list.append(list(range(0, w)))
    else:
        layers_to_skip_list = [_layers_from_window(start_layer, window_size)]

    # Group tasks by their num_fewshot setting (stable across iterations).
    tasks_by_fewshot = defaultdict(list)
    for task in tasks:
        fewshot = TASK_NUM_FEWSHOT.get(task)  # None uses task default
        tasks_by_fewshot[fewshot].append(task)

    # Use the resolved system prompt from the model (which applies model-specific
    # defaults like the InternVL2 Chinese instruction) rather than the raw CLI arg.
    resolved_system_prompt = lm._system_prompt or None

    results = None
    for layers_to_skip in layers_to_skip_list:
        configure_skip(layers_to_skip)

        if sweep_mode:
            if not layers_to_skip:
                cur_start, cur_window = 0, 0
                print("Evaluating model with no layers skipped (baseline)")
            else:
                cur_start = layers_to_skip[0]
                cur_window = len(layers_to_skip)
                print(
                    f"Evaluating model with layers {layers_to_skip[0]}-"
                    f"{layers_to_skip[-1]} skipped (W={cur_window})"
                )
        else:
            cur_start, cur_window = start_layer, window_size
            if window_size is not None:
                print(
                    f"Evaluating model with window of {window_size} layers "
                    f"skipped starting at {start_layer}"
                )
            else:
                print(f"Evaluating model with {start_layer} early layers skipped")

        # Run evaluations grouped by fewshot setting
        all_task_results = {}
        for num_fewshot, task_group in tasks_by_fewshot.items():
            print(f"Evaluating tasks {task_group} with num_fewshot={num_fewshot}")
            group_limit = limit
            if limit is None and task_group == ["triviaqa"]:
                cap = (
                    triviaqa_limit
                    if triviaqa_limit is not None
                    else _DEFAULT_TRIVIAQA_LIMIT
                )
                group_limit = None if cap == 0 else cap
                if group_limit is not None:
                    print(
                        f"Using triviaqa example limit={group_limit} "
                        "(lm-eval first-N subset)"
                    )

            group_results = evaluator.simple_evaluate(
                model=lm,
                tasks=task_group,
                batch_size=batch_size,
                confirm_run_unsafe_code=True,
                device=device,
                num_fewshot=num_fewshot,
                limit=group_limit,
                apply_chat_template=True,
                system_instruction=resolved_system_prompt,
            )
            all_task_results.update(group_results.get("results", {}))

        results = {"results": all_task_results}

        if output_dir and results:
            model_name = model_path.split("/")[-1]
            if sliding_window or evaluate_full_sliding_window:
                model_name += f"_sw{cur_window}"
            model_output_dir = os.path.join(output_dir, model_name)

            all_results = results.get("results", {})
            for task in tasks:
                task_results = {
                    k: v
                    for k, v in all_results.items()
                    if k == task or k.startswith(f"{task}_")
                }

                if task_results:
                    num_fewshot = TASK_NUM_FEWSHOT.get(task, 0)
                    fewshot_str = (
                        f"{num_fewshot}shot" if num_fewshot is not None else "default"
                    )
                    if sweep_mode:
                        filename = (
                            f"{task}_L{cur_start:02d}_W{cur_window}_{fewshot_str}.json"
                        )
                    else:
                        # Preserve legacy filename for single-run mode (e.g. window_size=None -> "WNone").
                        filename = (
                            f"{task}_L{start_layer:02d}_W{window_size}_{fewshot_str}.json"
                        )

                    task_output_dir = os.path.join(model_output_dir, task)
                    os.makedirs(task_output_dir, exist_ok=True)
                    output_path = os.path.join(task_output_dir, filename)

                    with open(output_path, "w") as f:
                        serializable = {
                            "config": {
                                "model_path": model_path,
                                "start_layer": cur_start,
                                "window_size": cur_window,
                                "task": task,
                                "num_fewshot": num_fewshot,
                            },
                            "results": task_results,
                        }
                        json.dump(serializable, f, indent=2, default=str)
                    print(f"Results for {task} saved to {output_path}")

        torch.cuda.empty_cache()

    del lm
    torch.cuda.empty_cache()

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM with early layer skipping using lm-evaluation-harness"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=0,
        help="First layer to skip.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["mmlu"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/layer_skip_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt to prepend (optional)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Size of the window to skip starting at start_layer",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for testing)",
    )
    parser.add_argument(
        "--triviaqa_limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When only triviaqa is run and --limit is not set, evaluate the first N examples "
            f"(default {_DEFAULT_TRIVIAQA_LIMIT}). Use 0 for the full validation split."
        ),
    )
    parser.add_argument(
        "--max_gen_toks",
        type=int,
        default=128,
        help="Max new tokens for generate_until tasks (lm-eval; default 256)",
    )
    parser.add_argument("--sliding_window", action="store_true")
    parser.add_argument(
        "--evaluate_full_sliding_window",
        action="store_true",
        help="Sweep a fixed-size window across all decoder layers "
        "(requires --window_size > 0).",
    )
    parser.add_argument(
        "--evaluate_full_progressive_ablation",
        action="store_true",
        help="Progressively skip layers [0:w] for w in "
        "range(restart_layer, num_layers + 1, interval).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Step size for sliding-window / progressive sweeps (default: 1).",
    )
    parser.add_argument(
        "--restart_layer",
        type=int,
        default=0,
        help="Starting offset for sweep modes — lets you resume an "
        "interrupted sweep without re-running prior iterations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start = time.time()

    evaluate_with_skipped_layers(
        model_path=args.model_path,
        start_layer=args.start_layer,
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        system_prompt=args.system_prompt,
        window_size=args.window_size,
        limit=args.limit,
        sliding_window=args.sliding_window,
        max_gen_toks=args.max_gen_toks,
        triviaqa_limit=args.triviaqa_limit,
        evaluate_full_sliding_window=args.evaluate_full_sliding_window,
        evaluate_full_progressive_ablation=args.evaluate_full_progressive_ablation,
        interval=args.interval,
        restart_layer=args.restart_layer,
    )
    print(f"Evaluation took: {(time.time()-start)/60:.2f} min")
