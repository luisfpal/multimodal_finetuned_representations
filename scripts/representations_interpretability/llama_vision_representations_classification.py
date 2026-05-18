"""LLaMA Vision 3.2 label-overlap experiment.

Loads a LLaMA Vision model once, then for each requested dataset:
  1. Builds a synthetic MCQ ClassificationDataset (filtered to n_choices classes).
  2. Extracts last-token residual-stream representations from all decoder layers.
  3. Computes per-layer label_overlap and saves to the standard results tree.

Representations are cached and reused across runs.

Usage:
  python llama_vision_representations_classification.py \\
    --model-path /path/to/Meta-Llama-3-90B-instruct \\
    --dataset-runs "ethz/food101|validation|generic_mcq_filter" \\
                   "tanganke/dtd|train|generic_mcq_filter" \\
    --n-choices 10 --class-fraction 0.3

Each --dataset-runs entry is "dataset_name|split|prompt_mode".
"""

from __future__ import annotations

import argparse
import gc
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

from utils import seed_all
from utils.classification_collators import _build_sample_signature
from utils.dataset import MCQ_FILTER_MODES, MCQ_MODES, SEED, ClassificationDataset
from utils.metrics_utils import classify_per_layer_label_overlap

load_dotenv()

MODEL_DTYPE = torch.bfloat16
# flash_attention_2 breaks with MllamaTextSelfAttention (no is_causal attr); sdpa is correct.
ATTN_IMPLEMENTATION = "sdpa"
INPUT_DEVICE = "cuda:0"
RESIDUAL_STREAM_TYPE = "output_layer"
POOLING_METHOD = "last"
GUIDE_TEXT = "Answer with the option's letter from the given choices directly."
METRIC_SHUFFLE_SEED = SEED

HOME_DIR = Path.home()
RESULTS_BASE = HOME_DIR / "multimodal_finetuned_representations" / "results"
REPRESENTATIONS_BASE = HOME_DIR / "scratch"


# ── Collate ───────────────────────────────────────────────────────────────────


def _collate_mcq(batch: list[dict], processor: AutoProcessor) -> tuple:
    """Apply Mllama chat template to MCQ questions with images and tokenize.

    Returns a 1-tuple (BatchFeature,) for compatibility with extraction loop.
    MllamaProcessor requires images as a nested list [[img0], [img1], ...].
    """
    questions = [item["question"] for item in batch]
    images = [[item["image"]] for item in batch]
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{q}\n{GUIDE_TEXT}".strip()},
            ]}],
            add_generation_prompt=True,
        )
        for q in questions
    ]
    tokenized = processor(
        images=images,
        text=prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    return (tokenized,)


# ── Representation extraction ─────────────────────────────────────────────────


@torch.no_grad()
def _extract_representations(
    model: MllamaForConditionalGeneration,
    dataloader: DataLoader,
) -> torch.Tensor:
    """Extract last-token hidden states from every language decoder layer.

    Args:
        model: LLaMA Vision model in eval mode.
        dataloader: yields 1-tuples (BatchFeature,), batch_size=1.

    Returns:
        Tensor of shape (num_layers, num_samples, hidden_size), dtype float32.
    """
    per_sample: list[torch.Tensor] = []

    for batch_tuple in tqdm(dataloader, desc="Extracting residual stream", unit="sample"):
        batch_enc = batch_tuple[0]
        inputs = {
            k: v.to(INPUT_DEVICE)
            for k, v in batch_enc.items()
            if isinstance(v, torch.Tensor)
        }

        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states[0] = embedding output; [1:] = decoder layer outputs
        # Each entry: (batch, seq_len, hidden_size)
        layer_reps = torch.stack([
            hs[0, -1, :].cpu().float()   # last token, single batch item
            for hs in outputs.hidden_states[1:]
        ])  # (num_layers, hidden_size)
        per_sample.append(layer_reps)

        del outputs, inputs, batch_enc
        gc.collect()
        torch.cuda.empty_cache()

    # Stack samples along dim 1: (num_layers, num_samples, hidden_size)
    return torch.stack(per_sample, dim=1)


# ── Path helpers ──────────────────────────────────────────────────────────────


def _build_repr_path(
    representations_base: Path,
    model_slug: str,
    dataset_slug: str,
    pooling_method: str,
    sample_set_tag: str,
) -> Path:
    return (
        representations_base
        / "representations"
        / model_slug
        / dataset_slug
        / RESIDUAL_STREAM_TYPE
        / pooling_method
        / sample_set_tag
        / "representations.safetensors"
    )


def _build_label_overlap_path(
    results_base: Path,
    model_slug: str,
    dataset_slug: str,
    split: str,
    pooling_method: str,
    file_suffix: str,
) -> Path:
    out_dir = (
        results_base
        / "representations_classification"
        / dataset_slug
        / split
        / model_slug
        / pooling_method
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"label_overlap{file_suffix}.safetensors"


# ── Split / mode routing (mirrors representations_classification.py) ──────────

_EFFECTIVE_SPLITS: dict[str, str] = {
    "mini-imagenet": "validation",
    "food101": "validation",
    "sun397": "test",
    "caltech101": "train",
    "oxford_flowers102": "test",
    "places365": "train",
    "dtd": "train",
}


def _effective_split(dataset_name: str, split: str) -> str:
    dn = dataset_name.lower()
    for key, override in _EFFECTIVE_SPLITS.items():
        if key == dn.split("/")[-1] or key in dn:
            return override
    return split


def _is_dataset_with_modes(dataset_name: str) -> bool:
    dn = dataset_name.lower()
    return any(
        k in dn for k in (
            "mini-imagenet", "food101", "sun397", "caltech101",
            "oxford_flowers102", "places365", "dtd",
        )
    )


def _build_suffixes(prompt_mode: str, n_choices: int, class_fraction: float | None) -> tuple[str, str]:
    """Return (sample_set_tag_suffix, label_overlap_file_suffix).

    Mirrors the suffix logic in representations_classification.py so that
    cached representations are reusable between scripts.
    """
    is_mcq = prompt_mode in MCQ_MODES
    mode_suffix = f"_{prompt_mode}"
    n_suffix = f"_n{n_choices}" if is_mcq else ""
    output_suffix = mode_suffix + n_suffix             # e.g. "_generic_mcq_filter_n10"

    lo_parts = [f"cf{class_fraction}"] if class_fraction is not None else []
    lo_suffix = ("_" + "_".join(lo_parts)) if lo_parts else ""
    file_suffix = output_suffix + lo_suffix            # e.g. "_generic_mcq_filter_n10_cf0.3"

    return output_suffix, file_suffix


def _metric_permutation(num_samples: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(METRIC_SHUFFLE_SEED)
    return torch.randperm(num_samples, generator=generator)


# ── Per-dataset logic ─────────────────────────────────────────────────────────


def run_dataset(
    *,
    model: MllamaForConditionalGeneration,
    processor: AutoProcessor,
    model_slug: str,
    dataset_name: str,
    split: str,
    prompt_mode: str,
    n_choices: int,
    class_fraction: float | None,
    force_recompute_representations: bool,
    force_recompute_metrics: bool,
    save_representations: bool,
    representations_base: Path,
    results_base: Path,
) -> None:
    split = _effective_split(dataset_name, split)
    dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()

    if not _is_dataset_with_modes(dataset_name):
        print(f"  [WARNING] {dataset_name} not a known image-classification dataset. "
              "Skipping.")
        return

    output_suffix, file_suffix = _build_suffixes(prompt_mode, n_choices, class_fraction)
    sample_set_tag = f"unique_images{output_suffix}"

    print(f"\n  Dataset:  {dataset_name}")
    print(f"  Split:    {split}")
    print(f"  Mode:     {prompt_mode} | n_choices={n_choices}")
    print(f"  Tag:      {sample_set_tag}")

    dataset = ClassificationDataset(
        dataset_name=dataset_name,
        split=split,
        prompt_mode=prompt_mode,
        n_choices=n_choices,
        seed=SEED,
    )
    labels = np.array(dataset.labels)
    sample_ids = list(dataset.sample_ids)
    sample_signature = _build_sample_signature(sample_ids=sample_ids, labels=labels)
    prompt_signature = dataset.prompt_signature

    print(f"  Samples:  {len(dataset)} | Classes: {len(set(dataset.labels))}")

    repr_path = _build_repr_path(
        representations_base, model_slug, dataset_slug, POOLING_METHOD, sample_set_tag
    )

    representations: torch.Tensor | None = None
    if repr_path.exists() and not force_recompute_representations:
        print(f"  Loading cached representations: {repr_path}")
        with safe_open(str(repr_path), framework="pt") as h:
            meta = h.metadata() or {}
            valid = (
                "residual_stream" in h.keys()
                and meta.get("sample_signature") == sample_signature
                and meta.get("prompt_signature") == prompt_signature
            )
            if valid:
                representations = h.get_tensor("residual_stream")
                print(f"  Cache valid. Shape: {tuple(representations.shape)}")
            else:
                print("  Cache signature mismatch — recomputing.")

    if representations is None:
        collate_fn = partial(_collate_mcq, processor=processor)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        representations = _extract_representations(model, dataloader)

        if save_representations:
            repr_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"residual_stream": representations.cpu().to(torch.float32)},
                str(repr_path),
                metadata={
                    "model_name": model_slug,
                    "dataset_name": dataset_name,
                    "split": split,
                    "tokens_pooling_method": POOLING_METHOD,
                    "num_samples": str(len(labels)),
                    "sample_signature": sample_signature,
                    "prompt_signature": prompt_signature,
                    "sample_set_tag": sample_set_tag,
                },
            )
            print(f"  Saved representations: {repr_path}")

    out_path = _build_label_overlap_path(
        results_base, model_slug, dataset_slug, split, POOLING_METHOD, file_suffix
    )
    if out_path.exists() and not force_recompute_metrics:
        print(f"  Skipping label_overlap (exists): {out_path}")
        return

    print("  Computing label_overlap...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Shuffle with fixed seed (mirrors representations_classification.py)
    perm = _metric_permutation(len(y))
    reps_perm = representations[:, perm, :]
    y_perm = y[perm.numpy()]

    lo_kwargs: dict = {}
    if class_fraction is not None:
        lo_kwargs["class_fraction"] = class_fraction

    layer_results = classify_per_layer_label_overlap(
        reps_perm.cpu().to(torch.float32), y_perm, **lo_kwargs
    )

    num_layers = representations.shape[0]
    lo_tensor = torch.tensor(
        [layer_results[i]["label_overlap"] for i in range(num_layers)],
        dtype=torch.float32,
    )
    save_file(
        {"label_overlap": lo_tensor},
        str(out_path),
        metadata={
            "model_name": model_slug,
            "dataset_name": dataset_name,
            "split": split,
            "pooling_method": POOLING_METHOD,
            "prompt_mode": prompt_mode,
            "n_choices": str(n_choices),
            "num_samples": str(len(y)),
            "class_fraction": str(class_fraction) if class_fraction is not None else "",
        },
    )
    print(f"  Saved label_overlap: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLaMA Vision label-overlap experiment (loads model once, runs all datasets).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or HuggingFace ID for the LLaMA Vision model.",
    )
    parser.add_argument(
        "--dataset-runs",
        nargs="+",
        required=True,
        metavar="DATASET|SPLIT|PROMPT_MODE",
        help=(
            "One or more pipe-delimited dataset run specs: "
            "'dataset_name|split|prompt_mode'. "
            "Example: 'ethz/food101|validation|generic_mcq_filter'"
        ),
    )
    parser.add_argument("--n-choices", type=int, default=10)
    parser.add_argument(
        "--class-fraction",
        type=float,
        default=None,
        help="class_fraction for label_overlap (default: use metric default).",
    )
    parser.add_argument(
        "--representations-dir",
        type=str,
        default=str(REPRESENTATIONS_BASE),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_BASE),
    )
    parser.add_argument(
        "--save-representations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-recompute-representations", action="store_true")
    parser.add_argument("--force-recompute-metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    model_slug = Path(args.model_path).name
    representations_base = Path(args.representations_dir).expanduser().resolve()
    results_base = Path(args.results_dir).expanduser().resolve()

    # Parse dataset run specs
    runs: list[tuple[str, str, str]] = []
    for spec in args.dataset_runs:
        parts = spec.split("|")
        if len(parts) != 3:
            raise ValueError(
                f"Each --dataset-runs entry must be 'dataset_name|split|prompt_mode', "
                f"got: {spec!r}"
            )
        runs.append((parts[0], parts[1], parts[2]))

    print(f"\n{'═' * 64}")
    print(f"  LLaMA Vision representations classification")
    print(f"  Model:       {model_slug}")
    print(f"  Datasets:    {len(runs)}")
    print(f"  n_choices:   {args.n_choices}")
    print(f"  cf:          {args.class_fraction}")
    print(f"{'═' * 64}")

    print(f"\n🤖 Loading model: {args.model_path}")
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=MODEL_DTYPE,
        device_map="auto",
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    print("  ✅ Model loaded.")

    try:
        for dataset_name, split, prompt_mode in runs:
            print(f"\n{'─' * 64}")
            print(f"  {dataset_name} | {split} | {prompt_mode}")
            print(f"{'─' * 64}")
            try:
                run_dataset(
                    model=model,
                    processor=processor,
                    model_slug=model_slug,
                    dataset_name=dataset_name,
                    split=split,
                    prompt_mode=prompt_mode,
                    n_choices=args.n_choices,
                    class_fraction=args.class_fraction,
                    force_recompute_representations=args.force_recompute_representations,
                    force_recompute_metrics=args.force_recompute_metrics,
                    save_representations=args.save_representations,
                    representations_base=representations_base,
                    results_base=results_base,
                )
            except Exception as exc:
                print(f"  ❌ FAILED: {type(exc).__name__}: {exc}")
                print("  Continuing with next dataset...")
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ All {len(runs)} datasets complete.")


if __name__ == "__main__":
    main()
