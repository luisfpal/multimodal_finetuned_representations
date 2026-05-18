import argparse
import gc
import os
from functools import partial
from typing import Any

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

from utils import compute_cross_layers_residual_stream_similarities, seed_all
from utils.dataset import (
    MULTIPLE_CHOICE_BENCHMARKS,
    MultipleChoiceDatasetBenchmark,
    OpenVQADataset,
)

load_dotenv()

MODEL_DTYPE = torch.bfloat16
# flash_attention_2 fails on this transformers version (4.57.6):
# MllamaTextSelfAttention has no is_causal attribute — sdpa is the correct choice.
ATTN_IMPLEMENTATION = "sdpa"
INPUT_DEVICE = "cuda:0"
RESIDUAL_STREAM_TYPE = "output_layer"
POOLING_METHOD = "last"

SIMILARITY_MEASURES_DEFAULT = [
    "linear_cka",
    "neighborhood_overlap",
    "information_imbalance",
    "information_imbalance_reversed",
]


# ── Dataset spec parsing ───────────────────────────────────────────────────────


def parse_dataset_spec(spec: str) -> dict[str, Any]:
    """Parse a pipe-delimited dataset run spec.

    Format: 'dataset_path|caption_source|dataset_split|downsample_size|repeated_images|open_ended'
    Example: 'l11p/MMBench_EN-dev-captioned|synthetic||2000|false|true'
    """
    parts = spec.split("|")
    if len(parts) != 6:
        raise ValueError(
            f"Dataset spec must have exactly 6 pipe-separated fields, got {len(parts)}: {spec!r}"
        )
    dataset_path, caption_source, dataset_split, downsample_size_s, repeated_images_s, open_ended_s = parts
    return {
        "dataset_path_or_name": dataset_path,
        "caption_source": caption_source if caption_source else "synthetic",
        "dataset_split": dataset_split if dataset_split else None,
        "downsample_size": int(downsample_size_s) if downsample_size_s else None,
        "repeated_images": repeated_images_s.strip().lower() == "true",
        "open_ended": open_ended_s.strip().lower() == "true",
    }


# ── Collate functions ──────────────────────────────────────────────────────────


def collate_llama_vision_txt(
    batch: list[dict],
    processor: AutoProcessor,
    guide_text: str,
) -> dict:
    """Text-only (caption) collate — no image, produces text representations."""
    questions = [item["question"] for item in batch]
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "text", "text": f"{q}\n{guide_text}\n".strip()}
            ]}],
            add_generation_prompt=True,
        )
        for q in questions
    ]
    return processor(text=prompts, return_tensors="pt", padding=True)


def collate_llama_vision_img(
    batch: list[dict],
    processor: AutoProcessor,
    guide_text: str,
) -> dict:
    """Image + question collate — produces image representations.

    MllamaProcessor requires a nested list [[img_per_item0], [img_per_item1], ...].
    A flat list fails for batch_size > 1.
    """
    questions = [item["question"] for item in batch]
    images = [[item["image"]] for item in batch]  # nested: one sub-list per batch item
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{q}\n{guide_text}\n".strip()},
            ]}],
            add_generation_prompt=True,
        )
        for q in questions
    ]
    return processor(
        images=images,
        text=prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )


# ── DataLoader builder ─────────────────────────────────────────────────────────


def build_dataloader(
    dataset_path_or_name: str,
    cache_dir: str | None,
    processor: AutoProcessor,
    guide_text: str,
    batch_size: int,
    seed: int,
    downsample_size: int | None,
    caption_source: str | None,
    unique_images: bool,
    open_ended: bool,
    dataset_split: str | None,
    txt_mode: bool,
) -> DataLoader:
    """Build a DataLoader for Llama Vision representation extraction."""
    is_mc = any(b in dataset_path_or_name.lower() for b in MULTIPLE_CHOICE_BENCHMARKS)

    if is_mc:
        dataset = MultipleChoiceDatasetBenchmark(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split="train",
            texts_qa=txt_mode,
            images_qa=not txt_mode,
            downsample_size=downsample_size,
            seed=seed,
            unique_images=unique_images,
            open_ended=open_ended,
        )
    else:
        dataset = OpenVQADataset(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split="train",
            downsample_size=downsample_size,
            seed=seed,
            images_qa=not txt_mode,
            texts_qa=txt_mode,
            caption_source=caption_source,
            dataset_split=dataset_split,
        )

    collate_fn = partial(
        collate_llama_vision_txt if txt_mode else collate_llama_vision_img,
        processor=processor,
        guide_text=guide_text,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # required for multi-GPU pipeline parallelism
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False,
    )


# ── Representation extraction ──────────────────────────────────────────────────


def extract_last_token_representations(
    model: MllamaForConditionalGeneration,
    dataloader: DataLoader,
) -> torch.Tensor:
    """Extract last-token hidden states from every decoder layer.

    Skips hidden_states[0] (raw token embeddings, before any transformer block).
    With left padding the last real token is always at position -1.

    Returns:
        Tensor of shape (num_layers, N, hidden_size) on CPU.
    """
    all_layers: list[list[torch.Tensor]] | None = None
    model.eval()

    dataset_size = len(dataloader.dataset)
    samples_processed = 0
    update_count = 0
    update_every = max(1, int(0.1 * dataset_size))

    progress_bar = tqdm(
        total=dataset_size,
        desc="Extracting",
        unit="sample",
    )

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                k: v.to(INPUT_DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            current_batch_size = inputs["input_ids"].shape[0]

            outputs = model(**inputs, output_hidden_states=True)

            if outputs.hidden_states is None:
                raise RuntimeError(
                    "output_hidden_states=True produced hidden_states=None. "
                    "MllamaForConditionalGeneration did not propagate the flag."
                )

            # hidden_states[0] is the embedding layer — skip it
            decoder_hidden_states = outputs.hidden_states[1:]

            if all_layers is None:
                all_layers = [[] for _ in decoder_hidden_states]

            for layer_idx, hs in enumerate(decoder_hidden_states):
                # hs: (batch, seq_len, hidden_size) on the layer's device
                # left padding → last real token is always at position -1
                last = hs[:, -1, :].cpu()  # move to CPU immediately — each layer lives on a different GPU
                all_layers[layer_idx].append(last)

            # Update progress tracking
            samples_processed += current_batch_size
            update_count += current_batch_size

            # Update progress bar and clean memory periodically
            if update_count % update_every == 0 or samples_processed == dataset_size:
                progress_bar.update(update_count)
                torch.cuda.empty_cache()
                gc.collect()
                update_count = 0

    progress_bar.close()
    assert all_layers is not None, "DataLoader was empty — no batches processed."
    return torch.stack([torch.cat(x, dim=0) for x in all_layers])


# ── Cache / similarity helpers ─────────────────────────────────────────────────


def build_similarity_filename(model_name: str, measure_name: str) -> str:
    return f"{model_name}_{measure_name}.safetensors"


def load_representation_from_cache(
    path: str,
    dry_run: bool,
    force: bool,
) -> torch.Tensor | None:
    if not dry_run and not force and os.path.exists(path):
        print(f"  💾 Found cached representation at: {path}")
        try:
            loaded = load_file(path)
            if "residual_stream" in loaded:
                tensor = loaded["residual_stream"]
                if tensor.ndim >= 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                    print(f"  ✅ Loaded successfully. Shape: {tensor.shape}")
                    return tensor[:]
                print(f"  ⚠️ Invalid tensor shape: {tensor.shape}. Re-extracting...")
            else:
                print("  ⚠️ Key 'residual_stream' not found. Re-extracting...")
        except (OSError, IOError, ValueError, KeyError) as e:
            print(f"  ⚠️ Error loading cache: {type(e).__name__}: {e}. Re-extracting...")
        except Exception as e:
            print(f"  ❌ Unexpected error loading cache: {type(e).__name__}: {e}")
            raise
    return None


def save_raw_representation(
    tensor: torch.Tensor,
    path: str,
    metadata: dict,
    dry_run: bool,
    force: bool,
) -> None:
    if dry_run or (not force and os.path.exists(path)):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    valid_meta = {k: str(v) for k, v in metadata.items()}
    try:
        save_file({"residual_stream": tensor.cpu()}, path, metadata=valid_meta)
        print(f"  💾 Saved raw: {path}")
    except Exception as e:
        print(f"  ❌ Error saving {path}: {type(e).__name__}: {e}")


def find_missing_measures(
    similarity_measures: list[str],
    similarities_dir: str,
    model_name: str,
    dry_run: bool,
    force_recompute: bool,
) -> list[str]:
    if dry_run or force_recompute:
        if force_recompute:
            print(f"    🔄 Force mode: recomputing all {len(similarity_measures)} measures")
        return list(similarity_measures)
    missing = []
    for measure_name in similarity_measures:
        filename = build_similarity_filename(model_name, measure_name)
        if not os.path.exists(os.path.join(similarities_dir, filename)):
            missing.append(measure_name)
        else:
            print(f"    ✅ Found existing: {filename}")
    return missing


def run_pair_analysis(
    res_a: torch.Tensor,
    res_b: torch.Tensor,
    modality_a: str,
    modality_b: str,
    similarities_dir: str,
    model_name: str,
    missing_measures: list[str],
    args: argparse.Namespace,
) -> None:
    """Compute and save all similarity measures for a (text, image) representation pair."""
    sim_metadata = {
        "task": "vqa",
        "row_modality": modality_a,
        "col_modality": modality_b,
        "description": (
            f"Rows represent {modality_a} layers, Columns represent {modality_b} layers"
        ),
        "pooling_method": POOLING_METHOD,
        "model_name": model_name,
        "residual_stream_type": RESIDUAL_STREAM_TYPE,
    }
    print(
        f"    ℹ️  Matrix orientation: Rows={modality_a.capitalize()}, "
        f"Columns={modality_b.capitalize()}"
    )

    for measure_name in missing_measures:
        print(f"    👉 Computing: {measure_name}")
        try:
            sim_results = compute_cross_layers_residual_stream_similarities(
                res_a,
                res_b,
                similarity_measures=[measure_name],
                maxk=30,
            )
            measure_val = sim_results[measure_name]
            print(f"    ✅ Shape: {measure_val.shape}")
            if not args.dry_run:
                save_path = os.path.join(
                    similarities_dir,
                    build_similarity_filename(model_name, measure_name),
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_file(
                    {measure_name: measure_val},
                    save_path,
                    metadata={k: str(v) for k, v in sim_metadata.items()},
                )
                print(f"    💾 Saved: {save_path}")
        except Exception as e:
            print(f"    ❌ Error computing {measure_name}: {type(e).__name__}: {e}")
            print(f"    ⚠️  Skipping {measure_name}...")


# ── Per-dataset processing ─────────────────────────────────────────────────────


def process_dataset(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    ensure_model: callable,
    datasets_cache_dir: str | None,
    model_name: str,
) -> None:
    """Extract representations and compute similarities for one dataset config."""
    dataset_path_or_name = cfg["dataset_path_or_name"]
    caption_source = cfg["caption_source"]
    dataset_split = cfg["dataset_split"]
    repeated_images = cfg["repeated_images"]
    open_ended = cfg["open_ended"]

    effective_downsample_size = cfg["downsample_size"]
    if dataset_split and cfg["downsample_size"] is not None:
        print("ℹ️  downsample_size ignored because dataset_split is active.")
        effective_downsample_size = None

    dataset_slug = dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    is_mc = any(b in dataset_slug for b in MULTIPLE_CHOICE_BENCHMARKS)

    if dataset_split and is_mc:
        raise ValueError(f"dataset_split is only valid for OpenVQA datasets, not MC: {dataset_path_or_name}")

    open_ended_tag = "open_ended" if (is_mc and open_ended) else ""
    sample_set_tag = "repeated_images" if (is_mc and repeated_images) else "unique_images"
    unique_images = is_mc and not repeated_images

    dataset_prefix = (
        f"{dataset_slug}_{open_ended_tag}_{caption_source}"
        if open_ended_tag
        else f"{dataset_slug}_{caption_source}"
    )
    if dataset_split:
        dataset_prefix = f"{dataset_prefix}_{dataset_split}"

    results_dir = os.path.expanduser(args.results_dir)
    similarities_dir = os.path.join(
        results_dir,
        dataset_prefix,
        f"{POOLING_METHOD}--{POOLING_METHOD}",
        sample_set_tag,
        "similarities",
    )
    os.makedirs(similarities_dir, exist_ok=True)

    def get_repr_path(dataset_n: str) -> str:
        repr_dir = os.path.join(
            os.path.expanduser(args.representations_dir),
            model_name,
            dataset_n,
            RESIDUAL_STREAM_TYPE,
            POOLING_METHOD,
            sample_set_tag,
        )
        os.makedirs(repr_dir, exist_ok=True)
        return os.path.join(repr_dir, "representations.safetensors")

    print(f"\n{'-' * 80}")
    print(f"🔍 Analysis: {RESIDUAL_STREAM_TYPE} | pooling={POOLING_METHOD} | {dataset_prefix}")
    print(f"{'-' * 80}")

    missing_measures = find_missing_measures(
        args.similarity_measures,
        similarities_dir,
        model_name,
        args.dry_run,
        args.force or args.force_metrics_only,
    )
    if not missing_measures:
        print(f"    ⏩ All measures already computed for {dataset_prefix}. Skipping.")
        return

    print(f"    ⚠️  Missing/forced measures: {missing_measures}")

    repr_path_txt = get_repr_path(f"{dataset_prefix}_txt")
    repr_path_img = get_repr_path(f"{dataset_prefix}_img")

    res_txt = load_representation_from_cache(repr_path_txt, args.dry_run, args.force)
    res_img = load_representation_from_cache(repr_path_img, args.dry_run, args.force)
    txt_from_cache = res_txt is not None
    img_from_cache = res_img is not None

    dl_kwargs = dict(
        dataset_path_or_name=dataset_path_or_name,
        cache_dir=datasets_cache_dir,
        guide_text=args.guide_text,
        batch_size=args.batch_size,
        seed=args.seed,
        downsample_size=effective_downsample_size,
        caption_source=caption_source,
        unique_images=unique_images,
        open_ended=open_ended,
        dataset_split=dataset_split,
    )

    if not txt_from_cache or not img_from_cache:
        model, processor = ensure_model()
        if not txt_from_cache:
            print(f"  👉 Extracting Text Representations ({dataset_prefix}_txt)...")
            dl_txt = build_dataloader(**dl_kwargs, processor=processor, txt_mode=True)
            res_txt = extract_last_token_representations(model, dl_txt)
            print(f"     Shape: {res_txt.shape}")
            del dl_txt
        if not img_from_cache:
            print(f"  👉 Extracting Image Representations ({dataset_prefix}_img)...")
            dl_img = build_dataloader(**dl_kwargs, processor=processor, txt_mode=False)
            res_img = extract_last_token_representations(model, dl_img)
            print(f"     Shape: {res_img.shape}")
            del dl_img
        gc.collect()
        torch.cuda.empty_cache()

    base_meta = {
        "model": model_name,
        "type": RESIDUAL_STREAM_TYPE,
        "pooling_method": POOLING_METHOD,
        "caption_source": caption_source,
        "open_ended": str(bool(open_ended_tag)),
        "repeated_images": str(sample_set_tag == "repeated_images"),
    }
    if not txt_from_cache and args.save_representations:
        save_raw_representation(
            res_txt, repr_path_txt,
            {**base_meta, "dataset": f"{dataset_prefix}_txt", "modality": "text"},
            args.dry_run, args.force,
        )
    if not img_from_cache and args.save_representations:
        save_raw_representation(
            res_img, repr_path_img,
            {**base_meta, "dataset": f"{dataset_prefix}_img", "modality": "image"},
            args.dry_run, args.force,
        )

    if args.representations_only:
        print("  ⏭️  --representations-only: skipping similarity computation.")
    else:
        print("  👉 Computing Similarities...")
        run_pair_analysis(
            res_txt, res_img,
            modality_a="text", modality_b="image",
            similarities_dir=similarities_dir,
            model_name=model_name,
            missing_measures=missing_measures,
            args=args,
        )
        print(f"  ✅ Results saved to {similarities_dir}")

    del res_txt, res_img
    gc.collect()
    torch.cuda.empty_cache()


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Text-vs-image residual stream similarity analysis for Llama 3.2 Vision 90B. "
            "Loads the model once and processes all dataset specs in sequence."
        )
    )
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--representations-dir",
        type=str,
        default=None,
        help="Base directory for caching representations (e.g. $HOME/scratch).",
    )
    parser.add_argument(
        "--dataset-runs",
        nargs="+",
        required=True,
        help=(
            "One or more dataset run specs: "
            "'dataset_path|caption_source|dataset_split|downsample_size|repeated_images|open_ended'. "
            "Example: 'l11p/GQA-val-balanced-noyesno-captioned|synthetic||2000|false|true'"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or Hugging Face ID for the Llama Vision model.",
    )
    parser.add_argument(
        "--guide-text",
        type=str,
        default="Answer the question using a single word or phrase.",
    )
    parser.add_argument(
        "--similarity-measures",
        nargs="+",
        default=SIMILARITY_MEASURES_DEFAULT,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite both representations and similarity results.",
    )
    parser.add_argument(
        "--force-metrics-only",
        action="store_true",
        help="Recompute similarities only; reuse cached representations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No load, no save. For debugging.",
    )
    parser.add_argument(
        "--save-representations",
        action="store_true",
        help="Persist raw representation tensors to disk.",
    )
    parser.add_argument(
        "--representations-only",
        action="store_true",
        help="Extract and save representations only; skip similarity computation.",
    )
    parser.add_argument(
        "--check-dataset-only",
        action="store_true",
        help=(
            "Build dataloaders, print one batch shape per dataset, then exit. "
            "Does NOT load the model. Use on nodes without enough VRAM."
        ),
    )

    args = parser.parse_args()

    if args.force and args.force_metrics_only:
        parser.error("--force and --force-metrics-only are mutually exclusive.")

    if args.dry_run:
        print("\n🧪 DRY-RUN MODE — no caching, no saving")
        args.results_dir = args.results_dir or "/tmp/debug_run"
        args.representations_dir = args.representations_dir or "/tmp/debug_run"
    elif not args.check_dataset_only:
        if args.results_dir is None or args.representations_dir is None:
            parser.error(
                "--results-dir and --representations-dir are required "
                "(unless --dry-run or --check-dataset-only is set)"
            )

    # Parse and validate all specs up-front so errors surface before loading the model
    try:
        dataset_configs = [parse_dataset_spec(s) for s in args.dataset_runs]
    except ValueError as e:
        parser.error(str(e))

    seed_all(args.seed)

    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE")
    model_path = os.path.expanduser(args.model_path)
    model_name = model_path.rstrip("/").split("/")[-1]

    # ── Check-dataset-only mode ────────────────────────────────────────────────
    if args.check_dataset_only:
        print("\n🔬 CHECK-DATASET-ONLY MODE — validating dataloaders, no model loaded")

        if os.path.exists(model_path):
            print(f"📦 Loading processor from {model_path} (tokenizer only, no GPU)...")
            processor = AutoProcessor.from_pretrained(model_path)
            processor.tokenizer.padding_side = "left"
        else:
            print(f"⚠️  Model path not accessible: {model_path}. Skipping collate check.")
            processor = None

        for cfg in dataset_configs:
            dataset_path_or_name = cfg["dataset_path_or_name"]
            is_mc = any(b in dataset_path_or_name.lower() for b in MULTIPLE_CHOICE_BENCHMARKS)
            unique_images = is_mc and not cfg["repeated_images"]
            print(f"\n  📊 {dataset_path_or_name} | caption={cfg['caption_source']} | open_ended={cfg['open_ended']}")
            dl_kwargs = dict(
                dataset_path_or_name=dataset_path_or_name,
                cache_dir=datasets_cache_dir,
                guide_text=args.guide_text,
                batch_size=args.batch_size,
                seed=args.seed,
                downsample_size=cfg["downsample_size"],
                caption_source=cfg["caption_source"],
                unique_images=unique_images,
                open_ended=cfg["open_ended"],
                dataset_split=cfg["dataset_split"],
            )
            for mode_label, txt_mode in [("txt", True), ("img", False)]:
                dl = build_dataloader(**dl_kwargs, processor=processor, txt_mode=txt_mode)
                print(f"    {mode_label}: {len(dl.dataset)} samples", end="")
                if processor is not None:
                    batch = next(iter(dl))
                    print(f" | input_ids={batch['input_ids'].shape}", end="")
                    if "pixel_values" in batch:
                        print(f" | pixel_values={batch['pixel_values'].shape}", end="")
                print()

        print("\n✅ Dataset check complete. Model was NOT loaded.")
        return

    # ── Lazy model loading (once, shared across all datasets) ──────────────────
    model: MllamaForConditionalGeneration | None = None
    processor: AutoProcessor | None = None

    def ensure_model() -> tuple[MllamaForConditionalGeneration, AutoProcessor]:
        nonlocal model, processor
        if model is not None:
            return model, processor

        print(f"\n🤖 Loading model: {model_path}")
        print(f"   dtype=bfloat16 | device_map=auto | attn_implementation={ATTN_IMPLEMENTATION}")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            dtype=MODEL_DTYPE,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_path)
        processor.tokenizer.padding_side = "left"
        print(f"   ✅ Model loaded. padding_side={processor.tokenizer.padding_side}")
        return model, processor

    # ── Process all datasets with model loaded once ────────────────────────────
    os.makedirs(os.path.expanduser(args.results_dir), exist_ok=True)

    for cfg in dataset_configs:
        print(f"\n{'═' * 64}")
        print(f"🌍 Dataset: {cfg['dataset_path_or_name']}")
        print(f"{'═' * 64}")
        try:
            process_dataset(cfg, args, ensure_model, datasets_cache_dir, model_name)
        except Exception as e:
            print(f"❌ Dataset {cfg['dataset_path_or_name']} failed: {type(e).__name__}: {e}")
            print("   Continuing with next dataset...")

    if model is not None:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    print("\n✅ All datasets complete.")


if __name__ == "__main__":
    main()
