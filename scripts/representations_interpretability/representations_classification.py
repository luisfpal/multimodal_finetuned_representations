import argparse
import gc
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from src.residual_stream_tracer import residual_stream_tracer
from utils import load_hf_model_and_processor_or_tokenizer, seed_all
from utils.classification_collators import (
    _build_sample_signature,
    _cambrian_icl_collate_fn,
    _cambrian_zero_shot_collate_fn,
    _icl_collate_fn,
    _internvl_icl_collate_fn,
    _internvl_zero_shot_collate_fn,
    _is_cambrian,
    _is_internvl,
    _llava_zero_shot_collate_fn,
    _load_mmmu_dev_pool,
)
from utils.cluster_labels import apply_cluster_labels
from utils.dataset import (
    MCQ_FILTER_MODES,
    MCQ_MODES,
    SEED,
    ClassificationDataset,
    _PROMPT_MODES,
)
from utils.metrics_utils import (
    classify_per_layer_knn,
    classify_per_layer_label_overlap,
    classify_per_layer_logistic_regression,
    compute_label_overlap,
    compute_neighborhood_overlap,
    homogeneity_score_per_layer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "flash_attention_2" if DEVICE.type == "cuda" else "sdpa"
HOME_DIR = Path.home()
RESULTS_BASE = HOME_DIR / "multimodal_finetuned_representations" / "results"
REPRESENTATIONS_BASE = HOME_DIR / "scratch"
RESIDUAL_STREAM_TYPE = "output_layer"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_BATCH_SIZE = 64
METRIC_SHUFFLE_SEED = SEED

SUPPORTED_METRICS = (
    "logistic_regression",
    "knn",
    "label_overlap",
    "homogeneity",
    "accuracy",
    "clip_neighborhood_overlap",
)
METRIC_TO_KEYS: dict[str, list[str]] = {
    "logistic_regression": ["f1_weighted", "f1_weighted_train", "mcc"],
    "knn": ["knn_f1_weighted", "knn_mcc"],
    "label_overlap": ["label_overlap"],
    "homogeneity": ["homogeneity_score", "adjusted_rand_score", "n_clusters_ratio"],
    "clip_neighborhood_overlap": ["clip_neighborhood_overlap"],
}


def get_representations_and_labels(
    model_name: str,
    dataset_name: str,
    split: str,
    guide_text: str,
    tokens_pooling_method: str,
    representation_cache_path: Path | None,
    force_recompute_representations: bool,
    save_representations: bool,
    sample_set_tag: str,
    label_granularity: str = "intermediate",
    max_samples: int | None = None,
    n_shots: int = 0,
    save_prompts_path: Path | None = None,
    prompt_mode: str = "describe",
    metrics_only: bool = False,
    n_choices: int = 10,
    subsample_size: int | None = None,
    label_mode: str = "class_name",
    collect_predictions: bool = False,
) -> tuple[torch.Tensor, np.ndarray, list[list[str]], list[str | None], str, str]:
    is_mini_imagenet = "mini-imagenet" in dataset_name.lower()
    is_imagenet_animals = "imagenet_short_text" in dataset_name.lower()
    _dn_lower = dataset_name.lower()
    is_generic_classification = any(
        ds in _dn_lower for ds in ("food101", "sun397", "caltech101")
    )
    is_dataset_with_modes = is_mini_imagenet or is_imagenet_animals or is_generic_classification

    if is_mini_imagenet:
        if split != "validation":
            print(f"  [mini-imagenet] Overriding split '{split}' → 'validation'")
            split = "validation"
        if prompt_mode not in MCQ_MODES:
            guide_text = ""
        n_shots = 0

    if is_imagenet_animals:
        if split != "train":
            print(f"  [imagenet_animals] Overriding split '{split}' → 'train'")
            split = "train"
        if prompt_mode not in MCQ_MODES:
            guide_text = ""
        n_shots = 0

    if is_generic_classification:
        if "food101" in _dn_lower and split != "validation":
            print(f"  [food101] Overriding split '{split}' → 'validation'")
            split = "validation"
        if "sun397" in _dn_lower and split != "test":
            print(f"  [sun397] Overriding split '{split}' → 'test'")
            split = "test"
        if "caltech101" in _dn_lower and split != "train":
            print(f"  [caltech101] Overriding split '{split}' → 'train'")
            split = "train"
        if prompt_mode not in MCQ_MODES:
            guide_text = ""
        n_shots = 0

    assistant_prefix = (
        _PROMPT_MODES.get(prompt_mode, _PROMPT_MODES["describe"])[1]
        if (is_mini_imagenet or is_imagenet_animals)
        else ""
    )

    dataset = ClassificationDataset(
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        label_granularity=label_granularity,
        prompt_mode=prompt_mode if is_dataset_with_modes else "describe",
        seed=SEED,
        n_choices=n_choices,
        subsample_size=subsample_size,
        label_mode=label_mode,
    )
    labels = np.array(dataset.labels)
    sample_ids = list(dataset.sample_ids)
    sample_signature = _build_sample_signature(sample_ids=sample_ids, labels=labels)
    prompt_signature = dataset.prompt_signature
    prompt_signature_required = (
        prompt_mode == "describe" and (is_mini_imagenet or is_imagenet_animals)
    )

    if (
        representation_cache_path is not None
        and representation_cache_path.exists()
        and not force_recompute_representations
    ):
        print(f"Loading cached representations: {representation_cache_path}")
        cached_representations: torch.Tensor | None = None
        metadata = {}
        with safe_open(str(representation_cache_path), framework="pt") as handle:
            metadata = handle.metadata() or {}
            if "residual_stream" in handle.keys():
                cached_representations = handle.get_tensor("residual_stream")

        if cached_representations is not None:
            cache_signature = metadata.get("sample_signature", "")
            cache_num_samples = metadata.get("num_samples", "")
            cache_pooling = metadata.get("tokens_pooling_method", "")
            cache_split = metadata.get("split", "")
            cache_sample_set = metadata.get("sample_set_tag", "")
            cache_prompt_signature = metadata.get("prompt_signature", "")
            prompt_signature_matches = cache_prompt_signature == prompt_signature
            if not prompt_signature_required and not cache_prompt_signature:
                prompt_signature_matches = True

            is_valid_cache = (
                cached_representations.shape[1] == len(labels)
                and cache_num_samples == str(len(labels))
                and cache_signature == sample_signature
                and cache_pooling == tokens_pooling_method
                and cache_split == split
                and cache_sample_set == sample_set_tag
                and prompt_signature_matches
            )

            if is_valid_cache:
                if save_prompts_path is not None:
                    print(
                        f"  [prompts] Representations loaded from cache — "
                        f"prompts were saved in the original run. "
                        f"Re-run with --force-recompute-representations to regenerate."
                    )
                if collect_predictions:
                    print(
                        "  [accuracy] Representations loaded from cache — "
                        "next-token predictions are only collected during extraction."
                    )
                return cached_representations, labels, [], [], sample_signature, prompt_signature

            if metrics_only:
                raise RuntimeError(
                    f"--metrics-only: cached representations at\n  {representation_cache_path}\n"
                    "exist but failed validation (shape/metadata/signature/prompt mismatch). "
                    "Re-run without --metrics-only to recompute."
                )
            print("Cached representations mismatch (shape/metadata/signature/prompt). Recomputing...")

    _prompt_log: list[str] = []
    _layer_token_log: list[list[str]] = []

    shots_pool: dict[str, list[dict]] | None = None
    if n_shots > 0 and "mmmu" in dataset_name.lower():
        print(f"Loading MMMU dev pool for {n_shots}-shot ICL ...")
        shots_pool = _load_mmmu_dev_pool(label_granularity)
        print(f"  Dev pool: {len(shots_pool)} categories, "
              f"{sum(len(v) for v in shots_pool.values())} instances.")

    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=model_name,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE,
    )

    if _is_internvl(model_name):
        if shots_pool is not None:
            collate_fn = partial(
                _internvl_icl_collate_fn,
                model=model,
                tokenizer=processor,
                shots_pool=shots_pool,
                n_shots=n_shots,
                label_granularity=label_granularity,
                assistant_prefix=assistant_prefix,
                prompt_log=_prompt_log,
            )
        else:
            collate_fn = partial(
                _internvl_zero_shot_collate_fn,
                model=model,
                tokenizer=processor,
                guide_text=guide_text,
                assistant_prefix=assistant_prefix,
                prompt_log=_prompt_log,
            )
    elif _is_cambrian(model_name):
        if shots_pool is not None:
            collate_fn = partial(
                _cambrian_icl_collate_fn,
                model=model,
                tokenizer=processor,
                shots_pool=shots_pool,
                n_shots=n_shots,
                label_granularity=label_granularity,
                guide_text=guide_text,
                assistant_prefix=assistant_prefix,
                prompt_log=_prompt_log,
            )
        else:
            collate_fn = partial(
                _cambrian_zero_shot_collate_fn,
                model=model,
                tokenizer=processor,
                guide_text=guide_text,
                assistant_prefix=assistant_prefix,
                prompt_log=_prompt_log,
            )
    elif shots_pool is not None:
        collate_fn = partial(
            _icl_collate_fn,
            processor=processor,
            shots_pool=shots_pool,
            n_shots=n_shots,
            guide_text=guide_text,
            label_granularity=label_granularity,
            assistant_prefix=assistant_prefix,
            prompt_log=_prompt_log,
        )
    else:
        collate_fn = partial(
            _llava_zero_shot_collate_fn,
            processor=processor,
            guide_text=guide_text,
            assistant_prefix=assistant_prefix,
            prompt_log=_prompt_log,
        )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    need_predictions = save_prompts_path is not None or collect_predictions
    representations = residual_stream_tracer(
        model=model,
        processed_dataloader=dataloader,
        processor=processor,
        return_dtype=torch.float32,
        tokens_pooling_method=tokens_pooling_method,
        residual_stream_type="output_layer",
        return_deepcopy=True,
        save_representations=False,
        layer_token_log=_layer_token_log if need_predictions else None,
    )

    if not isinstance(representations, torch.Tensor):
        raise ValueError(
            "Expected tensor representations from residual_stream_tracer. "
            f"Got: {type(representations)}"
        )

    if representation_cache_path is not None and save_representations:
        os.makedirs(representation_cache_path.parent, exist_ok=True)
        save_file(
            {"residual_stream": representations.cpu().to(torch.float32)},
            str(representation_cache_path),
            metadata={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "split": split,
                "tokens_pooling_method": tokens_pooling_method,
                "num_samples": str(len(labels)),
                "sample_signature": sample_signature,
                "prompt_signature": prompt_signature,
                "sample_set_tag": sample_set_tag,
            },
        )
        print(f"Saved representations cache: {representation_cache_path}")

    if save_prompts_path is not None and _prompt_log:
        option_letters = getattr(dataset, "_option_letters", None)
        records = [
            {
                "idx": i,
                "sample_id": sample_ids[i],
                "label": str(labels[i]),
                "class_name": str(labels[i]),
                "option_letter": option_letters[i] if option_letters else None,
                "formatted_prompt": prompt,
                "next_tokens_per_layer": _layer_token_log[i] if i < len(_layer_token_log) else [],
            }
            for i, prompt in enumerate(_prompt_log)
        ]
        save_prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_prompts_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Saved prompts: {save_prompts_path}")

    option_letters = list(getattr(dataset, "_option_letters", None) or [None] * len(labels))
    result = representations.cpu().to(torch.float32)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result, labels, _layer_token_log, option_letters, sample_signature, prompt_signature


def _compute_accuracy(
    layer_predictions: list[list[str]],
    option_letters: list[str | None],
) -> tuple[float, int, int]:
    """Exact-match accuracy at the last transformer layer (logit-lens prediction)."""
    valid = [
        (preds[-1], gt)
        for preds, gt in zip(layer_predictions, option_letters)
        if gt and preds
    ]
    if not valid:
        return 0.0, 0, 0
    hits = sum(p.strip().lstrip("▁").upper() == gt.upper() for p, gt in valid)
    return hits / len(valid), hits, len(valid)


def _effective_dataset_split(dataset_name: str, split: str) -> str:
    lower_name = dataset_name.lower()
    if "mini-imagenet" in lower_name:
        return "validation"
    if "imagenet_short_text" in lower_name:
        return "train"
    if "food101" in lower_name:
        return "validation"
    if "sun397" in lower_name:
        return "test"
    if "caltech101" in lower_name:
        return "train"
    return split


def _load_metric_dataset(args: argparse.Namespace) -> ClassificationDataset:
    return ClassificationDataset(
        dataset_name=args.dataset_name,
        split=_effective_dataset_split(args.dataset_name, args.split),
        max_samples=args.max_samples,
        label_granularity=args.mmmu_label_granularity,
        prompt_mode=args.prompt_mode,
        seed=SEED,
        n_choices=args.n_choices,
        subsample_size=args.subsample_size,
        label_mode=args.label_mode,
    )


@torch.no_grad()
def _compute_clip_image_embeddings(dataset: ClassificationDataset) -> torch.Tensor:
    print(f"Loading CLIP image encoder: {CLIP_MODEL_ID}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=MODEL_DTYPE).to(DEVICE).eval()

    all_embeddings: list[torch.Tensor] = []
    for start in range(0, len(dataset), CLIP_BATCH_SIZE):
        end = min(start + CLIP_BATCH_SIZE, len(dataset))
        images = [dataset[idx]["image"] for idx in range(start, end)]
        missing = [start + i for i, image in enumerate(images) if image is None]
        if missing:
            raise ValueError(f"CLIP embeddings require images; missing at indices {missing[:5]}")
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        embeddings = model.get_image_features(**inputs)
        all_embeddings.append(embeddings.cpu().to(torch.float32))

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)


def _compute_clip_neighborhood_overlap_per_layer(
    representations: torch.Tensor,
    clip_embeddings: torch.Tensor,
) -> dict[int, dict[str, float]]:
    num_layers = representations.shape[0]
    layer_results: dict[int, dict[str, float]] = {}
    for layer_idx in range(num_layers):
        overlap = compute_neighborhood_overlap(
            representations[layer_idx],
            clip_embeddings,
            maxk=30,
            use_dadapy=False,
            metric="euclidean",
        )
        layer_results[layer_idx] = {"clip_neighborhood_overlap": float(overlap)}
    return layer_results


def _metric_permutation(num_samples: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(METRIC_SHUFFLE_SEED)
    return torch.randperm(num_samples, generator=generator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="lbasile/llava_7b")
    parser.add_argument("--dataset-name", type=str, default="derek-thomas/ScienceQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--guide-text",
        type=str,
        default="Answer with the option's letter from the given choices directly.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help=(
            "Root directory for saving results. Use default to save under "
            "~/multimodal_finetuned_representations/results."
        ),
    )
    parser.add_argument(
        "--representations-dir",
        type=str,
        default=str(REPRESENTATIONS_BASE),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--tokens-pooling-method",
        type=str,
        default="last",
        choices=["last", "mean_image"],
    )
    parser.add_argument(
        "--save-representations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-recompute-representations", action="store_true")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["logistic_regression"],
        choices=list(SUPPORTED_METRICS),
    )
    parser.add_argument(
        "--mmmu-label-granularity",
        type=str,
        default="intermediate",
        choices=["intermediate", "detailed"],
        help=(
            "Label granularity for MMMU. "
            "'intermediate': subjects (~30 classes, default). "
            "'detailed': subfields with count >= mean count."
        ),
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=0,
        help=(
            "Number of in-context learning shots from the MMMU dev split. "
            "0 = zero-shot (default). Shots are matched by subject/subfield. "
            "Results and representations are saved with a _{n}shot suffix."
        ),
    )
    parser.add_argument(
        "--save-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save formatted prompts (after chat-template application) to "
            "prompts{icl_suffix}.json in the same directory as metric files."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="describe",
        choices=list(_PROMPT_MODES.keys()),
        help=(
            "Prompt mode for supported classification datasets. "
            "'describe' (default): detailed caption with a balanced seeded opening phrase. "
            "'continuation': short question + seeded assistant prefix for continuation. "
            "Non-default modes add a suffix to all output paths."
        ),
    )
    parser.add_argument(
        "--representations-only",
        action="store_true",
        help="Extract and save representations only — skip all metric computation.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help=(
            "Load representations from cache and compute metrics — skip GPU extraction. "
            "Fails fast if the representation cache does not exist."
        ),
    )
    parser.add_argument(
        "--cluster-labels",
        action="store_true",
        default=False,
        help=(
            "Replace dataset labels with CLIP-ADP cluster IDs and cap the dominant cluster "
            "to the size of the second-largest. Adds '_cluster' suffix to metric output paths."
        ),
    )
    parser.add_argument(
        "--class-fraction",
        type=float,
        default=None,
        help=(
            "class_fraction for compute_label_overlap (fraction of class samples used as k). "
            "Affects label_overlap and JSONL clip_label_overlap. Adds _cf{value} suffix "
            "to label_overlap output. "
            "Default (None): uses compute_label_overlap's built-in default (0.3)."
        ),
    )
    parser.add_argument(
        "--maxk",
        type=int,
        default=None,
        help=(
            "maxk for compute_label_overlap (max nearest neighbours). "
            "Affects label_overlap and JSONL clip_label_overlap. Adds _mk{value} "
            "suffix to label_overlap output when set. "
            "Default (None): uses compute_label_overlap's built-in default (30)."
        ),
    )
    parser.add_argument(
        "--n-choices",
        type=int,
        default=10,
        help=(
            "Synthetic MCQ width N for mcq / mcq_nocat[_filter] / animal_mcq[_filter] "
            "/ generic_mcq[_filter]. "
            "For category-based 'mcq', acts as cap (k=min(category_size, N)). "
            "For *_filter modes, also sets the number of classes the dataset is "
            "filtered to. Encoded as _n{N} suffix in MCQ output filenames. "
            "Ignored for non-MCQ modes and not used for ScienceQA/MMMU."
        ),
    )
    parser.add_argument(
        "--subsample-size",
        type=int,
        default=None,
        help=(
            "Seeded class-stratified random subsample size. Adds _ss{M} suffix to "
            "all output paths when set. Mutually exclusive with *_filter prompt "
            "modes (which already subsample by class)."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=["class_name", "option_letter"],
        default="class_name",
        help=(
            "Label source for metric computation. 'option_letter' uses the MCQ "
            "letter assigned to the correct answer (A/B/C/...) instead of the "
            "class name. Only valid for synthetic MCQ prompt modes on supported "
            "image-classification datasets. Adds _lbl_letter suffix to metric "
            "output files (representations cache is unchanged)."
        ),
    )
    return parser.parse_args()


def _resolve_results_base(results_dir_arg: str) -> Path:
    if results_dir_arg == "results":
        return RESULTS_BASE
    return Path(results_dir_arg).expanduser().resolve()


def _build_output_path(
    *,
    results_base: Path,
    dataset_name: str,
    split: str,
    model_name: str,
    pooling_method: str,
    metric_name: str,
    suffix: str = "",
    extension: str = "safetensors",
) -> Path:
    dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
    model_slug = Path(model_name).name
    out_dir = (
        results_base
        / "representations_classification"
        / dataset_slug
        / split
        / model_slug
        / pooling_method
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{metric_name}{suffix}.{extension}"


def _build_run_metrics_path(
    *,
    results_base: Path,
    dataset_name: str,
    split: str,
    model_name: str,
) -> Path:
    dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
    model_slug = Path(model_name).name
    out_dir = results_base / "representations_classification" / dataset_slug / split / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "run_metrics.jsonl"


def _base_run_metric_record(
    *,
    args: argparse.Namespace,
    metric_name: str,
    value: float,
    num_samples: int,
    metrics_suffix: str,
) -> dict:
    dataset_slug = args.dataset_name.split("/")[-1].replace("-", "_").lower()
    model_slug = Path(args.model_name).name
    record = {
        "metric": metric_name,
        "value": float(value),
        "model_name": args.model_name,
        "model_slug": model_slug,
        "dataset_name": args.dataset_name,
        "dataset_slug": dataset_slug,
        "split": args.split,
        "pooling_method": args.tokens_pooling_method,
        "prompt_mode": args.prompt_mode,
        "label_mode": args.label_mode,
        "cluster_labels": bool(args.cluster_labels),
        "subsample_size": args.subsample_size,
        "num_samples": int(num_samples),
        "experiment_suffix": metrics_suffix,
    }
    if args.prompt_mode in MCQ_MODES:
        record["n_choices"] = int(args.n_choices)
    else:
        record["n_choices"] = None
    return record


def _append_run_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Appended {record['metric']} ({record['value']:.4f}): {path}")


def _matching_accuracy_record_exists(
    path: Path,
    *,
    args: argparse.Namespace,
    metrics_suffix: str,
) -> bool:
    if not path.exists():
        return False
    expected_n_choices = int(args.n_choices) if args.prompt_mode in MCQ_MODES else None
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [accuracy] Ignoring corrupt JSONL line {line_number} in {path}")
                continue
            if record.get("metric") != "accuracy":
                continue
            if record.get("model_name") != args.model_name:
                continue
            if record.get("dataset_name") != args.dataset_name:
                continue
            if record.get("split") != args.split:
                continue
            if record.get("pooling_method") != args.tokens_pooling_method:
                continue
            if record.get("prompt_mode") != args.prompt_mode:
                continue
            if record.get("label_mode") != args.label_mode:
                continue
            if bool(record.get("cluster_labels", False)) != bool(args.cluster_labels):
                continue
            if record.get("subsample_size") != args.subsample_size:
                continue
            if record.get("experiment_suffix") != metrics_suffix:
                continue
            if record.get("n_choices") != expected_n_choices:
                continue
            return True
    return False


def _build_representation_cache_path(
    *,
    representations_base: Path,
    dataset_name: str,
    model_name: str,
    pooling_method: str,
    sample_set_tag: str,
) -> Path:
    dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
    model_slug = model_name.split("/")[-1]
    out_dir = (
        representations_base
        / "representations"
        / model_slug
        / dataset_slug
        / RESIDUAL_STREAM_TYPE
        / pooling_method
        / sample_set_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "representations.safetensors"


def _save_metric_results(
    *,
    out_path: Path,
    metric_name: str,
    layer_results: dict[int, dict[str, float]],
    num_layers: int,
    args: argparse.Namespace,
    num_samples: int,
    label_classes: np.ndarray,
) -> None:
    keys = METRIC_TO_KEYS[metric_name]
    tensors = {
        key: torch.tensor(
            [layer_results[i][key] for i in range(num_layers)], dtype=torch.float32
        )
        for key in keys
    }
    metadata = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "pooling_method": args.tokens_pooling_method,
        "guide_text": args.guide_text,
        "num_samples": str(num_samples),
        "classes": ", ".join(str(c) for c in label_classes),
        "metric": metric_name,
        "metric_keys": ", ".join(keys),
        "prompt_mode": args.prompt_mode,
        "label_mode": args.label_mode,
    }
    if args.prompt_mode in MCQ_MODES:
        metadata["n_choices"] = str(args.n_choices)
    if args.subsample_size is not None:
        metadata["subsample_size"] = str(args.subsample_size)
    if metric_name == "label_overlap":
        if args.class_fraction is not None:
            metadata["class_fraction"] = str(args.class_fraction)
        if args.maxk is not None:
            metadata["maxk"] = str(args.maxk)
    if metric_name == "clip_neighborhood_overlap":
        metadata["clip_model_name"] = CLIP_MODEL_ID
        metadata["maxk"] = "30"
    save_file(tensors, str(out_path), metadata=metadata)
    print(f"Saved {metric_name}: {out_path}")


def main() -> None:
    args = parse_args()
    effective_split = _effective_dataset_split(args.dataset_name, args.split)
    if effective_split != args.split:
        print(f"  [split] Overriding split '{args.split}' -> '{effective_split}'")
        args.split = effective_split
    results_base = _resolve_results_base(args.results_dir)
    representations_base = Path(args.representations_dir).expanduser().resolve()

    if args.subsample_size is not None and args.prompt_mode in MCQ_FILTER_MODES:
        raise ValueError(
            f"--subsample-size is mutually exclusive with filter modes; got "
            f"--prompt-mode={args.prompt_mode} --subsample-size={args.subsample_size}. "
            f"Filter modes already subsample by class."
        )

    icl_suffix = f"_{args.n_shots}shot" if args.n_shots > 0 else ""
    is_mini_imagenet = "mini-imagenet" in args.dataset_name.lower()
    is_imagenet_animals = "imagenet_short_text" in args.dataset_name.lower()
    is_food101 = "food101" in args.dataset_name.lower()
    is_sun397 = "sun397" in args.dataset_name.lower()
    is_caltech101 = "caltech101" in args.dataset_name.lower()
    is_generic_classification = is_food101 or is_sun397 or is_caltech101
    is_dataset_with_modes = is_mini_imagenet or is_imagenet_animals or is_generic_classification
    is_mcq = args.prompt_mode in MCQ_MODES and is_dataset_with_modes

    if args.label_mode == "option_letter" and not is_mcq:
        raise ValueError(
            "--label-mode=option_letter requires an MCQ prompt mode on a supported dataset; "
            f"got --prompt-mode={args.prompt_mode!r}"
        )
    mode_active = (
        (is_mini_imagenet and args.prompt_mode != "describe")
        or is_imagenet_animals
        or is_generic_classification
    )
    n_suffix = f"_n{args.n_choices}" if is_mcq else ""
    ss_suffix = (
        f"_ss{args.subsample_size}"
        if args.subsample_size is not None and is_dataset_with_modes
        else ""
    )
    lbl_suffix = "_lbl_letter" if (args.label_mode == "option_letter" and is_mcq) else ""
    mode_suffix = f"_{args.prompt_mode}" if mode_active else ""
    prompt_suffix = mode_suffix + n_suffix + ss_suffix
    output_suffix = icl_suffix + prompt_suffix        # representations cache — no lbl suffix
    metrics_suffix = output_suffix + lbl_suffix + ("_cluster" if args.cluster_labels else "")
    sample_set_tag = f"unique_images{output_suffix}"

    _lo_parts: list[str] = []
    if args.class_fraction is not None:
        _lo_parts.append(f"cf{args.class_fraction}")
    if args.maxk is not None:
        _lo_parts.append(f"mk{args.maxk}")
    label_overlap_suffix = ("_" + "_".join(_lo_parts)) if _lo_parts else ""

    if args.representations_only and args.metrics_only:
        raise ValueError("--representations-only and --metrics-only are mutually exclusive.")

    representation_cache_path = _build_representation_cache_path(
        representations_base=representations_base,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        pooling_method=args.tokens_pooling_method,
        sample_set_tag=sample_set_tag,
    )

    if args.metrics_only and not representation_cache_path.exists():
        raise FileNotFoundError(
            f"--metrics-only requires cached representations at:\n  {representation_cache_path}\n"
            "Run without --metrics-only first to extract and save representations."
        )

    prompts_save_path = (
        _build_output_path(
            results_base=results_base,
            dataset_name=args.dataset_name,
            split=args.split,
            model_name=args.model_name,
            pooling_method=args.tokens_pooling_method,
            metric_name="prompts",
            suffix=output_suffix,
            extension="json",
        )
        if args.save_prompts
        else None
    )

    (
        representations,
        labels,
        layer_predictions,
        option_letters,
        sample_signature,
        prompt_signature,
    ) = get_representations_and_labels(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        guide_text=args.guide_text,
        tokens_pooling_method=args.tokens_pooling_method,
        representation_cache_path=representation_cache_path,
        force_recompute_representations=args.force_recompute_representations,
        save_representations=args.save_representations,
        sample_set_tag=sample_set_tag,
        label_granularity=args.mmmu_label_granularity,
        max_samples=args.max_samples,
        n_shots=args.n_shots,
        save_prompts_path=prompts_save_path,
        prompt_mode=args.prompt_mode,
        metrics_only=args.metrics_only,
        n_choices=args.n_choices,
        subsample_size=args.subsample_size,
        label_mode=args.label_mode,
        collect_predictions="accuracy" in args.metrics,
    )

    if args.representations_only:
        print("--representations-only: representations saved. Skipping metric computation.")
        return

    clip_embeddings: torch.Tensor | None = None
    if "clip_neighborhood_overlap" in args.metrics:
        clip_dataset = _load_metric_dataset(args)
        clip_labels = np.array(clip_dataset.labels)
        clip_sample_signature = _build_sample_signature(
            sample_ids=list(clip_dataset.sample_ids),
            labels=clip_labels,
        )
        if clip_sample_signature != sample_signature:
            raise RuntimeError("CLIP dataset sample signature does not match representations.")
        if clip_dataset.prompt_signature != prompt_signature:
            raise RuntimeError("CLIP dataset prompt signature does not match representations.")
        clip_embeddings = _compute_clip_image_embeddings(clip_dataset)

    if args.cluster_labels:
        if clip_embeddings is not None:
            representations, labels, cluster_mask = apply_cluster_labels(
                representations, labels, args.dataset_name, return_mask=True
            )
            clip_embeddings = clip_embeddings[cluster_mask]
        else:
            representations, labels = apply_cluster_labels(
                representations, labels, args.dataset_name
            )

    encoded_labels = np.asarray(labels)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(encoded_labels)
    print(f'Number of unique labels: {label_encoder.classes_.size}')
    print(f'First 5 classes: {label_encoder.classes_[:5]}')

    num_layers, num_samples, _ = representations.shape

    non_accuracy_metrics = [metric for metric in args.metrics if metric != "accuracy"]
    if non_accuracy_metrics:
        permutation = _metric_permutation(num_samples)
        representations = representations[:, permutation, :]
        y = y[permutation.numpy()]
        if clip_embeddings is not None:
            clip_embeddings = clip_embeddings[permutation]

    for metric_name in args.metrics:
        if metric_name == "accuracy":
            continue  # handled separately below — needs layer_predictions, not representations

        print(f"\nComputing metric: {metric_name}")

        if metric_name == "logistic_regression":
            layer_results = classify_per_layer_logistic_regression(representations, y)
        elif metric_name == "knn":
            layer_results = classify_per_layer_knn(representations, y)
        elif metric_name == "label_overlap":
            lo_kwargs: dict = {}
            if args.maxk is not None:
                lo_kwargs["maxk"] = args.maxk
            if args.class_fraction is not None:
                lo_kwargs["class_fraction"] = args.class_fraction
            layer_results = classify_per_layer_label_overlap(representations, y, **lo_kwargs)
        elif metric_name == "clip_neighborhood_overlap":
            if clip_embeddings is None:
                raise RuntimeError("CLIP embeddings were not computed.")
            layer_results = _compute_clip_neighborhood_overlap_per_layer(
                representations,
                clip_embeddings,
            )
        else:  # homogeneity
            layer_results = homogeneity_score_per_layer(representations, y)

        metric_file_suffix = metrics_suffix + (
            label_overlap_suffix if metric_name == "label_overlap" else ""
        )
        out_path = _build_output_path(
            results_base=results_base,
            dataset_name=args.dataset_name,
            split=args.split,
            model_name=args.model_name,
            pooling_method=args.tokens_pooling_method,
            metric_name=metric_name,
            suffix=metric_file_suffix,
        )

        _save_metric_results(
            out_path=out_path,
            metric_name=metric_name,
            layer_results=layer_results,
            num_layers=num_layers,
            args=args,
            num_samples=num_samples,
            label_classes=label_encoder.classes_,
        )

        if metric_name == "clip_neighborhood_overlap":
            lo_kwargs = {}
            if args.maxk is not None:
                lo_kwargs["maxk"] = args.maxk
            if args.class_fraction is not None:
                lo_kwargs["class_fraction"] = args.class_fraction
            clip_label_overlap = compute_label_overlap(
                data=clip_embeddings,
                labels=y,
                metric="euclidean",
                **lo_kwargs,
            )
            run_metrics_path = _build_run_metrics_path(
                results_base=results_base,
                dataset_name=args.dataset_name,
                split=args.split,
                model_name=args.model_name,
            )
            record = _base_run_metric_record(
                args=args,
                metric_name="clip_label_overlap",
                value=clip_label_overlap,
                num_samples=num_samples,
                metrics_suffix=metrics_suffix,
            )
            record["clip_model_name"] = CLIP_MODEL_ID
            record["maxk"] = int(args.maxk) if args.maxk is not None else 30
            record["class_fraction"] = args.class_fraction
            _append_run_metric(run_metrics_path, record)

    if "accuracy" in args.metrics and is_mcq:
        print("\nComputing metric: accuracy")
        if not layer_predictions:
            run_metrics_path = _build_run_metrics_path(
                results_base=results_base,
                dataset_name=args.dataset_name,
                split=args.split,
                model_name=args.model_name,
            )
            if _matching_accuracy_record_exists(
                run_metrics_path,
                args=args,
                metrics_suffix=metrics_suffix,
            ):
                print(
                    "  [accuracy] Skipping: cached representations have a matching "
                    "accuracy record already."
                )
            else:
                print(
                    "  [accuracy] Skipping: no fresh next-token predictions are available. "
                    "Run without --metrics-only and with --force-recompute-representations "
                    "to create the scalar record."
                )
        elif len(layer_predictions) != len(option_letters):
            print(
                f"  [accuracy] Skipping: prediction count ({len(layer_predictions)}) "
                f"!= option_letters count ({len(option_letters)})."
            )
        else:
            acc, n_correct, n_total = _compute_accuracy(layer_predictions, option_letters)
            run_metrics_path = _build_run_metrics_path(
                results_base=results_base,
                dataset_name=args.dataset_name,
                split=args.split,
                model_name=args.model_name,
            )
            record = _base_run_metric_record(
                args=args,
                metric_name="accuracy",
                value=acc,
                num_samples=n_total,
                metrics_suffix=metrics_suffix,
            )
            record["n_correct"] = int(n_correct)
            record["n_total"] = int(n_total)
            _append_run_metric(run_metrics_path, record)
    elif "accuracy" in args.metrics:
        print(
            "\n[accuracy] Skipping: accuracy is only supported for synthetic MCQ "
            "prompt modes on image-classification datasets."
        )


if __name__ == "__main__":
    seed_all(SEED)
    main()
