"""Score image-caption semantic similarity with a CLIP-style model (clip or jina).

Inputs: --model (key of MODELS_CONFIG); datasets listed in DATASETS (HF_DATASETS_CACHE).
Outputs: one results file per (dataset, model), see get_output_path().
"""

import argparse
import base64
import json
import os
import sys
import warnings
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from typing import List, Union

from rich import print

# ---------------------------------------------------------------------------
# Environment  — must be set before any HuggingFace import
# ---------------------------------------------------------------------------
HOME_DIR = Path(os.path.expanduser("~"))
PROJECT_ROOT = HOME_DIR / "multimodal_finetuned_representations"
HF_HOME = HOME_DIR / "scratch" / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_DATASETS_CACHE = HF_HOME / "datasets"

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)

print(f"[blue]HF_HOME:            {HF_HOME}[/blue]")
print(f"[blue]HF_HUB_CACHE:       {HF_HUB_CACHE}[/blue]")
print(f"[blue]HF_DATASETS_CACHE:  {HF_DATASETS_CACHE}[/blue]")

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOWNSAMPLE_SIZE = 10_000
SPLIT = "train"
SEED = 42

# Display-only label — not used in any computation logic
REAL_DATASETS = {"l11p/COCOQA-restval-ICQA", "l11p/VQAv2-COCO-restval-ICQ"}

# Model Config
MODELS_CONFIG = {
    "clip": {
        "id": "openai/clip-vit-large-patch14-336",
        "max_tokens": 77,
        "needs_chunking": True,
        "trust_remote_code": False
    },
    "jina": {
        "id": "jinaai/jina-clip-v2",
        "max_tokens": 8192,
        "needs_chunking": False,
        "trust_remote_code": True,
        # Pin revision to avoid arbitrary code execution from upstream changes.
        # Update this hash deliberately after reviewing new releases.
        "revision": "344d954da76eb8ad47a7aaff42d012e30c15b8fe",
    }
}

# Dataset registry
# Entries with "hf_name" load from that HF repo but save results under the dict key.
# This lets us evaluate different caption fields from the same dataset separately.
DATASETS = {
    # "l11p/COCOQA-restval-ICQA": {
    #     "features": ["image", "captions", "image_id"],
    #     "load_config": {"remove_yes_no": True},
    # },
    # "l11p/VQAv2-COCO-restval-ICQ": {
    #     "features": ["image", "captions", "image_id"],
    #     "load_config": {"remove_yes_no": True},
    # },
    "l11p/COCOQA-restval-ICQA__synthetic": {
        "hf_name": "l11p/COCOQA-restval-ICQA",
        "features": ["image", "synthetic-caption", "image_id"],
        "load_config": {"remove_yes_no": True},
    },
    "l11p/VQAv2-COCO-restval-ICQ__synthetic": {
        "hf_name": "l11p/VQAv2-COCO-restval-ICQ",
        "features": ["image", "synthetic-caption", "image_id"],
        "load_config": {"remove_yes_no": True},
    },
    # "l11p/MMBench_EN-dev-captioned": {
    #     "features": ["image", "caption", "index"],
    #     "load_config": {"filter_mmbench_4_choices": True},
    # },
    # "l11p/MME-test-captioned": {
    #     "features": ["image", "caption", "question_id"],
    # },
    # "l11p/GQA-val-balanced-noyesno-captioned": {
    #     "features": ["image", "caption", "imageId"],
    # },
}

_MMBENCH_CHOICE_COLS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_raw_dataset(
    dataset_name: str,
    split: str = SPLIT,
    downsample_size: int = DOWNSAMPLE_SIZE,
    seed: int = SEED,
    remove_yes_no: bool = False,
    filter_mmbench_4_choices: bool = False,
):
    """Load a HuggingFace dataset and apply only the filters needed here."""
    dataset = load_dataset(dataset_name, split=split, cache_dir=str(HF_DATASETS_CACHE))

    if remove_yes_no and "answer" in dataset.column_names:
        before = len(dataset)
        dataset = dataset.filter(
            lambda x: str(x["answer"]).strip().lower() not in ["yes", "no"]
        )
        print(f"  [dim]remove_yes_no   : {before} → {len(dataset)}[/dim]")

    if filter_mmbench_4_choices:

        def _has_4_choices(ex):
            return (
                sum(
                    1
                    for c in _MMBENCH_CHOICE_COLS
                    if ex.get(c) is not None and str(ex.get(c)).lower() != "nan"
                )
                == 4
            )

        before = len(dataset)
        dataset = dataset.filter(_has_4_choices)
        print(f"  [dim]filter_4_choices: {before} → {len(dataset)}[/dim]")

    if downsample_size and downsample_size < len(dataset):
        print(f"  [dim]downsample      : {len(dataset)} → {downsample_size}[/dim]")
        dataset = dataset.shuffle(seed=seed).select(range(downsample_size))

    return dataset


# ---------------------------------------------------------------------------
# Output-path helpers
# ---------------------------------------------------------------------------
def get_output_path(dataset_name: str, model_key: str) -> Path:
    """Canonical save path: caption_evaluation / <dataset> / <model> / semantic_similarity.json"""
    # model_key handles the file naming now, e.g. "clip" or "jina"
    return (
        PROJECT_ROOT
        / "results"
        / "caption_evaluation"
        / dataset_name.replace("/", "_")
        / model_key
        / "semantic_similarity.json"
    )


def results_exist(dataset_name: str, model_key: str) -> bool:
    """True when the output JSON for this dataset already exists on disk."""
    return get_output_path(dataset_name, model_key).exists()


# ---------------------------------------------------------------------------
# Image normalisation
# ---------------------------------------------------------------------------
def _to_pil_rgb(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, str):
        return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(raw)}")


# ---------------------------------------------------------------------------
# Helper: Chunked Text Embedding
# ---------------------------------------------------------------------------
def get_text_embedding_chunked(text, model, processor, max_tokens, device, verbose=False):
    """
    Computes text embedding. If text is longer than max_tokens, chunks it and averages.
    """
    # 1. Tokenize without truncation to check length
    tokens = processor.tokenizer(text, return_tensors="pt", padding=False, truncation=False)
    input_ids = tokens["input_ids"][0]
    seq_len = len(input_ids)

    # 2. Check if chunking is needed
    if seq_len <= max_tokens:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
        return text_embeds

    # 3. Chunking logic
    if verbose:
        print(f"    [dim info] Chunking applied (length {seq_len} > {max_tokens}).[/dim info]")

    special_tokens_count = 2
    chunk_size = max_tokens - special_tokens_count
    stride = 20

    inner_ids = input_ids[1:-1]

    chunks = []
    start = 0
    while start < len(inner_ids):
        end = min(start + chunk_size, len(inner_ids))
        chunk_ids = inner_ids[start:end]

        full_chunk = torch.cat([
            input_ids[0].unsqueeze(0), # BOS
            chunk_ids,
            input_ids[-1].unsqueeze(0) # EOS
        ])
        chunks.append(full_chunk)

        if end == len(inner_ids):
            break
        start += (chunk_size - stride)

    # Pad chunks
    max_len_in_batch = max(len(c) for c in chunks)
    padded_chunks = []
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    for ch in chunks:
        if len(ch) < max_len_in_batch:
            padding = torch.tensor([pad_token_id] * (max_len_in_batch - len(ch)))
            ch = torch.cat([ch, padding])
        padded_chunks.append(ch)

    batch_input_ids = torch.stack(padded_chunks).to(device)
    batch_attention_mask = (batch_input_ids != pad_token_id).long().to(device)

    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            chunk_embeds = model.get_text_features(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        else:
            raise ValueError(f"Model {type(model)} does not have get_text_features")

        avg_embed = chunk_embeds.mean(dim=0, keepdim=True)

    return avg_embed


# ---------------------------------------------------------------------------
# Evaluator Class
# ---------------------------------------------------------------------------
class CaptionEvaluator:
    """Image↔text cosine similarity via a contrastive vision-language model."""

    def __init__(self, model_key: str, verbose: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose

        # Load Config
        self.model_config = MODELS_CONFIG[model_key]
        model_id = self.model_config["id"]

        print(f"  [dim]Initializing {model_id} on {self.device}...[/dim]")

        # Setup kwargs
        kwargs = {"use_safetensors": True}
        if self.model_config["trust_remote_code"]:
            kwargs["trust_remote_code"] = True

        # Jina-specific fix
        if model_key == "jina":
            sys.modules["flash_attn"] = None
            kwargs["attn_implementation"] = "eager"
            # Suppress Jina-internal deprecation (uses torch_dtype instead of dtype)
            warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

        # Pin revision for trust_remote_code models to avoid executing
        # arbitrary upstream code changes without review.
        revision = self.model_config.get("revision")
        if revision:
            kwargs["revision"] = revision

        try:
            # Load Processor
            proc_kwargs = {"trust_remote_code": self.model_config["trust_remote_code"], "use_fast": True}
            if revision:
                proc_kwargs["revision"] = revision
            self.processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

            # Load Model
            if "clip-vit" in model_id:
                self.model = CLIPModel.from_pretrained(model_id, **kwargs)
            else:
                self.model = AutoModel.from_pretrained(model_id, **kwargs)

            self.model.to(self.device).eval()

        except Exception as e:
            print(f"[red]Error loading model {model_id}: {e}[/red]")
            sys.exit(1)

        print(f"  [dim]Model loaded. Context window: {self.model_config['max_tokens']}[/dim]")

    @torch.no_grad()
    def _image_emb(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _text_emb(self, text: str) -> torch.Tensor:
        if self.model_config["needs_chunking"]:
            emb = get_text_embedding_chunked(
                text, self.model, self.processor,
                self.model_config["max_tokens"], self.device, self.verbose
            )
        else:
            # For models like Jina that handle long context natively
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config["max_tokens"],
            ).to(self.device)
            emb = self.model.get_text_features(**inputs)

        return emb / emb.norm(dim=-1, keepdim=True)

    def score(self, image: Image.Image, captions: Union[List[str], str]) -> float:
        """Cosine similarity between an image and its caption(s)."""
        text = " ".join(captions) if isinstance(captions, list) else captions
        return (self._image_emb(image) @ self._text_emb(text).T).item()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate_dataset(
    evaluator: CaptionEvaluator,
    dataset,  # raw HuggingFace dataset
    features: List[str],  # [image_col, caption_col, id_col]
) -> dict:
    img_col, caption_col, id_col = features
    results: dict = {}
    total = len(dataset)
    update_step = max(1, total // 10)

    for item in tqdm(dataset, desc="  scoring", total=total, miniters=update_step):
        raw_img = item[img_col]
        captions = item[caption_col]

        if raw_img is None or captions is None:
            continue

        image = _to_pil_rgb(raw_img)
        idx = str(item[id_col])  # JSON keys must be strings

        results[idx] = {
            "captions": captions,
            "similarity": evaluator.score(image, captions),
        }

    scores = [v["similarity"] for v in results.values()]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  [green]mean similarity: {avg:.4f}  ({len(scores)} scored)[/green]")
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_results(results: dict, dataset_name: str, model_key: str):
    """Write the per-sample results to the canonical JSON path."""
    path = get_output_path(dataset_name, model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  [blue]saved → {path}[/blue]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Semantically")
    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        choices=MODELS_CONFIG.keys(),
        help="Select model for evaluation (clip, jina)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (e.g., chunking reports)"
    )
    args = parser.parse_args()

    evaluator = CaptionEvaluator(model_key=args.model, verbose=args.verbose)

    for dataset_name, config in DATASETS.items():
        label = "Real" if dataset_name in REAL_DATASETS else "Synthetic"
        print(f"\n[bold]--- {dataset_name}  [{label}] ---[/bold]")

        # ── skip already-computed ──
        if results_exist(dataset_name, args.model):
            print("  [dim]already computed — skipping[/dim]")
            continue

        # ── load & preprocess ──
        # Use hf_name when available (virtual entries that share a HF repo)
        hf_name = config.get("hf_name", dataset_name)
        dataset = load_raw_dataset(hf_name, **config.get("load_config", {}))
        print(f"  [dim]{len(dataset)} samples after preprocessing[/dim]")

        # ── score & save ──
        results = evaluate_dataset(evaluator, dataset, config["features"])
        save_results(results, dataset_name, args.model)
