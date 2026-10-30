"""Module for scoring image-caption semantic similarity with a CLIP-like model and computing keep indices for filtering."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_ID = "jinaai/jina-clip-v2"
MODEL_REVISION = "344d954da76eb8ad47a7aaff42d012e30c15b8fe"


class CaptionSemanticScorer:
    """Image-caption cosine scorer using a CLIP-like model."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        import sys

        # Block flash_attn AND all its sub-modules so Jina's remote code
        # falls back to pure-PyTorch rotary embeddings and attention.
        # When a VLM (e.g. llava) is loaded first with flash_attention_2,
        # flash_attn.ops.triton.rotary is already in sys.modules. Jina's
        # code binds to its triton kernel, which is incompatible with the
        # installed PyTorch. Blocking all entries forces the fallback path.
        _fa_keys = [
            k for k in sys.modules if k == "flash_attn" or k.startswith("flash_attn.")
        ]
        _fa_backup = {k: sys.modules[k] for k in _fa_keys}
        for k in _fa_keys:
            sys.modules[k] = None

        warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=True,
            revision=MODEL_REVISION,
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_safetensors=True,
            revision=MODEL_REVISION,
            attn_implementation="eager",
        )
        self.model.to(self.device).eval()

        # Restore flash_attn for the main VLM that may still need it on GPU.
        # Jina's already-imported modules keep their fallback references.
        sys.modules.update(_fa_backup)

    @torch.no_grad()
    def _image_emb(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _text_emb(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(self.device)
        emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    def score(self, image: Image.Image, captions: str | list[str]) -> float:
        text = " ".join(captions) if isinstance(captions, list) else captions
        return (self._image_emb(image) @ self._text_emb(text).T).item()


def _to_pil_rgb(raw: Any) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(raw)}")


def _extract_caption_text(sample: dict[str, Any]) -> str:
    """Extract caption text with a FIXED priority: captions > caption > synthetic-caption.

    The priority is intentionally caption_source-independent so that semantic
    filter indices are deterministic per dataset, not per downstream task config.
    """
    if sample.get("captions") is not None:
        captions = sample["captions"]
        if isinstance(captions, list):
            return " ".join(str(c) for c in captions)
        return str(captions)

    if sample.get("caption") is not None:
        return str(sample["caption"])

    if sample.get("synthetic-caption") is not None:
        return str(sample["synthetic-caption"])

    raise ValueError(
        "No caption fields found (expected captions/caption/synthetic-caption)"
    )


def _try_load_cache(path: str) -> tuple[set[int], dict[str, float | int]] | None:
    """Attempt to load a single cache file. Returns None on any failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        indices = set(payload["indices"])
        stats = payload["stats"]
        return indices, stats
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"Invalid semantic filter cache at {path}: {e}")
        return None


def load_semantic_filter_cache(
    cache_path: str,
) -> tuple[set[int], dict[str, float | int]] | None:
    """Load cached semantic filter indices from *cache_path*.

    Returns (indices, stats) if the cache is valid, else None.
    """
    return _try_load_cache(cache_path)


def save_semantic_filter_cache(
    cache_path: str,
    indices: set[int],
    stats: dict[str, float | int],
    *,
    dataset_name: str,
    threshold_std_multiplier: float,
    num_dataset_rows: int,
) -> None:
    """Save semantic filter indices, stats, and provenance metadata to JSON."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "indices": sorted(indices),
        "stats": {k: v for k, v in stats.items()},
        "metadata": {
            "dataset": dataset_name,
            "threshold_std_multiplier": threshold_std_multiplier,
            "scorer_model": MODEL_ID,
            "scorer_revision": MODEL_REVISION,
            "num_dataset_rows": num_dataset_rows,
        },
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)


def compute_semantic_keep_indices(
    dataset: Any,
    scorer: CaptionSemanticScorer,
    threshold_std_multiplier: float = 1.0,
) -> tuple[set[int], dict[str, float | int]]:
    """Score every row in *dataset* and return the row indices to keep.

    Threshold: keep rows with score >= mean - threshold_std_multiplier * std.
    Caption resolution is fixed (captions > caption > synthetic-caption) so
    indices are deterministic per dataset, independent of downstream caption_source.
    """
    scores: list[tuple[int, float]] = []  # (row_index, score)

    total = len(dataset)
    update_every = max(1, int(0.15 * total))
    update_count = 0
    samples_processed = 0

    progress_bar = tqdm(
        total=total,
        desc="Semantic filtering",
        unit="sample",
    )

    for row_idx, sample in enumerate(dataset):
        image_raw = sample.get("image")
        if image_raw is None:
            samples_processed += 1
            update_count += 1
            if update_count % update_every == 0 or samples_processed == total:
                progress_bar.update(update_count)
                update_count = 0
            continue

        caption_text = _extract_caption_text(sample)
        if not caption_text.strip():
            samples_processed += 1
            update_count += 1
            if update_count % update_every == 0 or samples_processed == total:
                progress_bar.update(update_count)
                update_count = 0
            continue

        image = _to_pil_rgb(image_raw)
        score = scorer.score(image=image, captions=caption_text)
        scores.append((row_idx, float(score)))

        samples_processed += 1
        update_count += 1
        if update_count % update_every == 0 or samples_processed == total:
            progress_bar.update(update_count)
            update_count = 0

    progress_bar.close()

    if not scores:
        raise ValueError("No valid samples available for semantic filtering")

    all_scores = np.asarray([s for _, s in scores], dtype=np.float32)
    mean_score = float(all_scores.mean())
    std_score = float(all_scores.std())
    threshold = mean_score - threshold_std_multiplier * std_score

    keep_indices = {idx for idx, s in scores if s >= threshold}

    stats: dict[str, float | int] = {
        "num_candidates": len(scores),
        "num_kept": len(keep_indices),
        "mean_score": mean_score,
        "std_score": std_score,
        "threshold": threshold,
    }
    return keep_indices, stats
