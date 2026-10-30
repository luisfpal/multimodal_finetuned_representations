"""Provides CLIP image embedding extraction for samples in an image dataloader."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from PIL import Image

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_BATCH_SIZE = 64


def extract_clip_embeddings(dl_img: DataLoader, device: torch.device) -> torch.Tensor:
    """Extract CLIP image embeddings for every sample in the image dataloader.

    Accesses raw PIL images via ``dl_img.dataset.dataset`` — the underlying
    HuggingFace dataset stored inside the custom Dataset wrapper
    (``OpenVQADataset``, ``MultipleChoiceDatasetBenchmark``, etc.).

    This attribute is already subsampled / split-filtered with the same seed
    and ordering used during residual-stream extraction, guaranteeing 1-to-1
    sample alignment with ``res_img``.

    Note: CLIP inference is fast (seconds on A100 for 10k images).  Embeddings
    are computed at run time and do not need to be cached.

    Args:
        dl_img: Image dataloader produced by ``get_dataloader(images_qa=True)``.
            The underlying dataset must expose a ``dataset`` attribute that is
            either a HuggingFace ``Dataset`` with an ``"image"`` column of PIL
            Images or captioning records with ``"image_path"``.
        device: Target device for CLIP model inference.

    Returns:
        Float32 CPU tensor of shape ``(num_samples, clip_dim)``.
    """
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    # dl_img.dataset is the custom Dataset wrapper (e.g. OpenVQADataset).
    # .dataset on that wrapper is the underlying HF dataset, already subsampled
    # with the same seed/split as the representation extraction run.
    hf_dataset = dl_img.dataset.dataset
    num_samples = len(hf_dataset)

    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, num_samples, CLIP_BATCH_SIZE):
            end = min(start + CLIP_BATCH_SIZE, num_samples)
            images = []
            for idx in range(start, end):
                sample = hf_dataset[idx]
                image = sample.get("image")
                if image is None and sample.get("image_path"):
                    with Image.open(sample["image_path"]) as img:
                        image = img.convert("RGB")
                if image is not None:
                    images.append(image)
            if not images:
                continue
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            feats = clip_model.get_image_features(**inputs).float()
            embeddings.append(feats.cpu())

    del clip_model
    torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)  # (num_samples, clip_dim)
