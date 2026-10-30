"""Compute Linear CKA between an encoder (Jina CLIP or OpenAI CLIP) and VLM decoder residual stream.

For each decoder layer, computes:
  - cka_txt:    CKA(encoder_text_emb, decoder_text_residual[layer])
  - cka_img:    CKA(encoder_img_emb, decoder_img_residual[layer])
  - cka_concat: CKA(cat(enc_txt, enc_img), cat(dec_txt[layer], dec_img[layer]))

Encoder text = captions (synthetic or human) embedded by the chosen encoder.
Encoder image = raw images embedded by the chosen encoder.
Decoder = residual stream from VLM (LLaVA, InternVL, Cambrian, etc.).

Caption rules:
  - jina + synthetic: uses "synthetic-caption" column as single string
  - jina + human:     joins all captions in "captions" list with ", "
  - clip + human:     picks ONE caption at random from "captions" list (77-token limit)
  - clip + synthetic: NOT SUPPORTED (raises ValueError)
"""

import argparse
import base64
import gc
import json
import os
import random
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from rich import print
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt

from src.residual_stream_tracer import residual_stream_tracer
from utils import get_dataloader, load_hf_model_and_processor_or_tokenizer, seed_all
from utils.cambrian_utils import is_cambrian_model
from utils.dataset import _resolve_split_path
from utils.metrics_utils import compute_linear_cka_similarity, compute_linear_regression_r2_score

IMAGES_PATH = str(Path(__file__).resolve().parents[2] / "tinkering" / "plots")

def plot_images(images: list, ncols: int = 10, save_path: str = IMAGES_PATH,
                filename: str = "") -> None:
    """Plot and save image grid. No display."""
    ncols = min(ncols, len(images))
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()
    
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    
    for ax in axes[len(images):]:
        ax.axis("off")
    
    plt.tight_layout()
    
    if not filename:
        filename = f"images_grid_{len(images)}.png"
    
    output_path = Path(save_path) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

load_dotenv()

# ── Device & model config ─────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "flash_attention_2" if DEVICE.type == "cuda" else "sdpa"
RESIDUAL_STREAM_TYPE = "output_layer"

# ── Encoder model registry ───────────────────────────────────────────────
ENCODER_MODELS_CONFIG: dict[str, dict] = {
    "jina": {
        "id": "jinaai/jina-clip-v2",
        "revision": "344d954da76eb8ad47a7aaff42d012e30c15b8fe",
        "max_tokens": 8192,
        "trust_remote_code": True,
    },
    "clip": {
        "id": "openai/clip-vit-large-patch14-336",
        "max_tokens": 77,
        "trust_remote_code": False,
    },
}

# ── Pooling method registry (same as multimodality_similarity.py) ─────────
VIRTUAL_POOLING_DISPATCH: dict[str, tuple[str, str]] = {
    "mean_context": ("mean_text_context", "mean_image"),
    "mean_image_last_caption": ("last_caption", "mean_image"),
    "last_image_last_caption": ("last_caption", "last_image"),
}
_PASS_THROUGH_POOLING_METHODS: list[str] = ["last", "mean", "none"]
VALID_POOLING_METHODS: list[str] = _PASS_THROUGH_POOLING_METHODS + list(
    VIRTUAL_POOLING_DISPATCH.keys()
)

# ── Caption source mapping ────────────────────────────────────────────────
CAPTION_SOURCE_MAP: dict[str, str] = {
    "synthetic": "synthetic-caption",
    "human": "captions",
}


# ═════════════════════════════════════════════════════════════════════════
# Encoder helpers (from image_captions_alignment.py — standalone, not importable)
# ═════════════════════════════════════════════════════════════════════════


def _to_pil_rgb(raw: Image.Image | str) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, str):
        return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(raw)}")


def build_encoder_model(encoder_key: str, device: str) -> tuple[AutoModel, AutoProcessor]:
    """Load encoder model. Supports 'jina' (Jina CLIP v2) and 'clip' (OpenAI CLIP)."""
    config = ENCODER_MODELS_CONFIG[encoder_key]
    if encoder_key == "jina":
        sys.modules["flash_attn"] = None
        warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
        processor = AutoProcessor.from_pretrained(
            config["id"],
            trust_remote_code=True,
            use_fast=True,
            revision=config["revision"],
        )
        model = AutoModel.from_pretrained(
            config["id"],
            use_safetensors=True,
            trust_remote_code=True,
            attn_implementation="eager",
            revision=config["revision"],
        )
    else:  # clip
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(config["id"])
        model = CLIPModel.from_pretrained(
            config["id"],
            use_safetensors=True,
            torch_dtype=MODEL_DTYPE,
        )
    model.to(device).eval()
    return model, processor


def build_texts(
    dataset,
    source: Literal["captions", "synthetic-caption"],
    encoder_key: str,
) -> list[str]:
    """Extract caption strings from dataset.

    jina + synthetic-caption: single generated caption per sample
    jina + captions:          all human captions joined with ", "
    clip + captions:          one human caption chosen at random (77-token limit)
    clip + synthetic-caption: not supported — caller must reject before reaching here
    """
    if source == "synthetic-caption":
        return [str(item) for item in dataset["synthetic-caption"]]
    # source == "captions"
    if encoder_key == "clip":
        return [
            random.choice(item) if isinstance(item, list) and item else str(item)
            for item in dataset["captions"]
        ]
    # jina: join all captions
    return [
        ", ".join(item) if isinstance(item, list) else str(item)
        for item in dataset["captions"]
    ]


@torch.no_grad()
def embed_images(
    model: AutoModel,
    processor: AutoProcessor,
    images: list[Image.Image],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    total_samples = len(images)
    update_every = max(1, int(0.25 * total_samples))
    samples_processed = 0
    update_count = 0
    all_embeddings: list[torch.Tensor] = []
    progress_bar = tqdm(total=total_samples, desc="Embedding images", unit="sample")
    for start in range(0, total_samples, batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        image_emb = model.get_image_features(**inputs)
        # image_emb = F.normalize(image_emb, p=2, dim=-1)
        all_embeddings.append(image_emb.cpu())
        samples_processed += len(batch)
        update_count += len(batch)
        if update_count >= update_every or samples_processed == total_samples:
            progress_bar.update(update_count)
            update_count = 0
    progress_bar.close()
    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def embed_texts(
    model: AutoModel,
    processor: AutoProcessor,
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    total_samples = len(texts)
    update_every = max(1, int(0.25 * total_samples))
    samples_processed = 0
    update_count = 0
    all_embeddings: list[torch.Tensor] = []
    progress_bar = tqdm(total=total_samples, desc="Embedding texts", unit="sample")
    for start in range(0, total_samples, batch_size):
        batch = texts[start : start + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        text_emb = model.get_text_features(**inputs)
        # text_emb = F.normalize(text_emb, p=2, dim=-1)
        all_embeddings.append(text_emb.cpu())
        samples_processed += len(batch)
        update_count += len(batch)
        if update_count >= update_every or samples_processed == total_samples:
            progress_bar.update(update_count)
            update_count = 0
    progress_bar.close()
    return torch.cat(all_embeddings, dim=0)


# ═════════════════════════════════════════════════════════════════════════
# Decoder helpers
# ═════════════════════════════════════════════════════════════════════════


def extract_decoder_representation(
    dl,
    model,
    processor,
    pooling_method: str,
) -> torch.Tensor:
    """Extract residual stream from decoder. Returns (num_layers, N, d_dec)."""
    return residual_stream_tracer(
        model=model,
        processed_dataloader=dl,
        residual_stream_type=RESIDUAL_STREAM_TYPE,
        tokens_pooling_method=pooling_method,
        return_deepcopy=True,
        save_representations=False,
        processor=processor,
    )


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute encoder-decoder similarity (CKA or OLS R²) between an encoder "
            "(Jina CLIP or OpenAI CLIP) and VLM decoder residual stream representations."
        )
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="jina",
        choices=list(ENCODER_MODELS_CONFIG.keys()),
        help="Encoder model: 'jina' (Jina CLIP v2) or 'clip' (openai/clip-vit-large-patch14-336)",
    )
    parser.add_argument(
        "--dataset-path-or-name",
        type=str,
        required=True,
        help="HuggingFace dataset identifier (e.g. l11p/COCOQA-restval-ICQA)",
    )
    parser.add_argument(
        "--decoder-model",
        type=str,
        required=True,
        help="Decoder VLM identifier (e.g. lbasile/llava_7b)",
    )
    parser.add_argument(
        "--pooling-method",
        type=str,
        default="last",
        choices=VALID_POOLING_METHODS,
        help="Token pooling method for decoder residual stream",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="split01",
        help="Precomputed split name (e.g. split01) from data/splits/",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize all representations before CKA computation",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--caption-source",
        type=str,
        default="synthetic",
        choices=["human", "synthetic"],
        help="Caption column to use for encoder text embeddings",
    )
    parser.add_argument(
        "--guide-text",
        type=str,
        default="Answer the question using a single word or phrase.",
        help="Guide text for decoder VQA prompts",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Root results directory (default: PROJECT_ROOT/results)",
    )
    parser.add_argument(
        "--similarity-measure",
        type=str,
        default="cka",
        choices=["cka", "ols"],
        help="Similarity measure: 'cka' for Linear CKA, 'ols' for OLS linear regression R² (train+test)",
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    # ── Validate encoder/caption_source combo ─────────────────────────────
    if args.encoder_model == "clip" and args.caption_source == "synthetic":
        raise ValueError(
            "CLIP encoder does not support synthetic captions. Use --caption-source human."
        )

    encoder_config = ENCODER_MODELS_CONFIG[args.encoder_model]

    # ── Environment setup ─────────────────────────────────────────────────
    home = Path(os.path.expanduser("~"))
    project_root = home / "multimodal_finetuned_representations"
    hf_home = home / "scratch" / "huggingface"
    hf_hub_cache = str(hf_home / "hub")
    hf_datasets_cache = str(hf_home / "datasets")

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", hf_datasets_cache)

    model_cache_dir = os.getenv("HF_HUB_CACHE", hf_hub_cache)
    datasets_cache_dir = os.getenv("HF_DATASETS_CACHE", hf_datasets_cache)
    results_dir = args.results_dir or str(project_root / "results")
    device = str(DEVICE)

    seed_all(args.seed)

    # ── Load split indices ────────────────────────────────────────────────
    split_path = _resolve_split_path(args.dataset_path_or_name, args.dataset_split)
    with open(split_path) as f:
        all_indices = json.load(f)["indices"]
    indices = sorted(all_indices[: args.max_samples])
    indices_set = set(indices)
    num_samples = len(indices)

    print(f"\n{'=' * 70}")
    print("Encoder-Decoder CKA Analysis")
    print(f"{'=' * 70}")
    print(f"Dataset:        {args.dataset_path_or_name}")
    print(f"Encoder model:  {args.encoder_model} ({encoder_config['id']})")
    print(f"Decoder model:  {args.decoder_model}")
    print(f"Pooling:        {args.pooling_method}")
    print(f"Split:          {args.dataset_split} ({num_samples} samples)")
    print(f"Caption source: {args.caption_source}")
    print(f"Normalize:      {args.normalize}")
    print(f"Device:         {device}")
    print(f"{'=' * 70}")

    # ═════════════════════════════════════════════════════════════════════
    # ENCODER PHASE
    # ═════════════════════════════════════════════════════════════════════
    print("\n--- ENCODER PHASE ---")

    # Load raw HF dataset and select split samples
    print(f"Loading dataset: {args.dataset_path_or_name}")
    raw_dataset = load_dataset(
        args.dataset_path_or_name, split="train", cache_dir=datasets_cache_dir
    )
    raw_dataset = raw_dataset.select(indices)
    print(f"Selected {len(raw_dataset)} samples from split '{args.dataset_split}'")

    # Extract texts (captions) and images
    caption_column = CAPTION_SOURCE_MAP[args.caption_source]
    texts = build_texts(raw_dataset, caption_column, args.encoder_model)
    images = [_to_pil_rgb(img) for img in raw_dataset["image"]]
    print(f"Extracted {len(texts)} texts and {len(images)} images")

    # Load encoder and embed
    print(f"\nLoading encoder: {encoder_config['id']}")
    enc_model, enc_processor = build_encoder_model(encoder_key=args.encoder_model, device=device)

    enc_txt = embed_texts(
        enc_model, enc_processor, texts, device, args.batch_size,
        max_length=encoder_config["max_tokens"],
    )
    enc_img = embed_images(enc_model, enc_processor, images, device, args.batch_size)
    print(f"Encoder text embeddings:  {enc_txt.shape}")
    print(f"Encoder image embeddings: {enc_img.shape}")

    # Cleanup encoder
    del enc_model, enc_processor, texts, images, raw_dataset
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Restore flash_attn for decoder loading (only needed after Jina, which sets it to None)
    if args.encoder_model == "jina" and "flash_attn" in sys.modules:
        del sys.modules["flash_attn"]

    # ═════════════════════════════════════════════════════════════════════
    # DECODER PHASE
    # ═════════════════════════════════════════════════════════════════════
    print("\n--- DECODER PHASE ---")

    print(f"\nLoading decoder: {args.decoder_model}")
    dec_model, dec_processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.decoder_model,
        cache_dir=model_cache_dir,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE,
    )

    # Build dataloaders with same sample indices
    dl_args: dict = {
        "dataset_path_or_name": args.dataset_path_or_name,
        "seed": args.seed,
        "processor": dec_processor,
        "batch_size": args.batch_size,
        "cache_dir": datasets_cache_dir,
        "guide_text": args.guide_text,
        "split": "train",
        "caption_source": args.caption_source,
        "allowed_indices": indices_set,
    }

    if "intern" in dec_model.__class__.__name__.lower() or is_cambrian_model(
        model=dec_model
    ):
        dl_args["model"] = dec_model
    if is_cambrian_model(model=dec_model):
        dl_args["num_workers"] = 0

    print("Creating decoder dataloaders...")
    dl_txt = get_dataloader(texts_qa=True, **dl_args)
    dl_img = get_dataloader(images_qa=True, **dl_args)
    print(f"  text dataloader: {len(dl_txt.dataset)} samples")
    print(f"  image dataloader: {len(dl_img.dataset)} samples")
    
    # texts = [", ".join(instance['captions']) for instance in dl_txt.dataset.dataset]
    # ids_txt = [instance['image_id'] for instance in dl_txt.dataset.dataset]
    # ids_img = [instance['image_id'] for instance in dl_img.dataset.dataset]
    # ids_comparison = [ids_txt[i] == ids_img[i] for i in range(len(ids_txt))]
    # for idx, (txt, comparison) in enumerate(zip(texts, ids_comparison)):
    #     print(f"{idx}(img_dataset[{idx}]==txt_dataset[{idx}]={comparison}): {txt}")
    # images = [instance['image'] for instance in dl_img.dataset.dataset]
    # images = [_to_pil_rgb(img) for img in images]
    # plot_images(images, filename="decoder_images")

    # Dispatch pooling method
    pooling_txt, pooling_img = VIRTUAL_POOLING_DISPATCH.get(
        args.pooling_method, (args.pooling_method, args.pooling_method)
    )
    print(f"Pooling dispatch: txt={pooling_txt}, img={pooling_img}")

    # Extract decoder residual stream
    print("\nExtracting decoder text residual stream...")
    dec_txt = extract_decoder_representation(
        dl_txt, dec_model, dec_processor, pooling_txt
    )
    print(f"  dec_txt shape: {dec_txt.shape}")

    print("Extracting decoder image residual stream...")
    dec_img = extract_decoder_representation(
        dl_img, dec_model, dec_processor, pooling_img
    )
    print(f"  dec_img shape: {dec_img.shape}")

    # Truncate to match encoder if needed
    if dec_txt.shape[1] > num_samples:
        dec_txt = dec_txt[:, :num_samples, :]
        dec_img = dec_img[:, :num_samples, :]
        print(f"  Truncated decoder to {num_samples} samples")

    # Verify alignment
    assert dec_txt.shape[1] == enc_txt.shape[0], (
        f"Sample count mismatch: decoder={dec_txt.shape[1]}, encoder={enc_txt.shape[0]}"
    )

    num_layers = dec_txt.shape[0]

    # Cleanup decoder model
    del dec_model, dec_processor, dl_txt, dl_img
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════════════
    # SIMILARITY PHASE
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n--- SIMILARITY PHASE ({args.similarity_measure.upper()}) ---")

    # Optional normalization
    if args.normalize:
        print("L2-normalizing all representations...")
        enc_txt = F.normalize(enc_txt.float(), p=2, dim=-1)
        enc_img = F.normalize(enc_img.float(), p=2, dim=-1)
        dec_txt = F.normalize(dec_txt.float(), p=2, dim=-1)
        dec_img = F.normalize(dec_img.float(), p=2, dim=-1)

    # Pre-compute encoder concatenation (reused across layers)
    enc_concat = torch.cat([enc_txt, enc_img], dim=0)  # (2N, d_enc)

    if args.similarity_measure == "cka":
        cka_txt_vals = torch.zeros(num_layers)
        cka_img_vals = torch.zeros(num_layers)
        cka_concat_vals = torch.zeros(num_layers)

        for layer in tqdm(range(num_layers), desc="Computing CKA per layer"):
            cka_txt_vals[layer] = compute_linear_cka_similarity(enc_txt, dec_txt[layer])
            cka_img_vals[layer] = compute_linear_cka_similarity(enc_img, dec_img[layer])
            dec_concat = torch.cat([dec_txt[layer], dec_img[layer]], dim=0)
            cka_concat_vals[layer] = compute_linear_cka_similarity(enc_concat, dec_concat)

        print(f"\nCKA Text:   min={cka_txt_vals.min():.4f}, max={cka_txt_vals.max():.4f}")
        print(f"CKA Image:  min={cka_img_vals.min():.4f}, max={cka_img_vals.max():.4f}")
        print(f"CKA Concat: min={cka_concat_vals.min():.4f}, max={cka_concat_vals.max():.4f}")
        result_tensors = {
            "cka_txt": cka_txt_vals,
            "cka_img": cka_img_vals,
            "cka_concat": cka_concat_vals,
        }

    else:  # ols
        r2_test_txt = torch.zeros(num_layers)
        r2_train_txt = torch.zeros(num_layers)
        r2_test_img = torch.zeros(num_layers)
        r2_train_img = torch.zeros(num_layers)
        r2_test_concat = torch.zeros(num_layers)
        r2_train_concat = torch.zeros(num_layers)

        for layer in tqdm(range(num_layers), desc="Computing OLS R² per layer"):
            tr, te = compute_linear_regression_r2_score(
                enc_txt, dec_txt[layer], regression_type="ols", return_train_score=True
            )
            r2_train_txt[layer], r2_test_txt[layer] = tr, te

            tr, te = compute_linear_regression_r2_score(
                enc_img, dec_img[layer], regression_type="ols", return_train_score=True
            )
            r2_train_img[layer], r2_test_img[layer] = tr, te

            dec_concat = torch.cat([dec_txt[layer], dec_img[layer]], dim=0)
            tr, te = compute_linear_regression_r2_score(
                enc_concat, dec_concat, regression_type="ols", return_train_score=True
            )
            r2_train_concat[layer], r2_test_concat[layer] = tr, te

        print(f"\nR² Test  Text:   min={r2_test_txt.min():.4f}, max={r2_test_txt.max():.4f}")
        print(f"R² Train Text:   min={r2_train_txt.min():.4f}, max={r2_train_txt.max():.4f}")
        print(f"R² Test  Image:  min={r2_test_img.min():.4f}, max={r2_test_img.max():.4f}")
        print(f"R² Train Image:  min={r2_train_img.min():.4f}, max={r2_train_img.max():.4f}")
        result_tensors = {
            "r2_test_txt": r2_test_txt,
            "r2_train_txt": r2_train_txt,
            "r2_test_img": r2_test_img,
            "r2_train_img": r2_train_img,
            "r2_test_concat": r2_test_concat,
            "r2_train_concat": r2_train_concat,
        }

    # ═════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═════════════════════════════════════════════════════════════════════
    dataset_slug = args.dataset_path_or_name.split("/")[-1].replace("-", "_").lower()
    dataset_slug = f"{dataset_slug}_{args.dataset_split}"
    decoder_name = args.decoder_model.split("/")[-1]
    norm_tag = "normalized" if args.normalize else "unnormalized"

    out_dir = Path(results_dir) / "similarity_encoder_decoder" / dataset_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{decoder_name}_{args.encoder_model}_{args.pooling_method}"
        f"_{args.caption_source}_{norm_tag}_{args.similarity_measure}_N{num_samples}.safetensors"
    )
    out_path = out_dir / filename

    metadata = {
        "encoder_model": encoder_config["id"],
        "encoder_key": args.encoder_model,
        "decoder_model": args.decoder_model,
        "dataset": args.dataset_path_or_name,
        "dataset_split": args.dataset_split,
        "pooling_method": args.pooling_method,
        "pooling_txt": pooling_txt,
        "pooling_img": pooling_img,
        "normalized": str(args.normalize),
        "num_samples": str(num_samples),
        "num_layers": str(num_layers),
        "caption_source": args.caption_source,
        "seed": str(args.seed),
        "similarity_measure": args.similarity_measure,
    }

    save_file(result_tensors, str(out_path), metadata=metadata)
    print(f"\nResults saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
