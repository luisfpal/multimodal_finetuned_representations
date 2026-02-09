import os
import json
import random
import argparse
import threading
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Union

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from PIL import Image
from src.captioner import VlmCaptioner
from datasets import load_dataset

# Default configuration
DEFAULT_SEED = 42
# Project root is 3 levels up from this script (scripts/captioning/caption_dataset.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "captions"
DEFAULT_NUM_WORKERS = 16
DEFAULT_CACHE_DIR = os.path.expanduser("~/scratch/cache/huggingface/datasets")

CAPTIONING_PROMPT = (
    "URGENT: Generate a precise 3–5 sentence caption for this image.\n"
    "Rules:\n"
    "- Start immediately with the visual description\n"
    "- Describe ONLY what is visibly present\n"
    "- Focus on main objects, actions, and scene elements\n"
    "- NO reasoning, explanations, or meta-commentary\n"
    "- Output ONLY the final caption\n"
    "This is critical - follow these rules exactly."
)

console = Console()
file_lock = Lock()
thread_local_storage = threading.local()


# ============================================================================
# DATASET LOADERS
# ============================================================================


def load_mme_images(
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[Dict[str, Image.Image], str]:
    """
    Example loader for MME dataset from HuggingFace.

    Returns:
        (images_dict, dataset_name) where images_dict = {image_id: PIL.Image}
    """
    dataset = load_dataset("lmms-lab/MME", split="test", cache_dir=cache_dir)

    # Extract unique images (first occurrence of each ID)
    qid_to_idx = {}
    for idx, qid in enumerate(dataset["question_id"]):
        qid_to_idx.setdefault(qid, idx)

    images = {qid: dataset[idx]["image"] for qid, idx in qid_to_idx.items()}
    dataset_name = "lmms-lab_MME_test"

    return images, dataset_name


def load_gqa_images(
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[Dict[str, Image.Image], str]:
    """
    Loader for GQA validation balanced images dataset from HuggingFace.

    Dataset has 10,234 images with 'id' and 'image' fields.

    Returns:
        (images_dict, dataset_name) where images_dict = {image_id: PIL.Image}
    """
    dataset = load_dataset(
        "lmms-lab/GQA", "val_balanced_images", split="val", cache_dir=cache_dir
    )

    # All images are already unique (no duplicate IDs in this dataset)
    images = {sample["id"]: sample["image"] for sample in dataset}
    dataset_name = "lmms-lab_GQA_val_balanced"

    console.print(f"[green]Loaded {len(images)} unique images from GQA[/green]")

    return images, dataset_name


def load_mmbench_en_images(
    cache_dir: str = DEFAULT_CACHE_DIR,
    split: str = "dev",
) -> Tuple[Dict[str, Image.Image], str]:
    """
    Loader for MMBench_EN dataset from HuggingFace.

    Dataset has 4,377 images in dev split and 6,718 in test split.
    Uses 'index' field as unique identifier.

    Args:
        cache_dir: Cache directory for datasets
        split: Dataset split ('dev' or 'test')

    Returns:
        (images_dict, dataset_name) where images_dict = {image_id: PIL.Image}
    """
    dataset = load_dataset("lmms-lab/MMBench_EN", split=split, cache_dir=cache_dir)

    # All indices are unique - use index as image ID
    images = {str(sample["index"]): sample["image"] for sample in dataset}
    dataset_name = f"lmms-lab_MMBench_EN_{split}"

    console.print(
        f"[green]Loaded {len(images)} unique images from MMBench_EN ({split})[/green]"
    )

    return images, dataset_name


def load_vqav2_coco_restval_icq_images(
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[Dict[str, Image.Image], str]:
    """
    Loader for VQAv2-COCO-restval-ICQ dataset.
    
    Dataset has 'image_id' as identifier.
    Only 'train' split available.
    """
    dataset = load_dataset("l11p/VQAv2-COCO-restval-ICQ", split="train", cache_dir=cache_dir)
    
    # Use image_id as key.
    images = {str(sample["image_id"]): sample["image"] for sample in dataset}
    dataset_name = "l11p_VQAv2-COCO-restval-ICQ"
    
    console.print(f"[green]Loaded {len(images)} unique images from VQAv2-COCO-restval-ICQ[/green]")
    return images, dataset_name


def load_cocoqa_restval_icqa_images(
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[Dict[str, Image.Image], str]:
    """
    Loader for COCOQA-restval-ICQA dataset.
    
    Dataset has 'image_id' as identifier.
    Only 'train' split available.
    """
    dataset = load_dataset("l11p/COCOQA-restval-ICQA", split="train", cache_dir=cache_dir)
    
    # Use image_id as key.
    images = {str(sample["image_id"]): sample["image"] for sample in dataset}
    dataset_name = "l11p_COCOQA-restval-ICQA"
    
    console.print(f"[green]Loaded {len(images)} unique images from COCOQA-restval-ICQA[/green]")
    return images, dataset_name


def load_custom_images(
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[Dict[str, Image.Image], str]:
    """
    Template for custom dataset loader.

    Copy this function and modify for your dataset.
    Load images from anywhere: files, DB, other datasets, etc.

    Returns:
        (images_dict, dataset_name) where images_dict = {image_id: PIL.Image}
    """
    images = {}

    # YOUR CUSTOM LOGIC HERE
    # Example: Load from files
    # image_dir = Path("/path/to/images")
    # for img_path in image_dir.glob("*.jpg"):
    #     image_id = img_path.stem
    #     images[image_id] = Image.open(img_path)

    dataset_name = "my_custom_dataset"
    return images, dataset_name


# ============================================================================
# GENERIC CAPTIONING PIPELINE - Don't modify unless needed
# ============================================================================


def _load_existing_captions(output_path: Path) -> Dict:
    """Load existing captions from file."""
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def _save_captions(output_path: Path, captions: Dict):
    """Thread-safe caption saving."""
    with file_lock:
        with open(output_path, "w") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)


def _get_thread_captioner():
    """Get or create a thread-local VlmCaptioner instance."""
    if not hasattr(thread_local_storage, "captioner"):
        thread_local_storage.captioner = VlmCaptioner()
    return thread_local_storage.captioner


def _caption_single_image(args):
    """Caption a single image. Returns (image_id, caption, status, error_msg)."""
    image_id, image, output_path, captions, failed_log, seed = args

    error_msg = None
    try:
        # Optimization: Reuse captioner per thread instead of creating new one
        captioner = _get_thread_captioner()

        # Try with temperature=0.0 first (most deterministic)
        caption = captioner.caption_image(
            image=image,
            prompt=CAPTIONING_PROMPT,
            max_tokens=500,
            temperature=0.0,
            max_retries=5,
            seed=seed,
        )

        # If empty, try with slightly higher temperature
        if not caption:
            caption = captioner.caption_image(
                image=image,
                prompt=CAPTIONING_PROMPT,
                max_tokens=500,
                temperature=0.3,
                max_retries=5,
                seed=seed,
            )

        # Determine status
        if caption:
            status = "success"
        else:
            status = "failed"
            error_msg = "Empty caption after all retries (incomplete thinking or max retries reached)"
            caption = "FAILED"
    except Exception as e:
        status = "failed"
        error_msg = f"Exception: {str(e)}"
        caption = "FAILED"

    # Save immediately
    captions[image_id] = caption
    _save_captions(output_path, captions)

    # Log failures
    if status == "failed":
        with file_lock:
            failed_log.append({"image_id": image_id, "reason": error_msg})

    return image_id, caption, status, error_msg


def caption_pipeline(
    images: Union[Dict[str, Image.Image], List[Tuple[str, Image.Image]]],
    dataset_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    retry_failed: bool = False,
):
    """
    Generic captioning pipeline - works with any image source.

    Args:
        images: Either dict {image_id: PIL.Image} or list [(image_id, PIL.Image)]
        dataset_name: Name for output files (e.g., "lmms-lab_MME_test")
        output_dir: Directory to save caption files
        num_workers: Number of parallel workers
        seed: Random seed for reproducibility
        retry_failed: If True, only retry images that previously failed
    """
    # Set seeds
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Convert list to dict if needed
    if isinstance(images, list):
        images = dict(images)

    # Setup output paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_captions.json"
    failed_log_path = output_dir / f"{dataset_name}_failed.json"

    # Load existing captions for resume
    captions = _load_existing_captions(output_path)

    # Load existing failed log
    failed_log = []
    if failed_log_path.exists():
        try:
            with open(failed_log_path, "r") as f:
                failed_log = json.load(f)
        except:
            failed_log = []

    # Handle retry-failed mode
    if retry_failed:
        failed_ids = [
            img_id for img_id, caption in captions.items() if caption == "FAILED"
        ]
        if failed_ids:
            console.print(
                f"[yellow]Retry mode: Found {len(failed_ids)} failed captions[/yellow]"
            )
            # Remove failed entries so they get retried
            for img_id in failed_ids:
                del captions[img_id]
            # Clear failed log - will be repopulated if still failing
            failed_log = []
        else:
            console.print("[green]No failed captions to retry![/green]")
            return

    # Filter out already processed
    all_ids = list(images.keys())
    remaining_ids = [img_id for img_id in all_ids if img_id not in captions]

    console.print(f"[cyan]Dataset: {dataset_name}[/cyan]")
    console.print(f"[green]Total unique images: {len(all_ids)}[/green]")
    console.print(f"[green]Already processed: {len(captions)}[/green]")
    console.print(f"[yellow]Remaining to process: {len(remaining_ids)}[/yellow]")

    if not remaining_ids:
        console.print("[green]All images already captioned![/green]")
        return

    # Prepare tasks
    tasks = [
        (img_id, images[img_id], output_path, captions, failed_log, seed)
        for img_id in remaining_ids
    ]

    # Process in parallel
    stats = {"success": 0, "failed": 0}

    console.print(f"[cyan]Processing with {num_workers} parallel workers...[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Captioning images...", total=len(remaining_ids))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_caption_single_image, t): t[0] for t in tasks}

            for future in as_completed(futures):
                img_id, caption, status, error_msg = future.result()
                stats[status] += 1

                # Update progress
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Captioning... (✓{stats['success']} ✗{stats['failed']})",
                )

    # Save failed log
    if failed_log:
        with open(failed_log_path, "w") as f:
            json.dump(failed_log, f, indent=2, ensure_ascii=False)

    # Final summary
    console.print("\n[bold green]Captioning complete![/bold green]")
    console.print(f"  • Total processed: {len(remaining_ids)}")
    console.print(
        f"  • Successful: {stats['success']} ({stats['success'] / len(remaining_ids) * 100:.1f}%)"
    )
    console.print(
        f"  • Failed: {stats['failed']} ({stats['failed'] / len(remaining_ids) * 100:.1f}%)"
    )
    console.print(f"  • Total in file: {len(captions)}")
    console.print(f"  • Output: {output_path}")
    if failed_log:
        console.print(f"  • Failed log: {failed_log_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

DATASET_LOADERS = {
    "mme": load_mme_images,
    "gqa": load_gqa_images,
    "mmbench_en_dev": lambda cache_dir: load_mmbench_en_images(cache_dir, split="dev"),
    "mmbench_en_test": lambda cache_dir: load_mmbench_en_images(cache_dir, split="test"),
    "vqav2_coco": load_vqav2_coco_restval_icq_images,
    "cocoqa": load_cocoqa_restval_icqa_images,
    "custom": load_custom_images,
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Caption images from a dataset")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="gqa", 
        choices=list(DATASET_LOADERS.keys()),
        help="Dataset to caption (default: gqa)"
    )
    return parser.parse_args()


def main():
    """
    Main execution - swap out the loader for different datasets.
    """
    args = parse_args()

    # STEP 1: Load images using your chosen loader
    console.print(f"[bold]Loading dataset:[/bold] {args.dataset}")
    loader_func = DATASET_LOADERS[args.dataset]
    images, dataset_name = loader_func(cache_dir=args.cache_dir)

    # STEP 2: Run the generic pipeline
    caption_pipeline(
        images=images,
        dataset_name=dataset_name,
        output_dir=Path(args.output_dir),
        num_workers=args.num_workers,
        seed=args.seed,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()