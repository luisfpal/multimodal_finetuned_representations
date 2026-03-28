#!/usr/bin/env python3
"""Download images for ARO order subtasks (COCO val2014 + Flickr30k).

Idempotent: skips images already present.
Resume-capable: uses HF cache for Flickr30k, parallel wget for COCO.

Usage:
    python scripts/data/download_aro_images.py                      # download both
    python scripts/data/download_aro_images.py --dataset coco       # COCO only
    python scripts/data/download_aro_images.py --dataset flickr30k  # Flickr30k only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_ARO_ROOT = (
    Path.home() / "scratch" / "datasets" / "vision-language-models-are-bows"
)

COCO_IMAGE_URL = "http://images.cocodataset.org/val2014"
COCO_ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"

FLICKR_HF_REPO = "nlphuji/flickr30k"
FLICKR_HF_FILE = "flickr30k-images.zip"
FLICKR_MIN_IMAGES = 30000


def _count_jpgs(directory: Path) -> int:
    """Count .jpg files in a directory (non-recursive)."""
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix == ".jpg")


def _download_one_image(args: tuple[str, Path]) -> bool:
    """Download a single image. Returns True on success."""
    url, dest = args
    if dest.exists():
        return True
    try:
        urlretrieve(url, dest)
        return True
    except Exception:
        return False


def download_coco_val2014(aro_root: Path, workers: int = 16) -> None:
    """Download only the COCO val2014 images needed by ARO annotations.

    Downloads individual images in parallel (~5000 images, ~1GB total)
    instead of the full val2014.zip (~6GB, 41K images).
    Falls back to full zip if individual downloads fail.
    """
    target = aro_root / "val2014"

    # Parse annotation to find needed images
    ann_path = aro_root / "coco_karpathy_test.json"
    if not ann_path.exists():
        print(f"Error: annotation not found: {ann_path}")
        sys.exit(1)

    ann = json.loads(ann_path.read_text())
    needed_images = sorted({item["image"] for item in ann})  # e.g. val2014/COCO_...jpg
    print(f"COCO val2014: {len(needed_images)} images needed by annotations")

    # Check which are already downloaded
    target.mkdir(parents=True, exist_ok=True)
    present = {f"val2014/{f.name}" for f in target.iterdir() if f.suffix == ".jpg"}
    to_download = [img for img in needed_images if img not in present]

    if not to_download:
        print(f"✓ COCO val2014: all {len(needed_images)} images already present")
        return

    print(f"  {len(present)} already present, {len(to_download)} to download")

    # Build download tasks: (url, dest_path)
    tasks = [
        (f"{COCO_IMAGE_URL}/{img.split('/')[-1]}", target / img.split("/")[-1])
        for img in to_download
    ]

    # Parallel download
    failed = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one_image, t): t for t in tasks}
        for future in as_completed(futures):
            completed += 1
            if not future.result():
                failed += 1
            if completed % 500 == 0 or completed == len(tasks):
                print(f"  [{completed}/{len(tasks)}] downloaded ({failed} failed)")

    if failed > 0:
        print(f"⚠ {failed} images failed to download")
        print("  Falling back to full val2014.zip download...")
        _download_coco_zip_fallback(aro_root)
        return

    n_final = _count_jpgs(target)
    print(f"✓ COCO val2014: {n_final} images")


def _download_coco_zip_fallback(aro_root: Path) -> None:
    """Fallback: download full val2014.zip."""
    zip_path = aro_root / "val2014.zip"
    target = aro_root / "val2014"

    print(f"Downloading {COCO_ZIP_URL} ...")
    subprocess.run(
        ["wget", "-c", "--progress=bar:force", COCO_ZIP_URL, "-O", str(zip_path)],
        check=True,
    )
    print("Extracting val2014.zip ...")
    subprocess.run(
        ["unzip", "-o", "-q", str(zip_path), "-d", str(aro_root)],
        check=True,
    )
    zip_path.unlink(missing_ok=True)

    n_images = _count_jpgs(target)
    print(f"✓ COCO val2014: {n_images} images (from zip)")


def download_flickr30k(aro_root: Path) -> None:
    """Download Flickr30k images from HuggingFace (~4.3GB zip, ~31K images)."""
    target = aro_root / "flickr30k-images"
    n_existing = _count_jpgs(target)
    if n_existing > FLICKR_MIN_IMAGES:
        print(f"✓ Flickr30k already present ({n_existing} images)")
        return

    print(f"Downloading Flickr30k from HuggingFace ({FLICKR_HF_REPO}) ...")

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        FLICKR_HF_REPO, FLICKR_HF_FILE, repo_type="dataset"
    )

    print("Extracting flickr30k-images.zip ...")
    subprocess.run(
        ["unzip", "-o", "-q", zip_path, "-d", str(aro_root)],
        check=True,
        env={**__import__("os").environ, "UNZIP_DISABLE_ZIPBOMB_DETECTION": "TRUE"},
    )

    # Validate against annotations
    ann_path = aro_root / "flickr30k_test.json"
    if ann_path.exists():
        ann = json.loads(ann_path.read_text())
        needed = {item["image"].replace("flickr30k-images/", "") for item in ann}
        present_set = {f.name for f in target.iterdir() if f.suffix == ".jpg"}
        missing = needed - present_set
        if missing:
            print(f"⚠ {len(missing)} annotation images missing from flickr30k-images/")

    n_images = _count_jpgs(target)
    if n_images < FLICKR_MIN_IMAGES:
        print(f"⚠ Expected >{FLICKR_MIN_IMAGES} Flickr30k images, got {n_images}")
        sys.exit(1)

    print(f"✓ Flickr30k: {n_images} images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download images for ARO order subtasks."
    )
    parser.add_argument(
        "--aro-data-root",
        type=Path,
        default=DEFAULT_ARO_ROOT,
        help="Root directory of the ARO dataset repo",
    )
    parser.add_argument(
        "--dataset",
        choices=["coco", "flickr30k", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel download workers for COCO (default: 16)",
    )
    args = parser.parse_args()

    if not args.aro_data_root.is_dir():
        print(f"Error: ARO data root not found: {args.aro_data_root}")
        sys.exit(1)

    if args.dataset in ("coco", "all"):
        download_coco_val2014(args.aro_data_root, workers=args.workers)

    if args.dataset in ("flickr30k", "all"):
        download_flickr30k(args.aro_data_root)

    print(
        "\nDone. You can now enable coco_order / flickr30k_order "
        "in launch_compositionality.sh"
    )


if __name__ == "__main__":
    main()
