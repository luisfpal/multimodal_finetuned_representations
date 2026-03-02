"""Update MME/MMBench datasets with imageHash (unique image hash)."""

import argparse
import base64
import hashlib
import os
import struct
from io import BytesIO

from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


def get_cache_dir():
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        return os.path.expanduser(os.path.expandvars(cache_dir))
    return None


def get_num_proc(requested: int) -> int:
    if requested and requested > 0:
        return requested
    count = os.cpu_count() or 1
    return max(1, count // 2)


def _open_image(image):
    if image is None:
        raise ValueError("Image is None")
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        raw = image.get("bytes")
        if raw:
            return Image.open(BytesIO(raw))
        path = image.get("path")
        if path:
            return Image.open(path)
        raise ValueError("Unsupported image dict payload: expected bytes or path")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(image))
    if isinstance(image, str):
        try:
            raw = base64.b64decode(image, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 image string") from exc
        return Image.open(BytesIO(raw))
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def hash_image_pixels(image):
    img = _open_image(image)
    try:
        img = img.convert("RGB")
        digest = hashlib.sha256()
        digest.update(struct.pack(">II", img.size[0], img.size[1]))
        digest.update(img.tobytes())
        return digest.hexdigest()
    except Exception as exc:
        raise RuntimeError("Failed to hash image pixels") from exc


def add_image_hash(batch):
    return {"imageHash": [hash_image_pixels(img) for img in batch["image"]]}


def update_dataset(repo_id: str, push_to_hub: bool, num_proc: int, batch_size: int):
    cache_dir = get_cache_dir()
    print(f"Loading dataset: {repo_id}")
    dataset = load_dataset(repo_id, split="train", cache_dir=cache_dir)

    print(f"Samples: {len(dataset)}")
    # Ensure old column name is removed before writing the new one.
    if "imageId" in dataset.column_names:
        print("Removing legacy imageId column...")
        dataset = dataset.remove_columns(["imageId"])
    if "imageHash" in dataset.column_names:
        print("imageHash already exists; it will be recomputed.")

    print("Computing imageHash (sha256 over RGB pixels)...")
    updated = dataset.map(
        add_image_hash,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    image_hashes = updated["imageHash"]
    unique_image_hashes = set(image_hashes)
    print(f"Unique images: {len(unique_image_hashes)}")

    if len(updated) > 0:
        sample = updated[0]
        print(f"Sample: imageHash={sample['imageHash']}")

    if push_to_hub:
        token = os.getenv("HF_TOKEN")
        print(f"Pushing to Hub: {repo_id}")
        if token:
            updated.push_to_hub(repo_id, token=token)
        else:
            updated.push_to_hub(repo_id)
        print("Done.")
    else:
        print("Skipping push to hub (use --push-to-hub to push)")

    return updated


def update_mme(push_to_hub: bool, hub_username: str, num_proc: int, batch_size: int):
    repo_id = f"{hub_username}/MME-test-captioned"
    return update_dataset(repo_id, push_to_hub, num_proc, batch_size)


def update_mmbench(push_to_hub: bool, hub_username: str, num_proc: int, batch_size: int):
    repo_id = f"{hub_username}/MMBench_EN-dev-captioned"
    return update_dataset(repo_id, push_to_hub, num_proc, batch_size)


def main():
    parser = argparse.ArgumentParser(description="Update datasets with imageHash")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mme", "mmbench"],
        help="Dataset to update",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-username",
        default="l11p",
        help="HuggingFace username",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=-1,
        help="Worker processes (default: half CPU cores)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for hashing",
    )
    args = parser.parse_args()

    num_proc = get_num_proc(args.num_proc)

    if args.dataset == "mme":
        update_mme(args.push_to_hub, args.hub_username, num_proc, args.batch_size)
    elif args.dataset == "mmbench":
        update_mmbench(args.push_to_hub, args.hub_username, num_proc, args.batch_size)

    print("\nMISSION ACCOMPLISHED")


if __name__ == "__main__":
    main()
