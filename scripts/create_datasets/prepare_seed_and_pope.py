"""Prepare SEED-Bench and POPE with imageHash/questionHash and optional Hub push."""

import argparse
import base64
import hashlib
import os
import random
import struct
import time
import warnings
from io import BytesIO

from datasets import Image as HFImage, load_dataset
from dotenv import load_dotenv
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image

load_dotenv()
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def get_cache_dir():
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        return os.path.expanduser(os.path.expandvars(cache_dir))
    return None


def get_num_proc(requested: int) -> int:
    if requested and requested > 0:
        return requested
    count = os.cpu_count() or 1
    default_workers = max(1, count // 2)
    default_cap = int(os.getenv("DATASET_HASH_DEFAULT_MAX_PROC", "16"))
    return min(default_workers, default_cap)


def _open_image(image):
    if image is None:
        return None, False
    if isinstance(image, list):
        if not image:
            return None, False
        image = image[0]
    if isinstance(image, Image.Image):
        return image, False
    if isinstance(image, dict):
        raw = image.get("bytes")
        if raw:
            return Image.open(BytesIO(raw)), True
        path = image.get("path")
        if path:
            return Image.open(path), True
        return None, False
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(image)), True
    if isinstance(image, str):
        try:
            raw = base64.b64decode(image)
            return Image.open(BytesIO(raw)), True
        except Exception:
            try:
                return Image.open(image), True
            except Exception:
                return None, False
    return None, False


def hash_image_pixels(image):
    img, should_close = _open_image(image)
    if img is None:
        return None
    rgb = None
    try:
        rgb = img.convert("RGB")
        digest = hashlib.sha256()
        digest.update(struct.pack(">II", rgb.size[0], rgb.size[1]))
        digest.update(rgb.tobytes())
        return digest.hexdigest()
    except Exception:
        return None
    finally:
        if rgb is not None:
            rgb.close()
        if should_close:
            img.close()


def hash_question_text(question):
    if question is None:
        return None
    if not isinstance(question, str):
        question = str(question)
    text = question.strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_image_hash(batch):
    return {"imageHash": [hash_image_pixels(img) for img in batch["image"]]}


def add_question_hash(batch):
    return {"questionHash": [hash_question_text(q) for q in batch["question"]]}


def flatten_seed_image(batch):
    flattened = []
    for item in batch["image"]:
        if isinstance(item, list):
            flattened.append(item[0] if item else None)
        else:
            flattened.append(item)
    return {"image": flattened}


def robust_map(dataset, fn, batch_size, num_proc):
    attempted = []
    candidates = [num_proc, min(16, num_proc), min(8, num_proc), 1]
    candidates = [w for w in candidates if w >= 1]
    candidates = list(dict.fromkeys(candidates))

    for workers in candidates:
        attempted.append(workers)
        try:
            return dataset.map(
                fn,
                batched=True,
                batch_size=batch_size,
                num_proc=workers,
            )
        except RuntimeError as exc:
            if workers == 1:
                raise
            if "subprocesses has abruptly died" not in str(exc).lower():
                raise
            print(
                f"Map failed with num_proc={workers}. Retrying with fewer workers..."
            )
    raise RuntimeError(f"Map failed for all worker settings: {attempted}")


def maybe_push(
    dataset,
    repo_id: str,
    push_to_hub: bool,
    push_retries: int,
    retry_base_wait: int,
    max_shard_size: str,
):
    if not push_to_hub:
        print("Skipping push to hub (use --push-to-hub to push)")
        return
    token = os.getenv("HF_TOKEN")
    print(f"Pushing to Hub: {repo_id} (max_shard_size={max_shard_size})")

    for attempt in range(1, push_retries + 1):
        try:
            if token:
                dataset.push_to_hub(
                    repo_id, token=token, max_shard_size=max_shard_size
                )
            else:
                dataset.push_to_hub(repo_id, max_shard_size=max_shard_size)
            print("Done.")
            return
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            transient = status is not None and (status >= 500 or status == 429)
            if (not transient) or attempt == push_retries:
                raise
            wait_s = retry_base_wait * (2 ** (attempt - 1))
            print(
                f"Push failed with status={status} on attempt "
                f"{attempt}/{push_retries}. Retrying in {wait_s}s..."
            )
            time.sleep(wait_s)


def prepare_seed(
    split: str,
    num_proc: int,
    batch_size: int,
    push_to_hub: bool,
    output_repo: str,
    push_retries: int,
    retry_base_wait: int,
    max_shard_size: str,
):
    cache_dir = get_cache_dir()
    print(f"\nLoading lmms-lab/SEED-Bench [{split}]")
    dataset = load_dataset("lmms-lab/SEED-Bench", split=split, cache_dir=cache_dir)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    image_row_indices = [
        i for i, dtype in enumerate(dataset["data_type"]) if dtype == "image"
    ]
    filtered = dataset.select(image_row_indices)
    print(f"Filtered samples (data_type=image): {len(filtered)}")

    # SEED stores images as single-item lists for image rows.
    # Flatten to plain Image objects to keep downstream captioning stable.
    filtered = robust_map(
        filtered,
        flatten_seed_image,
        batch_size=batch_size,
        num_proc=num_proc,
    )
    filtered = filtered.cast_column("image", HFImage(decode=True))

    image_batch_size = min(batch_size, 32)
    with_image_hash = robust_map(
        filtered,
        add_image_hash,
        batch_size=image_batch_size,
        num_proc=num_proc,
    )
    with_ids = robust_map(
        with_image_hash,
        add_question_hash,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    unique_image_hashes = {x for x in with_ids["imageHash"] if x is not None}
    unique_question_hashes = {x for x in with_ids["questionHash"] if x is not None}
    print(f"Unique imageHash count: {len(unique_image_hashes)}")
    print(f"Unique questionHash count: {len(unique_question_hashes)}")

    maybe_push(
        with_ids,
        output_repo,
        push_to_hub,
        push_retries=push_retries,
        retry_base_wait=retry_base_wait,
        max_shard_size=max_shard_size,
    )
    return with_ids


def select_random_index_per_observed_pair(question_hashes, image_hashes, seed):
    rng = random.Random(seed)
    pair_seen_count = {}
    pair_selected_index = {}

    for idx, (question_hash, image_hash) in enumerate(zip(question_hashes, image_hashes)):
        if question_hash is None or image_hash is None:
            continue

        pair = (question_hash, image_hash)
        seen = pair_seen_count.get(pair, 0) + 1
        pair_seen_count[pair] = seen

        # Reservoir sampling of size 1 per pair.
        if seen == 1 or rng.randrange(seen) == 0:
            pair_selected_index[pair] = idx

    return sorted(pair_selected_index.values())


def prepare_pope(
    split: str,
    num_proc: int,
    batch_size: int,
    random_seed: int,
    push_to_hub: bool,
    output_repo: str,
    push_retries: int,
    retry_base_wait: int,
    max_shard_size: str,
):
    cache_dir = get_cache_dir()
    print(f"\nLoading lmms-lab/POPE [{split}]")
    dataset = load_dataset("lmms-lab/POPE", split=split, cache_dir=cache_dir)
    print(f"Total samples: {len(dataset)}")

    image_batch_size = min(batch_size, 32)
    with_image_hash = robust_map(
        dataset,
        add_image_hash,
        batch_size=image_batch_size,
        num_proc=num_proc,
    )
    with_ids = robust_map(
        with_image_hash,
        add_question_hash,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    unique_question_hashes = {x for x in with_ids["questionHash"] if x is not None}
    unique_question_text = len(set(with_ids["question"]))
    unique_image_hashes = {x for x in with_ids["imageHash"] if x is not None}
    print(f"Unique question text count: {unique_question_text}")
    print(f"Unique questionHash count: {len(unique_question_hashes)}")
    print(f"Unique imageHash count: {len(unique_image_hashes)}")

    max_cartesian_pairs = len(unique_question_hashes) * len(unique_image_hashes)
    sampled_indices = select_random_index_per_observed_pair(
        with_ids["questionHash"], with_ids["imageHash"], seed=random_seed
    )
    observed_pair_dataset = with_ids.select(sampled_indices)
    print(
        "Rows after random sampling over observed (questionHash, imageHash) pairs: "
        f"{len(observed_pair_dataset)}"
    )
    print(
        "Observed pair coverage vs full cartesian: "
        f"{len(observed_pair_dataset)}/{max_cartesian_pairs}"
    )

    maybe_push(
        observed_pair_dataset,
        output_repo,
        push_to_hub,
        push_retries=push_retries,
        retry_base_wait=retry_base_wait,
        max_shard_size=max_shard_size,
    )
    return observed_pair_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SEED/POPE with imageHash and POPE observed-pair sampling."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["seed", "pope", "both"],
        help="Dataset pipeline to run",
    )
    parser.add_argument(
        "--seed-split",
        default="test",
        help="Split for SEED-Bench",
    )
    parser.add_argument(
        "--pope-split",
        default="test",
        help="Split for POPE",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=-1,
        help="Worker processes (default: half CPU cores, capped for stability)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for map operations",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for selecting one row per observed (questionHash, imageHash).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push resulting dataset to Hub",
    )
    parser.add_argument(
        "--push-retries",
        type=int,
        default=5,
        help="Retries for transient Hub push failures (5xx/429).",
    )
    parser.add_argument(
        "--push-retry-base-wait",
        type=int,
        default=30,
        help="Base wait seconds for exponential backoff between push retries.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="300MB",
        help="Shard size passed to dataset.push_to_hub.",
    )
    parser.add_argument(
        "--hub-username",
        default="l11p",
        help="Hub username used when output repos are not explicitly set",
    )
    parser.add_argument(
        "--seed-output-repo",
        default=None,
        help="Hub repo for processed SEED-Bench",
    )
    parser.add_argument(
        "--pope-output-repo",
        default=None,
        help="Hub repo for processed POPE",
    )
    args = parser.parse_args()

    num_proc = get_num_proc(args.num_proc)
    seed_output_repo = (
        args.seed_output_repo
        if args.seed_output_repo
        else f"{args.hub_username}/SEED-Bench-images-captioned"
    )
    pope_output_repo = (
        args.pope_output_repo
        if args.pope_output_repo
        else f"{args.hub_username}/POPE-test-captioned"
    )

    if args.dataset in {"seed", "both"}:
        prepare_seed(
            split=args.seed_split,
            num_proc=num_proc,
            batch_size=args.batch_size,
            push_to_hub=args.push_to_hub,
            output_repo=seed_output_repo,
            push_retries=args.push_retries,
            retry_base_wait=args.push_retry_base_wait,
            max_shard_size=args.max_shard_size,
        )

    if args.dataset in {"pope", "both"}:
        prepare_pope(
            split=args.pope_split,
            num_proc=num_proc,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            push_to_hub=args.push_to_hub,
            output_repo=pope_output_repo,
            push_retries=args.push_retries,
            retry_base_wait=args.push_retry_base_wait,
            max_shard_size=args.max_shard_size,
        )

    print("\nMISSION ACCOMPLISHED")


if __name__ == "__main__":
    main()
