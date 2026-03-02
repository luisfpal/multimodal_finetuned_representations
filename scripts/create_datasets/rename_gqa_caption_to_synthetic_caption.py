"""Rename GQA caption column from caption to synthetic-caption and push to Hub."""

import argparse
import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def get_cache_dir():
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        return os.path.expanduser(os.path.expandvars(cache_dir))
    return None


def update_gqa_dataset(repo_id: str, split: str, push_to_hub: bool):
    print(f"Loading dataset: {repo_id} (split={split})")
    dataset = load_dataset(repo_id, split=split, cache_dir=get_cache_dir())

    print(f"Samples: {len(dataset)}")
    columns = dataset.column_names
    print(f"Columns before: {columns}")

    if "caption" in columns and "synthetic-caption" in columns:
        print("Both caption and synthetic-caption found. Replacing synthetic-caption with caption.")
        dataset = dataset.remove_columns(["synthetic-caption"])
        dataset = dataset.rename_column("caption", "synthetic-caption")
    elif "caption" in columns:
        print("Renaming caption -> synthetic-caption")
        dataset = dataset.rename_column("caption", "synthetic-caption")
    elif "synthetic-caption" in columns:
        print("caption is already renamed. No schema change needed.")
    else:
        raise ValueError("Neither caption nor synthetic-caption exists in dataset columns.")

    if "caption" in dataset.column_names:
        raise RuntimeError("caption column still exists after update.")
    if "synthetic-caption" not in dataset.column_names:
        raise RuntimeError("synthetic-caption column is missing after update.")

    print(f"Columns after: {dataset.column_names}")

    if push_to_hub:
        token = os.getenv("HF_TOKEN")
        print(f"Pushing to Hub: {repo_id}")
        if token:
            dataset.push_to_hub(repo_id, token=token)
        else:
            dataset.push_to_hub(repo_id)
        print("Done.")
    else:
        print("Skipping push to hub (use --push-to-hub to push)")


def main():
    parser = argparse.ArgumentParser(
        description="Rename GQA caption column from caption to synthetic-caption"
    )
    parser.add_argument(
        "--repo-id",
        default="l11p/GQA-val-balanced-noyesno-captioned",
        help="Dataset repo id to update",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to update",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push updated dataset to HuggingFace Hub",
    )
    args = parser.parse_args()

    update_gqa_dataset(args.repo_id, args.split, args.push_to_hub)
    print("\nMISSION ACCOMPLISHED")


if __name__ == "__main__":
    main()
