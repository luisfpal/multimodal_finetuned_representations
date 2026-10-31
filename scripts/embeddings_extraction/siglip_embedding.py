import argparse
import json
import os
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

from utils.embeddings_extraction.imagenet_loader import ImageFolder


def load_and_validate_detail23k(json_path: str, image_dir: str) -> List[Dict]:
    """Load detail_23k.json, verify every referenced image exists, return list of
    {image_path, image_id} preserving the file order — same order the qwen/llava
    pipeline in embeddings_coco.py uses."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [
        os.path.join(image_dir, f"COCO_train2014_{entry['image']}")
        for entry in data
        if not os.path.isfile(os.path.join(image_dir, f"COCO_train2014_{entry['image']}"))
    ]
    if missing:
        sample = "\n  ".join(missing[:10])
        tail = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} image(s) missing from {image_dir}:\n  {sample}{tail}"
        )

    return [
        {
            "image_path": os.path.join(image_dir, f"COCO_train2014_{entry['image']}"),
            "image_id": entry["image"],
        }
        for entry in data
    ]


class SiglipImageDataset(torch.utils.data.Dataset):
    """Image-only dataset: processes each image on the fly inside DataLoader workers."""

    def __init__(self, records: List[Dict], processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image = Image.open(self.records[idx]["image_path"]).convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"][0]}


def collate_images(instances: Sequence[Dict]) -> Dict:
    return {
        "pixel_values": torch.stack(
            [inst["pixel_values"] for inst in instances], dim=0
        )
    }


class SiglipTransform:
    """Picklable wrapper so the processor can be used as an ImageFolder transform
    across DataLoader workers."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


class SiglipImagenetDataset(torch.utils.data.Dataset):
    """ImageNet-style dataset backed by utils.imagenet_loader.ImageFolder. Either
    samples num_classes folders (reproducibly via seed) or picks the classes at
    the given positions in the alphabetically-sorted directory list."""

    def __init__(
        self,
        root,
        processor,
        num_classes=None,
        num_images_per_class=None,
        seed=0,
        class_indices=None,
    ):
        kwargs = dict(
            root=root,
            nimg_cat=num_images_per_class,
            transform=SiglipTransform(processor),
        )
        if class_indices is not None:
            kwargs["class_indices"] = class_indices
        else:
            kwargs["nclasses"] = num_classes
            kwargs["seed"] = seed
        self.folder = ImageFolder(**kwargs)

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        pixel_values, target = self.folder[idx]
        return {"pixel_values": pixel_values, "target": target}


def _resolve_class_subset(subset_name: str) -> List[int]:
    """Look up a named list of class indices in animal_lists.py (e.g. animals_100)."""
    import animal_lists

    if not hasattr(animal_lists, subset_name):
        available = sorted(
            n for n in dir(animal_lists)
            if not n.startswith("_") and isinstance(getattr(animal_lists, n), list)
        )
        raise ValueError(
            f"Unknown class subset '{subset_name}'. Available: {available}"
        )
    indices = getattr(animal_lists, subset_name)
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise TypeError(f"animal_lists.{subset_name} must be a list of ints")
    return indices


@torch.inference_mode()
def encode_images(model, dataloader, device, dtype) -> torch.Tensor:
    model.eval()
    out = []
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(
            device=device, dtype=dtype, non_blocking=True
        )
        feats = model.get_image_features(pixel_values=pixel_values)
        out.append(feats.float().cpu())
        if step % 500 == 0:
            print(f"  batch {step}/{len(dataloader)}")
    return torch.cat(out, dim=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/siglip2-so400m-patch14-384",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["coco", "imagenet"],
        default="coco",
    )
    parser.add_argument(
        "--detail_json",
        type=str,
        default=None,
        help="Path to detail_23k.json (required when --dataset=coco).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="For --dataset=coco: directory with COCO_train2014_* images. "
             "For --dataset=imagenet: root folder with one subdir per class (n01440764/, ...).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of ImageNet classes to sample (required with --dataset=imagenet "
             "unless --class_subset is set).",
    )
    parser.add_argument(
        "--class_subset",
        type=str,
        default=None,
        help="Name of a list in animal_lists.py giving indices into the sorted "
             "list of ImageNet classes (e.g. animals_100). Overrides "
             "--num_classes / --seed.",
    )
    parser.add_argument(
        "--num_images_per_class",
        type=int,
        default=None,
        help="Images per class to load. If None, loads every image in the class.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/embedding_similarity",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="",
        help="Optional suffix appended to the output .pt filename.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=None,
        help="If set, only encode the first N images (coco mode only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = (
        AutoModel.from_pretrained(args.model_name_or_path, dtype=dtype)
        .to(device)
        .eval()
    )

    model = torch.compile(model)
    model_name = "siglip2"

    if args.dataset == "coco":
        if not args.detail_json:
            raise ValueError("--detail_json is required when --dataset=coco")
        print(f"Validating images in {args.image_dir} ...")
        records = load_and_validate_detail23k(args.detail_json, args.image_dir)
        if args.max_num_samples is not None:
            records = records[: args.max_num_samples]
        print(f"All {len(records)} images found.")
        dataset = SiglipImageDataset(records, processor)
        base_name = f"{model_name}_llava_coco"
    else:
        if args.class_subset is None and args.num_classes is None:
            raise ValueError(
                "--num_classes or --class_subset is required when --dataset=imagenet"
            )
        if args.class_subset is not None:
            class_indices = _resolve_class_subset(args.class_subset)
            print(
                f"Loading ImageNet from {args.image_dir} "
                f"(class_subset={args.class_subset} with {len(class_indices)} classes, "
                f"num_images_per_class={args.num_images_per_class}) ..."
            )
            dataset = SiglipImagenetDataset(
                root=args.image_dir,
                processor=processor,
                class_indices=class_indices,
                num_images_per_class=args.num_images_per_class,
            )
        else:
            print(
                f"Loading ImageNet from {args.image_dir} "
                f"(num_classes={args.num_classes}, num_images_per_class={args.num_images_per_class}, seed={args.seed}) ..."
            )
            dataset = SiglipImagenetDataset(
                root=args.image_dir,
                processor=processor,
                num_classes=args.num_classes,
                num_images_per_class=args.num_images_per_class,
                seed=args.seed,
            )
        print(f"Loaded {len(dataset)} images across {len(dataset.folder.classes)} classes.")
        base_name = f"{model_name}_imagenet"

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    embeddings = encode_images(model, dataloader, device=device, dtype=dtype)
    assert embeddings.shape[0] == len(dataset), (embeddings.shape, len(dataset))
    print(f"embeddings shape : {embeddings.shape}")

    if args.dataset == "coco":
        payload = {
            "embeddings": embeddings,
            "image_ids": [r["image_id"] for r in records],
            "image_paths": [r["image_path"] for r in records],
        }
    else:
        targets = torch.tensor(
            [t for _, t in dataset.folder.samples], dtype=torch.long
        )
        paths = [p for p, _ in dataset.folder.samples]
        idx_to_class = {v: k for k, v in dataset.folder.class_to_idx.items()}
        imagenet_codes = [idx_to_class[int(t)] for t in targets.tolist()]
        payload = {
            "embeddings": embeddings,
            "targets": targets,
            "imagenet_codes": imagenet_codes,
            "image_paths": paths,
            "classes": dataset.folder.classes,
            "class_to_idx": dataset.folder.class_to_idx,
        }

    filename = base_name
    if args.filename:
        filename += f"_{args.filename}"
    if args.dataset == "coco" and args.max_num_samples is not None:
        filename += f"_{args.max_num_samples}"
    if args.dataset == "imagenet":
        if args.class_subset is not None:
            filename += f"_{args.class_subset}"
        else:
            filename += f"_nc{args.num_classes}_seed{args.seed}"
        if args.num_images_per_class is not None:
            filename += f"_npc{args.num_images_per_class}"
    out_path = f"{args.out_dir}/{filename}.pt"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
