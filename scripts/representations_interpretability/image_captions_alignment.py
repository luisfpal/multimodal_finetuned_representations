import argparse
import base64
import gc
import json
import os
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from PIL import Image
from rich import print
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

HOME_DIR = Path(os.path.expanduser("~"))
PROJECT_ROOT = HOME_DIR / "multimodal_finetuned_representations"
HF_HOME = HOME_DIR / "scratch" / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_DATASETS_CACHE = HF_HOME / "datasets"

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)

MODEL_ID = "jinaai/jina-clip-v2"
MODEL_REVISION = "344d954da76eb8ad47a7aaff42d012e30c15b8fe"
DEFAULT_DATASET = "l11p/VQAv2-COCO-restval-ICQ"
DEFAULT_SPLIT = "train"
DEFAULT_SUBSET_SIZES = [10, 100, 1000]
DEFAULT_SEED = 42
HISTOGRAM_SUBSET_SIZES = {1000, 10000}


def _to_pil_rgb(raw: Image.Image | str) -> Image.Image:
	if isinstance(raw, Image.Image):
		return raw.convert("RGB")
	if isinstance(raw, str):
		return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
	raise TypeError(f"Unsupported image type: {type(raw)}")


def load_alignment_dataset(dataset_name: str, split: str) -> Dataset:
	dataset = load_dataset(dataset_name, split=split, cache_dir=str(HF_DATASETS_CACHE))
	required_columns = {"image", "captions", "synthetic-caption"}
	missing = required_columns.difference(dataset.column_names)
	if missing:
		raise ValueError(
			f"Dataset {dataset_name} is missing required columns: {sorted(missing)}"
		)
	return dataset


def select_subset(dataset: Dataset, size: int, seed: int) -> Dataset:
	if size > len(dataset):
		raise ValueError(
			f"Requested subset size {size} exceeds dataset size {len(dataset)}"
		)
	return dataset.shuffle(seed=seed).select(range(size))


def build_model_and_processor(device: str) -> tuple[AutoModel, AutoProcessor]:
	sys.modules["flash_attn"] = None
	warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

	model_kwargs = {
		"use_safetensors": True,
		"trust_remote_code": True,
		"attn_implementation": "eager",
		"revision": MODEL_REVISION,
	}
	processor_kwargs = {
		"trust_remote_code": True,
		"use_fast": True,
		"revision": MODEL_REVISION,
	}

	processor = AutoProcessor.from_pretrained(MODEL_ID, **processor_kwargs)
	model = AutoModel.from_pretrained(MODEL_ID, **model_kwargs)
	model.to(device).eval()
	return model, processor


@torch.no_grad()
def embed_images(
	model: AutoModel,
	processor: AutoProcessor,
	images: list[Image.Image],
	device: str,
	batch_size: int,
) -> torch.Tensor:
	all_embeddings: list[torch.Tensor] = []
	total_samples = len(images)
	update_every = max(1, int(0.1 * total_samples))
	update_count = 0

	with tqdm(total=total_samples, desc="Embedding images", leave=False) as progress_bar:
		for start in range(0, len(images), batch_size):
			batch = images[start : start + batch_size]
			inputs = processor(images=batch, return_tensors="pt").to(device)
			image_emb = model.get_image_features(**inputs)
			image_emb = F.normalize(image_emb, p=2, dim=-1)
			all_embeddings.append(image_emb.cpu())

			update_count += len(batch)
			if update_count >= update_every or start + batch_size >= total_samples:
				progress_bar.update(update_count)
				update_count = 0

	return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def embed_texts(
	model: AutoModel,
	processor: AutoProcessor,
	texts: list[str],
	device: str,
	batch_size: int,
	max_length: int = 8192,
) -> torch.Tensor:
	all_embeddings: list[torch.Tensor] = []
	total_samples = len(texts)
	update_every = max(1, int(0.1 * total_samples))
	update_count = 0

	with tqdm(total=total_samples, desc="Embedding texts", leave=False) as progress_bar:
		for start in range(0, len(texts), batch_size):
			batch = texts[start : start + batch_size]
			inputs = processor(
				text=batch,
				return_tensors="pt",
				padding=True,
				truncation=True,
				max_length=max_length,
			).to(device)
			text_emb = model.get_text_features(**inputs)
			text_emb = F.normalize(text_emb, p=2, dim=-1)
			all_embeddings.append(text_emb.cpu())

			update_count += len(batch)
			if update_count >= update_every or start + batch_size >= total_samples:
				progress_bar.update(update_count)
				update_count = 0

	return torch.cat(all_embeddings, dim=0)


def build_texts(
	dataset: Dataset,
	source: Literal["captions", "synthetic-caption"],
) -> list[str]:
	if source == "captions":
		return [", ".join(item) if isinstance(item, list) else str(item) for item in dataset["captions"]]
	return [str(item) for item in dataset["synthetic-caption"]]


def compute_similarity_matrix(
	image_embeddings: torch.Tensor,
	text_embeddings: torch.Tensor,
) -> torch.Tensor:
	return image_embeddings @ text_embeddings.T


def compute_rank_outputs(similarity_matrix: torch.Tensor) -> dict[str, float | list[int] | list[float]]:
	if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
		raise ValueError("Rank computation requires square matrix with paired rows/columns")

	diagonal = similarity_matrix.diag().unsqueeze(1)
	greater_counts = (similarity_matrix > diagonal).sum(dim=1)
	equal_counts = (similarity_matrix == diagonal).sum(dim=1) - 1

	ranks_min = (greater_counts + 1).to(torch.int64)
	ranks_avg = ranks_min.to(torch.float32) + (equal_counts.to(torch.float32) / 2.0)
	argmax_indices = similarity_matrix.argmax(dim=1).to(torch.int64)

	mrr = (1.0 / ranks_min.to(torch.float32)).mean().item()
	r_at_1 = (ranks_min == 1).to(torch.float32).mean().item()
	mean_rank = ranks_avg.mean().item()
	median_rank = torch.median(ranks_avg).item()

	return {
		"argmax_index_per_row": argmax_indices.tolist(),
		"diagonal_rank_min_per_row": ranks_min.tolist(),
		"diagonal_rank_avg_per_row": [float(x) for x in ranks_avg.tolist()],
		"mrr": float(mrr),
		"r_at_1": float(r_at_1),
		"mean_rank": float(mean_rank),
		"median_rank": float(median_rank),
	}


def save_histograms(similarity_matrix: torch.Tensor, out_dir: Path, title_prefix: str) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	similarity_matrix = similarity_matrix.to(torch.float32)
	matrix_np = similarity_matrix.numpy()

	diagonal_values = similarity_matrix.diag().numpy()
	upper_indices = torch.triu_indices(
		similarity_matrix.shape[0], similarity_matrix.shape[1], offset=1
	)
	off_diagonal_upper_values = similarity_matrix[
		upper_indices[0], upper_indices[1]
	].numpy()
	all_values = matrix_np.ravel()

	plt.figure(figsize=(18, 5))
	plt.subplot(1, 3, 1)
	plt.hist(diagonal_values, bins=50)
	plt.title(f"{title_prefix}\nDiagonal values")
	plt.xlabel("Cosine similarity")
	plt.ylabel("Count")

	plt.subplot(1, 3, 2)
	plt.hist(off_diagonal_upper_values, bins=50)
	plt.title(f"{title_prefix}\nOff-diagonal upper values")
	plt.xlabel("Cosine similarity")
	plt.ylabel("Count")

	plt.subplot(1, 3, 3)
	plt.hist(all_values, bins=50)
	plt.title(f"{title_prefix}\nAll values")
	plt.xlabel("Cosine similarity")
	plt.ylabel("Count")

	plt.tight_layout()
	plt.savefig(out_dir / "similarity_histograms.png", dpi=200)
	plt.close()


def save_results(
	out_dir: Path,
	dataset_name: str,
	source: str,
	subset_size: int,
	rank_outputs: dict[str, float | list[int] | list[float]],
	similarity_matrix: torch.Tensor,
) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)

	result_payload = {
		"dataset": dataset_name,
		"model": MODEL_ID,
		"caption_source": source,
		"subset_size": subset_size,
		"matrix_shape": list(similarity_matrix.shape),
		"metrics": {
			"mrr": rank_outputs["mrr"],
			"r_at_1": rank_outputs["r_at_1"],
			"mean_rank": rank_outputs["mean_rank"],
			"median_rank": rank_outputs["median_rank"],
		},
		"per_row": {
			"argmax_index": rank_outputs["argmax_index_per_row"],
			"diagonal_rank_min": rank_outputs["diagonal_rank_min_per_row"],
			"diagonal_rank_avg": rank_outputs["diagonal_rank_avg_per_row"],
		},
	}

	with open(out_dir / "rank_metrics.json", "w", encoding="utf-8") as fp:
		json.dump(result_payload, fp, indent=2)

	torch.save(similarity_matrix, out_dir / "similarity_matrix.pt")


def run_single_configuration(
	subset: Dataset,
	image_embeddings: torch.Tensor,
	model: AutoModel,
	processor: AutoProcessor,
	device: str,
	dataset_name: str,
	source: Literal["captions", "synthetic-caption"],
	subset_size: int,
	batch_size_texts: int,
	output_root: Path,
) -> None:
	texts = build_texts(subset, source)

	print(f"[cyan]Computing embeddings for source={source}, N={subset_size}[/cyan]")
	text_embeddings = embed_texts(model, processor, texts, device, batch_size_texts)

	similarity_matrix = compute_similarity_matrix(image_embeddings, text_embeddings)
	rank_outputs = compute_rank_outputs(similarity_matrix)

	run_dir = (
		output_root
		/ dataset_name.replace("/", "_")
		/ "jina"
		/ source.replace("-", "_")
		/ f"subset_{subset_size}"
	)
	save_results(run_dir, dataset_name, source, subset_size, rank_outputs, similarity_matrix)
	if subset_size in HISTOGRAM_SUBSET_SIZES:
		save_histograms(
			similarity_matrix,
			run_dir,
			title_prefix=f"{source} | N={subset_size}",
		)

	print(
		"[green]saved[/green] "
		f"source={source}, N={subset_size}, "
		f"MRR={rank_outputs['mrr']:.4f}, R@1={rank_outputs['r_at_1']:.4f}"
	)

	del text_embeddings
	del similarity_matrix
	gc.collect()
	if device == "cuda":
		torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Image-text alignment matrix and rank analysis with jinaai/jina-clip-v2"
	)
	parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
	parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
	parser.add_argument(
		"--subset-sizes",
		type=int,
		nargs="+",
		default=DEFAULT_SUBSET_SIZES,
		help="Subset sizes to evaluate",
	)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument("--batch-size-images", type=int, default=32)
	parser.add_argument("--batch-size-texts", type=int, default=32)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=PROJECT_ROOT / "results" / "caption_evaluation" / "alignment",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = "cuda" if torch.cuda.is_available() else "cpu"

	print(f"[blue]HF_HOME:            {HF_HOME}[/blue]")
	print(f"[blue]HF_HUB_CACHE:       {HF_HUB_CACHE}[/blue]")
	print(f"[blue]HF_DATASETS_CACHE: {HF_DATASETS_CACHE}[/blue]")
	print(f"[blue]Device:             {device}[/blue]")

	dataset = load_alignment_dataset(args.dataset, args.split)
	model, processor = build_model_and_processor(device=device)

	for subset_size in args.subset_sizes:
		subset = select_subset(dataset, subset_size, seed=args.seed)
		images = [_to_pil_rgb(img) for img in subset["image"]]

		print(f"[cyan]Computing image embeddings once for N={subset_size}[/cyan]")
		image_embeddings = embed_images(
			model,
			processor,
			images,
			device,
			args.batch_size_images,
		)

		for source in ["captions", "synthetic-caption"]:
			run_single_configuration(
				subset=subset,
				image_embeddings=image_embeddings,
				model=model,
				processor=processor,
				device=device,
				dataset_name=args.dataset,
				source=source,
				subset_size=subset_size,
				batch_size_texts=args.batch_size_texts,
				output_root=args.output_root,
			)

		del image_embeddings
		del images
		del subset
		gc.collect()
		if device == "cuda":
			torch.cuda.empty_cache()


if __name__ == "__main__":
	main()
