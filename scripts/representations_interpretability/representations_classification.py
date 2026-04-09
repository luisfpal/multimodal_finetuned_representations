import argparse
import math
import time
import warnings
from collections import Counter
from functools import partial, wraps
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from safetensors.torch import save_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import load_hf_model_and_processor_or_tokenizer, seed_all
from utils.dataset import _build_llava_prompts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "flash_attention_2" if DEVICE.type == "cuda" else "sdpa"
POOLING_METHOD = "last"
HOME_DIR = Path.home()
RESULTS_BASE = HOME_DIR / "multimodal_finetuned_representations" / "results"
MIN_TOPIC_SAMPLES = 70

SUPPORTED_METRICS = ("logistic_regression", "knn", "label_overlap")
METRIC_TO_KEY = {
    "logistic_regression": "f1_weighted",
    "knn": "knn_f1_weighted",
    "label_overlap": "label_overlap",
}


def format_time_smart(seconds: int) -> str:
    if seconds < 0:
        return "0s"

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def timed_metric(metric_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            layer_results = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            print(
                f"Metric {metric_name} completed in "
                f"{format_time_smart(int(elapsed_time))}."
            )
            return layer_results

        return wrapper

    return decorator


class ClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "derek-thomas/ScienceQA",
        split: str = "test",
        max_samples: int | None = None,
    ) -> None:
        dataset = load_dataset(dataset_name, split=split)

        topics_counts = Counter(dataset["topic"])

        def filter_min_counts_and_image(example):
            return (
                topics_counts[example["topic"]] >= MIN_TOPIC_SAMPLES
                and example["image"] is not None
            )

        filtered_dataset = dataset.filter(filter_min_counts_and_image)
        topics_counts_filtered = Counter(filtered_dataset["topic"])

        def filter_min_counts(example):
            return topics_counts_filtered[example["topic"]] >= MIN_TOPIC_SAMPLES

        self.dataset = filtered_dataset.filter(filter_min_counts)

        if max_samples is not None:
            self.dataset = self.dataset.select(
                range(min(max_samples, len(self.dataset)))
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, str | Image.Image]:
        instance = self.dataset[idx]
        context_text = instance.get("hint", None)
        context_text = "" if context_text == "nan" else context_text
        choices_list = instance["choices"]
        image = instance["image"]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        prompt = ""
        if context_text:
            prompt += f"Context: {context_text} "
        if choices_list is not None:
            last_choice = choices_list[-1]
            for letter_ord, choice_text in zip(
                range(ord("A"), ord("Z") + 1), choices_list
            ):
                prompt += f"{chr(letter_ord)}. {choice_text}"
                prompt += "\n" if choice_text != last_choice else ""

        return {
            "question": prompt,
            "image": image,
            "label": instance["topic"],
        }


def get_dataloader(
    dataset_name: str,
    split: str,
    guide_text: str,
    processor,
    max_samples: int | None = None,
) -> DataLoader:
    def collator_fn(batch, processor, guide_text):
        questions = [item["question"] for item in batch]
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]
        processor_kwargs = {
            "return_tensors": "pt",
            "text": _build_llava_prompts(
                questions=questions,
                images=images,
                processor=processor,
                guide_text=guide_text,
            ),
            "images": images,
        }
        tokenized = processor(**processor_kwargs)
        return tokenized, labels

    collate_fn = partial(collator_fn, processor=processor, guide_text=guide_text)
    dataset = ClassificationDataset(dataset_name, split, max_samples=max_samples)
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


def _resolve_decoder_config(model):
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "config"):
        return language_model.config

    decoder_model = getattr(model, "model", None)
    if decoder_model is not None and hasattr(decoder_model, "config"):
        return decoder_model.config

    if hasattr(model, "config"):
        return model.config

    raise ValueError(
        "Could not resolve decoder config for residual stream tracing. "
        f"Model class: {model.__class__.__name__}"
    )


def get_representations_and_labels(
    model_name: str,
    dataset_name: str,
    split: str,
    guide_text: str,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=model_name,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE,
    )

    decoder_config = _resolve_decoder_config(model)
    num_layers = decoder_config.num_hidden_layers
    hidden_size = decoder_config.hidden_size

    dataloader = get_dataloader(
        dataset_name, split, guide_text, processor, max_samples=max_samples
    )
    num_samples = len(dataloader.dataset)

    hidden_states = torch.zeros(
        num_layers, num_samples, hidden_size, device="cpu", dtype=torch.float32
    )
    labels = [None] * num_samples

    update_every = max(1, int(0.1 * num_samples))
    updates_pending = 0
    with tqdm(total=num_samples, desc="Extracting representations", unit="sample") as pbar:
        for idx, (inputs, label) in enumerate(dataloader):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)

            stacked = torch.stack(
                [h[0, -1, :].detach().float() for h in outputs.hidden_states[1:]]
            )
            hidden_states[:, idx, :] = stacked.cpu()
            labels[idx] = label[0]
            updates_pending += 1
            if (idx + 1) % update_every == 0 or (idx + 1) == num_samples:
                pbar.update(updates_pending)
                updates_pending = 0

    return hidden_states, np.array(labels)


def preprocess_data_label_overlap(
    data: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(data, torch.Tensor):
        np_data = data.detach().cpu().to(torch.float32).numpy()
    else:
        np_data = np.asarray(data, dtype=np.float32)

    if isinstance(labels, torch.Tensor):
        np_labels = labels.detach().cpu().numpy()
    else:
        np_labels = np.asarray(labels)

    _, unique_indices = np.unique(np_data, axis=0, return_index=True)
    unique_data = np_data[unique_indices]
    filtered_labels = np_labels[unique_indices]
    return unique_data, filtered_labels


def compute_label_overlap(
    data: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    maxk: int = 30,
    use_dadapy: bool = True,
    metric: str = "euclidean",
    class_fraction: float | None = 0.2,
    weighted: bool = True,
) -> float:
    if isinstance(data, torch.Tensor):
        data_t = data.detach().to(torch.float32)
    else:
        data_t = torch.from_numpy(np.asarray(data, dtype=np.float32))

    if isinstance(labels, torch.Tensor):
        labels_t = labels.detach().long()
    else:
        labels_t = torch.from_numpy(np.asarray(labels)).long()

    data_t = data_t.to(DEVICE)
    labels_t = labels_t.to(DEVICE)

    n_samples = data_t.shape[0]
    effective_k = min(maxk, n_samples - 1)
    if effective_k <= 0:
        return 0.0

    unique_labels, class_counts = np.unique(labels_t.detach().cpu().numpy(), return_counts=True)
    class_count_map = {
        int(label): int(count) for label, count in zip(unique_labels, class_counts)
    }
    class_weight_map = {
        label: 1.0 / count for label, count in class_count_map.items()
    }

    if class_fraction is not None:
        k_per_class: dict[int, int] = {}
        for label, count in class_count_map.items():
            class_k = int(count * class_fraction)
            if class_k == 0:
                class_k = 1
                warnings.warn(
                    (
                        f"max_k < 1 for label {label}. max_k set to 1. "
                        "Consider increasing class_fraction."
                    ),
                    stacklevel=2,
                )
            k_per_class[label] = class_k

        effective_k = min(max(k_per_class.values()), n_samples - 1)
        k_per_sample_np = np.array(
            [min(k_per_class[int(label)], effective_k) for label in labels_t.detach().cpu().numpy()]
        )
    else:
        k_per_sample_np = np.full((n_samples,), effective_k, dtype=np.int64)

    if use_dadapy:
        from dadapy import Data

        with torch.no_grad():
            if metric == "cosine":
                normed = F.normalize(data_t, p=2, dim=1)
                sim = normed @ normed.T
                sim_no_self = sim.clone()
                sim_no_self.fill_diagonal_(-float("inf"))
                nn_sim, nn_idx = torch.topk(sim_no_self, effective_k, dim=1, largest=True)
                nn_dist = torch.clamp(1.0 - nn_sim, min=0.0, max=2.0)
            else:
                dist = torch.cdist(data_t, data_t)
                dist_no_self = dist.clone()
                dist_no_self.fill_diagonal_(float("inf"))
                nn_dist, nn_idx = torch.topk(
                    dist_no_self,
                    effective_k,
                    dim=1,
                    largest=False,
                )

        self_idx = torch.arange(n_samples, device=DEVICE).unsqueeze(1)
        self_dist = torch.zeros((n_samples, 1), dtype=nn_dist.dtype, device=DEVICE)

        dist_for_dadapy = torch.cat([self_dist, nn_dist], dim=1).cpu().numpy()
        idx_for_dadapy = torch.cat([self_idx, nn_idx], dim=1).cpu().numpy()

        clean_labels = labels_t.detach().cpu().numpy().astype(int)
        d = Data(distances=(dist_for_dadapy, idx_for_dadapy), maxk=effective_k)
        overlap = d.return_label_overlap(
            clean_labels,
            k=effective_k,
            class_fraction=class_fraction,
            weighted=weighted,
        )
        return float(overlap)

    with torch.no_grad():
        if metric == "cosine":
            normed = F.normalize(data_t, p=2, dim=1)
            sim = normed @ normed.T
            sim.fill_diagonal_(-float("inf"))
            neighbors = torch.topk(sim, effective_k, dim=1, largest=True).indices
        else:
            dist = torch.cdist(data_t, data_t)
            dist.fill_diagonal_(float("inf"))
            neighbors = torch.topk(
                dist,
                effective_k,
                dim=1,
                largest=False,
            ).indices

        target_labels = labels_t.view(n_samples, 1)
        neighbor_labels = labels_t[neighbors]
        overlaps = (neighbor_labels == target_labels)

        if class_fraction is not None:
            nearest_neighbor_rank = torch.arange(effective_k, device=DEVICE).unsqueeze(0)
            k_per_sample_t = torch.from_numpy(k_per_sample_np).to(DEVICE).unsqueeze(1)
            mask = nearest_neighbor_rank >= k_per_sample_t
            overlaps = overlaps & (~mask)

        overlaps_per_sample = overlaps.sum(dim=1).to(torch.float32) / torch.from_numpy(
            k_per_sample_np
        ).to(DEVICE)

        if weighted:
            sample_weights_np = np.array(
                [class_weight_map[int(label)] for label in labels_t.detach().cpu().numpy()],
                dtype=np.float32,
            )
            sample_weights_t = torch.from_numpy(sample_weights_np).to(DEVICE)
            weighted_overlap = (overlaps_per_sample * sample_weights_t).sum() / sample_weights_t.sum()
            return float(weighted_overlap.item())

        return float(overlaps_per_sample.mean().item())


@timed_metric("knn")
def classify_per_layer_knn(
    representations: torch.Tensor,
    y: np.ndarray,
    n_neighbors: int = 30,
) -> dict[int, float]:
    num_layers, num_samples, _ = representations.shape
    use_stratify = bool(np.all(np.bincount(y) >= 2))
    layer_results: dict[int, float] = {}

    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric knn", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            X_train_gpu = representations[layer_idx, train_idx].to(
                device=DEVICE,
                dtype=torch.float32,
            )
            X_test_gpu = representations[layer_idx, test_idx].to(
                device=DEVICE,
                dtype=torch.float32,
            )

            X_train_norm = F.normalize(X_train_gpu, p=2, dim=1)
            X_test_norm = F.normalize(X_test_gpu, p=2, dim=1)

            train_sim = torch.mm(X_train_norm, X_train_norm.t())
            test_sim = torch.mm(X_test_norm, X_train_norm.t())

            D_train = torch.clamp(1.0 - train_sim, min=0.0, max=2.0).cpu().numpy()
            D_test = torch.clamp(1.0 - test_sim, min=0.0, max=2.0).cpu().numpy()
            np.fill_diagonal(D_train, 0.0)
            D_train = np.ascontiguousarray(D_train, dtype=np.float32)
            D_test = np.ascontiguousarray(D_test, dtype=np.float32)

            n_neighbors_effective = min(n_neighbors, len(train_idx))
            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors_effective,
                metric="precomputed",
                weights="distance",
                algorithm="brute",
            )
            knn.fit(D_train, y[train_idx])
            y_pred = knn.predict(D_test)

            layer_results[layer_idx] = float(
                f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)
            )

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results


@timed_metric("label_overlap")
def classify_per_layer_label_overlap(
    representations: torch.Tensor,
    y: np.ndarray,
    maxk: int = 30,
) -> dict[int, float]:
    num_layers = representations.shape[0]
    layer_results: dict[int, float] = {}

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric label_overlap", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            layer_data = representations[layer_idx]
            overlap = compute_label_overlap(
                data=layer_data,
                labels=y,
                maxk=maxk,
                use_dadapy=True,
                metric="cosine",
            )
            layer_results[layer_idx] = float(overlap)

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results


def _torch_logistic_regression_l1(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    num_classes: int,
    C: float = 0.5,
    max_iter: int = 1000,
    lr: float = 1.0,
) -> np.ndarray:
    n_samples, hidden_size = X_train.shape
    lam = 1.0 / (C * n_samples)

    class_counts = torch.bincount(y_train, minlength=num_classes).float()
    class_weights = n_samples / (num_classes * class_counts.clamp(min=1))
    sample_weights = class_weights[y_train].unsqueeze(1)

    W = torch.zeros(num_classes, hidden_size, device=X_train.device)
    b = torch.zeros(num_classes, device=X_train.device)

    W_y, b_y = W.clone(), b.clone()
    t = 1.0

    for _ in range(max_iter):
        logits = X_train @ W_y.T + b_y
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(y_train, num_classes=num_classes).float()

        grad_logits = (probs - targets_one_hot) * sample_weights
        grad_W = grad_logits.T @ X_train / n_samples
        grad_b = grad_logits.sum(dim=0) / n_samples

        W_next = W_y - lr * grad_W
        b_next = b_y - lr * grad_b

        threshold = lr * lam
        W_next = torch.sign(W_next) * torch.clamp(W_next.abs() - threshold, min=0.0)

        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = (t - 1.0) / t_next

        W_y = W_next + momentum * (W_next - W)
        b_y = b_next + momentum * (b_next - b)

        W, b, t = W_next, b_next, t_next

    with torch.no_grad():
        logits_test = X_test @ W.T + b
        y_pred = logits_test.argmax(dim=1)

    return y_pred.cpu().numpy()


@timed_metric("logistic_regression")
def classify_per_layer_logistic_regression(
    representations: torch.Tensor,
    y: np.ndarray,
    backend: str,
) -> dict[int, float]:
    num_layers, num_samples, _ = representations.shape
    use_stratify = bool(np.all(np.bincount(y) >= 2))
    num_classes = int(y.max()) + 1
    layer_results: dict[int, float] = {}

    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    if backend == "pytorch":
        y_train_t = torch.tensor(y[train_idx], dtype=torch.long, device=DEVICE)
        update_every = max(1, int(0.1 * num_layers))
        updates_pending = 0
        with tqdm(total=num_layers, desc="Metric logistic_regression", unit="layer") as pbar:
            for layer_idx in range(num_layers):
                X_train = representations[layer_idx, train_idx].to(DEVICE)
                X_test = representations[layer_idx, test_idx].to(DEVICE)

                y_pred = _torch_logistic_regression_l1(
                    X_train,
                    y_train_t,
                    X_test,
                    num_classes,
                    C=0.1,
                    max_iter=5000,
                )
                layer_results[layer_idx] = float(
                    f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)
                )

                updates_pending += 1
                if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                    pbar.update(updates_pending)
                    updates_pending = 0
    else:
        warnings.filterwarnings("ignore", category=UserWarning)
        update_every = max(1, int(0.1 * num_layers))
        updates_pending = 0
        with tqdm(total=num_layers, desc="Metric logistic_regression", unit="layer") as pbar:
            for layer_idx in range(num_layers):
                X = representations[layer_idx].numpy()
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model = Pipeline(
                    [
                        (
                            "classifier",
                            LogisticRegression(
                                penalty="l1",
                                solver="liblinear",
                                C=0.1,
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                )

                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_train, y_train)
                    if captured:
                        print(f"Warning: Layer {layer_idx} failed to converge.")

                y_pred = model.predict(X_test)
                layer_results[layer_idx] = float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                )

                updates_pending += 1
                if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                    pbar.update(updates_pending)
                    updates_pending = 0

    return layer_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="lbasile/llava_7b")
    parser.add_argument("--dataset-name", type=str, default="derek-thomas/ScienceQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--guide-text",
        type=str,
        default="Answer with the option's letter from the given choices directly.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help=(
            "Root directory for saving results. Use default to save under "
            "~/multimodal_finetuned_representations/results."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of samples for quick debug runs. Omit for full dataset.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sklearn",
        choices=["pytorch", "sklearn"],
        help="Backend used only by logistic_regression.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["logistic_regression"],
        choices=list(SUPPORTED_METRICS),
        help="Metrics to compute.",
    )
    return parser.parse_args()


def _resolve_results_base(results_dir_arg: str) -> Path:
    if results_dir_arg == "results":
        return RESULTS_BASE
    return Path(results_dir_arg).expanduser().resolve()


def _build_output_path(
    *,
    results_base: Path,
    dataset_name: str,
    split: str,
    model_name: str,
    pooling_method: str,
    metric_name: str,
) -> Path:
    dataset_slug = Path(dataset_name).name
    model_slug = Path(model_name).name
    out_dir = (
        results_base
        / "representations_classification"
        / dataset_slug
        / split
        / model_slug
        / pooling_method
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{metric_name}.safetensors"


def _save_metric_results(
    *,
    out_path: Path,
    metric_name: str,
    layer_results: dict[int, float],
    num_layers: int,
    args: argparse.Namespace,
    num_samples: int,
    label_classes: np.ndarray,
) -> None:
    metric_key = METRIC_TO_KEY[metric_name]
    metric_tensor = torch.tensor(
        [layer_results[i] for i in range(num_layers)], dtype=torch.float32
    )

    metadata = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "pooling_method": POOLING_METHOD,
        "guide_text": args.guide_text,
        "num_samples": str(num_samples),
        "classes": ", ".join(label_classes),
        "metric": metric_name,
        "metric_key": metric_key,
        "backend": args.backend if metric_name == "logistic_regression" else "n/a",
    }
    save_file({metric_key: metric_tensor}, str(out_path), metadata=metadata)
    print(f"Saved {metric_name}: {out_path}")


def main() -> None:
    args = parse_args()

    representations, labels = get_representations_and_labels(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        guide_text=args.guide_text,
        max_samples=args.max_samples,
    )

    encoded_labels = np.asarray(labels)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(encoded_labels)

    num_layers, num_samples, _ = representations.shape
    results_base = _resolve_results_base(args.results_dir)

    for metric_name in args.metrics:
        print(f"\nComputing metric: {metric_name}")

        if metric_name == "logistic_regression":
            layer_results = classify_per_layer_logistic_regression(
                representations,
                y,
                backend=args.backend,
            )
        elif metric_name == "label_overlap":
            layer_results = classify_per_layer_label_overlap(representations, y)
        else:
            layer_results = classify_per_layer_knn(representations, y)

        out_path = _build_output_path(
            results_base=results_base,
            dataset_name=args.dataset_name,
            split=args.split,
            model_name=args.model_name,
            pooling_method=POOLING_METHOD,
            metric_name=metric_name,
        )

        classes = label_encoder.classes_
        _save_metric_results(
            out_path=out_path,
            metric_name=metric_name,
            layer_results=layer_results,
            num_layers=num_layers,
            args=args,
            num_samples=num_samples,
            label_classes=classes,
        )


if __name__ == "__main__":
    seed_all(42)
    main()
