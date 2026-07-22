"""
Compute imbalance and intrinsic dimension (ID) between text and image
representations extracted from multimodal language models.

Usage
-----
python compute_id_imbalance.py --model_name llava_7b --dataset_name vqav2 \
    --ft_mode ft --compute_imbalance

python compute_id_imbalance.py --model_name llava_7b --dataset_name vqav2 \
    --ft_mode pre --compute_id --pooling_strategy avg
"""

import argparse
import os
import time

import numpy as np
import torch
from dadapy import data
from safetensors import safe_open

from utils.embeddings_extraction.imbalance_adapter import (
    _return_imbalance,
    compute_imbalance_torch,
)

# ---------------------------------------------------------------------------
# Global random state
# ---------------------------------------------------------------------------
rng = np.random.default_rng()

# ---------------------------------------------------------------------------
# Configuration: paths and name mappings
# ---------------------------------------------------------------------------
REPRESENTATIONS_ROOT = os.environ.get(
    "REPRESENTATIONS_DIR", os.path.expanduser("~/scratch/representations")
)

MODEL_MAP = {
    "llava_7b": {"pre": "llava-v1.5-7b-pretrained-lbasile", "ft": "llava_7b"},
    "llava_13b": {"pre": "llava-v1.5-13b-pretrained-lbasile", "ft": "llava_13b"},
    "internvl2_8b": {"pre": "InternVL2-8B-Pretrain", "ft": "InternVL2-8B"},
    #"llava_onevision_8b": {"pre": "LLaVA-OneVision-1.5-8B-stage1", "ft": "LLaVA-OneVision-1.5-8B-stage2-2600"},
    "llava_onevision_8b_hf": {"pre": "LLaVA-OneVision-1.5-8B-Base", "ft": "LLaVA-OneVision-1.5-8B-Instruct"},
    "llava_onevision_4b_hf": {"pre": "LLaVA-OneVision-1.5-4B-Base", "ft": "LLaVA-OneVision-1.5-4B-Instruct"},
    "cambrian_8b": {"pre": "cambrian-8b-pretrained", "ft": "cambrian-8b"},
    "llama_90b":{"ft": "Meta-Llama-3-90B-instruct"}
    
}

DATASET_MAP = {
    "vqav2": "vqav2_coco_restval_icq",
    "cocoqa": "cocoqa_restval_icqa",
    "gqa": "gqa_val_balanced_noyesno_captioned",
    "mme": "mme_test_captioned",
    "mmbench": "mmbench_en_dev_captioned",
    "pope": "pope_test_captioned",
    "seed": "seed_bench_images_captioned",
}

# Per-dataset suffixes that distinguish the text vs. image representation files.
DATA_TEXT_POSTFIX = {
    "vqav2": "split01_human_txt",
    "cocoqa": "split01_human_txt",
    "gqa": "synthetic_txt",
    "mme": "synthetic_txt",
    "mmbench": "synthetic_txt",
    "seed": "synthetic_txt",
    "pope": "synthetic_txt",
}

DATA_IMG_POSTFIX = {
    "vqav2": "split01_human_img",
    "cocoqa": "split01_human_img",
    "gqa": "synthetic_img",
    "mme": "synthetic_img",
    "mmbench": "synthetic_img",
    "seed": "synthetic_img",
    "pope": "synthetic_img",
}

POOLING_FOLDER = {
    "last": "last",
    "avg": "mean_context",
}

ALL_DATA = {"vqav2", "cocoqa", "gqa", "mmbench", "seed"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute imbalance / intrinsic dimension between"
            " text and image representations"
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(MODEL_MAP),
        help="Model whose representations to analyse",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=list(DATASET_MAP),
        help="Dataset whose representations to analyse",
    )
    parser.add_argument(
        "--ft_mode",
        type=str,
        required=True,
        choices=["pre", "ft"],
        help="'pre' for pretrained checkpoint, 'ft' for fine-tuned checkpoint",
    )
    parser.add_argument(
        "--compute_imbalance",
        action="store_true",
        help="Compute cross-layer imbalance between text and image representations",
    )
    parser.add_argument(
        "--compute_id",
        action="store_true",
        help="Compute intrinsic dimension of text and image representations",
    )
    parser.add_argument(
        "--pooling_strategy",
        default="last",
        choices=list(POOLING_FOLDER),
        help="Token pooling strategy used when loading representations",
    )
    parser.add_argument(
        "--open_ended",
        action="store_true",
    )
    parser.add_argument(
        "--all_data",
        action="store_true",
    )
    parser.add_argument(
        "--test_imbalance",
        action="store_true",
    )
    parser.add_argument(
        "--nsamples_dataset",
        type=int, 
        default=5000
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _get_nn_indices(X, k=None):
    """Return k-nearest-neighbour indices and distances for every point in *X*.

    Parameters
    ----------
    X:
        Input data matrix of shape ``(N, D)``.  Accepts NumPy arrays and
        PyTorch tensors.
    k:
        Number of neighbours to return.  Defaults to ``N - 1`` (every other
        point).

    Returns
    -------
    nn_indices:
        ``(N, k+1)`` integer array.  Column 0 is the point itself (distance 0).
    distances:
        ``(N, k+1)`` float array of sorted distances.
    """
    if isinstance(X, np.ndarray):
        x_tensor = torch.from_numpy(X).float()
    else:
        x_tensor = X.float()

    device = torch.device("cuda")
    x_tensor = x_tensor.to(device)

    N = x_tensor.shape[0]

    dist_matrix = torch.cdist(x_tensor, x_tensor, p=2)
    nn_indices = torch.argsort(dist_matrix, dim=1, descending=False, stable=True)
    sorted_distances = dist_matrix.gather(1, nn_indices)

    return nn_indices.cpu().numpy(), sorted_distances.cpu().numpy()


def _first_occurrence_mask(X):
    """Boolean mask that is ``True`` for the first occurrence of each unique row.

    Order of the surviving rows is preserved.
    Works with PyTorch tensors (any dtype) and NumPy arrays.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    _, inverse = torch.unique(X, dim=0, sorted=False, return_inverse=True)
    seen = set()
    mask = []
    for idx in inverse.tolist():
        mask.append(idx not in seen)
        seen.add(idx)
    return torch.tensor(mask, dtype=torch.bool)


def _remove_paired_duplicates(x_img, x_txt, x_img_ref=None, x_txt_ref=None):
    """Remove rows that are duplicated in *either* modality.

    Keeps the paired relationship intact: row *i* of ``x_img`` and row *i* of
    ``x_txt`` refer to the same data point.

    If ``x_img_ref`` / ``x_txt_ref`` are provided, the deduplication mask is
    computed on those tensors (e.g. avg-token representations) but applied to
    ``x_img`` / ``x_txt`` (e.g. last-token representations).  When the ref
    tensors have a different number of rows they are silently ignored.

    Returns
    -------
    (x_img_clean, x_txt_clean)
    """
    if x_img_ref is not None and x_img_ref.shape[0] != x_img.shape[0]:
        print(
            f"    Warning: x_img_ref has {x_img_ref.shape[0]} rows but x_img has "
            f"{x_img.shape[0]} rows; ignoring ref for deduplication"
        )
        x_img_ref = None
    if x_txt_ref is not None and x_txt_ref.shape[0] != x_txt.shape[0]:
        print(
            f"    Warning: x_txt_ref has {x_txt_ref.shape[0]} rows but x_txt has "
            f"{x_txt.shape[0]} rows; ignoring ref for deduplication"
        )
        x_txt_ref = None

    mask_img = _first_occurrence_mask(x_img_ref if x_img_ref is not None else x_img)
    mask_txt = _first_occurrence_mask(x_txt_ref if x_txt_ref is not None else x_txt)
    keep = mask_img & mask_txt

    n_removed = (~keep).sum().item()
    if n_removed > 0:
        print(f"    deduplication: removed {n_removed} / {len(keep)} rows")

    np_mask = keep.numpy()
    if isinstance(x_img, np.ndarray):
        return x_img[np_mask], x_txt[np_mask]
    return x_img[keep], x_txt[keep]


def _load_safetensor_layers(filepath: str, layer_dict_old=None, n_samples=3000):
    """Load a multi-layer safetensors file and split it into per-layer tensors.

    Parameters
    ----------
    filepath:
        Path to the ``.safetensors`` file.  Expected to hold a single key
        whose tensor has shape ``(num_layers, N, D)``.
    label:
        Human-readable description used in log messages.

    Returns
    -------
    layer_dict:
        Mapping ``{layer_index: tensor}`` where each tensor has shape ``(N, D)``.
    num_layers:
        Total number of layers found in the file.
    """
    with safe_open(filepath, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"    {key}: shape {tensor.shape}, dtype {tensor.dtype}")

    num_layers = tensor.shape[0]
    n_take = min(n_samples, tensor.shape[1])
    if layer_dict_old is not None:
        layer_dict = {
            i: torch.cat([layer_dict_old[i], tensor[i][:n_take]])
            for i in range(num_layers)
        }
    else:
        layer_dict = {i: tensor[i][:n_take] for i in range(num_layers)}
    return layer_dict, num_layers


def load_all_data(
    data_type,
    open_ended,
    model_dir,
    pooling_folder,
    n_samples_dataset,
):
    for i, dataset_name in enumerate(ALL_DATA):
        dataset = DATASET_MAP[dataset_name]
        if data_type == "txt":
            postfix = DATA_TEXT_POSTFIX[dataset_name]
        else:
            postfix = DATA_IMG_POSTFIX[dataset_name]

        print(f"  data type: {data_type} ({postfix})")
        if dataset_name in ["mmbench", "seed"] and open_ended:
            dataset += "_open_ended"
        filepath = f"{REPRESENTATIONS_ROOT}/{model_dir}/{dataset}_{postfix}/output_layer/{pooling_folder}/unique_images/representations.safetensors"

        if i == 0:
            representations = None

        representations, num_layers = _load_safetensor_layers(
            filepath,
            layer_dict_old=representations,
            n_samples=n_samples_dataset,
        )
    return representations, num_layers


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_imbalance(X, coordinates, x_avg=None, coord_avg=None):
    """Compute the pairwise imbalance between two representation matrices.

    Parameters
    ----------
    X:
        First representation matrix ``(N, D1)``.
    coordinates:
        Second representation matrix ``(N, D2)``.
    x_avg, coord_avg:
        Optional avg-token counterparts used to drive deduplication while the
        actual imbalance is computed on ``X`` / ``coordinates``.

    Returns
    -------
    (imb_ij, imb_ji):
        Imbalance from *X* to *coordinates* and vice-versa.
    """
    if X.shape[0] != coordinates.shape[0]:
        raise ValueError(
            f"Both datasets must have the same number of samples, "
            f"got {X.shape[0]} vs {coordinates.shape[0]}"
        )

    X, coordinates = _remove_paired_duplicates(X, coordinates, x_avg, coord_avg)

    N = X.shape[0]
    dist_indices_base, _ = _get_nn_indices(X, N - 1)
    dist_indices_other, _ = _get_nn_indices(coordinates, N - 1)

    assert np.all(
        dist_indices_base[:, 0] == np.arange(N)
    ), "Each point should be its own nearest neighbour"
    assert dist_indices_base.shape[0] == dist_indices_other.shape[0]

    imb_ij = _return_imbalance(dist_indices_base, dist_indices_other, rng, k=1)
    imb_ji = _return_imbalance(dist_indices_other, dist_indices_base, rng, k=1)
    return imb_ij, imb_ji


def test_imbalance(X1, X2):

    dtype1 = X1.dtype
    dtype2 = X2.dtype
    assert dtype1 == dtype2

    atol, rtol = torch.testing._comparison.default_tolerances(dtype1)
    print(f"testing imbalances dtype: {dtype1}, atol {atol} rtol {rtol}")

    t1 = time.time()
    i_dada_12, i_dada_21 = compute_imbalance(X1, X2)
    t2 = time.time()
    i_torch_12, i_torch_21 = compute_imbalance_torch(X1, X2, exact_rank=False)
    t3 = time.time()
    print(f"mixed time: {t2-t1:.1f}s")
    print(f"gpu time: {t3-t2:.1f}s")

    torch.testing.assert_close(i_dada_12, i_torch_12, rtol=rtol, atol=atol)
    torch.testing.assert_close(i_dada_21, i_torch_21, rtol=rtol, atol=atol)


def compute_id(X):
    """Estimate the intrinsic dimension of *X* with the GRIDE estimator.

    *X* is expected to contain only unique rows; callers are responsible for
    deduplication before calling this function.

    Returns
    -------
    ids:
        Array of ID estimates at different GRIDE scales.
    """
    if isinstance(X, np.ndarray):
        X = X.astype(np.float64)
    else:
        X = X.to(torch.float64).cpu().numpy()

    d = data.Data(coordinates=X)
    d.compute_distances(maxk=200)
    ids, _, _ = d.return_id_scaling_gride(range_max=100)
    return ids


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------


def test_imbalances(args):
    ft_mode = args.ft_mode
    model_dir = MODEL_MAP[args.model_name][ft_mode]

    pooling_folder = POOLING_FOLDER[args.pooling_strategy]
    txt_postfix = DATA_TEXT_POSTFIX[args.dataset_name]
    img_postfix = DATA_IMG_POSTFIX[args.dataset_name]

    representations: dict = {}

    for data_type, postfix in (("txt", txt_postfix), ("img", img_postfix)):

        dataset = DATASET_MAP[args.dataset_name]
        if args.dataset_name in ["mmbench", "seed"] and args.open_ended:
            dataset += "_open_ended"
        filepath = f"{REPRESENTATIONS_ROOT}/{model_dir}/{dataset}_{postfix}/output_layer/{pooling_folder}/unique_images/representations.safetensors"

        print(f"{args.dataset_name} {data_type} ({args.pooling_strategy}-token)")
        representations[postfix], _ = _load_safetensor_layers(filepath)

    for i, x_txt in representations[txt_postfix].items():
        print(f"testing imbalances for layer {i}")
        for j, x_img in representations[img_postfix].items():
            test_imbalance(x_txt, x_img)


def compute_imbalance_layer(args: argparse.Namespace) -> None:
    """Load representations and compute cross-layer imbalance matrices."""
    ft_mode = args.ft_mode
    model_dir = MODEL_MAP[args.model_name][ft_mode]

    pooling_folder = POOLING_FOLDER[args.pooling_strategy]
    txt_postfix = DATA_TEXT_POSTFIX[args.dataset_name]
    img_postfix = DATA_IMG_POSTFIX[args.dataset_name]

    representations: dict = {}
    avg_representations: dict = {}

    for data_type, postfix in (("txt", txt_postfix), ("img", img_postfix)):

        dataset = DATASET_MAP[args.dataset_name]
        if args.dataset_name in ["mmbench", "seed"] and args.open_ended:
            dataset += "_open_ended"
        filepath = f"{REPRESENTATIONS_ROOT}/{model_dir}/{dataset}_{postfix}/output_layer/{pooling_folder}/unique_images/representations.safetensors"

        print(f"{args.dataset_name} {data_type} ({args.pooling_strategy}-token)")
        representations[postfix], num_layers = _load_safetensor_layers(filepath)

    ii_txt_img = np.zeros((num_layers, num_layers))
    ii_img_txt = np.zeros((num_layers, num_layers))

    for i, x_txt in representations[txt_postfix].items():
        print(f"computing imbalances for layer {i}")
        for j, x_img in representations[img_postfix].items():
            imb_txt_img, imb_img_txt = compute_imbalance_torch(x_txt, x_img)
            ii_txt_img[i, j] = imb_txt_img
            ii_img_txt[i, j] = imb_img_txt

    output_dir = (
        f"results/imbalance/{model_dir}/{args.dataset_name}"
        f"/{args.pooling_strategy}_token"
    )
    os.makedirs(output_dir, exist_ok=True)
    postfix = ""
    if args.open_ended:
        postfix += "_open_ended"
    np.save(f"{output_dir}/{ft_mode}_ii_txt_img_dedup{postfix}.npy", ii_txt_img)
    np.save(f"{output_dir}/{ft_mode}_ii_img_txt_dedup{postfix}.npy", ii_img_txt)
    print(f"Saved imbalance results to {output_dir}/")


def compute_imbalance_all(args: argparse.Namespace) -> None:
    """Load representations and compute cross-layer imbalance matrices."""
    ft_mode = args.ft_mode
    model_dir = MODEL_MAP[args.model_name][ft_mode]
    pooling_folder = POOLING_FOLDER[args.pooling_strategy]
    txt_postfix = DATA_TEXT_POSTFIX[args.dataset_name]
    img_postfix = DATA_IMG_POSTFIX[args.dataset_name]

    representations: dict = {}
    avg_representations: dict = {}

    for data_type, postfix in (("txt", txt_postfix), ("img", img_postfix)):

        representations[postfix], num_layers = load_all_data(
            data_type,
            args.open_ended,
            model_dir,
            pooling_folder,
            n_samples_dataset=args.nsamples_dataset,
        )
        print(f"loaded all {data_type} data. Shape: {representations[postfix][0].shape[0]}")

    ii_txt_img = np.zeros((num_layers, num_layers))
    ii_img_txt = np.zeros((num_layers, num_layers))

    for i, x_txt in representations[txt_postfix].items():
        print(f"computing imbalances for layer {i}")
        for j, x_img in representations[img_postfix].items():
            imb_txt_img, imb_img_txt = compute_imbalance_torch(x_txt, x_img)
            ii_txt_img[i, j] = imb_txt_img
            ii_img_txt[i, j] = imb_img_txt

    output_dir = f"results/imbalance/{model_dir}/all_data/{args.pooling_strategy}_token"
    os.makedirs(output_dir, exist_ok=True)
    postfix = ""
    if args.open_ended:
        postfix += "_open_ended"
    np.save(f"{output_dir}/{ft_mode}_ii_txt_img_dedup{postfix}.npy", ii_txt_img)
    np.save(f"{output_dir}/{ft_mode}_ii_img_txt_dedup{postfix}.npy", ii_img_txt)
    print(f"Saved imbalance results to {output_dir}/")


def compute_id_layer(args: argparse.Namespace) -> None:
    """Load representations and compute per-layer intrinsic dimension."""
    ft_mode = args.ft_mode
    model_dir = MODEL_MAP[args.model_name][ft_mode]
    pooling_folder = POOLING_FOLDER[args.pooling_strategy]
    print(f"Computing ID for model: {model_dir}")

    for data_type_, postfix in (
        ("txt", DATA_TEXT_POSTFIX[args.dataset_name]),
        ("img", DATA_IMG_POSTFIX[args.dataset_name]),
    ):

        dataset = DATASET_MAP[args.dataset_name]
        print(f"  data type: {data_type_} ({postfix})")
        if args.dataset_name in ["mmbench", "seed"] and args.open_ended:
            dataset += "_open_ended"
        filepath = f"{REPRESENTATIONS_ROOT}/{model_dir}/{dataset}_{postfix}/output_layer/{pooling_folder}/unique_images/representations.safetensors"

        print(f"{args.dataset_name} {data_type_} ({args.pooling_strategy}-token)")
        representations, _ = _load_safetensor_layers(filepath)

        ids_all_layers = []
        for i, layer_unique in representations.items():
            print(f"computing ids for layer {i}")
            ids_all_layers.append(compute_id(layer_unique))

        id_results = np.array(ids_all_layers)  # shape: (num_layers, num_gride_scales)

        output_dir = (
            f"results/intrinsic_dim/{model_dir}/{args.dataset_name}"
            f"/{args.pooling_strategy}_token"
        )
        os.makedirs(output_dir, exist_ok=True)
        postfix = ""
        if args.open_ended:
            postfix += "_open_ended"
        np.save(f"{output_dir}/{ft_mode}_id_{data_type_}{postfix}.npy", id_results)
        print(f"  Saved ID ({data_type_}) results to {output_dir}/")


def compute_id_all(args):
    ft_mode = args.ft_mode
    model_dir = MODEL_MAP[args.model_name][ft_mode]
    pooling_folder = POOLING_FOLDER[args.pooling_strategy]
    print(f"Computing ID for model: {model_dir}")

    for data_type in ["txt", "img"]:

        representations, _ = load_all_data(
            data_type,
            args.open_ended,
            model_dir,
            pooling_folder,
            n_samples_dataset=args.nsamples_dataset,
        )
        print(f"loaded all {data_type} data. Shape: {representations[0].shape[0]}")
        ids_all_layers = []
        for i, layer_unique in representations.items():
            print(f"computing ids for layer {i}")
            ids_all_layers.append(compute_id(layer_unique))

        id_results = np.array(ids_all_layers)  # shape: (num_layers, num_gride_scales)

        output_dir = (
            f"results/intrinsic_dim/{model_dir}/all_data/{args.pooling_strategy}_token"
        )
        os.makedirs(output_dir, exist_ok=True)
        postfix = ""
        if args.open_ended:
            postfix += "_open_ended"

        np.save(f"{output_dir}/{ft_mode}_id_{data_type}{postfix}.npy", id_results)
        print(f"Saved ID ({data_type}) results to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not args.compute_imbalance and not args.compute_id:
        print("Nothing to do. Pass --compute_imbalance and/or --compute_id.")
        return

    if args.all_data:
        print(args.ft_mode, f"computing all data jointly: {args.nsamples_dataset}\n")
    else:
        print(args.dataset_name, args.ft_mode, "\n")

    if args.compute_imbalance:
        if args.test_imbalance:
            test_imbalances(args)
        elif args.all_data:
            compute_imbalance_all(args)
        else:
            compute_imbalance_layer(args)

    if args.compute_id:
        if args.all_data:
            compute_id_all(args)
        else:
            compute_id_layer(args)


if __name__ == "__main__":
    main()
