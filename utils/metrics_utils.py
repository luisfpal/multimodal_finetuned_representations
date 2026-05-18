import math
import time
import warnings
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from dadapy.data import Data
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    homogeneity_completeness_v_measure,
    matthews_corrcoef,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# from anatome.similarity import svcca_distance
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_device(
    *tensors: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Move tensors to GPU if available, otherwise keep on CPU.

    Args:
        *tensors: Variable number of torch tensors
        device: Optional target device. Defaults to CUDA when available.
        dtype: Optional dtype conversion.
        non_blocking: Forwarded to ``Tensor.to`` when a copy is needed.

    Returns:
        Single tensor if one input, tuple of tensors if multiple inputs
    """
    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    moved_tensors = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        target_dtype = dtype if dtype is not None else tensor.dtype
        if tensor.device == target_device and tensor.dtype == target_dtype:
            moved_tensors.append(tensor)
        else:
            moved_tensors.append(
                tensor.to(
                    device=target_device,
                    dtype=target_dtype,
                    non_blocking=non_blocking,
                )
            )
    moved_tensors = tuple(moved_tensors)

    if len(moved_tensors) == 1:
        return moved_tensors[0]
    return moved_tensors


def _as_metric_tensor(
    data: Union[np.ndarray, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
        tensor = _ensure_device(tensor, device=device, dtype=torch.float32)
        return tensor.contiguous()

    array = np.asarray(data, dtype=np.float32)
    tensor = torch.tensor(array, dtype=torch.float32)
    return _ensure_device(tensor, device=device)


def preprocess_data(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    output_format: str = "tensor",
    remove_duplicates: bool = True,
    return_indices: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    """Common preprocessing for similarity computation.

    Args:
        data1: First array
        data2: Second array
        output_format: 'tensor' or 'numpy'
        remove_duplicates: Whether to remove duplicates and match indices
        return_indices: When True, return the valid index array as a third element.
            The indices are relative to the original (pre-filtering) sample axis and
            can be used to align an additional array (e.g. CLIP embeddings) to the
            same subset.  Backward-compatible default is False.
        device: Optional target device for tensor output. Defaults to CUDA when available.

    Returns:
        (data1_out, data2_out) when return_indices=False (default).
        (data1_out, data2_out, valid_indices) when return_indices=True, where
        valid_indices is a 1-D numpy int array of the retained sample positions.
    """
    # Ensure both arrays have the same number of samples
    if data1.shape[0] != data2.shape[0]:
        raise ValueError(
            f"Data1 has {data1.shape[0]} samples, Data2 has {data2.shape[0]} samples"
        )

    if output_format == "tensor" and not remove_duplicates:
        unique_data1 = _as_metric_tensor(data1, device=device)
        matched_data2 = _as_metric_tensor(data2, device=device)
        valid_indices = np.arange(data1.shape[0])
        if return_indices:
            return unique_data1, matched_data2, valid_indices
        return unique_data1, matched_data2

    # Convert to float32 and move to CPU if it's a tensor, or ensure float32 for numpy
    if isinstance(data1, torch.Tensor):
        np_data1 = data1.detach().cpu().to(torch.float32).numpy()
    else:
        np_data1 = np.asarray(data1, dtype=np.float32)

    if isinstance(data2, torch.Tensor):
        np_data2 = data2.detach().cpu().to(torch.float32).numpy()
    else:
        np_data2 = np.asarray(data2, dtype=np.float32)

    # Remove duplicates and match indices if requested
    if remove_duplicates:
        # Get unique indices for BOTH arrays
        _, idx1 = np.unique(np_data1, axis=0, return_index=True)
        _, idx2 = np.unique(np_data2, axis=0, return_index=True)

        # Keep only indices that are unique in BOTH arrays.
        # np.intersect1d naturally returns a sorted 1D array.
        valid_indices = np.intersect1d(idx1, idx2)

        original_num_samples = np_data1.shape[0]
        unique_num_samples = valid_indices.shape[0]
        if unique_num_samples < original_num_samples:
            print(
                f"Warning: {original_num_samples - unique_num_samples} duplicate samples removed."
            )

        # Apply the exact same valid indices mask to both arrays to guarantee 1-to-1 sync
        unique_data1 = np_data1[valid_indices]
        matched_data2 = np_data2[valid_indices]
    else:
        unique_data1 = np_data1
        matched_data2 = np_data2
        valid_indices = np.arange(np_data1.shape[0])

    # Convert to requested output format
    if output_format == "tensor":
        unique_data1 = torch.tensor(unique_data1, dtype=torch.float32)
        matched_data2 = torch.tensor(matched_data2, dtype=torch.float32)
        unique_data1, matched_data2 = _ensure_device(
            unique_data1,
            matched_data2,
            device=device,
        )
    # else: keep as numpy arrays

    if return_indices:
        return unique_data1, matched_data2, valid_indices
    return unique_data1, matched_data2


def _overlap_neighbor_indices(
    data: torch.Tensor,
    maxk: int,
    metric: str = "euclidean",
) -> torch.Tensor:
    if metric == "cosine":
        data = F.normalize(data, dim=1)
        return torch.topk(data @ data.T, maxk + 1, dim=1).indices[:, 1:]

    return torch.topk(
        torch.cdist(data, data), maxk + 1, dim=1, largest=False
    ).indices[:, 1:]


def _neighborhood_overlap_from_indices(
    x_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    maxk: int,
) -> torch.Tensor:
    if maxk == 0:
        raise ZeroDivisionError("float division by zero")
    n_samples = x_neighbors.shape[0]
    overlap = (
        x_neighbors.unsqueeze(2)
        .eq(y_neighbors.unsqueeze(1))
        .sum(dtype=torch.float64)
    )
    return overlap / (maxk * n_samples)


def _information_distance_matrix(data: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    if metric == "cosine":
        data = F.normalize(data, dim=1)
    return torch.cdist(data, data)


def _first_neighbor_indices(data: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    distances = _information_distance_matrix(data, metric=metric)
    indices = torch.topk(distances, k=2, dim=1, largest=False).indices[:, 1]
    del distances
    return indices


def _information_imbalance_from_cached(
    source_neighbor_indices: torch.Tensor,
    target_distances: torch.Tensor,
) -> torch.Tensor:
    n_samples = target_distances.shape[0]
    row_indices = torch.arange(n_samples, device=target_distances.device)
    dists_to_nn = target_distances[row_indices, source_neighbor_indices].unsqueeze(1)
    ranks = (target_distances < dists_to_nn).sum(dim=1).float()
    return 2 * ranks.mean() / n_samples


def _center_gram_inplace(kernel: torch.Tensor) -> torch.Tensor:
    row_mean = kernel.mean(dim=1, keepdim=True)
    col_mean = kernel.mean(dim=0, keepdim=True)
    grand_mean = kernel.mean()
    kernel.sub_(row_mean)
    kernel.sub_(col_mean)
    kernel.add_(grand_mean)
    return kernel


def _has_duplicate_rows_like_preprocess(data: torch.Tensor) -> bool:
    data = data.detach().to(dtype=torch.float32)
    if torch.isnan(data).any().item():
        return True
    return torch.unique(data, dim=0).shape[0] != data.shape[0]


def compute_neighborhood_overlap(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    maxk: int = 30,
    use_dadapy: bool = False,
    metric: str = "euclidean",
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Compute neighborhood overlap between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        maxk: Maximum number of nearest neighbors to consider
        use_dadapy: Whether to use dadapy for distance computation
        metric: Distance metric to use ('euclidean' or 'cosine')
        device: Optional target device for tensor computations.

    Returns:
        Overlap score between the two datasets

    Raises:
        ValueError: If the number of samples in data1 and data2 do not match.
    """
    if use_dadapy:
        # For dadapy, we need numpy arrays and should remove duplicates
        processed_data1, processed_data2 = preprocess_data(
            data1, data2, output_format="numpy", remove_duplicates=True
        )
        d1 = Data(processed_data1)
        d1.compute_distances(maxk=maxk)
        return d1.return_data_overlap(processed_data2)

    # For manual computation, preprocess as tensors with duplicate removal
    data1, data2 = preprocess_data(
        data1,
        data2,
        output_format="tensor",
        remove_duplicates=True,
        device=device,
    )

    with torch.no_grad():
        X_tensor, Y_tensor = _ensure_device(
            data1,
            data2,
            device=device,
            dtype=torch.float32,
        )
        x_neighbors = _overlap_neighbor_indices(X_tensor, maxk=maxk, metric=metric)
        y_neighbors = _overlap_neighbor_indices(Y_tensor, maxk=maxk, metric=metric)
        return _neighborhood_overlap_from_indices(
            x_neighbors,
            y_neighbors,
            maxk=maxk,
        ).item()


def compute_information_imbalance(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    metric: str = "euclidean",
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Compute Information Imbalance from data1 to data2.
    Measures how well distances in data1 predict neighbors in data2.

    Args:
        data1: First dataset/representation (Space A)
        data2: Second dataset/representation (Space B)
        metric: Distance metric 'euclidean' or 'cosine'
        device: Optional target device for tensor computations.

    Returns:
        Information Imbalance score (lower is better)
    """
    # Preprocess and move to GPU if available
    # We keep duplicates to preserve the exact structure of the representation
    data1, data2 = preprocess_data(
        data1,
        data2,
        output_format="tensor",
        remove_duplicates=False,
        device=device,
    )
    data1, data2 = _ensure_device(data1, data2, device=device, dtype=torch.float32)

    with torch.no_grad():
        nn_indices_1 = _first_neighbor_indices(data1, metric=metric)
        d2 = _information_distance_matrix(data2, metric=metric)
        return _information_imbalance_from_cached(nn_indices_1, d2).item()


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def compute_distance_correlation(
    latent: Union[np.ndarray, torch.Tensor], control: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute distance correlation between two datasets.

    Args:
        latent: First dataset/representation
        control: Second dataset/representation

    Returns:
        Distance correlation score between the two datasets
    """
    # Preprocess and move to GPU if available
    latent, control = preprocess_data(
        latent, control, output_format="tensor", remove_duplicates=False
    )

    # Ensure tensors are on GPU if available
    latent, control = _ensure_device(latent, control)

    def normalize(data):
        if len(data.shape) > 1 and data.shape[1] > 1:
            return torch.nn.functional.normalize(data, dim=1)
        else:
            return data

    with torch.no_grad():
        latent = normalize(latent)
        control = normalize(control)
        matrix_a = torch.cdist(latent, latent)
        matrix_b = torch.cdist(control, control)
        matrix_A = (
            matrix_a
            - torch.mean(matrix_a, dim=0, keepdim=True)
            - torch.mean(matrix_a, dim=1, keepdim=True)
            + torch.mean(matrix_a)
        )
        matrix_B = (
            matrix_b
            - torch.mean(matrix_b, dim=0, keepdim=True)
            - torch.mean(matrix_b, dim=1, keepdim=True)
            + torch.mean(matrix_b)
        )

        Gamma_XY = torch.sum(matrix_A * matrix_B) / (
            matrix_A.shape[0] * matrix_A.shape[1]
        )
        Gamma_XX = torch.sum(matrix_A * matrix_A) / (
            matrix_A.shape[0] * matrix_A.shape[1]
        )
        Gamma_YY = torch.sum(matrix_B * matrix_B) / (
            matrix_A.shape[0] * matrix_A.shape[1]
        )

        correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r.item()


# def compute_svcca_similarity(
#     X: Union[np.ndarray, torch.Tensor],
#     Y: Union[np.ndarray, torch.Tensor],
#     accept_rate: float = 0.95,
# ) -> float:
#     """
#     Compute SVCCA similarity between two datasets.

#     Args:
#         X: First dataset/representation
#         Y: Second dataset/representation

#     Returns:
#         SVCCA similarity score between the two datasets
#     """
#     # Preprocess and move to GPU if available
#     X, Y = preprocess_data(X, Y, output_format="tensor", remove_duplicates=False)

#     # Ensure tensors are on GPU if available
#     X, Y = _ensure_device(X, Y)

#     similarity = 1 - svcca_distance(X, Y, accept_rate=accept_rate, backend="svd")
#     return similarity.item()


def compute_linear_cka_similarity(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Compute Linear CKA similarity between two datasets.

    Args:
        X: First dataset/representation
        Y: Second dataset/representation

    Returns:
        Linear CKA similarity score between the two datasets
    """
    # Preprocess and move to GPU if available
    X, Y = preprocess_data(
        X,
        Y,
        output_format="tensor",
        remove_duplicates=False,
        device=device,
    )

    # Ensure tensors are on GPU if available
    X, Y = _ensure_device(X, Y, device=device, dtype=torch.float32)

    return cka(X, Y, hsic=linear_hsic).item()


def compute_rbf_cka_similarity(
    X: torch.Tensor, Y: torch.Tensor, sigma: float = None
) -> float:
    """
    Compute RBF CKA similarity between two datasets.

    Args:
        X: First dataset/representation
        Y: Second dataset/representation
        sigma: Sigma for the RBF kernel

    Returns:
        RBF CKA similarity score between the two datasets
    """
    X, Y = preprocess_data(X, Y, output_format="tensor", remove_duplicates=False)

    # Ensure tensors are on GPU if available
    X, Y = _ensure_device(X, Y)
    return cka(X, Y, hsic=kernel_hsic, sigma=sigma).item()


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def cka(
    space1: torch.Tensor,
    space2: torch.Tensor,
    hsic: callable,
    sigma: float = None,
    tolerance=1e-6,
) -> torch.Tensor:
    assert space1.shape[0] == space2.shape[0], (
        "X and Y must have the same number of samples."
    )

    numerator = hsic(space1, space2, sigma)

    var1 = torch.sqrt(hsic(space1, space1, sigma))
    var2 = torch.sqrt(hsic(space2, space2, sigma))

    cka_result = numerator / (var1 * var2)

    assert 0 - tolerance <= cka_result <= 1 + tolerance, (
        "CKA value must be between 0 and 1."
    )

    return cka_result


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def linear_hsic(X: torch.Tensor, Y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Compute HSIC for linear kernels.

    This method is used in the computation of linear CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.

    Returns:
        The computed HSIC value.
    """
    # inter-sample similarity matrices for both spaces ~(N, N)
    L_X = _center_gram_inplace(X @ X.T)
    L_Y = _center_gram_inplace(Y @ Y.T)

    return torch.sum(L_X * L_Y)


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def kernel_hsic(X: torch.Tensor, Y: torch.Tensor, sigma) -> torch.Tensor:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) for RBF kernels.

    This is used in the computation of kernel CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.
        sigma: The RBF kernel width.

    Returns:
        The computed HSIC value.
    """
    return torch.sum(
        center_kernel_matrix(rbf(X, sigma)) * center_kernel_matrix(rbf(Y, sigma))
    )


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def center_kernel_matrix(K: torch.Tensor) -> torch.Tensor:
    """Center the kernel matrix K using the centering matrix H = I_n - (1/n) 1 * 1^T. (Eq. 3 in the paper).

    This method is used in the calculation of HSIC.

    Args:
        K: The kernel matrix to be centered.

    Returns:
        The centered kernel matrix.
    """
    n = K.shape[0]
    unit = torch.ones([n, n]).type_as(K)
    identity_mat = torch.eye(n).type_as(K)
    H = identity_mat - unit / n

    return H @ K @ H


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def rbf(X: torch.Tensor, sigma=None) -> torch.Tensor:
    """Compute the RBF (Radial Basis Function) kernel for a matrix X.

    If sigma is not provided, it is computed based on the median distance.

    Args:
        X: The input matrix (num_samples, embedding_dim).
        sigma: Optional parameter to specify the RBF kernel width.

    Returns:
        The RBF kernel matrix.
    """
    GX = X @ X.T
    KX = torch.diag(GX).type_as(X) - GX + (torch.diag(GX) - GX).T

    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)

    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)

    return KX


def standardize(data: torch.Tensor) -> torch.Tensor:
    """
    Standardize the data (zero mean, unit variance).

    Args:
        data: Input tensor

    Returns:
        Standardized tensor
    """
    return (data - data.mean(0)) / (data.std(0) + 1e-9)


def compute_id_correlation_similarity(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    algorithm: str = "twoNN",
    k: int = 100,
    fraction: float = 0.9,
    return_pvalue: bool = False,
    permutations: int = 100,
    id1: float = None,
    id2: float = None,
) -> float:
    """
    Compute ID correlation between two datasets.

    Args:
        data1: First dataset/representation
        data2: Second dataset/representation
        algorithm: Algorithm to use for ID estimation ('twoNN', 'MLE', etc.)
        k: Number of nearest neighbors for MLE
        fraction: Fraction of points for twoNN
        return_pvalue: Whether to compute and return p-value (warning: expensive)
        permutations: Number of permutations for p-value computation
        id1: Precomputed ID for data1 (optional optimization)
        id2: Precomputed ID for data2 (optional optimization)

    Returns:
        ID correlation score (and optionally p-value, but currently only score is returned to fit interface)
    """
    # Preprocess and move to GPU if available
    data1, data2 = preprocess_data(
        data1, data2, output_format="tensor", remove_duplicates=False
    )
    data1, data2 = _ensure_device(data1, data2)

    data1 = standardize(data1)
    data2 = standardize(data2)

    # Compute individual IDs if not provided
    id_1 = (
        id1
        if id1 is not None
        else compute_intrinsic_dimension(
            data1, algorithm=algorithm, k=k, fraction=fraction
        )
    )
    id_2 = (
        id2
        if id2 is not None
        else compute_intrinsic_dimension(
            data2, algorithm=algorithm, k=k, fraction=fraction
        )
    )

    upper_bound = id_1 + id_2
    lower_bound = min(id_1, id_2)

    # Compute joint ID
    # Concatenate along feature dimension
    joint_data = torch.cat([data1, data2], dim=1)
    original_id = compute_intrinsic_dimension(
        joint_data, algorithm=algorithm, k=k, fraction=fraction
    )

    corr = (upper_bound - original_id) / (upper_bound - lower_bound)

    if return_pvalue:
        # Note: This is computationally expensive, especially for cross-layer analysis
        count = 0
        for _ in range(permutations):
            # Shuffle data2
            shuffled_data2 = data2[torch.randperm(data2.size(0))]
            joint_shuffled = torch.cat([data1, shuffled_data2], dim=1)
            shuffled_id = compute_intrinsic_dimension(
                joint_shuffled, algorithm=algorithm, k=k, fraction=fraction
            )
            if shuffled_id < original_id:
                count += 1

        # p_value = (count + 1) / (permutations + 1)
        # For now, we only return correlation to match the compute_similarity signature
        pass

    return corr


def compute_linear_regression_r2_score(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    regression_type: str = "ridge",
    alphas: Union[np.ndarray, list, None] = None,
    fit_intercept: bool = True,
    random_state: int = 42,
    return_train_score: bool = False,
) -> "float | tuple[float, float]":
    """
    Compute test R² using a single 80/20 train-test split.

    Supports:
    - 'ols'    → ordinary least squares (no regularization)
    - 'ridge'  → ridge regression (closed-form + holdout alpha selection)
    - 'lasso'  → multitask lasso via ISTA (group l21 penalty)

    Computation uses PyTorch and runs on GPU when available.
    Final scoring uses sklearn.metrics.r2_score (uniform average).

    Args:
        data1: Input features X (n_samples, n_features)
        data2: Target values Y (n_samples, n_targets)
        regression_type: 'ols', 'ridge', or 'lasso'
        alphas: Optional list/array of alpha values to grid-search (ridge/lasso only)
        fit_intercept: Whether to include bias term
        random_state: For reproducible train/test split
        return_train_score: If True, return (train_r2, test_r2) tuple; otherwise return test_r2 only

    Returns:
        test_r2 (float) when return_train_score=False, or (train_r2, test_r2) tuple when True
    """

    def score_uniform_average(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        return float(r2_score(y_true_np, y_pred_np, multioutput="uniform_average"))

    def fit_multitask_lasso_ista(
        X: torch.Tensor,
        Y: torch.Tensor,
        alpha: float,
        max_iter: int,
        tol: float,
        beta_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_samples, n_features = X.shape
        _, n_targets = Y.shape

        if beta_init is None:
            beta = torch.zeros(
                n_features,
                n_targets,
                device=X.device,
                dtype=X.dtype,
            )
        else:
            beta = beta_init.clone()

        spectral_norm = torch.linalg.matrix_norm(X, ord=2)
        lipschitz = (spectral_norm.item() ** 2) / max(n_samples, 1)
        step_size = 1.0 / (lipschitz + 1e-8)

        for _ in range(max_iter):
            prev_beta = beta

            residual = X @ beta - Y
            gradient = (X.T @ residual) / n_samples
            beta_temp = beta - step_size * gradient

            row_norms = torch.linalg.norm(beta_temp, ord=2, dim=1, keepdim=True)
            shrink = torch.clamp(row_norms - alpha * step_size, min=0.0)
            beta = beta_temp * (shrink / row_norms.clamp(min=1e-12))

            delta = torch.linalg.norm(beta - prev_beta) / (
                torch.linalg.norm(prev_beta) + 1e-8
            )
            if delta.item() < tol:
                break

        return beta

    regression_type = regression_type.lower()
    X, Y = preprocess_data(
        data1,
        data2,
        output_format="tensor",
        remove_duplicates=False,
    )
    X, Y = _ensure_device(X, Y)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    elif Y.ndim != 2:
        raise ValueError(f"Y must be 1D or 2D. Got shape: {Y.shape}")

    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    device = X.device

    n_samples, n_features = X.shape

    if regression_type not in {"ols", "ridge", "lasso"}:
        raise ValueError("regression_type must be 'ols', 'ridge', or 'lasso'")

    if regression_type == "ridge" and n_samples <= n_features:
        raise ValueError(
            f"Require N > d for {regression_type}: got N={n_samples}, d={n_features}"
        )

    # 80/20 train-test split
    generator = torch.Generator(device="cpu").manual_seed(random_state)
    perm = torch.randperm(n_samples, generator=generator)
    train_size = int(0.8 * n_samples)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    if regression_type == "ridge" and len(train_idx) <= n_features:
        raise ValueError(
            "Require N > d in train split for "
            f"{regression_type}: got N_train={len(train_idx)}, d={n_features}"
        )

    x_mean = X_train.mean(dim=0, keepdim=True)
    x_std = X_train.std(dim=0, unbiased=False, keepdim=True).clamp_(min=1e-8)
    X_train_scaled = (X_train - x_mean) / x_std
    X_test_scaled = (X_test - x_mean) / x_std

    y_mean = Y_train.mean(dim=0, keepdim=True)
    y_std = Y_train.std(dim=0, unbiased=False, keepdim=True).clamp_(min=1e-8)
    Y_train_scaled = (Y_train - y_mean) / y_std
    Y_test_scaled = (Y_test - y_mean) / y_std

    if regression_type == "ols":
        with torch.no_grad():
            if fit_intercept:
                ones_train = torch.ones((X_train_scaled.shape[0], 1), device=device)
                ones_test = torch.ones((X_test_scaled.shape[0], 1), device=device)
                X_train_fit = torch.cat([X_train_scaled, ones_train], dim=1)
                X_test_fit = torch.cat([X_test_scaled, ones_test], dim=1)
            else:
                X_train_fit = X_train_scaled
                X_test_fit = X_test_scaled

            beta, *_ = torch.linalg.lstsq(X_train_fit, Y_train_scaled, rcond=None)
            test_r2 = score_uniform_average(Y_test_scaled, X_test_fit @ beta)
            if return_train_score:
                train_r2 = score_uniform_average(Y_train_scaled, X_train_fit @ beta)
                return train_r2, test_r2
            return test_r2

    train_count = X_train_scaled.shape[0]
    val_count = max(1, int(0.2 * train_count))
    has_validation_split = train_count - val_count >= 2

    if has_validation_split:
        split_generator = torch.Generator(device="cpu").manual_seed(random_state + 1)
        split_perm = torch.randperm(train_count, generator=split_generator)
        val_idx = split_perm[:val_count]
        subtrain_idx = split_perm[val_count:]

        X_subtrain = X_train_scaled[subtrain_idx]
        Y_subtrain = Y_train_scaled[subtrain_idx]
        X_val = X_train_scaled[val_idx]
        Y_val = Y_train_scaled[val_idx]
    else:
        X_subtrain = X_train_scaled
        Y_subtrain = Y_train_scaled
        X_val = None
        Y_val = None

    if fit_intercept:
        ones_subtrain = torch.ones((X_subtrain.shape[0], 1), device=device)
        X_subtrain = torch.cat([X_subtrain, ones_subtrain], dim=1)

        ones_train = torch.ones((X_train_scaled.shape[0], 1), device=device)
        X_train_fit = torch.cat([X_train_scaled, ones_train], dim=1)

        ones_test = torch.ones((X_test_scaled.shape[0], 1), device=device)
        X_test_fit = torch.cat([X_test_scaled, ones_test], dim=1)

        if X_val is not None:
            ones_val = torch.ones((X_val.shape[0], 1), device=device)
            X_val = torch.cat([X_val, ones_val], dim=1)
    else:
        X_train_fit = X_train_scaled
        X_test_fit = X_test_scaled

    if regression_type == "ridge":
        alpha_grid = (
            np.logspace(-4, 2, 9) if alphas is None else np.asarray(alphas, dtype=float)
        )
        best_alpha = float(alpha_grid[0])
        best_score = -float("inf")

        with torch.no_grad():
            eye = torch.eye(X_subtrain.shape[1], device=device, dtype=torch.float32)
            XtX = X_subtrain.T @ X_subtrain
            XtY = X_subtrain.T @ Y_subtrain

            for alpha_value in alpha_grid:
                beta = torch.linalg.solve(XtX + float(alpha_value) * eye, XtY)
                if X_val is not None and Y_val is not None:
                    current_score = score_uniform_average(Y_val, X_val @ beta)
                else:
                    current_score = score_uniform_average(Y_subtrain, X_subtrain @ beta)

                if current_score > best_score:
                    best_score = current_score
                    best_alpha = float(alpha_value)

            eye_train = torch.eye(X_train_fit.shape[1], device=device, dtype=torch.float32)
            beta = torch.linalg.solve(
                X_train_fit.T @ X_train_fit + best_alpha * eye_train,
                X_train_fit.T @ Y_train_scaled,
            )
            test_r2 = score_uniform_average(Y_test_scaled, X_test_fit @ beta)
            if return_train_score:
                train_r2 = score_uniform_average(Y_train_scaled, X_train_fit @ beta)
                return train_r2, test_r2
            return test_r2

    if regression_type == "lasso":
        alpha_grid = (
            np.logspace(-4, 0, 9) if alphas is None else np.asarray(alphas, dtype=float)
        )
        best_alpha = float(alpha_grid[0])
        best_score = -float("inf")
        warm_beta = None

        with torch.no_grad():
            for alpha_value in np.sort(alpha_grid)[::-1]:
                beta = fit_multitask_lasso_ista(
                    X_subtrain,
                    Y_subtrain,
                    alpha=float(alpha_value),
                    max_iter=3000,
                    tol=1e-5,
                    beta_init=warm_beta,
                )
                warm_beta = beta

                if X_val is not None and Y_val is not None:
                    current_score = score_uniform_average(Y_val, X_val @ beta)
                else:
                    current_score = score_uniform_average(Y_subtrain, X_subtrain @ beta)

                if current_score > best_score:
                    best_score = current_score
                    best_alpha = float(alpha_value)

            best_beta = fit_multitask_lasso_ista(
                X_train_fit,
                Y_train_scaled,
                alpha=best_alpha,
                max_iter=5000,
                tol=1e-6,
            )
            test_r2 = score_uniform_average(Y_test_scaled, X_test_fit @ best_beta)
            if return_train_score:
                train_r2 = score_uniform_average(Y_train_scaled, X_train_fit @ best_beta)
                return train_r2, test_r2
            return test_r2

    raise ValueError("regression_type must be 'ols', 'ridge', or 'lasso'")

def compute_similarity(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    measure: str,
    **kwargs,
) -> float:
    """
    Unified function to compute similarity between two neural network representations.

    Args:
        data1: First dataset/representation
        data2: Second dataset/representation
        measure: Similarity measure name. Options:
            - 'neighborhood_overlap': Neighborhood overlap measure
            - 'distance_correlation': Distance correlation measure
            - 'svcca': SVCCA similarity measure
            - 'linear_cka': Linear CKA similarity measure
            - 'rbf_cka': RBF CKA similarity measure
            - 'id_correlation': ID correlation measure
        **kwargs: Additional parameters specific to each measure:
            For 'neighborhood_overlap':
                - maxk (int): Maximum number of nearest neighbors (default: 30)
                - use_dadapy (bool): Whether to use dadapy (default: False)
                - metric (str): Distance metric 'euclidean' or 'cosine' (default: 'euclidean')
            For 'id_correlation':
                - algorithm (str): ID estimation algorithm (default: 'twoNN')
                - return_pvalue (bool): Whether to compute p-value (default: False)

    Returns:
        Similarity score between the two representations

    Raises:
        ValueError: If measure name is not recognized or data shapes don't match
    """
    available_measures = {
        "neighborhood_overlap",
        "distance_correlation",
        # "svcca",
        "linear_cka",
        "rbf_cka",
        "id_correlation",
        "information_imbalance",
        "information_imbalance_reversed",
        "information_imbalance_mean",
        "linear_regression_r2",
        "linear_regression_r2_reversed",
    }

    if measure not in available_measures:
        raise ValueError(
            f"Unknown measure '{measure}'. Available measures: {available_measures}"
        )

    if measure == "neighborhood_overlap":
        maxk = kwargs.get("maxk", 30)
        use_dadapy = kwargs.get("use_dadapy", False)
        metric = kwargs.get("metric", "euclidean")
        device = kwargs.get("device", None)
        return compute_neighborhood_overlap(
            data1,
            data2,
            maxk,
            use_dadapy,
            metric,
            device=device,
        )

    elif measure == "information_imbalance":
        metric = kwargs.get("metric", "euclidean")
        device = kwargs.get("device", None)
        return compute_information_imbalance(data1, data2, metric, device=device)

    elif measure == "information_imbalance_reversed":
        metric = kwargs.get("metric", "euclidean")
        device = kwargs.get("device", None)
        return compute_information_imbalance(data2, data1, metric, device=device)

    elif measure == "information_imbalance_mean":
        metric = kwargs.get("metric", "euclidean")
        device = kwargs.get("device", None)
        return 0.5 * (
            compute_information_imbalance(data1, data2, metric, device=device)
            + compute_information_imbalance(data2, data1, metric, device=device)
        )

    elif measure == "linear_regression_r2":
        regression_type = kwargs.get("regression_type", "ridge")
        alphas = kwargs.get("alphas", None)
        fit_intercept = kwargs.get("fit_intercept", True)
        random_state = kwargs.get("random_state", 0)
        return compute_linear_regression_r2_score(
            data1,
            data2,
            regression_type=regression_type,
            alphas=alphas,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )

    elif measure == "linear_regression_r2_reversed":
        regression_type = kwargs.get("regression_type", "ridge")
        alphas = kwargs.get("alphas", None)
        fit_intercept = kwargs.get("fit_intercept", True)
        random_state = kwargs.get("random_state", 0)
        return compute_linear_regression_r2_score(
            data2,
            data1,
            regression_type=regression_type,
            alphas=alphas,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )

    elif measure == "distance_correlation":
        return compute_distance_correlation(data1, data2)

    # elif measure == "svcca":
    #     accept_rate = kwargs.get("accept_rate", 0.95)
    #     return compute_svcca_similarity(data1, data2, accept_rate)

    elif measure == "linear_cka":
        device = kwargs.get("device", None)
        return compute_linear_cka_similarity(data1, data2, device=device)

    elif measure == "rbf_cka":
        sigma = kwargs.get("sigma", None)
        return compute_rbf_cka_similarity(data1, data2, sigma)

    elif measure == "id_correlation":
        algorithm = kwargs.get("algorithm", "twoNN")
        return_pvalue = kwargs.get("return_pvalue", False)
        id1 = kwargs.get("id1", None)
        id2 = kwargs.get("id2", None)
        # Pass other kwargs that might be relevant for ID estimation
        return compute_id_correlation_similarity(
            data1,
            data2,
            algorithm=algorithm,
            return_pvalue=return_pvalue,
            id1=id1,
            id2=id2,
        )


def compute_intrinsic_dimension(
    X: torch.Tensor,
    algorithm: str = "twoNN",
    fraction: float = 0.9,
    full_output: bool = False,
    range_max: int = 100,
    k: int = 100,
) -> float:
    """
    Compute ID (Intrinsic Dimension) of a dataset.

    Args:
        X: Dataset/representation
        algorithm: Algorithm to use for ID estimation
        fraction: Fraction of points to consider for twoNN algorithm
        full_output: Whether to return the full output or just the mean
        range_max: Maximum neighbor rank to consider for scaling_gride algorithm
        k: Number of nearest neighbors rank to consider for MLE/scaling_gride algorithm
    """
    if isinstance(X, torch.Tensor):
        X = X.to(torch.float32).cpu().numpy()
        X, _ = np.unique(X, axis=0, return_index=True)
        X = torch.tensor(X, dtype=torch.float32)

    if algorithm == "twoNN":
        return twoNN(X, fraction).item()
    elif algorithm == "MLE":
        return MLE(X, k, full_output).item()
    elif algorithm == "scaling_gride":
        return id_scaling_gride(X, k, range_max).item()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/intrinsic_dimension.py
def MLE(X: torch.Tensor, k: int = 100, full_output: bool = False) -> torch.Tensor:
    """
    Compute MLE (Maximum Likelihood Estimation) intrinsic dimension.

    Args:
        X: Dataset/representation
        k: Number of nearest neighbors to consider
        full_output: Whether to return the full output or just the mean

    Returns:
        Intrinsic dimension score between the two representations
    """
    X = X.float()
    X = _ensure_device(X)
    X = torch.cdist(X, X)
    Y = torch.topk(X, k + 1, dim=1, largest=False)[0][:, 1:]
    mask = (Y != 0).all(dim=1)
    Y = Y[mask]
    Y = torch.log(torch.reciprocal(torch.div(Y, Y[:, -1].reshape(-1, 1))))
    dim = torch.reciprocal(1 / (k - 1) * torch.sum(Y, dim=1))
    return dim if full_output else dim.mean()


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/intrinsic_dimension.py
def twoNN(
    X: torch.Tensor, fraction: float = 0.9, distances: bool = False
) -> torch.Tensor:
    """
    Compute twoNN (Two-Nearest-Neighbors) intrinsic dimension.

    Args:
        X: Dataset/representation
        fraction: Fraction of points to consider for the linear regression
        distances: Whether to compute distances between points

    Returns:
        Intrinsic dimension score between the two representations
    """
    X = _ensure_device(X)
    if not distances:
        X = torch.cdist(X, X)
    Y = torch.topk(X, 3, dim=1, largest=False)[0]
    # clean data
    k1 = Y[:, 1]
    k2 = Y[:, 2]
    # remove zeros and degeneracies (k1==k2)
    old_k1 = k1
    k1 = k1[old_k1 != 0]
    k2 = k2[old_k1 != 0]
    old_k1 = k1
    k1 = k1[old_k1 != k2]
    k2 = k2[old_k1 != k2]
    # n.of points to consider for the linear regression
    npoints = int(np.floor(len(k1) * fraction))
    # define mu and Femp
    N = len(k1)
    mu, _ = torch.sort(torch.divide(k2, k1).flatten())
    Femp = (torch.arange(1, N + 1, dtype=X.dtype)) / N
    # take logs (leave out the last element because 1-Femp is zero there)
    x = torch.log(mu[:-1])[0:npoints]
    y = -torch.log(1 - Femp[:-1])[0:npoints]
    x, y = _ensure_device(x, y)
    slope = torch.linalg.lstsq(x.unsqueeze(-1), y.unsqueeze(-1))
    return slope.solution.squeeze()


def id_scaling_gride(
    X: torch.Tensor, k: int = 8, range_max: int = 100
) -> torch.Tensor:
    """
    Compute ID scaling grid.

    Args:
        X: Dataset/representation
        k: Density estimation scale used for ID estimation.
        range_max: Maximum neighbor rank to consider for ID estimation.
                      Creates scales {1,2,4,8,...,2^floor(log2(range_max))}.
                      Default 64 provides 7 scales: {1,2,4,8,16,32,64}.
                      Larger values give more scales but increased computation.
    """
    X = X.float()
    X = _ensure_device(X)
    distances = torch.cdist(X, X).cpu().numpy()
    data = Data(distances=distances)
    ids, _, _ = data.return_id_scaling_gride(range_max=range_max)
    id_index = int(math.log2(k))
    data.set_id(ids[id_index])
    return torch.tensor(data.intrinsic_dim, dtype=torch.float32)


def id_scaling_gride_full(
    X: torch.Tensor, range_max: int = 550
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ID estimates at ALL neighbor-pair scales from the GRIDE algorithm.

    Unlike ``id_scaling_gride``, which selects a single k, this returns the full
    profile so that the caller can analyse how ID varies with neighbourhood scale.

    Exact-duplicate rows are removed before computation (consistent with
    ``compute_intrinsic_dimension``).

    Args:
        X: Dataset/representation of shape (N, D).
        range_max: Maximum neighbour rank. The number of returned scales is
            ``floor(log2(min(N, range_max)))``.  With N≥550 and range_max=550
            this yields 9 scales whose neighbour pairs are
            (1,2), (2,4), (4,8), (8,16), (16,32), (32,64), (64,128), (128,256), (256,512).

    Returns:
        ids:     shape (n_scales,) — intrinsic dimension per scale.
        ids_err: shape (n_scales,) — GRIDE error estimates per scale.

    Scale labels: for scale index i, label as ``k={2**i}-{2**(i+1)}``.
    """
    X = X.float()
    X = _ensure_device(X)
    X_np, _ = np.unique(X.cpu().numpy(), axis=0, return_index=True)
    X_dedup = torch.tensor(X_np, dtype=torch.float32)
    distances = torch.cdist(X_dedup, X_dedup).cpu().numpy()
    # maxk must be >= range_max so dadapy can access the required neighbours.
    # Default maxk=100 would cap effective range_max at 100.
    N = X_np.shape[0]
    effective_maxk = min(N - 1, range_max)
    data = Data(distances=distances, maxk=effective_maxk)
    ids, ids_err, _ = data.return_id_scaling_gride(range_max=range_max)
    return (
        torch.tensor(ids, dtype=torch.float32),
        torch.tensor(ids_err, dtype=torch.float32),
    )


def normalize(R: torch.Tensor) -> torch.Tensor:
    """
    Mean-center and L2-normalize rows of a 2D or 3D tensor.

    - If R is 2D (N, D): subtract the mean across rows and normalize each row to unit L2 norm.
    - If R is 3D (L, M, D): apply the above independently for each of the L slices along the first dimension.

    Args:
        R: Input tensor of shape (L, M, D) or (M, D)

    Returns:
        Normalized tensor of the same shape as input
    """
    with torch.no_grad():
        if R.dim() == 2:
            # Original 2D case from https://github.com/waltonfuture/Matrix-Entropy
            mean = R.mean(dim=0)
            R = R - mean
            norms = torch.norm(R, p=2, dim=1, keepdim=True)
            R = R / norms
        elif R.dim() == 3:
            mean = R.mean(dim=1, keepdim=True)  # Shape: (L, 1, D)
            R = R - mean
            norms = torch.norm(R, p=2, dim=2, keepdim=True)  # Shape: (L, M, 1)
            R = R / norms
        else:
            raise ValueError("Input tensor must be 2D or 3D")
    return R


# inspired by https://github.com/uk-cliplab/representation-itl/blob/main/src/repitl/matrix_itl.py
# https://github.com/OFSkean/information_flow/blob/main/experiments/utils/metrics/metric_functions.py
# https://github.com/OFSkean/information_flow/blob/main/experiments/utils/metrics/metric_calling.py
def compute_matrix_based_renyi_entropy(
    Z: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """
    Computes the Rényi entropy of order `alpha` for a matrix Z ∈ ℝ^{L × M × D}
    S_alpha(K) = 1 / (1 - alpha) * log(sum_{i=1}^D (lambda_i(K)/trace(K))^alpha)

    Args:
        Z (torch.Tensor): Input matrix of shape (L, M, D) or (M, D), where L = layers, M = samples or tokens and D = hidden_size.
        alpha (float): Rényi entropy order (α = 1 for Shannon).

    Returns:
        Tensor of shape (L,) or (1,), where L = layers.
    """
    # verify that Z is a 3D tensor
    if Z.dim() == 2:
        Z = Z.unsqueeze(0)
    elif Z.dim() != 3:
        raise ValueError("Z must be a 2D or 3D tensor")
    Z = Z.double()

    # shape: (num_layers, num_samples, hidden_size)
    layers, _, D = Z.shape

    entropies = torch.zeros(layers, dtype=torch.float64)

    # Pre-compute some constants
    is_shannon = abs(alpha - 1.0) < 1e-6
    alpha_factor = 1.0 / (1 - alpha) if not is_shannon else 1.0

    # Remove duplicates
    Z_layers = []
    unique_tensors = set()
    for layer in range(layers):
        # shape: (num_samples, hidden_size)
        Z_layer = Z[layer]
        # !Different layers could have different unique number of unique tokens/samples
        Z_layer = torch.unique(Z_layer, dim=0)
        Z_layers.append(Z_layer)
        unique_tensors.add(Z_layer.shape[0])

    if len(unique_tensors) == 1:
        # shape: (num_layers, num_samples, hidden_size)
        Z_layers = torch.stack(Z_layers, dim=0)

        # Project each row to the unit sphere removing shared components in the direction of the mean
        Z_layers = normalize(Z_layers)

        M = Z_layers.shape[1]  # num_samples

        Z_layers = _ensure_device(Z_layers)  # (num_layers, num_samples, hidden_size)

        if M < D:  # ! This is the most likely case for my experiments
            K = torch.matmul(Z_layers, Z_layers.transpose(1, 2))
            # (L, M, D) @ (L, D, M) -> (L, M, M)
        else:
            K = torch.matmul(Z_layers.transpose(1, 2), Z_layers)
            # (L, D, M) @ (L, M, D) -> (L, D, D)

        # Clamp negative eigenvalues due to numerical precision
        K = torch.clamp(K, min=0.0)  # (L, M, M) or (L, D, D)

        # Normalize kernel matrices by their traces (layer-wise)
        traces = torch.einsum("lii->l", K)  # Trace per layer
        K /= traces.view(-1, 1, 1)  # Broadcasting over last two dims

        # Compute eigenvalues (batched)
        eigenvals = torch.linalg.eigvalsh(K)  # (L, M) or (L, D)

        # Keep only positive eigenvalues
        eigenvals = torch.where(
            eigenvals > 0, eigenvals, torch.tensor(0.0, device=eigenvals.device)
        )

        # Normalize eigenvalues to get probabilities
        probs = eigenvals / eigenvals.sum(dim=-1, keepdim=True)  # (L, M) or (L, D)

        if is_shannon:
            entropy = -torch.sum(
                probs * torch.log(probs + 1e-12), dim=-1
            )  # Avoid log(0)
        else:
            entropy = alpha_factor * torch.log(torch.sum(probs**alpha, dim=-1))

        # Normalize entropy
        log_min_dim = math.log(min(M, D))
        entropies = entropy / log_min_dim  # (L,)

    else:
        for layer in range(layers):
            # shape: (num_samples, hidden_size)
            Z_layer = Z_layers[layer]

            # Project each row to the unit sphere removing shared components in the direction of the mean
            Z_layer = normalize(Z_layer)

            M = Z_layer.shape[0]  # num_samples

            Z_layer = _ensure_device(Z_layer)

            if M < D:  # ! This is the most likely case for my experiments
                K = Z_layer @ Z_layer.T  # M x M
            else:
                K = Z_layer.T @ Z_layer  # D x D

            # Clamp negative eigenvalues due to numerical noise (due to numerical precision)
            K = torch.clamp(K, min=0.0)

            # Pre-normalize the kernel matrix -> surrogate Renyi entropy
            K /= torch.trace(K)

            # Compute eigenvalues
            # ! Different layers could have different number of eigenvalues
            eigenvals = torch.linalg.eigvalsh(K)
            eigenvals = eigenvals[eigenvals > 0]
            probs = eigenvals / eigenvals.sum()
            if is_shannon:
                entropy = -torch.sum(probs * torch.log(probs + 1e-12))  # Avoid log(0)
            else:
                entropy = alpha_factor * torch.log(torch.sum(probs**alpha))
            # maxEntropy normalization
            # the entropy of a uniform distribution over the eigenvalues is log(min(M, D))
            entropies[layer] = entropy / min(math.log(M), math.log(D))

    return entropies.cpu()


def compute_prompt_entropy_sequential(
    samples: List[torch.Tensor], alpha: float = 1.0
) -> torch.Tensor:
    """
    Compute matrix-based Rényi entropy for multiple samples sequentially,
    updating tqdm only every 5% of the total samples.

    Args:
        samples: List of tensors, each of shape (num_layers, seq_len_i, dim)
        alpha: Rényi entropy parameter

    Returns:
        Average entropy per layer across all samples
    """
    num_samples = len(samples)
    num_layers = samples[0].shape[0]
    layers_entropies = torch.zeros(num_samples, num_layers, dtype=torch.float64)

    update_every = max(1, int(0.1 * num_samples))  # Update every 10%
    progress_bar = tqdm(total=num_samples, desc="Processing samples", unit="sample")
    last_update = 0

    for idx, sample in enumerate(samples):
        result = compute_matrix_based_renyi_entropy(sample, alpha)
        layers_entropies[idx] = result

        # Update only every `update_every` steps or at the end
        if (idx + 1) % update_every == 0 or (idx + 1) == num_samples:
            progress_bar.update((idx + 1) - last_update)
            last_update = idx + 1

    progress_bar.close()
    return torch.mean(layers_entropies, dim=0)


def compute_layers_residual_stream_entropy(
    residual_stream: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
    entropy_type: str,
) -> torch.Tensor:
    if isinstance(residual_stream, torch.Tensor) and entropy_type == "prompt-entropy":
        raise ValueError(
            "Prompt entropy is not supported for a single tensor. Please provide a list of tensors."
        )
    elif (
        isinstance(residual_stream, list) or isinstance(residual_stream, dict)
    ) and entropy_type == "dataset-entropy":
        raise ValueError(
            "Dataset entropy is not supported for a list of tensors. Please provide a single tensor."
        )

    if isinstance(residual_stream, dict):
        residual_stream = list(residual_stream.values())

    if entropy_type == "prompt-entropy":
        # list of num_samples tensors of shape (num_layers, seq_len(idx), hidden_size)
        return compute_prompt_entropy_sequential(residual_stream)
        # shape: (num_layers,)
    elif entropy_type == "dataset-entropy":
        # shape: (num_layers, num_samples, hidden_size)
        return compute_matrix_based_renyi_entropy(
            residual_stream
        )  # shape: (num_layers,)
    else:
        raise ValueError(f"Invalid entropy type: {entropy_type}")


def compute_layers_residual_stream_similarities(
    residual_stream_network1: torch.Tensor,
    residual_stream_network2: torch.Tensor,
    similarity_measures: List[str],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Compute measures of the residual stream of two networks.

    Args:
        residual_stream_network1: Tensor of shape (num_layers, num_samples, hidden_size)
        residual_stream_network2: Tensor of shape (num_layers, num_samples, hidden_size)
        similarity_measures:
            - neighborhood_overlap
            - linear_cka
            - svcca
        **kwargs:
            - maxk: int = 30 for neighborhood_overlap
            - accept_rate: float = 0.95 for svcca

    Returns:
        Dict of tensors of measures of the residual stream of shape (num_layers,)
    """
    num_layers, num_samples, _ = residual_stream_network1.shape
    num_layers_2, num_samples_2, _ = residual_stream_network2.shape
    if num_layers != num_layers_2 or num_samples != num_samples_2:
        raise ValueError(
            "The number of layers and samples of the two networks must be the same"
        )
    residual_stream_measures = {}
    for measure in similarity_measures:
        residual_stream_measures[measure] = torch.zeros(num_layers, dtype=torch.float64)
        for layer in range(num_layers):
            residual_stream_measures[measure][layer] = compute_similarity(
                residual_stream_network1[layer],
                residual_stream_network2[layer],
                measure,
                **kwargs,
            )
    return residual_stream_measures


def _maybe_update_progress(
    progress_bar: tqdm,
    current_comparison: int,
    last_update: int,
    total_comparisons: int,
) -> int:
    update_every = max(1, int(0.1 * total_comparisons))
    if (
        current_comparison - last_update >= update_every
        or current_comparison == total_comparisons
    ):
        progress_bar.update(current_comparison - last_update)
        return current_comparison
    return last_update


def _precompute_linear_cka_states(
    network: torch.Tensor,
    device: Optional[Union[str, torch.device]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build reusable linear-CKA states for a stack of layers.

    Each layer becomes one centered Gram matrix ``K = X X^T`` flattened to a row
    vector. For two layers, linear CKA is then just
    ``dot(Kx, Ky) / (||Kx|| * ||Ky||)``. Preallocating ``states`` avoids the
    extra peak memory from building a Python list of large Gram tensors and then
    stacking it.
    """
    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    num_layers, num_samples, _ = network.shape
    states = torch.empty(
        num_layers,
        num_samples * num_samples,
        device=target_device,
        dtype=torch.float32,
    )
    norms = torch.empty(num_layers, device=target_device, dtype=torch.float32)

    for layer_idx, layer in enumerate(network):
        layer_tensor = _as_metric_tensor(layer, device=device)
        # Store each centered Gram matrix flattened so all pairwise CKA scores
        # reduce to one state matrix multiply when memory allows it.
        gram = _center_gram_inplace(layer_tensor @ layer_tensor.T)
        states[layer_idx].copy_(gram.reshape(-1))
        norms[layer_idx] = torch.sqrt(torch.sum(gram * gram))
        del layer_tensor, gram
    return states, norms


def _compute_cross_layer_linear_cka(
    residual_stream_network1: torch.Tensor,
    residual_stream_network2: torch.Tensor,
    lower_triangular_only: bool,
    total_comparisons: int,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    """Compute the cross-layer linear CKA matrix with bounded GPU memory.

    A single CKA state costs ``num_samples ** 2 * 4`` bytes. At N=9848 that is
    about 0.36 GiB, so caching all states for an 80x100 comparison would need
    about 65 GiB before scratch space. The strategy below keeps the exact same
    centered-Gram CKA formula but chooses the largest safe cache layout:
    full cache -> smaller-side cache -> paired row/column blocks.
    """
    num_layers_1 = residual_stream_network1.shape[0]
    num_layers_2 = residual_stream_network2.shape[0]
    num_samples = residual_stream_network1.shape[1]
    result = torch.zeros(num_layers_1, num_layers_2, dtype=torch.float64)

    with torch.no_grad():
        target_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        float32_bytes = torch.empty((), dtype=torch.float32).element_size()
        state_bytes = num_samples * num_samples * float32_bytes
        all_cache_bytes = (num_layers_1 + num_layers_2) * state_bytes

        # Keep the cache budget conservative: state construction also needs one
        # transient Gram matrix, and PyTorch matmul needs workspace/allocator slack.
        if target_device.type == "cuda" and torch.cuda.is_available():
            try:
                with torch.cuda.device(target_device):
                    free_bytes, _ = torch.cuda.mem_get_info()
                cache_budget_bytes = int(0.45 * free_bytes)
                block_budget_bytes = max(int(0.25 * free_bytes), 3 * state_bytes)
            except RuntimeError:
                cache_budget_bytes = 8 * (1024**3)
                block_budget_bytes = 2 * (1024**3)
        else:
            cache_budget_bytes = 8 * (1024**3)
            block_budget_bytes = 2 * (1024**3)

        # Full cache is fastest, but only safe if all states plus one scratch
        # Gram matrix fit inside the conservative cache budget.
        cache_all_states = all_cache_bytes + state_bytes <= cache_budget_bytes

        with tqdm(
            total=total_comparisons,
            desc="Computing linear_cka",
            unit="pair",
        ) as pbar:
            last_update = 0
            current_comparison = 0

            if cache_all_states:
                states1, norms1 = _precompute_linear_cka_states(
                    residual_stream_network1, device
                )
                states2, norms2 = _precompute_linear_cka_states(
                    residual_stream_network2, device
                )
                # Batched exact CKA: numerator is state dot product; denominator
                # is the product of centered-Gram Frobenius norms.
                scores = (states1 @ states2.T) / (norms1[:, None] * norms2[None, :])

                valid_scores = [
                    scores[i, j]
                    for i in range(num_layers_1)
                    for j in range(num_layers_2)
                    if not lower_triangular_only or i >= j
                ]
                if valid_scores:
                    checked = torch.stack(valid_scores)
                    tolerance = 1e-6
                    if not torch.all(
                        (checked >= -tolerance) & (checked <= 1 + tolerance)
                    ):
                        raise AssertionError("CKA value must be between 0 and 1.")

                scores_cpu = scores.to(device="cpu", dtype=torch.float64)
                for i in range(num_layers_1):
                    cols = [
                        j
                        for j in range(num_layers_2)
                        if not lower_triangular_only or i >= j
                    ]
                    if cols:
                        result[i, cols] = scores_cpu[i, cols]
                        current_comparison += len(cols)
                        last_update = _maybe_update_progress(
                            pbar,
                            current_comparison,
                            last_update,
                            total_comparisons,
                        )
                del states1, norms1, states2, norms2, scores, scores_cpu
            else:
                row_cache_bytes = num_layers_1 * state_bytes
                col_cache_bytes = num_layers_2 * state_bytes
                # Prefer caching the smaller axis once. The +2 states leave room
                # for a streamed block state and the transient Gram used to build it.
                cache_rows = (
                    num_layers_1 <= num_layers_2
                    and row_cache_bytes + 2 * state_bytes <= cache_budget_bytes
                )
                cache_cols = (
                    not cache_rows
                    and col_cache_bytes + 2 * state_bytes <= cache_budget_bytes
                )

                if cache_rows or cache_cols:
                    cached_bytes = row_cache_bytes if cache_rows else col_cache_bytes
                    remaining_bytes = max(
                        cache_budget_bytes - cached_bytes,
                        2 * state_bytes,
                    )
                    # One state is reserved as scratch while constructing the
                    # streamed block; the rest determines the block length.
                    states_per_block = (remaining_bytes - state_bytes) // max(
                        state_bytes, 1
                    )
                    stream_block_layers = max(1, min(16, int(states_per_block)))

                    cached_axis = "row" if cache_rows else "column"
                    streamed_axis = "column" if cache_rows else "row"
                    print(
                        "    - linear_cka full Gram cache needs "
                        f"{all_cache_bytes / (1024**3):.2f} GiB; using "
                        f"{cached_axis}-cache with {stream_block_layers}-layer "
                        f"{streamed_axis} blocks."
                    )

                if cache_rows:
                    # Cache the smaller row side once; stream column-layer states.
                    states1, norms1 = _precompute_linear_cka_states(
                        residual_stream_network1, device
                    )
                    for j0 in range(0, num_layers_2, stream_block_layers):
                        j1 = min(j0 + stream_block_layers, num_layers_2)
                        states2, norms2 = _precompute_linear_cka_states(
                            residual_stream_network2[j0:j1], device
                        )
                        scores = (states1 @ states2.T) / (
                            norms1[:, None] * norms2[None, :]
                        )

                        valid_block_scores = []
                        valid_count = 0
                        scores_cpu = scores.to(device="cpu", dtype=torch.float64)
                        for i in range(num_layers_1):
                            cols = []
                            local_cols = []
                            for local_j, j in enumerate(range(j0, j1)):
                                if lower_triangular_only and j > i:
                                    continue
                                cols.append(j)
                                local_cols.append(local_j)

                            if cols:
                                valid_block_scores.append(scores[i, local_cols])
                                result[i, cols] = scores_cpu[i, local_cols]
                                valid_count += len(cols)

                        if valid_block_scores:
                            checked = torch.cat(
                                [score.reshape(-1) for score in valid_block_scores]
                            )
                            tolerance = 1e-6
                            if not torch.all(
                                (checked >= -tolerance) & (checked <= 1 + tolerance)
                            ):
                                raise AssertionError(
                                    "CKA value must be between 0 and 1."
                                )

                            current_comparison += valid_count
                            last_update = _maybe_update_progress(
                                pbar,
                                current_comparison,
                                last_update,
                                total_comparisons,
                            )

                        del states2, norms2, scores, scores_cpu

                    del states1, norms1
                elif cache_cols:
                    # Symmetric case: cache columns once; stream row-layer states.
                    states2, norms2 = _precompute_linear_cka_states(
                        residual_stream_network2, device
                    )
                    for i0 in range(0, num_layers_1, stream_block_layers):
                        i1 = min(i0 + stream_block_layers, num_layers_1)
                        states1, norms1 = _precompute_linear_cka_states(
                            residual_stream_network1[i0:i1], device
                        )
                        scores = (states1 @ states2.T) / (
                            norms1[:, None] * norms2[None, :]
                        )

                        valid_block_scores = []
                        valid_count = 0
                        scores_cpu = scores.to(device="cpu", dtype=torch.float64)
                        for local_i, i in enumerate(range(i0, i1)):
                            cols = []
                            for j in range(num_layers_2):
                                if lower_triangular_only and j > i:
                                    continue
                                cols.append(j)

                            if cols:
                                valid_block_scores.append(scores[local_i, cols])
                                result[i, cols] = scores_cpu[local_i, cols]
                                valid_count += len(cols)

                        if valid_block_scores:
                            checked = torch.cat(
                                [score.reshape(-1) for score in valid_block_scores]
                            )
                            tolerance = 1e-6
                            if not torch.all(
                                (checked >= -tolerance) & (checked <= 1 + tolerance)
                            ):
                                raise AssertionError(
                                    "CKA value must be between 0 and 1."
                                )

                            current_comparison += valid_count
                            last_update = _maybe_update_progress(
                                pbar,
                                current_comparison,
                                last_update,
                                total_comparisons,
                            )

                        del states1, norms1, scores, scores_cpu

                    del states2, norms2
                else:
                    # Last-resort path for very tight memory: both sides are blocked.
                    # Reserve one scratch Gram, then split the remaining budget
                    # between row states and column states.
                    states_per_side = (block_budget_bytes - state_bytes) // max(
                        2 * state_bytes, 1
                    )
                    paired_block_layers = max(1, min(8, int(states_per_side)))

                    print(
                        "    - linear_cka full Gram cache needs "
                        f"{all_cache_bytes / (1024**3):.2f} GiB; using "
                        f"{paired_block_layers}-layer row/column blocks."
                    )

                    for i0 in range(0, num_layers_1, paired_block_layers):
                        i1 = min(i0 + paired_block_layers, num_layers_1)
                        states1, norms1 = _precompute_linear_cka_states(
                            residual_stream_network1[i0:i1], device
                        )

                        for j0 in range(0, num_layers_2, paired_block_layers):
                            j1 = min(j0 + paired_block_layers, num_layers_2)
                            states2, norms2 = _precompute_linear_cka_states(
                                residual_stream_network2[j0:j1], device
                            )
                            scores = (states1 @ states2.T) / (
                                norms1[:, None] * norms2[None, :]
                            )

                            valid_block_scores = []
                            valid_count = 0
                            scores_cpu = scores.to(device="cpu", dtype=torch.float64)
                            for local_i, i in enumerate(range(i0, i1)):
                                cols = []
                                local_cols = []
                                for local_j, j in enumerate(range(j0, j1)):
                                    if lower_triangular_only and j > i:
                                        continue
                                    cols.append(j)
                                    local_cols.append(local_j)

                                if cols:
                                    valid_block_scores.append(
                                        scores[local_i, local_cols]
                                    )
                                    result[i, cols] = scores_cpu[local_i, local_cols]
                                    valid_count += len(cols)

                            if valid_block_scores:
                                checked = torch.cat(
                                    [score.reshape(-1) for score in valid_block_scores]
                                )
                                tolerance = 1e-6
                                if not torch.all(
                                    (checked >= -tolerance)
                                    & (checked <= 1 + tolerance)
                                ):
                                    raise AssertionError(
                                        "CKA value must be between 0 and 1."
                                    )

                                current_comparison += valid_count
                                last_update = _maybe_update_progress(
                                    pbar,
                                    current_comparison,
                                    last_update,
                                    total_comparisons,
                                )

                            del states2, norms2, scores, scores_cpu

                        del states1, norms1

    return result


def _precompute_overlap_neighbors_if_duplicate_free(
    network: torch.Tensor,
    maxk: int,
    metric: str,
    device: Optional[Union[str, torch.device]],
) -> Optional[List[torch.Tensor]]:
    neighbors = []
    with torch.no_grad():
        for layer in network:
            layer_tensor = _as_metric_tensor(layer, device=device)
            # Duplicate rows must keep preprocess_data(remove_duplicates=True)
            # semantics, so the cached fast path is only valid when none exist.
            if _has_duplicate_rows_like_preprocess(layer_tensor):
                return None
            neighbors.append(
                _overlap_neighbor_indices(layer_tensor, maxk=maxk, metric=metric)
            )
    return neighbors


def _compute_cross_layer_neighborhood_overlap(
    residual_stream_network1: torch.Tensor,
    residual_stream_network2: torch.Tensor,
    lower_triangular_only: bool,
    total_comparisons: int,
    maxk: int,
    metric: str,
    device: Optional[Union[str, torch.device]],
) -> Optional[torch.Tensor]:
    num_layers_1 = residual_stream_network1.shape[0]
    num_layers_2 = residual_stream_network2.shape[0]
    neighbors1 = _precompute_overlap_neighbors_if_duplicate_free(
        residual_stream_network1,
        maxk=maxk,
        metric=metric,
        device=device,
    )
    if neighbors1 is None:
        return None
    neighbors2 = _precompute_overlap_neighbors_if_duplicate_free(
        residual_stream_network2,
        maxk=maxk,
        metric=metric,
        device=device,
    )
    if neighbors2 is None:
        return None

    result = torch.zeros(num_layers_1, num_layers_2, dtype=torch.float64)
    with torch.no_grad():
        with tqdm(total=total_comparisons, desc="Computing neighborhood_overlap", unit="pair") as pbar:
            last_update = 0
            current_comparison = 0
            for i in range(num_layers_1):
                cols = []
                vals = []
                for j in range(num_layers_2):
                    if lower_triangular_only and j > i:
                        continue
                    cols.append(j)
                    vals.append(
                        _neighborhood_overlap_from_indices(
                            neighbors1[i],
                            neighbors2[j],
                            maxk=maxk,
                        )
                    )
                if vals:
                    result[i, cols] = torch.stack(vals).to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    current_comparison += len(vals)
                    last_update = _maybe_update_progress(
                        pbar,
                        current_comparison,
                        last_update,
                        total_comparisons,
                    )

    return result


def _precompute_first_neighbors(
    network: torch.Tensor,
    metric: str,
    device: Optional[Union[str, torch.device]],
) -> List[torch.Tensor]:
    neighbors = []
    with torch.no_grad():
        for layer in network:
            layer_tensor = _as_metric_tensor(layer, device=device)
            neighbors.append(_first_neighbor_indices(layer_tensor, metric=metric))
    return neighbors


def _compute_cross_layer_information_imbalance(
    residual_stream_network1: torch.Tensor,
    residual_stream_network2: torch.Tensor,
    lower_triangular_only: bool,
    total_comparisons: int,
    metric: str,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    num_layers_1 = residual_stream_network1.shape[0]
    num_layers_2 = residual_stream_network2.shape[0]
    result = torch.zeros(num_layers_1, num_layers_2, dtype=torch.float64)
    source_neighbors = _precompute_first_neighbors(
        residual_stream_network1,
        metric=metric,
        device=device,
    )

    with torch.no_grad():
        with tqdm(total=total_comparisons, desc="Computing information_imbalance", unit="pair") as pbar:
            last_update = 0
            current_comparison = 0
            for j in range(num_layers_2):
                target_tensor = _as_metric_tensor(residual_stream_network2[j], device=device)
                target_distances = _information_distance_matrix(target_tensor, metric=metric)
                rows = []
                vals = []
                for i in range(num_layers_1):
                    if lower_triangular_only and j > i:
                        continue
                    rows.append(i)
                    vals.append(
                        _information_imbalance_from_cached(
                            source_neighbors[i],
                            target_distances,
                        )
                    )
                if vals:
                    result[rows, j] = torch.stack(vals).to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    current_comparison += len(vals)
                    last_update = _maybe_update_progress(
                        pbar,
                        current_comparison,
                        last_update,
                        total_comparisons,
                    )

    return result


def _compute_cross_layer_information_imbalance_reversed(
    residual_stream_network1: torch.Tensor,
    residual_stream_network2: torch.Tensor,
    lower_triangular_only: bool,
    total_comparisons: int,
    metric: str,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    num_layers_1 = residual_stream_network1.shape[0]
    num_layers_2 = residual_stream_network2.shape[0]
    result = torch.zeros(num_layers_1, num_layers_2, dtype=torch.float64)
    source_neighbors = _precompute_first_neighbors(
        residual_stream_network2,
        metric=metric,
        device=device,
    )

    with torch.no_grad():
        with tqdm(total=total_comparisons, desc="Computing information_imbalance_reversed", unit="pair") as pbar:
            last_update = 0
            current_comparison = 0
            for i in range(num_layers_1):
                target_tensor = _as_metric_tensor(residual_stream_network1[i], device=device)
                target_distances = _information_distance_matrix(target_tensor, metric=metric)
                cols = []
                vals = []
                for j in range(num_layers_2):
                    if lower_triangular_only and j > i:
                        continue
                    cols.append(j)
                    vals.append(
                        _information_imbalance_from_cached(
                            source_neighbors[j],
                            target_distances,
                        )
                    )
                if vals:
                    result[i, cols] = torch.stack(vals).to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    current_comparison += len(vals)
                    last_update = _maybe_update_progress(
                        pbar,
                        current_comparison,
                        last_update,
                        total_comparisons,
                    )

    return result


def compute_cross_layers_residual_stream_similarities(
    residual_stream_network1: torch.Tensor,
    residual_stream_network2: torch.Tensor,
    similarity_measures: List[str],
    lower_triangular_only: bool = False,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Compute measures of the residual stream of two networks, comparing every layer of network1
    with every layer of network2.

    Args:
        residual_stream_network1: Tensor of shape (num_layers_1, num_samples, hidden_size)
        residual_stream_network2: Tensor of shape (num_layers_2, num_samples, hidden_size)
        similarity_measures:
            - neighborhood_overlap
            - linear_cka
            - svcca
        lower_triangular_only: If True, compute only when row_idx > col_idx (i > j).
                               Useful for intra-model symmetric validation.
        **kwargs:
            - maxk: int = 30 for neighborhood_overlap
            - accept_rate: float = 0.95 for svcca

    Returns:
        Dict of tensors of measures of the residual stream of shape (num_layers_1, num_layers_2)
    """
    # print(f"    ℹ️  Computing similarity for pooling method: {tokens_pooling_method}") # Removed
    num_layers_1, num_samples, _ = residual_stream_network1.shape
    num_layers_2, num_samples_2, _ = residual_stream_network2.shape

    if num_samples != num_samples_2:
        raise ValueError("The number of samples of the two networks must be the same")

    residual_stream_measures = {}

    # Pre-calculate update interval for progress bar
    if lower_triangular_only:
        # If strict lower triangular (i >= j), valid pairs are i * (i+1) / 2 for symmetric NxN
        # Assumes num_layers_1 == num_layers_2 typically, but logic holds (pairs where i>=j)
        total_comparisons = 0
        for i in range(num_layers_1):
            for j in range(num_layers_2):
                if i >= j:
                    total_comparisons += 1
    else:
        total_comparisons = num_layers_1 * num_layers_2

    update_every = max(1, int(0.1 * total_comparisons)) if total_comparisons > 0 else 1

    for measure in similarity_measures:
        print(f"    - Computing {measure} matrix ({num_layers_1}x{num_layers_2})...")
        residual_stream_measures[measure] = torch.zeros(
            num_layers_1, num_layers_2, dtype=torch.float64
        )
        if total_comparisons == 0:
            continue
        device = kwargs.get("device", None)

        if measure == "linear_cka":
            residual_stream_measures[measure] = _compute_cross_layer_linear_cka(
                residual_stream_network1,
                residual_stream_network2,
                lower_triangular_only=lower_triangular_only,
                total_comparisons=total_comparisons,
                device=device,
            )
            continue

        if measure == "neighborhood_overlap" and not kwargs.get("use_dadapy", False):
            fast_overlap = _compute_cross_layer_neighborhood_overlap(
                residual_stream_network1,
                residual_stream_network2,
                lower_triangular_only=lower_triangular_only,
                total_comparisons=total_comparisons,
                maxk=kwargs.get("maxk", 30),
                metric=kwargs.get("metric", "euclidean"),
                device=device,
            )
            if fast_overlap is not None:
                residual_stream_measures[measure] = fast_overlap
                continue
            print(
                "    - Duplicate rows detected; falling back to pairwise "
                "neighborhood_overlap to preserve duplicate-removal behavior."
            )

        if measure == "information_imbalance":
            residual_stream_measures[measure] = _compute_cross_layer_information_imbalance(
                residual_stream_network1,
                residual_stream_network2,
                lower_triangular_only=lower_triangular_only,
                total_comparisons=total_comparisons,
                metric=kwargs.get("metric", "euclidean"),
                device=device,
            )
            continue

        if measure == "information_imbalance_reversed":
            residual_stream_measures[measure] = _compute_cross_layer_information_imbalance_reversed(
                residual_stream_network1,
                residual_stream_network2,
                lower_triangular_only=lower_triangular_only,
                total_comparisons=total_comparisons,
                metric=kwargs.get("metric", "euclidean"),
                device=device,
            )
            continue

        if measure == "information_imbalance_mean":
            forward = _compute_cross_layer_information_imbalance(
                residual_stream_network1,
                residual_stream_network2,
                lower_triangular_only=lower_triangular_only,
                total_comparisons=total_comparisons,
                metric=kwargs.get("metric", "euclidean"),
                device=device,
            )
            backward = _compute_cross_layer_information_imbalance_reversed(
                residual_stream_network1,
                residual_stream_network2,
                lower_triangular_only=lower_triangular_only,
                total_comparisons=total_comparisons,
                metric=kwargs.get("metric", "euclidean"),
                device=device,
            )
            residual_stream_measures[measure] = 0.5 * (forward + backward)
            continue

        # Precompute IDs if dealing with id_correlation to avoid redundant calculations
        precomputed_ids1 = None
        precomputed_ids2 = None
        if measure == "id_correlation":
            algorithm = kwargs.get("algorithm", "twoNN")
            k = kwargs.get("k", 100)
            fraction = kwargs.get("fraction", 0.9)

            # Helper for ID precomputation
            def _precompute_ids(network):
                ids = torch.zeros(network.shape[0], dtype=torch.float64)
                for idx in range(network.shape[0]):
                    data = network[idx]
                    data = _ensure_device(data)
                    data = standardize(data)
                    ids[idx] = compute_intrinsic_dimension(
                        data, algorithm=algorithm, k=k, fraction=fraction
                    )
                return ids

            precomputed_ids1 = _precompute_ids(residual_stream_network1)
            precomputed_ids2 = _precompute_ids(residual_stream_network2)

        progress_bar = tqdm(
            total=total_comparisons, desc=f"Computing {measure}", unit="pair"
        )
        last_update = 0
        current_comparison = 0

        for i in range(num_layers_1):
            for j in range(num_layers_2):
                # Optimization for Lower Triangular (Strict: i >= j)
                if lower_triangular_only and j > i:
                    continue

                current_comparison += 1

                # Inject precomputed IDs into kwargs if available
                current_kwargs = kwargs.copy()
                if precomputed_ids1 is not None:
                    current_kwargs["id1"] = precomputed_ids1[i].item()
                    current_kwargs["id2"] = precomputed_ids2[j].item()

                residual_stream_measures[measure][i, j] = compute_similarity(
                    residual_stream_network1[i],
                    residual_stream_network2[j],
                    measure,
                    **current_kwargs,
                )

                # Update progress bar
                if current_comparison % update_every == 0 or current_comparison == total_comparisons:
                    progress_bar.update(current_comparison - last_update)
                    last_update = current_comparison

        progress_bar.close()

    return residual_stream_measures


def compute_layers_intrinsic_dimension(
    residual_stream: torch.Tensor,
    algorithm: str = "scaling_gride",
    **kwargs,
) -> torch.Tensor:
    """
    Compute intrinsic dimension of the residual stream.
    """
    num_layers = residual_stream.shape[0]
    intrinsic_dimension = torch.zeros(num_layers, dtype=torch.float64)
    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc=f"Intrinsic dimension ({algorithm})", unit="layer") as pbar:
        for layer in range(num_layers):
            intrinsic_dimension[layer] = compute_intrinsic_dimension(
                residual_stream[layer], algorithm=algorithm, **kwargs
            )
            updates_pending += 1
            if (layer + 1) % update_every == 0 or (layer + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0
    return intrinsic_dimension


def compute_heads_projection_residual_stream_similarities(
    residual_stream_network1: torch.Tensor,
    residual_stream_network2: torch.Tensor,
    similarity_measures: List[str],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Compute measures of the heads projection residual stream of two networks.

    Args:
        residual_stream_network1: Tensor of shape (num_layers, num_samples, num_heads, hidden_size)
        residual_stream_network2: Tensor of shape (num_layers, num_samples, num_heads, hidden_size)
        similarity_measures:
            - neighborhood_overlap
            - linear_cka
            - svcca
        **kwargs:
            - maxk: int = 30 for neighborhood_overlap
            - accept_rate: float = 0.95 for svcca

    Returns:
        Dict of tensors of measures of the heads projection residual stream of shape (num_layers, num_heads)
    """
    num_layers, num_samples, num_heads, _ = residual_stream_network1.shape
    num_layers_2, num_samples_2, num_heads_2, _ = residual_stream_network2.shape
    if (
        num_layers != num_layers_2
        or num_samples != num_samples_2
        or num_heads != num_heads_2
    ):
        raise ValueError(
            "The number of layers, samples and heads of the two networks must be the same"
        )
    residual_stream_measures = {}
    for measure in similarity_measures:
        residual_stream_measures[measure] = torch.zeros(
            num_layers, num_heads, dtype=torch.float64
        )

    update_every = max(1, int(0.1 * num_heads))  # Update every 10%
    progress_bar = tqdm(
        total=num_heads, desc="Computing heads projection similarities", unit="head"
    )
    last_update = 0

    for h in range(num_heads):
        # Shape: (num_layers, num_samples, hidden_size)
        res_h1 = residual_stream_network1[:, :, h, :]
        res_h2 = residual_stream_network2[:, :, h, :]

        for measure in similarity_measures:
            for layer in range(num_layers):
                residual_stream_measures[measure][layer, h] = compute_similarity(
                    res_h1[layer],
                    res_h2[layer],
                    measure,
                    **kwargs,
                )

        if (h + 1) % update_every == 0 or (h + 1) == num_heads:
            progress_bar.update((h + 1) - last_update)
            last_update = h + 1

    progress_bar.close()
    return residual_stream_measures


def compute_consecutive_layers_similarities(
    residual_stream: torch.Tensor,
    similarity_measures: List[str],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Compute similarity measures between consecutive layers of a single network.

    Args:
        residual_stream: Tensor of shape (num_layers, num_samples, hidden_size)
        similarity_measures: List of similarity measures to compute
        **kwargs: Additional arguments for similarity computation

    Returns:
        Dict of tensors of shape (num_layers - 1,) containing similarities
        between layer i and layer i+1.
    """
    num_layers, num_samples, _ = residual_stream.shape
    if num_layers < 2:
        raise ValueError("Network must have at least 2 layers for consecutive comparison")

    residual_stream_measures = {}
    for measure in similarity_measures:
        residual_stream_measures[measure] = torch.zeros(
            num_layers - 1, dtype=torch.float64
        )
        for layer in range(num_layers - 1):
            residual_stream_measures[measure][layer] = compute_similarity(
                residual_stream[layer],
                residual_stream[layer + 1],
                measure,
                **kwargs,
            )
    return residual_stream_measures


_CLIP_SUPPORTED_MEASURES = frozenset([
    "neighborhood_overlap",
    "information_imbalance",
    "information_imbalance_reversed",
])


def compute_clip_residual_stream_similarities_per_layer(
    representations: torch.Tensor,
    clip_embeddings: torch.Tensor,
    similarity_measures: List[str],
    maxk: int = 30,
) -> Dict[str, torch.Tensor]:
    """Compare each decoder layer against a fixed 2-D CLIP embedding tensor.

    Unlike ``compute_cross_layers_residual_stream_similarities`` (which compares
    two 3-D tensors layer-by-layer), this function takes:
      - ``representations``: shape ``(num_layers, N, hidden_size)``
      - ``clip_embeddings``: shape ``(N, clip_dim)``   — already filtered to match N

    and returns one scalar per layer for each requested measure.

    Supported measures:
      - ``"neighborhood_overlap"``:          compute_neighborhood_overlap(repr[layer], clip)
      - ``"information_imbalance"``:         compute_information_imbalance(repr[layer], clip)
      - ``"information_imbalance_reversed"``: compute_information_imbalance(clip, repr[layer])

    Args:
        representations: Residual-stream tensor, shape (num_layers, N, hidden_size).
        clip_embeddings:  CLIP feature tensor, shape (N, clip_dim).
        similarity_measures: Subset of _CLIP_SUPPORTED_MEASURES.
        maxk: Neighbourhood size for neighborhood_overlap.

    Returns:
        Dict mapping measure name → 1-D float64 Tensor of shape (num_layers,).
    """
    unsupported = set(similarity_measures) - _CLIP_SUPPORTED_MEASURES
    if unsupported:
        raise ValueError(
            f"Unsupported CLIP measures: {unsupported}. "
            f"Supported: {_CLIP_SUPPORTED_MEASURES}"
        )

    num_layers = representations.shape[0]
    results: Dict[str, List[float]] = {m: [] for m in similarity_measures}

    for layer_idx in range(num_layers):
        layer_repr = representations[layer_idx]   # (N, hidden_size)
        for measure in similarity_measures:
            if measure == "neighborhood_overlap":
                val = compute_neighborhood_overlap(layer_repr, clip_embeddings, maxk=maxk)
            elif measure == "information_imbalance":
                val = compute_information_imbalance(layer_repr, clip_embeddings)
            elif measure == "information_imbalance_reversed":
                val = compute_information_imbalance(clip_embeddings, layer_repr)
            results[measure].append(float(val))

    return {m: torch.tensor(v, dtype=torch.float64) for m, v in results.items()}


@torch.no_grad()
def bootstrap_median_from_cos(
    cos: torch.Tensor, n_boot: int = 1000, sample_size: int = None, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bootstrap the per-layer median using only the (layers, samples) cosine matrix.

    Parameters
    ----------
    cos : (L, N)
    n_boot : number of bootstrap replicates
    sample_size : m in m-out-of-n bootstrap (defaults to N for classic bootstrap)
    seed : RNG seed

    Returns
    -------
    mean_of_medians : (L,)
    std_of_medians  : (L,)
    """
    if cos.ndim != 2:
        raise ValueError("cos must be 2D: (layers, samples)")

    L, N = cos.shape
    device = cos.device
    sample_size = N if sample_size is None else sample_size

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # (B, sample_size)
    idx = torch.randint(
        low=0, high=N, size=(n_boot, sample_size), generator=g, device=device
    )
    # Broadcast to (B, L, sample_size) and gather from (L, N):
    cos_expanded = cos.unsqueeze(0).expand(n_boot, L, N)  # (B, L, N)
    idx_expanded = idx.unsqueeze(1).expand(n_boot, L, sample_size)  # (B, L, S)
    sampled = torch.gather(cos_expanded, dim=2, index=idx_expanded)  # (B, L, S)

    boot_medians = sampled.median(dim=2).values  # (B, L)

    mean_of_medians = boot_medians.mean(dim=0)  # (L,)
    std_of_medians = boot_medians.std(dim=0, unbiased=True)  # (L,)

    return torch.stack([mean_of_medians, std_of_medians], dim=1)  # (L, 2)


@torch.no_grad()
def compute_layers_cosine_similarity(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    bootstrap: bool = False,
    n_boot: int = 1000,
) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors across layers and return median similarity per layer.

    Args:
        tensor1: First tensor of shape (layers, samples, hidden_dim)
        tensor2: Second tensor of shape (layers, samples, hidden_dim)
        bootstrap: Whether to bootstrap the median similarity per layer
        n_boot: Number of bootstrap replicates

    Returns:
        If bootstrap is False:
            Tensor of shape (layers,) containing median cosine similarity for each layer
        If bootstrap is True:
            Tensor of shape (layers, 2) containing the mean and std of the median cosine similarity for each layer
    """
    # Validate input shapes
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensors must have the same shape. Got {tensor1.shape} and {tensor2.shape}"
        )

    if len(tensor1.shape) != 3:
        raise ValueError(f"Expected 3D tensors, got shape {tensor1.shape}")

    # Compute cosine similarity using F.cosine_similarity
    # Shape: (layers, samples)
    cosine_similarities = F.cosine_similarity(tensor1, tensor2, dim=-1)

    # Compute median across samples for each layer
    # Shape: (layers,)
    if bootstrap:
        return bootstrap_median_from_cos(cosine_similarities, n_boot=n_boot)
    else:
        return torch.median(cosine_similarities, dim=1).values


def compute_layers_homogeneity_score(
    X_tensor: torch.Tensor,
    Y_tensor: torch.Tensor,
    metric: str = "euclidean",
    maxk: int = 128,
    range_max: int = 128,
    k: int = 16,
    Z: int = 1.65,
) -> dict:
    """
    Compute homogeneity score between two tensors across layers and return homogeneity score per layer.

    Args:
        X_tensor: First tensor of shape (layers, samples, hidden_dim)
        Y_tensor: Second tensor of shape (layers, samples, hidden_dim)

    Returns:
        Tensor of shape (layers,) containing homogeneity score for each layer
    """
    # Safety check
    if metric not in ["euclidean", "cosine"]:
        raise ValueError(f"Invalid metric: {metric}")

    # Validate input shapes
    if X_tensor.shape != Y_tensor.shape:
        raise ValueError(
            f"Tensors must have the same shape. Got {X_tensor.shape} and {Y_tensor.shape}"
        )

    if len(X_tensor.shape) != 3:
        raise ValueError(f"Expected 3D tensors, got shape {X_tensor.shape}")

    num_layers = X_tensor.shape[0]
    num_samples = X_tensor.shape[1]
    homogeneity_scores = torch.zeros(num_layers, dtype=torch.float64)

    with torch.no_grad():
        # Unify the data into the same sample space
        data_tensor = torch.cat([X_tensor, Y_tensor], dim=1)

        for layer in range(num_layers):
            layer_data = data_tensor[layer, :, :]
            # Preprocess data
            layer_data, _ = preprocess_data(
                layer_data,
                layer_data,
                output_format="tensor",
                remove_duplicates=False,
            )
            layer_data = _ensure_device(layer_data)
            layer_data = layer_data.to(torch.float64)

            # Compute distance matrix
            if metric == "cosine":
                layer_data = F.normalize(layer_data, dim=1)
            distances = torch.cdist(layer_data, layer_data, p=2).cpu().numpy()

            # Compute intrinsic dimension
            data_dadapy = Data(distances=distances, maxk=maxk)
            id_list, _, _ = data_dadapy.return_id_scaling_gride(range_max=range_max)
            id_index = int(math.log2(k))
            data_dadapy.set_id(id_list[id_index])
            data_dadapy.compute_density_kNN(k=k)
            clusters_AdvancedDensityPeak = data_dadapy.compute_clustering_ADP(Z=Z)

            # Create ground truth labels for the clustering
            ground_truth_labels = np.zeros(num_samples * 2)
            ground_truth_labels[:num_samples] = 1.0

            # Compute homogeneity score
            homogeneity_score, _, _ = homogeneity_completeness_v_measure(
                ground_truth_labels,
                clusters_AdvancedDensityPeak,
            )
            homogeneity_scores[layer] = homogeneity_score

    return homogeneity_scores


# ---------------------------------------------------------------------------
# Classification metrics — layer-wise probing
# ---------------------------------------------------------------------------


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


def compute_label_overlap(
    data: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    maxk: int = 30,
    use_dadapy: bool = True,
    metric: str = "euclidean",
    class_fraction: float | None = 0.3,
    weighted: bool = True,
) -> float:
    """Compute DADApy label overlap using GPU-computed nearest neighbours."""
    if not use_dadapy:
        warnings.warn(
            "compute_label_overlap now always delegates label-overlap scoring to DADApy.",
            stacklevel=2,
        )
    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported metric '{metric}'. Expected 'euclidean' or 'cosine'.")

    if isinstance(data, torch.Tensor):
        data_t = data.detach().to(device=DEVICE, dtype=torch.float32)
    else:
        data_t = torch.as_tensor(np.asarray(data, dtype=np.float32), device=DEVICE)

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    labels_np = labels_np.reshape(-1).astype(int)

    n_samples = data_t.shape[0]
    if labels_np.shape[0] != n_samples:
        raise ValueError(
            f"Data has {n_samples} samples, labels has {labels_np.shape[0]} samples"
        )
    if n_samples <= 1:
        return 0.0

    # Ask DADApy how many neighbours its class_fraction logic will need, then
    # hand the public scoring call a GPU-computed neighbour table of that width.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, required_k = Data()._label_imbalance_helper(labels_np, maxk, class_fraction)
    effective_k = min(int(required_k), n_samples - 1)
    if effective_k <= 0:
        return 0.0

    with torch.no_grad():
        if metric == "cosine":
            normed = F.normalize(data_t, p=2, dim=1)
            sim = normed @ normed.T
            sim.fill_diagonal_(-float("inf"))
            nn_sim, nn_idx = torch.topk(sim, effective_k, dim=1, largest=True)
            nn_dist = torch.clamp(1.0 - nn_sim, min=0.0, max=2.0)
        else:
            dist = torch.cdist(data_t, data_t)
            dist.fill_diagonal_(float("inf"))
            nn_dist, nn_idx = torch.topk(dist, effective_k, dim=1, largest=False)

    self_idx = torch.arange(n_samples, device=data_t.device).unsqueeze(1)
    self_dist = torch.zeros((n_samples, 1), dtype=nn_dist.dtype, device=data_t.device)
    dist_for_dadapy = torch.cat([self_dist, nn_dist], dim=1).cpu().numpy()
    idx_for_dadapy = torch.cat([self_idx, nn_idx], dim=1).cpu().numpy()

    d = Data(distances=(dist_for_dadapy, idx_for_dadapy), maxk=effective_k)
    overlap = d.return_label_overlap(
        labels_np,
        k=effective_k,
        class_fraction=class_fraction,
        weighted=weighted,
    )
    return float(overlap)


@timed_metric("knn")
def classify_per_layer_knn(
    representations: torch.Tensor,
    y: np.ndarray,
    n_neighbors: int = 30,
) -> dict[int, dict[str, float]]:
    num_layers, num_samples, _ = representations.shape
    use_stratify = bool(np.all(np.bincount(y) >= 2))
    layer_results: dict[int, dict[str, float]] = {}

    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y if use_stratify else None,
    )

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric knn", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            X_train_gpu = representations[layer_idx, train_idx].to(device=DEVICE, dtype=torch.float32)
            X_test_gpu = representations[layer_idx, test_idx].to(device=DEVICE, dtype=torch.float32)

            X_train_gpu = F.normalize(X_train_gpu, p=2, dim=1)
            X_test_gpu = F.normalize(X_test_gpu, p=2, dim=1)

            train_sim = torch.mm(X_train_gpu, X_train_gpu.t())
            test_sim = torch.mm(X_test_gpu, X_train_gpu.t())

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
            y_true = y[test_idx]

            layer_results[layer_idx] = {
                "knn_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "knn_mcc": float(matthews_corrcoef(y_true, y_pred)),
            }

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
    class_fraction: float | None = None,
) -> dict[int, dict[str, float]]:
    num_layers = representations.shape[0]
    layer_results: dict[int, dict[str, float]] = {}

    lo_kwargs = {} if class_fraction is None else {"class_fraction": class_fraction}
    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric label_overlap", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            overlap = compute_label_overlap(
                data=representations[layer_idx],
                labels=y,
                maxk=maxk,
                metric="euclidean",
                **lo_kwargs,
            )
            layer_results[layer_idx] = {"label_overlap": float(overlap)}

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results


@timed_metric("logistic_regression")
def classify_per_layer_logistic_regression(
    representations: torch.Tensor,
    y: np.ndarray,
) -> dict[int, dict[str, float]]:
    num_layers, num_samples, _ = representations.shape
    use_stratify = bool(np.all(np.bincount(y) >= 2))
    layer_results: dict[int, dict[str, float]] = {}

    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y if use_stratify else None,
    )

    if np.unique(y[train_idx]).shape[0] < 2:
        warnings.warn("Logistic regression requires at least 2 classes in train split.", stacklevel=2)
        return {i: {"f1_weighted": 0.0, "f1_weighted_train": 0.0, "mcc": 0.0} for i in range(num_layers)}

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric logistic_regression", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            X = representations[layer_idx].numpy()
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            n_train_samples, n_features = X_train.shape
            if n_train_samples < n_features:
                classifier = LogisticRegression(
                    penalty="l1", solver="liblinear", C=0.1, tol=1e-2,
                    max_iter=2000, class_weight="balanced", random_state=42,
                )
            else:
                classifier = LogisticRegression(
                    penalty=None, solver="lbfgs", max_iter=2000,
                    class_weight="balanced", random_state=42,
                )

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(X_train, y_train)
                if captured:
                    print(f"Warning: Layer {layer_idx} failed to converge.")

            y_pred = classifier.predict(X_test)
            y_pred_train = classifier.predict(X_train)
            layer_results[layer_idx] = {
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_weighted_train": float(f1_score(y_train, y_pred_train, average="weighted", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_test, y_pred)),
            }

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results


@timed_metric("homogeneity")
def homogeneity_score_per_layer(
    representations: torch.Tensor,
    y: np.ndarray,
    K: int = 16,
    Z: float = 1.65,
) -> dict[int, dict[str, float]]:
    num_layers, n_samples, _ = representations.shape
    layer_results: dict[int, dict[str, float]] = {}

    if n_samples <= K:
        warnings.warn(f"n_samples ({n_samples}) <= K ({K}). Returning 0.0 for all layers.", stacklevel=2)
        return {
            layer_idx: {"homogeneity_score": 0.0, "adjusted_rand_score": 0.0, "n_clusters_ratio": 0.0}
            for layer_idx in range(num_layers)
        }

    n_labels = len(np.unique(y))
    id_index = int(math.log2(K))

    update_every = max(1, int(0.1 * num_layers))
    updates_pending = 0
    with tqdm(total=num_layers, desc="Metric homogeneity", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            try:
                layer_data = F.normalize(
                    representations[layer_idx].to(DEVICE, dtype=torch.float32), p=2, dim=1
                )
                with torch.no_grad():
                    distances = torch.cdist(layer_data, layer_data).cpu().numpy().astype(np.float64)

                d = Data(distances=distances)
                d.distances = d.remove_zero_dists(d.distances)
                id_list, _, _ = d.return_id_scaling_gride()
                d.set_id(id_list[id_index])
                d.compute_density_kNN(k=K)
                cluster_labels = d.compute_clustering_ADP(Z=Z)

                score, _, _ = homogeneity_completeness_v_measure(y, cluster_labels)
                layer_results[layer_idx] = {
                    "homogeneity_score": float(score),
                    "adjusted_rand_score": float(adjusted_rand_score(y, cluster_labels)),
                    "n_clusters_ratio": float(d.N_clusters / n_labels),
                }
            except Exception as exc:
                print(f"Warning: Homogeneity failed at layer {layer_idx}: {exc}")
                layer_results[layer_idx] = {
                    "homogeneity_score": 0.0,
                    "adjusted_rand_score": 0.0,
                    "n_clusters_ratio": 0.0,
                }

            updates_pending += 1
            if (layer_idx + 1) % update_every == 0 or (layer_idx + 1) == num_layers:
                pbar.update(updates_pending)
                updates_pending = 0

    return layer_results
