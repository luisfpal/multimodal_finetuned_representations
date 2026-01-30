import math
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from dadapy.data import Data
from sklearn.metrics import homogeneity_completeness_v_measure

# from anatome.similarity import svcca_distance
from tqdm import tqdm


def _ensure_device(
    *tensors: torch.Tensor,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Move tensors to GPU if available, otherwise keep on CPU.

    Args:
        *tensors: Variable number of torch tensors

    Returns:
        Single tensor if one input, tuple of tensors if multiple inputs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved_tensors = tuple(tensor.to(device) for tensor in tensors)

    if len(moved_tensors) == 1:
        return moved_tensors[0]
    return moved_tensors


def _preprocess_data(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    output_format: str = "tensor",
    remove_duplicates: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Common preprocessing for similarity computation.

    Args:
        data1: First dataset
        data2: Second dataset
        output_format: 'tensor' or 'numpy'
        remove_duplicates: Whether to remove duplicates and match indices

    Returns:
        Tuple of preprocessed data1 and data2
    """
    # Ensure both datasets have the same number of samples
    if data1.shape[0] != data2.shape[0]:
        raise ValueError(
            f"Data1 has {data1.shape[0]} samples, Data2 has {data2.shape[0]} samples"
        )

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
        unique_data1, indices1 = np.unique(np_data1, axis=0, return_index=True)
        original_num_samples = np_data1.shape[0]
        unique_num_samples = unique_data1.shape[0]
        if unique_num_samples < original_num_samples:
            print(
                f"Warning: {original_num_samples - unique_num_samples} duplicate samples removed."
            )
        matched_data2 = np_data2[indices1]
    else:
        unique_data1 = np_data1
        matched_data2 = np_data2

    # Convert to requested output format
    if output_format == "tensor":
        unique_data1 = torch.tensor(unique_data1, dtype=torch.float32)
        matched_data2 = torch.tensor(matched_data2, dtype=torch.float32)
        unique_data1, matched_data2 = _ensure_device(unique_data1, matched_data2)
        return unique_data1, matched_data2
    else:
        return unique_data1, matched_data2


def compute_neighborhood_overlap(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    maxk: int = 30,
    use_dadapy: bool = False,
    metric: str = "euclidean",
) -> float:
    """
    Compute neighborhood overlap between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        maxk: Maximum number of nearest neighbors to consider
        use_dadapy: Whether to use dadapy for distance computation
        metric: Distance metric to use ('euclidean' or 'cosine')

    Returns:
        Overlap score between the two datasets

    Raises:
        ValueError: If the number of samples in data1 and data2 do not match.
    """
    if use_dadapy:
        # For dadapy, we need numpy arrays and should remove duplicates
        processed_data1, processed_data2 = _preprocess_data(
            data1, data2, output_format="numpy", remove_duplicates=True
        )
        d1 = Data(processed_data1)
        d1.compute_distances(maxk=maxk)
        return d1.return_data_overlap(processed_data2)

    # For manual computation, preprocess as tensors with duplicate removal
    data1, data2 = _preprocess_data(
        data1, data2, output_format="tensor", remove_duplicates=True
    )

    with torch.no_grad():
        X_tensor, Y_tensor = _ensure_device(data1, data2)
        X_tensor = X_tensor.to(torch.float32)
        Y_tensor = Y_tensor.to(torch.float32)

        if metric == "cosine":
            X_tensor = torch.nn.functional.normalize(X_tensor, dim=1)
            Y_tensor = torch.nn.functional.normalize(Y_tensor, dim=1)
            x_neighbors = torch.topk(X_tensor @ X_tensor.T, maxk + 1, dim=1).indices[
                :, 1:
            ]
            y_neighbors = torch.topk(Y_tensor @ Y_tensor.T, maxk + 1, dim=1).indices[
                :, 1:
            ]
        else:
            x_neighbors = torch.topk(
                torch.cdist(X_tensor, X_tensor), maxk + 1, dim=1, largest=False
            ).indices[:, 1:]
            y_neighbors = torch.topk(
                torch.cdist(Y_tensor, Y_tensor), maxk + 1, dim=1, largest=False
            ).indices[:, 1:]

        N = x_neighbors.shape[0]
        if X_tensor.device.type == "cuda":
            indicator_X = torch.zeros((N, N), dtype=torch.int16, device=X_tensor.device)
            indicator_Y = torch.zeros((N, N), dtype=torch.int16, device=X_tensor.device)
            indicator_X.scatter_(dim=1, index=x_neighbors, value=1)
            indicator_Y.scatter_(dim=1, index=y_neighbors, value=1)
            return (indicator_X * indicator_Y).sum().item() / (maxk * N)

        x_neighbors = x_neighbors.cpu()
        y_neighbors = y_neighbors.cpu()
        overlap_sum = 0.0
        for i in range(N):
            xi = set(x_neighbors[i].tolist())
            yi = set(y_neighbors[i].tolist())
            overlap_sum += len(xi & yi)

        return overlap_sum / (maxk * N)


def compute_information_imbalance(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    metric: str = "euclidean",
) -> float:
    """
    Compute Information Imbalance from data1 to data2.
    Measures how well distances in data1 predict neighbors in data2.

    Args:
        data1: First dataset/representation (Space A)
        data2: Second dataset/representation (Space B)
        metric: Distance metric 'euclidean' or 'cosine'

    Returns:
        Information Imbalance score (lower is better)
    """
    # Preprocess and move to GPU if available
    # We keep duplicates to preserve the exact structure of the representation
    data1, data2 = _preprocess_data(
        data1, data2, output_format="tensor", remove_duplicates=False
    )
    data1, data2 = _ensure_device(data1, data2)

    N = data1.shape[0]

    with torch.no_grad():
        if metric == "cosine":
            data1 = F.normalize(data1, dim=1)
            data2 = F.normalize(data2, dim=1)
            # Euclidean distance on normalized vectors is monotonic with cosine distance

        # 1. Find Nearest Neighbors in Space A (data1)
        # Distances in A
        d1 = torch.cdist(data1, data1)

        # Get index of first NN (k=2, index 1 because index 0 is self)
        # largest=False -> smallest distances
        _, indices_1 = torch.topk(d1, k=2, dim=1, largest=False)
        nn_indices_1 = indices_1[:, 1]  # shape (N,)

        del d1  # Save memory

        # 2. Compute Ranks in Space B (data2)
        d2 = torch.cdist(data2, data2)

        # Extract distances to the specific neighbors identified in A
        # d2[i, nn_indices_1[i]]
        row_indices = torch.arange(N, device=d2.device)
        dists_to_nn = d2[row_indices, nn_indices_1].unsqueeze(1)  # shape (N, 1)

        # Count how many points in B are strictly closer than this distance
        # This corresponds to the rank (0-based)
        ranks = (d2 < dists_to_nn).sum(dim=1).float()

        # 3. Compute Imbalance
        # The factor 2/N normalizes the rank (which averages N/2 for random) to 1.
        imb = 2 * ranks.mean() / N

        return imb.item()


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
    latent, control = _preprocess_data(
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
#     X, Y = _preprocess_data(X, Y, output_format="tensor", remove_duplicates=False)

#     # Ensure tensors are on GPU if available
#     X, Y = _ensure_device(X, Y)

#     similarity = 1 - svcca_distance(X, Y, accept_rate=accept_rate, backend="svd")
#     return similarity.item()


def compute_linear_cka_similarity(
    X: Union[np.ndarray, torch.Tensor], Y: Union[np.ndarray, torch.Tensor]
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
    X, Y = _preprocess_data(X, Y, output_format="tensor", remove_duplicates=False)

    # Ensure tensors are on GPU if available
    X, Y = _ensure_device(X, Y)

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
    X, Y = _preprocess_data(X, Y, output_format="tensor", remove_duplicates=False)

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
    L_X = X @ X.T
    L_Y = Y @ Y.T

    return torch.sum(center_kernel_matrix(L_X) * center_kernel_matrix(L_Y))


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
    data1, data2 = _preprocess_data(
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
    }

    if measure not in available_measures:
        raise ValueError(
            f"Unknown measure '{measure}'. Available measures: {available_measures}"
        )
        
    if measure == "neighborhood_overlap":
        maxk = kwargs.get("maxk", 30)
        use_dadapy = kwargs.get("use_dadapy", False)
        metric = kwargs.get("metric", "euclidean")
        return compute_neighborhood_overlap(data1, data2, maxk, use_dadapy, metric)

    elif measure == "information_imbalance":
        metric = kwargs.get("metric", "euclidean")
        return compute_information_imbalance(data1, data2, metric)

    elif measure == "information_imbalance_reversed":
        metric = kwargs.get("metric", "euclidean")
        return compute_information_imbalance(data2, data1, metric)

    elif measure == "information_imbalance_mean":
        metric = kwargs.get("metric", "euclidean")
        return 0.5 * (
            compute_information_imbalance(data1, data2, metric)
            + compute_information_imbalance(data2, data1, metric)
        )

    elif measure == "distance_correlation":
        return compute_distance_correlation(data1, data2)

    # elif measure == "svcca":
    #     accept_rate = kwargs.get("accept_rate", 0.95)
    #     return compute_svcca_similarity(data1, data2, accept_rate)

    elif measure == "linear_cka":
        return compute_linear_cka_similarity(data1, data2)

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
        X = X.cpu().numpy()
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
    X: torch.Tensor, k: int = 16, range_max: int = 100
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
    for layer in range(num_layers):
        id_layer = compute_intrinsic_dimension(
            residual_stream[layer], algorithm=algorithm, **kwargs
        )
        intrinsic_dimension[layer] = id_layer
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
            layer_data, _ = _preprocess_data(
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
