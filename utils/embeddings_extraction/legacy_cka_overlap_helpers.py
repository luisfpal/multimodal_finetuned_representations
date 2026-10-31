import numpy as np
import torch
import torch.nn.functional as F


def _to_gpu(x):
    """Convert numpy array or CPU tensor to float32 CUDA tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().cuda()
    return x.float().cuda()


@torch.inference_mode()
def compute_overlap_torch(X1, X2, ks=(10,), normalize=False):
    """Mean per-sample fraction of X1 k-NN ids that appear among X2 k-NN ids, for each k in ks."""
    if isinstance(ks, int):
        ks = (ks,)

    X1 = _to_gpu(X1)
    X2 = _to_gpu(X2)

    if normalize:
        X1 = F.normalize(X1, p=2, dim=1)
        X2 = F.normalize(X2, p=2, dim=1)

    d1 = torch.cdist(X1, X1)
    del X1
    torch.cuda.empty_cache()

    d2 = torch.cdist(X2, X2)
    del X2
    torch.cuda.empty_cache()

    ovs = []
    for k_ in ks:
        _, indices_1 = torch.topk(d1, k=k_ + 1, dim=1, largest=False)
        base_neighbors = indices_1[:, 1 : k_ + 1]

        _, indices_2 = torch.topk(d2, k=k_ + 1, dim=1, largest=False)
        other_neighbors = indices_2[:, 1 : k_ + 1]

        matches = base_neighbors.unsqueeze(2) == other_neighbors.unsqueeze(1)
        overlaps_row = matches.any(dim=2).sum(dim=1).float() / k_
        ovs.append(overlaps_row.mean().item())

    return ovs


@torch.inference_mode()
def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel on GPU.

    Args:
      x: A num_examples x num_features matrix (numpy array or torch tensor).

    Returns:
      A symmetric num_examples x num_examples CUDA tensor.
    """
    x = _to_gpu(x)
    gram = x @ x.T
    return (gram + gram.T) / 2


@torch.inference_mode()
def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel on GPU.

    Args:
      x: A num_examples x num_features matrix (numpy array or torch tensor).
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth.

    Returns:
      A symmetric num_examples x num_examples CUDA tensor.
    """
    x = _to_gpu(x)
    dot_products = x @ x.T
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_distances = (sq_distances + sq_distances.T) / 2
    sq_distances.clamp_(min=0.0)
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


@torch.inference_mode()
def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix on GPU.

    This is equivalent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric CUDA tensor.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric CUDA tensor (float64) with centered columns and rows.
    """
    gram = gram.double().clone()

    if unbiased:
        n = gram.shape[0]
        gram.fill_diagonal_(0)
        means = gram.sum(0) / (n - 2)
        means -= means.sum() / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)
    else:
        means = gram.mean(0)
        means -= means.mean() / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


@torch.inference_mode()
def cka(gram_x, gram_y, debiased=False):
    """Compute CKA on GPU.

    Args:
      gram_x: A num_examples x num_examples Gram CUDA tensor.
      gram_y: A num_examples x num_examples Gram CUDA tensor.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      CKA value as a Python float.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    scaled_hsic = (gram_x * gram_y).sum()
    normalization_x = torch.linalg.norm(gram_x)
    normalization_y = torch.linalg.norm(gram_y)
    return (scaled_hsic / (normalization_x * normalization_y)).item()


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


@torch.inference_mode()
def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space, on GPU.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
      features_x: A num_examples x num_features matrix (numpy array or torch tensor).
      features_y: A num_examples x num_features matrix (numpy array or torch tensor).
      debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
      CKA value as a Python float.
    """
    features_x = _to_gpu(features_x)
    features_y = _to_gpu(features_y)

    features_x = features_x - features_x.mean(0, keepdim=True)
    features_y = features_y - features_y.mean(0, keepdim=True)

    dot_product_similarity = torch.linalg.norm(features_x.T @ features_y) ** 2
    normalization_x = torch.linalg.norm(features_x.T @ features_x)
    normalization_y = torch.linalg.norm(features_y.T @ features_y)

    if debiased:
        n = features_x.shape[0]
        sum_squared_rows_x = torch.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = torch.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = sum_squared_rows_x.sum()
        squared_norm_y = sum_squared_rows_y.sum()

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = torch.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = torch.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return (dot_product_similarity / (normalization_x * normalization_y)).item()
