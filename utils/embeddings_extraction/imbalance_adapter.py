"""Thin adapter over the shared imbalance math in utils.metrics_utils.

compute_id_imbalance.py and friends were written against Diego's original
utils.imbalance_utils API (compute_imbalance_torch returning both directions
in one call). utils.metrics_utils.compute_information_imbalance implements
the identical algorithm (verified: same torch.topk(k=2)/rank-count/2*mean/N
formula) but only computes one direction per call, so this wraps it to match
the original two-direction call signature instead of duplicating the math.

_return_ranks/_return_imbalance operate on precomputed dadapy neighbour-index
arrays (a different calling convention from metrics_utils.py's tensor-based
API) and have no equivalent there, so they are kept as-is.
"""

import numpy as np

from utils.metrics_utils import compute_information_imbalance


def compute_imbalance_torch(x1, x2, exact_rank=False):
    imb_1_to_2 = compute_information_imbalance(x1, x2)
    imb_2_to_1 = compute_information_imbalance(x2, x1)
    return imb_1_to_2, imb_2_to_1


def _return_ranks(dist_indices_1, dist_indices_2, rng, k=1):
    """Ranks according to distance 2 of the neighbours according to distance 1."""
    N = dist_indices_1.shape[0]
    maxk_2 = dist_indices_2.shape[1]

    conditional_ranks = np.zeros((N, k))

    for i in range(N):
        idx_k_d1 = dist_indices_1[i, 1 : k + 1]

        wr = [
            np.where(idx_k_d1[k_neighbor] == dist_indices_2[i])[0]
            for k_neighbor in range(k)
        ]

        for k_neighbor in range(k):
            if len(wr[k_neighbor]) == 0:
                conditional_ranks[i, k_neighbor] = rng.integers(
                    low=maxk_2, high=N, size=1
                )
            else:
                conditional_ranks[i, k_neighbor] = wr[k_neighbor][0]

    return conditional_ranks


def _return_imbalance(dist_indices_1, dist_indices_2, rng, k=1):
    """Information imbalance between two precomputed dadapy distance-index arrays."""
    assert dist_indices_1.shape[0] == dist_indices_2.shape[0]

    N = dist_indices_1.shape[0]
    ranks = _return_ranks(dist_indices_1, dist_indices_2, rng=rng, k=k)
    return np.mean(ranks) / (N / 2.0)
