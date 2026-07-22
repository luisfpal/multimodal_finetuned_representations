import matplotlib.pyplot as plt
import numpy as np
from dadapy import Data
from safetensors import safe_open


def remove_close_rows(x):
    from sklearn.metrics import pairwise_distances

    x_np = x.float().cpu().numpy()
    dist = pairwise_distances(x_np, x_np, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    thresh = 100 * np.finfo(x_np.dtype).eps
    close = dist <= thresh
    keep = ~np.triu(close, 1).any(axis=0)
    return x[keep]


def _load_safetensor_layers(filepath):
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
    layer_dict = {i: tensor[i] for i in range(num_layers)}
    return layer_dict, num_layers


representations, num_layers = _load_safetensor_layers(
    "results/representations/representations.safetensors",
)


def compute_id_gride_multiscale(mus):
    # array of ids (as a function of the average distance to a point)
    ids_scaling = np.zeros(mus.shape[1])
    # array of error estimates (via fisher information)
    ids_scaling_err = np.zeros(mus.shape[1])

    for i in range(mus.shape[1]):
        n1 = 2**i

        mus_ = mus[:, i]
        indx = np.nonzero(mus_ == 1)

        mus[indx] += 10 * np.finfo(mus_.dtype).eps

        q3, q2 = np.percentile(mus_, [95, 50])
        mu_max = (
            5 * (q3 - q2) + q2
        )  # very generous threshold: to accomodate fat tailed distributions
        select = (
            mus_ < mu_max
        )  # remove high mu values related very likely to overlapping datapoints
        mus_ = mus_[select]

        intrinsic_dim, id_error = data._compute_id_gride_single_scale(
            0.001, 1000, mus_, n1, 2 * n1, 1e-7
        )

        ids_scaling[i] = intrinsic_dim
        ids_scaling_err[i] = id_error

    return ids_scaling


ids = []
for i in range(0, len(representations)):
    print(f"layer {i}")
    X = representations[i]
    n_before = X.shape[0]
    X = remove_close_rows(X)
    print(f"  removed {n_before - X.shape[0]} near-duplicate rows")
    data = Data(X.float().cpu().numpy())

    distances, dist_indices, mus, rs = data._return_mus_scaling(range_scaling=550)
    ids_scaling = compute_id_gride_multiscale(mus)
    ids.append(ids_scaling)


id_arr = np.array(ids)
for i in [5, 6]:
    plt.plot(id_arr[:, i], label=f"k={2**i}-{2**(i+1)}")
plt.legend()
