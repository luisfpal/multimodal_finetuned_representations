"""Provides a seed_all function that sets random seeds for reproducibility across Python, NumPy, PyTorch, and Hugging Face, and configures deterministic PyTorch behavior."""

import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed


def seed_all(seed: int):
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """

    # 1. Python built-in random module
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    # 4. Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # must be False for reproducibility

    # 5. Hugging Face
    hf_set_seed(seed)
