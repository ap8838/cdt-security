import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (GPU, if available)

    # Ensure deterministic behavior in cudnn (for CNNs etc.)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Also propagate to env
    os.environ["PYTHONHASHSEED"] = str(seed)
