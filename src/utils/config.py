import os
import random
import numpy as np
import torch

SEED = int(os.getenv("CDT_SEED", 42))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
