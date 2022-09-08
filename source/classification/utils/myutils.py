import os
import numpy as np
import torch
import random


def seed_everything(seed_value):

    # set python seed
    random.seed(seed_value)

    # seed the global NumPy RNG
    np.random.seed(seed_value)

    # seed the RNG for all devices (both CPU and CUDA):
    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
