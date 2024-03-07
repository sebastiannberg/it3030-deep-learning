import numpy as np
import random
import torch

from config import test_config


def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

def main():
    set_random_seeds(seed_value=test_config["random_seed"])

if __name__ == "__main__":
    main()