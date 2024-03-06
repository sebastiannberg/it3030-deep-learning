import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import os

from data.datasets import PowerConsumptionDataset

def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

def main():
    set_random_seeds(seed_value=42)
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "raw", "consumption_and_temperatures.csv")
    train_dataset_no1 = PowerConsumptionDataset(
        csv_file=dataset_path,
        sequence_length=24,
        forecast_horizon=24,
        bidding_area="NO1",
        mode='train'
    )
    train_dataloader_no1 = DataLoader(train_dataset_no1, batch_size=32, shuffle=False)
    train_features, train_targets = next(iter(train_dataloader_no1))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_targets.size()}")


if __name__ == "__main__":
    main()