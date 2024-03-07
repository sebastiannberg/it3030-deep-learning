import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PowerConsumptionDataset(Dataset):

    def __init__(self, csv_file, sequence_length=24, forecast_horizon=24, bidding_area="NO1", mode="train"):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.bidding_area = bidding_area
        self.mode = mode
        self.temperature_col = f"{bidding_area}_temperature"
        self.consumption_col = f"{bidding_area}_consumption"

    def __len__(self):
        # TODO double check this method
        if self.mode == "train":
            # In "n in, 1 out" training, ensuring each sequence has a single next step as the target
            return len(self.data) - self.sequence_length - 1
        elif self.mode == "test":
            return len(self.data) - self.sequence_length
        else:
            raise ValueError(f"{self.mode} not supported in dataset")

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length

        historical_temps = self.data[self.temperature_col][start_idx:end_idx].to_numpy(dtype=np.float32)
        historical_consumption = self.data[self.consumption_col][start_idx:end_idx].to_numpy(dtype=np.float32)
        # Features stacked horizontally (time steps x features)
        features = np.hstack([historical_temps.reshape(-1, 1), historical_consumption.reshape(-1, 1)])

        if self.mode == "train":
            # Fetch the single target value immediately following the historical sequence
            target = np.array(self.data[self.consumption_col][end_idx], dtype=np.float32).reshape(1)
        elif self.mode == "test":
            # Fetch the next sequence of target values of length forecast_horizon
            targets_start_idx = end_idx
            targets_end_idx = targets_start_idx + self.forecast_horizon
            target = self.data[self.consumption_col][targets_start_idx:targets_end_idx].to_numpy(dtype=np.float32)

        return torch.from_numpy(features), torch.from_numpy(target)
