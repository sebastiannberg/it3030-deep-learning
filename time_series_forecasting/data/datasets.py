import torch
from torch.utils.data import Dataset
import numpy as np


class PowerConsumptionDataset(Dataset):

    def __init__(self, data, sequence_length=24, forecast_horizon=24, bidding_area="NO1"):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.bidding_area = bidding_area
        self.temperature_col = f"{bidding_area}_temperature"
        self.consumption_col = f"{bidding_area}_consumption"

    def __len__(self):
        # TODO double check this method
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx):
        start_idx = idx
        end_sequence_idx = idx + self.sequence_length
        end_forecast_idx = end_sequence_idx + self.forecast_horizon

        historical_temps = self.data[self.temperature_col].iloc[start_idx:end_sequence_idx].to_numpy(dtype=np.float32)
        historical_consumption = self.data[self.consumption_col].iloc[start_idx:end_sequence_idx].to_numpy(dtype=np.float32)
        # Features stacked horizontally (time steps x features)
        features = np.hstack([historical_temps.reshape(-1, 1), historical_consumption.reshape(-1, 1)])

        # Forecast proxy temperatures fore the forecast horizon
        forecast_proxy_temps = self.data[self.temperature_col].iloc[end_sequence_idx:end_forecast_idx].to_numpy(dtype=np.float32).reshape(-1, 1)

        # Fetch the forecast horizon of target values immediately following the historical sequence
        targets = self.data[self.consumption_col].iloc[end_sequence_idx:end_forecast_idx].to_numpy(dtype=np.float32)

        return torch.from_numpy(features), torch.from_numpy(forecast_proxy_temps), torch.from_numpy(targets)
