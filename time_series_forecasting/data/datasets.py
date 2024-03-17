import torch
from torch.utils.data import Dataset
import numpy as np


class PowerConsumptionDataset(Dataset):

    def __init__(self, data, sequence_length=24, forecast_horizon=24, target_column="NO1_consumption", temperature_column="NO1_temperature"):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        self.temperature_column = temperature_column

    def __len__(self):
        return len(self.data) - self.sequence_length - self.forecast_horizon - 1

    def __getitem__(self, idx):
        start_idx = idx
        end_sequence_idx = idx + self.sequence_length
        end_forecast_idx = end_sequence_idx + self.forecast_horizon

        # Assemble features
        feature_columns = [self.target_column, self.temperature_column] + [col for col in self.data.columns if col not in ["timestamp", self.target_column, self.temperature_column]]
        features_df = self.data[feature_columns].iloc[start_idx:end_sequence_idx]
        features = features_df.to_numpy(dtype=np.float32)

        # Forecast features for the forecast horizon
        forecast_columns = [self.temperature_column] + [col for col in self.data.columns if col not in ["timestamp", self.target_column, self.temperature_column]]
        forecast_features = self.data[forecast_columns].iloc[end_sequence_idx:end_forecast_idx].to_numpy(dtype=np.float32)

        # Fetch the forecast horizon of target values immediately following the historical sequence
        targets = self.data[self.target_column].iloc[end_sequence_idx:end_forecast_idx].to_numpy(dtype=np.float32)

        # Retreive all timestamps to be used for forecasting plots
        timestamps = self.data["timestamp"].iloc[start_idx:end_forecast_idx].to_numpy(dtype=np.datetime64)
        # Convert timestamps to Unix epoch time (seconds) as int64
        timestamps_in_seconds = timestamps.astype('datetime64[s]').view('int64')

        return torch.from_numpy(features), torch.from_numpy(forecast_features), torch.from_numpy(targets), timestamps_in_seconds
