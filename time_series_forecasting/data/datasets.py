import torch
from torch.utils.data import Dataset
import numpy as np


class PowerConsumptionDataset(Dataset):

    def __init__(self, data, sequence_length=24, forecast_horizon=24, target_column="NO1_consumption", sequence_features=["NO1_consumption", "NO1_temperature"], forecast_features=["NO1_temperature"]):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        self.sequence_features = sequence_features
        self.forecast_features = forecast_features

        # Create a mapping of feature names to their indices in sequence features and forecast features
        self.feature_indices = {
            feature: {
                'sequence_index': self.sequence_features.index(feature),
                'forecast_index': self.forecast_features.index(feature) if feature in self.forecast_features else None
            }
            for feature in self.sequence_features
        }

    def __len__(self):
        return len(self.data) - self.sequence_length - self.forecast_horizon - 1

    def __getitem__(self, idx):
        start_idx = idx
        end_sequence_idx = idx + self.sequence_length
        end_forecast_idx = end_sequence_idx + self.forecast_horizon

        # Assemble sequence features
        sequence_features_df = self.data[self.sequence_features].iloc[start_idx:end_sequence_idx]
        sequence_features = sequence_features_df.to_numpy(dtype=np.float32)

        # Forecast features for the forecast horizon
        forecast_features = self.data[self.forecast_features].iloc[end_sequence_idx:end_forecast_idx].to_numpy(dtype=np.float32)

        # Fetch the forecast horizon of target values immediately following the historical sequence
        targets = self.data[self.target_column].iloc[end_sequence_idx:end_forecast_idx].to_numpy(dtype=np.float32)

        # Retreive all timestamps to be used for forecasting plots
        timestamps = self.data["timestamp"].iloc[start_idx:end_forecast_idx].to_numpy(dtype=np.datetime64)
        # Convert timestamps to Unix epoch time (seconds) as int64
        timestamps_in_seconds = timestamps.astype('datetime64[s]').view('int64')

        return torch.from_numpy(sequence_features), torch.from_numpy(forecast_features), torch.from_numpy(targets), timestamps_in_seconds
