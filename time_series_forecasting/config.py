import torch.nn as nn
import torch.optim as optim

from models.cnn_forecasting_model import CNNForecastingModel


global_config = {
    "csv_filename": "consumption_and_temperatures.csv",
    "sequence_length": 24,
    "forecast_horizon": 24,
    "bidding_area": "NO1",
    "model": "cnn_forecasting_model",
    "load_model": False,
    "load_model_filename": "14-03-2024-192257_epoch_1.pt",
    "save_model": True,
    "train": True,
    "test": True,
    "visualize": True
}

cnn_config = {
    "model": CNNForecastingModel,
    "optimizer": optim.Adam,
    "loss_function": nn.L1Loss,
    "lr": 0.0001,
    "epochs": 1
}
