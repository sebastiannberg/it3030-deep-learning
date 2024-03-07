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
    "save_model": False,
    "train": True,
    "test": True,
    "visualize": False
}

test_config = {
    "random_seed": 42
}

cnn_config = {
    "model": CNNForecastingModel,
    "optimizer": optim.Adam,
    "loss_function": nn.MSELoss,
    "lr": 0.001,
    "epochs": 5
}
