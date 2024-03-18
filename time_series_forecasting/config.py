import torch.nn as nn
import torch.optim as optim

from models.cnn_forecasting_model import CNNForecastingModel


global_config = {
    "csv_filename": "consumption_and_temperatures.csv",
    "sequence_length": 24,
    "forecast_horizon": 24,
    "bidding_area": "NO1",
    "model": CNNForecastingModel,
    "load_model": False,
    "load_model_filename": "17-03-2024-164326_NO1_epoch_2.pt",
    "save_model": True,
    "train": True,
    "test": True,
    # TODO implement test_using_all_data # Set train to False if used
    "visualize": True,
    "compare": True,
    "compare_filenames": (
        ("17-03-2024-164326_NO1_epoch_2.pt", CNNForecastingModel),
        ("17-03-2024-164326_NO1_epoch_2.pt", CNNForecastingModel),
    )
}

cnn_config = {
    "model": CNNForecastingModel,
    "optimizer": optim.Adam,
    "loss_function": nn.L1Loss,
    "lr": 0.0001,
    "epochs": 2
}
