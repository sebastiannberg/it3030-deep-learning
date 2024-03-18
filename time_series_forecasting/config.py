import torch.nn as nn
import torch.optim as optim

from models.cnn_forecasting_model import CNNForecastingModel


global_config = {
    "csv_filename": "consumption_and_temperatures.csv",
    "model": CNNForecastingModel,
    "visualize": True,
    # Enable only one of TRAIN, LOAD or COMPARE
    "TRAIN": {
        "enabled": True,
        "sequence_length": 24,
        "forecast_horizon": 24,
        "bidding_area": "NO1",
        "optimizer": optim.Adam,
        "loss_function": nn.L1Loss,
        "lr": 0.0001,
        "epochs": 2,
        "save_model": True
    },
    "LOAD": {
        "enabled": False,
        "load_model_filename": "17-03-2024-164326_NO1_epoch_2.pt",
        "test_using_all_data": False,
        "test_bidding_area": "NO1"
    },
    "COMPARE": {
        "enabled": False,
        "compare_filenames": (
            ("17-03-2024-164326_NO1_epoch_2.pt", CNNForecastingModel),
            ("17-03-2024-164326_NO1_epoch_2.pt", CNNForecastingModel),
        )
    }
}
