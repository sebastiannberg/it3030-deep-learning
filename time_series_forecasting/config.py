import torch.nn as nn
import torch.optim as optim

from models.cnn_forecasting_model import CNNForecastingModel


global_config = {
    "csv_filename": "consumption_and_temperatures.csv",
    "model": CNNForecastingModel,
    "visualize": True,
    # Enable only one of TRAIN, LOAD or COMPARE
    "TRAIN": {
        "enabled": False,
        "bidding_area": "NO1",
        "sequence_length": 24,
        "forecast_horizon": 24,
        "optimizer": optim.Adam,
        "loss_function": nn.L1Loss,
        "lr": 0.0001,
        "epochs": 2,
        "save_model": True
    },
    "LOAD": {
        "enabled": False,
        "load_model_filename": "19-03-2024-150330_NO1_epoch_2.pt",
        "test_using_all_data": False,
        "test_bidding_area": "NO1"
    },
    "COMPARE": {
        "enabled": True,
        "compare_filenames": (
            ("19-03-2024-150330_NO1_epoch_2.pt", CNNForecastingModel),
            ("19-03-2024-170009_NO1_epoch_2.pt", CNNForecastingModel),
        ),
        "test_using_all_data": False,
        "test_bidding_area": "NO1"
    }
}
