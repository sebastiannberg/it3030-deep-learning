global_config = {
    "csv_filename": "consumption_and_temperatures.csv",
    "sequence_length": 24,
    "forecast_horizon": 1,
    "bidding_area": "NO1",
    "model": "cnn_forecasting_model",
    "load_model": False,
    "save_model": False,
    "train": True,
    "test": True
}

test_config = {
    "random_seed": 42
}

cnn_config = {
    "lr": 0.001,
    "epochs": 5
}
