import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import time
import os

from config import global_config, cnn_config
from data.datasets import PowerConsumptionDataset

def setup():
    """
    Setup selected model and create datasets based on the configuration file
    """
    print("\033[1;32m" + "="*15 + " Setup " + "="*15 + "\033[0m")

    if global_config["load_model"]:
        raise NotImplementedError()

    if global_config["save_model"]:
        raise NotImplementedError()

    if global_config["visualize"]:
        raise NotImplementedError()

    # Load data
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "raw", global_config["csv_filename"])
    print(f"Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    dataset = PowerConsumptionDataset(
        data=df,
        sequence_length=global_config["sequence_length"],
        forecast_horizon=global_config["forecast_horizon"],
        bidding_area=global_config["bidding_area"],
    )

    train_size = int(0.7 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size

    train_dataset = Subset(dataset, range(train_size))
    validation_dataset = Subset(dataset, range(train_size, train_size + validation_size))
    test_dataset = Subset(dataset, range(train_size + validation_size, train_size + validation_size + test_size))

    print(f"Train dataset created with {len(train_dataset)} samples from bidding area {global_config['bidding_area']}")
    print(f"Validation dataset created with {len(validation_dataset)} samples from bidding area {global_config['bidding_area']}")
    print(f"Test dataset created with {len(test_dataset)} samples from bidding area {global_config['bidding_area']}")

    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

    # Set nn_config to correct model configuration
    if global_config["model"] == "cnn_forecasting_model":
        nn_config = cnn_config
    else:
        raise ValueError(f"{global_config['model']} in global config is not valid")

    # Instantiate model and optimizer
    # TODO num_features not fixed
    model = nn_config["model"](num_features=2, sequence_length=global_config["sequence_length"])
    optimizer = nn_config["optimizer"](model.parameters(), lr=nn_config["lr"])
    loss_function = nn_config["loss_function"]()
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Loss function: {loss_function.__class__.__name__}")

    return model, optimizer, loss_function, train_data_loader, validation_data_loader, test_data_loader, nn_config

def n_in_one_out(model, features, temperature_forecasts, targets):
    # Instantiate consumption_forecasts tensor for minibatch
    consumption_forecasts = torch.zeros(targets.size())

    # Add a row with forecast values (temp_forecast_i+1, 0)
    forecast_row = torch.zeros(32, 1, 2)
    forecast_row[:, :, 0] = temperature_forecasts[:, 0, :]
    features = torch.cat((features, forecast_row), dim=1)
    initial_consumption_forecast = model(features)
    consumption_forecasts[:, 0] = initial_consumption_forecast[:, 0]

    # n in, 1 out
    for i in range(1, global_config["forecast_horizon"]):
        # Remove oldest features
        features = features[:, 1:, :]

        # Remove previous forecast rows
        features = features[:, :-1, :]

        # Create row for current timestep
        current_timestep_row = torch.zeros(32, 1, 2)
        # Current timestep temperature is previous timestep forecast
        current_timestep_row[:, :, 0] = temperature_forecasts[:, i-1, :]
        # Current timestep consumption is previous consumption prediction
        previous_prediction = consumption_forecasts[:, i-1].reshape(-1, 1)
        current_timestep_row[:, :, 1] = previous_prediction[:, :]
        # Add current timestep to features (now size is (sequence_length, features))
        features = torch.cat((features, current_timestep_row), dim=1)

        # Add a row with forecast values (temp_forecast_i+1, 0)
        forecast_row = torch.zeros(32, 1, 2)
        forecast_row[:, :, 0] = temperature_forecasts[:, i, :]
        # Size is now (sequence_length + 1, features)
        features = torch.cat((features, forecast_row), dim=1)

        # Get consumption forecast for next timestep
        consumption_forecast = model(features)
        # Add to consumption_forecasts tensor
        consumption_forecasts[:, i] = consumption_forecast[:, 0]

    return consumption_forecasts

def train_one_epoch(model, optimizer, loss_function, train_data_loader):
    running_loss = []

    for features, temperature_forecasts, targets in train_data_loader:
        # Zeroing gradients for each minibatch
        optimizer.zero_grad()

        # Perform n in, 1 out
        consumption_forecasts = n_in_one_out(model, features, temperature_forecasts, targets)

        # Compute loss and gradients
        loss = loss_function(consumption_forecasts, targets)
        loss.backward()
        # Adjust weights
        optimizer.step()
        # Gather data
        running_loss.append(loss.item())

    return np.mean(running_loss)

def train(model, optimizer, loss_function, train_data_loader, validation_data_loader, epochs):
    start_time = time.time()

    print("\033[1;32m" + "="*15 + " Training " + "="*15 + "\033[0m")
    for epoch in range(epochs):
        # Training loop
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, loss_function, train_data_loader)
        model.train(False)

        # Validation loop
        running_validation_loss = []
        for features, temperature_forecasts, targets in validation_data_loader:
            consumption_forecasts = n_in_one_out(model, features, temperature_forecasts, targets)
            validation_loss = loss_function(consumption_forecasts, targets)
            running_validation_loss.append(validation_loss.item())
        avg_validation_loss = np.mean(running_validation_loss)

        print(f"Epoch {epoch+1}, Training Loss: {avg_loss}, Validation Loss: {avg_validation_loss}")

    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)

    print(f"Total training time: {int(minutes)} minutes and {round(seconds, 2)} seconds")

def test(model, loss_function, test_data_loader):
    print("\033[1;32m" + "="*15 + " Testing " + "="*15 + "\033[0m")

    model.eval()
    running_test_loss = []
    for features, temperature_forecasts, targets in test_data_loader:
        consumption_forecasts = n_in_one_out(model, features, temperature_forecasts, targets)
        test_loss = loss_function(consumption_forecasts, targets)
        running_test_loss.append(test_loss.item())

    print(f"Test Loss: {np.mean(running_test_loss)}")

def main():
    model, optimizer, loss_function, train_data_loader, validation_data_loader, test_data_loader, nn_config = setup()

    if global_config["train"]:
        train(model, optimizer, loss_function, train_data_loader, validation_data_loader, epochs=nn_config["epochs"])

    if global_config["test"]:
        test(model, loss_function, test_data_loader)


if __name__ == "__main__":
    main()
