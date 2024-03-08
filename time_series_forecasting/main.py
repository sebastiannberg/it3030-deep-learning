import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import os

from config import global_config, cnn_config
from data.datasets import PowerConsumptionDataset

def setup():
    """
    Setup selected model and create datasets based on the configuration file
    """
    print("\033[1;32m" + "="*15 + " Setup " + "="*15 + "\033[0m")

    # Set nn_config to correct model configuration
    if global_config["model"] == "cnn_forecasting_model":
        nn_config = cnn_config
    else:
        raise ValueError(f"{global_config['model']} in global config is not valid")

    if global_config["load_model"]:
        raise NotImplementedError()

    if global_config["save_model"]:
        raise NotImplementedError()

    if global_config["visualize"]:
        raise NotImplementedError()

    # Instantiate model and optimizer
    model = nn_config["model"]()
    optimizer = nn_config["optimizer"](model.parameters(), lr=nn_config["lr"])
    print(f"{model.__class__.__name__} and {optimizer.__class__.__name__} optimizer instantiated")
    loss_function = nn_config["loss_function"]()
    print(f"Using loss function {loss_function.__class__.__name__}")

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

    return model, optimizer, loss_function, train_data_loader, validation_data_loader, test_data_loader, nn_config

def train_one_epoch(model, optimizer, loss_function, train_data_loader):
    running_loss = []

    for features, temperature_forecasts, targets in train_data_loader:
        # Zeroing gradients for each minibatch
        optimizer.zero_grad()

        # Instantiate consumption_forecasts tensor for minibatch
        consumption_forecasts = torch.zeros(targets.size())
        initial_consumption_forecast = model(features)
        consumption_forecasts[:, 0] = initial_consumption_forecast[:, 0]

        # n in, 1 out
        for i in range(1, global_config["forecast_horizon"]):
            # Remove oldest features
            features = features[:, 1:, :]
            # Insert temperetature forecast and previous prediction
            new_row_features = torch.zeros(32, 1, 2)
            # Temperature is proxy forecast
            new_row_features[:, :, 0] = temperature_forecasts[:, i-1, :]
            # Get previous prediction
            previous_prediction = consumption_forecasts[:, i-1].reshape(-1, 1)
            new_row_features[:, :, 1] = previous_prediction[:, :]
            # Add forecasts to features
            features = torch.cat((features, new_row_features), dim=1)
            # Forecast consumption
            consumption_forecast = model(features)
            # Add to consumption_forecasts tensor
            consumption_forecasts[:, i] = consumption_forecast[:, 0]

        # Compute loss and gradients
        loss = loss_function(consumption_forecasts, targets)
        loss.backward()
        # Adjust weights
        optimizer.step()
        # Gather data
        running_loss.append(loss.item())

    return np.mean(running_loss)

def train(model, optimizer, loss_function, train_data_loader, validation_data_loader, epochs):
    print("\033[1;32m" + "="*15 + " Training " + "="*15 + "\033[0m")
    for epoch in range(epochs):
        # Training loop
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, loss_function, train_data_loader)
        model.train(False)
        # Validation loop
        running_validation_loss = []
        for features, targets in validation_data_loader:
            validation_predictions = model(features)
            validation_loss = loss_function(validation_predictions, targets)
            running_validation_loss.append(validation_loss.item())

        avg_validation_loss = np.mean(running_validation_loss)

        print(f"Epoch {epoch+1}, Training Loss: {avg_loss}, Validation Loss: {avg_validation_loss}")

def test(model, loss_function, test_data_loader):
    # TODO remove fixed sizes on tensors
    print("\033[1;32m" + "="*15 + " Testing " + "="*15 + "\033[0m")
    model.eval()
    running_test_loss = []
    k = 1

    for features, targets in test_data_loader:
        # n in, 1 out
        forecasts = torch.zeros(targets.size())
        print(features.size())
        print(targets.size())
        forecast = model(features)
        forecasts[:, 0] = forecast[:, 0]
        print(forecasts)
        print(forecasts.size())
        for i in range(1, global_config["forecast_horizon"]):
            # remove oldest features
            features = features[:, 1:, :]
            print(features.size())
            # add previous forecast as newest feature
            # features[:, 23, :] = forecasts[:, i-1]
            pred_as_feature = torch.zeros(32, 1, 2)
            pred_as_feature[:, 0, 1] = forecasts[:, i-1]
            print(pred_as_feature)
            features = torch.cat((features, pred_as_feature), dim=2)
            print(features.size())
            # get new forecast
            # add to forecasts
        print(forecasts)
        # compare forecasts to targets (should be same shape)
        test_loss = loss_function(forecasts, targets)
        # add loss to running_test_loss
        running_test_loss.append(test_loss.item())
        if k == 1:
            break


def main():
    model, optimizer, loss_function, train_data_loader, validation_data_loader, test_data_loader, nn_config = setup()

    if global_config["train"]:
        train(model, optimizer, loss_function, train_data_loader, validation_data_loader, epochs=nn_config["epochs"])

    if global_config["test"]:
        test(model, loss_function, test_data_loader)


if __name__ == "__main__":
    main()
