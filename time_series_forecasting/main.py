import torch
from torch.utils.data import DataLoader
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

    # Choose dataset path if needed
    if global_config["train"] or global_config["test"]:
        dataset_path = os.path.join(os.path.dirname(__file__), "data", "raw", global_config["csv_filename"])
        print(f"Loading data from {dataset_path}")

    # Initialize data loaders to None in case they're not used
    train_data_loader = None
    test_data_loader = None

    if global_config["train"]:
        train_dataset = PowerConsumptionDataset(
            csv_file=dataset_path,
            sequence_length=global_config["sequence_length"],
            forecast_horizon=global_config["forecast_horizon"],
            bidding_area=global_config["bidding_area"],
            mode="train"
        )
        print(f"Train dataset created with {len(train_dataset)} samples from bidding area {global_config['bidding_area']}")
        train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    if global_config["test"]:
        test_dataset = PowerConsumptionDataset(
            csv_file=dataset_path,
            sequence_length=global_config["sequence_length"],
            forecast_horizon=global_config["forecast_horizon"],
            bidding_area=global_config["bidding_area"],
            mode="test"
        )
        print(f"Test dataset created with {len(test_dataset)} samples from bidding area {global_config['bidding_area']}")
        test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return model, optimizer, loss_function, train_data_loader, test_data_loader, nn_config

def train(model, optimizer, loss_function, data_loader, epochs):
    print("\033[1;32m" + "="*15 + " Training " + "="*15 + "\033[0m")
    model.train()
    for epoch in range(epochs):
        for features, targets in data_loader:
            optimizer.zero_grad()
            loss = loss_function(features, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def main():
    model, optimizer, loss_function, train_data_loader, test_data_loader, nn_config = setup()

    if global_config["train"]:
        train(model, optimizer, loss_function, train_data_loader, epochs=nn_config["epochs"])



if __name__ == "__main__":
    main()
