from torch.utils.data import DataLoader
import os

from config import global_config, cnn_config
from data.datasets import PowerConsumptionDataset

def setup():
    print("\033[1;32m" + "="*15 + " Setup " + "="*15 + "\033[0m")

    # Set nn_config to correct model configuration
    if global_config["model"] == "cnn_forecasting_model":
        nn_config = cnn_config
    else:
        raise ValueError(f"{global_config['model']} model in globals config not defined")

    # Instantiate model and optimizer
    model = nn_config["model"]()
    optimizer = nn_config["optimizer"](model.parameters(), lr=nn_config["lr"])
    if model and optimizer:
        print(f"{model.__class__.__name__} and {optimizer.__class__.__name__} optimizer instantiated")

    # Choose dataset path if needed
    if global_config["train"] or global_config["test"]:
        dataset_path = os.path.join(os.path.dirname(__file__), "data", "raw", global_config["csv_filename"])
        print(f"Loading data from {dataset_path}")

    if global_config["train"]:
        train_dataset = PowerConsumptionDataset(
            csv_file=dataset_path,
            sequence_length=global_config["sequence_length"],
            forecast_horizon=global_config["forecast_horizon"],
            bidding_area=global_config["bidding_area"],
            mode="train"
        )
        print(f"Training dataset created with {len(train_dataset)} samples from bidding area {global_config['bidding_area']}")
        train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    if global_config["test"]:
        test_dataset = PowerConsumptionDataset(
            csv_file=dataset_path,
            sequence_length=global_config["sequence_length"],
            forecast_horizon=global_config["forecast_horizon"],
            bidding_area=global_config["bidding_area"],
            mode="test"
        )
        print(f"Testing dataset created with {len(test_dataset)} samples from bidding area {global_config['bidding_area']}")
        test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)




def main():
    setup()



if __name__ == "__main__":
    main()
