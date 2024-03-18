import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
from typing import Dict

from config import global_config, cnn_config
from data.datasets import PowerConsumptionDataset
from data.feature_engineering import FeatureEngineering
from data.preprocessing import Preprocessor
from utils.visualize import TrainingVisualizer, ForecastVisualizer, EvaluationVisualizer
from models.cnn_forecasting_model import CNNForecastingModel

class ModelController:

    def __init__(self, config: Dict) -> None:
        pass

    def setup(self):
        pass


def setup(preprocessor: Preprocessor, feature_engineering: FeatureEngineering):
    """
    Setup selected model and create datasets based on the configuration file
    """
    print("\033[1;32m" + "="*15 + " Setup " + "="*15 + "\033[0m")

    # Load data
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "raw", global_config["csv_filename"])
    print(f"Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    bidding_area = global_config["bidding_area"]
    df.interpolate(method='polynomial', order=2, inplace=True, limit_direction='both')

    # Feature engineering
    df = feature_engineering.add_features(df, bidding_area)

    # Feature selection
    sequence_features = [f"{bidding_area}_consumption", f"{bidding_area}_temperature", "hour_sin", "weekday_cos"]
    forecast_features = [f"{bidding_area}_temperature", "hour_sin"]
    features_to_preprocess = [f"{bidding_area}_consumption", f"{bidding_area}_temperature"]

    # Split dataset into training, validation and test
    train_size = int(0.7 * len(df))
    validation_size = int(0.1 * len(df))
    train_df = df[:train_size]
    validation_df = df[train_size:(train_size + validation_size)]
    test_df = df[(train_size + validation_size):]

    # Preprocessing
    print("Preprocessing training data")
    # Applying spike removal before standardization, because standardization would be affected by missing values
    # Fitting params to training data
    train_df = preprocessor.remove_spikes(train_df, features=features_to_preprocess, fit=True)
    # Using Z-score standardization because data is approximately normally distributed
    # Results in range of mean around 0 and standard deviation of 1
    train_df = preprocessor.standardize(train_df, features=features_to_preprocess, fit=True)
    print("Preprocessing validation data")
    validation_df = preprocessor.remove_spikes(validation_df, features=features_to_preprocess)
    validation_df = preprocessor.standardize(validation_df, features=features_to_preprocess)
    print("Preprocessing test data")
    test_df = preprocessor.remove_spikes(test_df, features=features_to_preprocess)
    test_df = preprocessor.standardize(test_df, features=features_to_preprocess)

    # Create datasets
    train_dataset = PowerConsumptionDataset(
        data=train_df,
        sequence_length=global_config["sequence_length"],
        forecast_horizon=global_config["forecast_horizon"],
        target_column=f"{bidding_area}_consumption",
        sequence_features=sequence_features,
        forecast_features=forecast_features
    )
    validation_dataset = PowerConsumptionDataset(
        data=validation_df,
        sequence_length=global_config["sequence_length"],
        forecast_horizon=global_config["forecast_horizon"],
        target_column=f"{bidding_area}_consumption",
        sequence_features=sequence_features,
        forecast_features=forecast_features
    )
    test_dataset = PowerConsumptionDataset(
        data=test_df,
        sequence_length=global_config["sequence_length"],
        forecast_horizon=global_config["forecast_horizon"],
        target_column=f"{bidding_area}_consumption",
        sequence_features=sequence_features,
        forecast_features=forecast_features
    )

    feature_indices = None
    if train_dataset:
        feature_indices = train_dataset.feature_indices
    elif test_dataset:
        feature_indices = test_dataset.feature_indices

    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

    print(f"Train dataset created with {len(train_dataset)} samples from bidding area {global_config['bidding_area']}")
    print(f"Validation dataset created with {len(validation_dataset)} samples from bidding area {global_config['bidding_area']}")
    print(f"Test dataset created with {len(test_dataset)} samples from bidding area {global_config['bidding_area']}")

    # Set nn_config to correct model configuration
    if global_config["model"] == CNNForecastingModel:
        nn_config = cnn_config
    else:
        raise ValueError(f"{global_config['model']} in global config is not valid")

    # Instantiate model and optimizer
    num_features = len(sequence_features)
    model = nn_config["model"](num_features=num_features, sequence_length=global_config["sequence_length"])
    optimizer = nn_config["optimizer"](model.parameters(), lr=nn_config["lr"])
    loss_function = nn_config["loss_function"]()
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Loss Function: {loss_function.__class__.__name__}")

    if global_config["load_model"]:
        model_load_path = os.path.join(os.path.dirname(__file__), "saved_models", model.__class__.__name__, global_config["load_model_filename"])
        if os.path.exists(model_load_path):
            model.load_state_dict(torch.load(model_load_path))
            print(f"Model loaded from {model_load_path}")
        else:
            raise FileNotFoundError(f"No model found at {model_load_path}")

    return model, optimizer, loss_function, train_data_loader, validation_data_loader, test_data_loader, feature_indices, nn_config

def n_in_one_out(model, sequence_features, forecast_features, targets, feature_indices):
    """
    Consumption are always first column, and temperature are always second column
    """
    # Instantiate consumption_forecasts tensor for minibatch
    consumption_forecasts = torch.zeros(targets.size())

    initial_consumption_forecast = model(sequence_features)
    consumption_forecasts[:, 0] = initial_consumption_forecast[:, 0]

    # n in, 1 out
    for i in range(1, global_config["forecast_horizon"]):
        # Remove oldest sequence_features timestep
        sequence_features = sequence_features[:, 1:, :]

        # Create row for current timestep
        current_timestep_row = sequence_features[:, -1:, :].clone()  # Clone the last timestep
        # Current timestep consumption is previous prediction
        current_timestep_row[:, :, 0] = consumption_forecasts[:, i-1].reshape(-1, 1)

        # Insert forecast_features to matching sequence_features
        for feature_name, indices in feature_indices.items():
            forecast_index = indices['forecast_index']
            sequence_index = indices['sequence_index']

            if forecast_index is not None:
                # Replace the feature at the sequence index with the forecast feature value
                current_timestep_row[:, :, sequence_index] = forecast_features[:, i-1, forecast_index].reshape(-1, 1)

        # Add current timestep to features (now size is (sequence_length, features))
        sequence_features = torch.cat((sequence_features, current_timestep_row), dim=1)

        # Get consumption forecast for next timestep
        consumption_forecast = model(sequence_features)
        # Add to consumption_forecasts tensor
        consumption_forecasts[:, i] = consumption_forecast[:, 0]
    return consumption_forecasts

def train_one_epoch(model, optimizer, loss_function, train_data_loader, feature_indices, training_visualizer, preprocessor: Preprocessor):
    running_loss = []

    for sequence_features, forecast_features, targets, _ in train_data_loader:
        # Zeroing gradients for each minibatch
        optimizer.zero_grad()

        # Perform n in, 1 out
        consumption_forecasts = n_in_one_out(model, sequence_features, forecast_features, targets, feature_indices)

        consumption_forecasts_reversed = preprocessor.reverse_standardize_targets(consumption_forecasts)
        targets_reversed = preprocessor.reverse_standardize_targets(targets)

        # Compute loss and gradients
        loss = loss_function(consumption_forecasts_reversed, targets_reversed)
        loss.backward()
        # Adjust weights
        optimizer.step()
        # Gather data
        training_visualizer.add_minibatch_datapoint(loss.item())
        running_loss.append(loss.item())

    return np.mean(running_loss)

def train(model, optimizer, loss_function, train_data_loader, validation_data_loader, feature_indices, epochs, training_visualizer, save: bool, preprocessor: Preprocessor):
    start_time = time.time()

    print("\033[1;32m" + "="*15 + " Training " + "="*15 + "\033[0m")
    for epoch in range(epochs):
        # Training loop
        model.train()
        avg_loss = train_one_epoch(model, optimizer, loss_function, train_data_loader, feature_indices, training_visualizer, preprocessor)

        # Validation loop
        model.eval()
        running_validation_loss = []

        with torch.no_grad():
            for sequence_features, forecast_features, targets, _ in validation_data_loader:
                consumption_forecasts = n_in_one_out(model, sequence_features, forecast_features, targets, feature_indices)
                consumption_forecasts_reversed = preprocessor.reverse_standardize_targets(consumption_forecasts)
                targets_reversed = preprocessor.reverse_standardize_targets(targets)
                validation_loss = loss_function(consumption_forecasts_reversed, targets_reversed)
                running_validation_loss.append(validation_loss.item())
        avg_validation_loss = np.mean(running_validation_loss)

        training_visualizer.add_epoch_datapoint(avg_loss, avg_validation_loss)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss}, Validation Loss: {avg_validation_loss}")

        if save:
            current_time = datetime.now().strftime("%d-%m-%Y-%H%M%S")
            model_save_path = os.path.join(os.path.dirname(__file__), "saved_models", model.__class__.__name__, f"{current_time}_{global_config['bidding_area']}_epoch_{epoch+1}.pt")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)

    print(f"Total training time: {int(minutes)} minutes and {round(seconds, 2)} seconds")

def test(model, loss_function, test_data_loader, feature_indices, forecast_visualizer, preprocessor: Preprocessor):
    print("\033[1;32m" + "="*15 + " Testing " + "="*15 + "\033[0m")

    model.eval()
    running_test_loss = []

    with torch.no_grad():
        for sequence_features, forecast_features, targets, timestamps in test_data_loader:
            consumption_forecasts = n_in_one_out(model, sequence_features, forecast_features, targets, feature_indices)
            consumption_forecasts_reversed = preprocessor.reverse_standardize_targets(consumption_forecasts)
            targets_reversed = preprocessor.reverse_standardize_targets(targets)
            test_loss = loss_function(consumption_forecasts_reversed, targets_reversed)
            for i in range(len(sequence_features)):
                # TODO remove fixed values
                historical_consumption = sequence_features[i, :, 0]
                historical_consumption_reversed = preprocessor.reverse_standardize_targets(historical_consumption)
                forecast_visualizer.add_datapoint(historical_consumption_reversed, consumption_forecasts_reversed[i], targets_reversed[i], timestamps[i])
            running_test_loss.append(test_loss.item())

    print(f"Test Loss: {np.mean(running_test_loss)}")

def compare_models(test_data_loader, feature_indices, compare_filenames, preprocessor: Preprocessor):
    print("\033[1;32m" + "="*15 + " Comparing " + "="*15 + "\033[0m")
    metrics_results = {}

    for i, (filename, model_class) in enumerate(compare_filenames):
        model_load_path = os.path.join(os.path.dirname(__file__), "saved_models", model_class.__name__, filename)
        if not os.path.exists(model_load_path):
            print(f"Model file {model_load_path} not found.")
            continue

        # TODO fix hard coded value
        model = model_class(num_features=2, sequence_length=global_config["sequence_length"])
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        with torch.no_grad():
            all_predictions = []
            all_targets = []
            for sequence_features, temperature_forecasts, targets, _ in test_data_loader:
                consumption_forecasts = n_in_one_out(model, sequence_features, temperature_forecasts, targets, feature_indices)
                consumption_forecasts_reversed = preprocessor.reverse_standardize_targets(consumption_forecasts)
                targets_reversed = preprocessor.reverse_standardize_targets(targets)
                all_predictions.append(consumption_forecasts_reversed)
                all_targets.append(targets_reversed)

            # Concatenate all batch predictions and targets
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            rmse = torch.sqrt(torch.nn.functional.mse_loss(all_predictions, all_targets))
            mae = torch.nn.functional.l1_loss(all_predictions, all_targets)
            mape = torch.mean(torch.abs((all_predictions - all_targets) / all_targets)) * 100

        metrics_results[f"{i+1}. {model_class.__name__}"] = {
            "RMSE": rmse.item(),
            "MAE": mae.item(),
            "MAPE": mape.item()
        }

    return metrics_results


def main():
    preprocessor = Preprocessor()
    feature_engineering = FeatureEngineering()

    model, optimizer, loss_function, train_data_loader, validation_data_loader, test_data_loader, feature_indices, nn_config = setup(preprocessor, feature_engineering)
    training_visualizer = TrainingVisualizer()
    forecast_visualizer = ForecastVisualizer()
    evaluation_visualizer = EvaluationVisualizer()

    if global_config["train"]:
        train(model, optimizer, loss_function, train_data_loader, validation_data_loader, feature_indices, nn_config["epochs"], training_visualizer, global_config["save_model"], preprocessor)

    if global_config["test"]:
        test(model, loss_function, test_data_loader, feature_indices, forecast_visualizer, preprocessor)

    if global_config["visualize"]:
        if global_config["train"]:
            training_visualizer.plot_training_progress()
        if global_config["test"]:
            forecast_visualizer.plot_consumption_forecast()
            forecast_visualizer.plot_error_statistics()

    if global_config["compare"]:
        metrics_result = compare_models(test_data_loader, feature_indices, global_config["compare_filenames"], preprocessor)
        print(metrics_result)
        evaluation_visualizer.plot_summary(metrics_result)


if __name__ == "__main__":
    main()
