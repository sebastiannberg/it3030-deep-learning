import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrainingVisualizer:

    def __init__(self):
        self.minibatches = []
        self.epochs = []

    def add_minibatch_datapoint(self, training_loss):
        self.minibatches.append(training_loss)

    def add_epoch_datapoint(self, training_loss, validation_loss):
        self.epochs.append((training_loss, validation_loss))

    def plot_training_progress(self):
        plt.figure()
        plt.title("Learning Progression")
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")
        minibatch_x_values = np.arange(1, len(self.minibatches) + 1)
        plt.plot(minibatch_x_values, np.array(self.minibatches), label="Train", color="blue")
        plt.legend()

        plt.figure()
        plt.title("Learning Progression")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        epoch_x_values = np.arange(1, len(self.epochs) + 1)
        plt.plot(epoch_x_values, np.array([loss[0] for loss in self.epochs]), label="Train", color="blue")
        plt.plot(epoch_x_values, np.array([loss[1] for loss in self.epochs]), label="Validation", color="orange")
        plt.legend()

        plt.show()

class ForecastVisualizer:

    def __init__(self):
        self.data = []

    def add_datapoint(self, historical_consumption, consumption_forecasts, targets, timestamps):
        self.data.append((historical_consumption, consumption_forecasts, targets, timestamps))

    def plot_consumption_forecast(self, indexes=(0, 24, 288, 504)):
        for i in indexes:
            historical_consumption, consumption_forecasts, targets, timestamps = self.data[i]
            # Convert seconds from epoch to datetime
            timestamps = pd.to_datetime(timestamps, unit='s')

            plt.figure()
            plt.title("Forecast")
            plt.xlabel("Timestamp")
            plt.ylabel("Consumption (MWh)")

            # Plot historical consumption in blue
            plt.plot(timestamps[:len(historical_consumption)], historical_consumption, label="Historical Consumption", color="blue")
            # Plot targets in green
            plt.plot(timestamps[-len(targets):], targets, label="Actual Consumption", color="green")
            # Plot consumption forecasts in orange
            plt.plot(timestamps[-len(consumption_forecasts):], consumption_forecasts, label="Forecasted Consumption", color="orange")

            plt.legend()
            plt.xticks(rotation=75)
            plt.tight_layout()

        plt.show()

    def plot_error_statistics(self):
        errors = np.zeros((len(self.data), len(self.data[0][1])))

        for i, (_, consumption_forecasts, targets, _) in enumerate(self.data):
            abs_errors = np.abs(consumption_forecasts - targets)
            errors[i, :] = abs_errors

        # Calculate mean and standard deviation of errors for each hour in forecast horizon
        mean_errors = np.mean(errors, axis=0)
        std_errors = np.std(errors, axis=0)

        plt.figure()
        plt.title("Forecast Error Plot")
        plt.xlabel("Forecast Hour")
        plt.ylabel("Error")
        plt.plot(mean_errors, label="Mean", color="blue")
        plt.plot(std_errors, label="Standard Deviation", color="green")
        plt.legend()
        plt.show()

class EvaluationVisualizer:

    def plot_summary(self, metrics_result):
        metrics = ["RMSE", "MAE", "MAPE"]

        # Initialize a figure and axis for plotting
        fig, ax = plt.subplots()
        bar_width = 0.25  # Width of the bars
        n_metrics = len(metrics)
        n_models = len(metrics_result)

        # Setting up positions for the bars
        positions = np.arange(n_metrics)

        for i, (model, results) in enumerate(metrics_result.items()):
            # Extracting the metrics in order
            measurements = [results[metric] for metric in metrics]
            # Calculating the position for each model's bars
            offset = (i - n_models / 2) * bar_width + bar_width / 2
            # Plotting
            rects = ax.bar(positions + offset, measurements, bar_width, label=model)

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Scores")
        ax.set_title("Model Comparison")
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics)
        ax.legend()

        plt.show()