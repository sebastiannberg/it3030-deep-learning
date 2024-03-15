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
        plt.plot(np.array(self.minibatches), label="Train", color="blue")
        plt.legend()

        plt.figure()
        plt.title("Learning Progression")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(np.array([loss[0] for loss in self.epochs]), label="Train", color="blue")
        plt.plot(np.array([loss[1] for loss in self.epochs]), label="Validation", color="orange")
        plt.legend()

        plt.show()

class ForecastVisualizer:

    def __init__(self):
        self.data = []

    def add_datapoint(self, historical_consumption, consumption_forecasts, targets, timestamps):
        self.data.append((historical_consumption.numpy(), consumption_forecasts.numpy(), targets.numpy(), timestamps.numpy()))

    def plot_consumption_forecast(self, indexes=(0, 27, 124, 578)):
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
        data = {model: (metrics_result[model][metric] for metric in metrics_result[model].keys()) for model in metrics_result.keys()}

        x = np.arange(len(metrics))
        bar_width = 0.25
        multiplier = 0

        fig, ax = plt.subplots(layout="constrained")

        for attribute, measurement in data.items():
            offset = bar_width * multiplier
            rects = ax.bar(x + offset, measurement, bar_width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_ylabel("Score")
        ax.set_title("Evaluation Summary")
        ax.set_xticks(x + bar_width, metrics)
        ax.legend(loc='upper left')

        plt.show()
