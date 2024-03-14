import matplotlib.pyplot as plt
import numpy as np


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

    def __init__(self, sequence_length, forecast_horizon):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon