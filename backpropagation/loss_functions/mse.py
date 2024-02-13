import numpy as np


def mse(predicted_outputs, targets):
    squared_errors = (predicted_outputs - targets)**2
    return np.mean(squared_errors, axis=0)

def mse_derivative(predicted_outputs, targets):
    n = targets.shape[0] # Number of output neurons
    return (2 / n) * (predicted_outputs - targets)
