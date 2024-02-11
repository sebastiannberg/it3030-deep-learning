import numpy as np


def mse(predicted_outputs, targets):
    return np.mean(np.sum((predicted_outputs - targets)**2, axis=1))

def mse_derivative(predicted_outputs, targets):
    n = targets.shape[0] # Number of samples
    return (2 / n) * (predicted_outputs - targets)
