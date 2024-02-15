import numpy as np


def mse(X, y):
    squared_errors = (X - y)**2
    return np.mean(squared_errors, axis=0, keepdims=True)

def mse_derivative(X, y):
    n = y.shape[0] # Number of output neurons
    # TODO ensure correct output shape when returning
    return (2 / n) * (X - y)
