import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    output_X = sigmoid(X)
    return output_X * (1 - output_X)
