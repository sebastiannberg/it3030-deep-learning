import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    output_x = sigmoid(x)
    return output_x * (1 - output_x)
