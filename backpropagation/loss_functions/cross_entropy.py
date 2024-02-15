import numpy as np


def cross_entropy(X, y):
    cross_entropy = -np.sum(y * np.log2(X), axis=0, keepdims=True)
    return cross_entropy

def cross_entropy_derivative(X, y):
    return - y / X
