import numpy as np


def cross_entropy(predicted_outputs, targets):
    cross_entropy = -np.sum(targets * np.log(predicted_outputs), axis=1)
    return cross_entropy
