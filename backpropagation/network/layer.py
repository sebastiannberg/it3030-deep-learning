import numpy as np

from activation_functions.identity import *


class Layer:

    def __init__(self, layer_type, neurons, activation_function=None, weight_range=None, bias_range=None, learning_rate=None):
        self.layer_type = layer_type
        self.neurons = neurons
        self.activation_function = activation_function
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.learning_rate = learning_rate
        self.previous_layer = None

    def set_previous_layer(self, layer):
        self.previous_layer = layer

    def init_parameters(self):
        if self.layer_type == "input":
            return
        else:
            if self.weight_range == "glorot":
                print("TODO implement glorot")
            else:
                self.weights = np.random.uniform(low=self.weight_range[0],
                                                 high=self.weight_range[1],
                                                 size=(self.neurons, self.previous_layer.neurons))
                self.bias = np.random.uniform(low=self.bias_range[0],
                                              high=self.bias_range[1],
                                              size=(self.neurons, 1))

    def forward_pass(self, input_tensor):
        if self.layer_type == "input" and input_tensor.shape == (self.neurons, 1):
            return input_tensor
        elif self.layer_type == "input" and not input_tensor.shape == (self.neurons, 1):
            raise ValueError(
                f"Size of input {input_tensor.shape} does not match input layer size: {(self.neurons, 1)}")

        ting = np.dot(self.weights, input_tensor)
        ting += self.bias

        return ting
