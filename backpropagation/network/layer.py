import numpy as np

from activation_functions.identity import identity, identity_derivative
from activation_functions.relu import relu, relu_derivative
from activation_functions.softmax import softmax


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

    def apply_activation_function(self, x):
        if self.activation_function == "identity":
            return identity(x)
        elif self.activation_function == "relu":
            return relu(x)
        elif self.activation_function == "softmax":
            return softmax(x)

    def apply_activation_function_derivative(self, x):
        pass

    def forward_pass(self, input_tensor):
        if self.layer_type == "input" and input_tensor.shape[0] == self.neurons:
            return input_tensor
        elif self.layer_type == "input" and not input_tensor.shape[0] == self.neurons:
            raise ValueError(
                f"Size of input {input_tensor.shape} does not match input layer size: {(self.neurons, 1)}")
        print(self.weights.shape, input_tensor.shape)
        z = np.dot(self.weights, input_tensor)
        z += self.bias
        output = self.apply_activation_function(z)
        return output
