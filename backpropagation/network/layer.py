import numpy as np

from activation_functions.identity import identity, identity_derivative
from activation_functions.relu import relu, relu_derivative
from activation_functions.sigmoid import sigmoid, sigmoid_derivative
from utils.softmax import softmax
from activation_functions.tanh import tanh, tanh_derivative


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

    # Used for testing
    def _set_weights(self, W):
        self.weights = W

    # Used for testing
    def _set_bias(self, bias):
        self.bias = bias

    # Used for testing
    def _cache_Y_sum_output(self, Y, sum, output):
        self.Y = Y
        self.sum = sum
        self.output = output

    def init_parameters(self):
        """
        Initializing weight matrix w_ij meaning weight from neuron i to neuron j
        """
        if self.layer_type == "input" or self.layer_type == "softmax":
            return
        else:
            if self.weight_range == "glorot":
                # TODO
                print("TODO implement glorot")
            else:
                self.weights = np.random.uniform(low=self.weight_range[0],
                                                 high=self.weight_range[1],
                                                 size=(self.previous_layer.neurons, self.neurons))
                self.bias = np.random.uniform(low=self.bias_range[0],
                                              high=self.bias_range[1],
                                              size=(1, self.neurons))

    def compute_sum(self, X):
        """
        Assuming X is in column vector shape (features, cases)
        """
        return np.dot(self.weights.T, X) + self.bias.T

    def compute_output(self, X):
        return self.apply_activation_function(X)

    def compute_jacobian_Z_sum(self):
        """
        Shape of jacobian will be (i, j, k ,l)
        where (i, j) is shape of sum and (k, l) is shape of Z
        """
        jacobian = np.zeros(shape=(self.sum.shape[0], self.sum.shape[1], self.output.shape[0], self.output.shape[1]))
        derivatives = self.apply_activation_function_derivative(self.sum)
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                for k in range(jacobian.shape[2]):
                    for l in range(jacobian.shape[3]):
                        if i == k and j == l:
                            jacobian[i, j, k, l] = derivatives[i, j]
        return jacobian

    def compute_jacobian_Z_Y(self, jacobian_Z_sum):
        jacobian_Z_Y = np.dot(jacobian_Z_sum.T, self.weights.T).T
        return jacobian_Z_Y

    def compute_jacobian_Z_W(self, jacobian_Z_sum):
        jacobian_Z_W = np.dot(self.Y, jacobian_Z_sum.T)
        transposed = np.transpose(jacobian_Z_W, axes=(0, 2, 3, 1))
        return transposed

    def compute_jacobian_Z_B(self, jacobian_Z_sum):
        """
        Shape of jacobian will be (i, j, k, l)
        where (i, j) is shape of bias and (k, l) is shape of Z
        """
        jacobian_Z_B = np.sum(jacobian_Z_sum, axis=1)[np.newaxis, :, :, :]
        return jacobian_Z_B

    def compute_jacobian_L_W(self, jacobian_L_Z, jacobian_Z_W):
        jacobian_L_W = np.tensordot(jacobian_Z_W, jacobian_L_Z)
        return jacobian_L_W

    def compute_jacobian_L_B(self, jacobian_L_Z, jacobian_Z_B):
        jacobian_L_B = np.tensordot(jacobian_Z_B, jacobian_L_Z)
        return jacobian_L_B

    def compute_jacobian_L_Y(self, jacobian_L_Z, jacobian_Z_Y):
        jacobian_L_Y = np.tensordot(jacobian_Z_Y, jacobian_L_Z)
        return jacobian_L_Y

    def compute_jacobian_S_Y(self):
        """
        Computes jacobian for softmax layer
        """
        jacobian = np.zeros(shape=(self.Y.shape[0], self.Y.shape[1], self.output.shape[0], self.output.shape[1]))
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                for k in range(jacobian.shape[2]):
                    for l in range(jacobian.shape[3]):
                        if i == k and j == l:
                            jacobian[i,j,k,l] = self.output[i,j] * (1 - self.output[i,j])
                        else:
                            jacobian[i,j,k,l] = - self.output[i,j] * self.output[k,l]
        return jacobian

    def apply_activation_function(self, X):
        if self.activation_function == "identity":
            return identity(X)
        elif self.activation_function == "relu":
            return relu(X)
        elif self.activation_function == "sigmoid":
            return sigmoid(X)
        elif self.activation_function == "tanh":
            return tanh(X)
        else:
            raise ValueError(f"Received unsupported activation function: {self.activation_function}")

    def apply_activation_function_derivative(self, X):
        if self.activation_function == "identity":
            return identity_derivative(X)
        elif self.activation_function == "relu":
            return relu_derivative(X)
        elif self.activation_function == "sigmoid":
            return sigmoid_derivative(X)
        elif self.activation_function == "tanh":
            return tanh_derivative(X)
        else:
            raise ValueError(f"Received unsupported activation function: {self.activation_function}")

    def forward_pass(self, X):
        """
        Assuming X is in column vector shape (features, cases)
        """
        if self.layer_type == "input":
            # TODO add check that input layer corresponds in shape to input vectors
            return X
        elif self.layer_type == "softmax":
            self.Y = X
            self.output = softmax(X)
            return self.output
        else:
            self.Y = X
            self.sum = self.compute_sum(X)
            self.output = self.compute_output(self.sum)
            return self.output

    def backward_pass(self, jacobian_L_Z):
        """
        Computing all needed jacobians in denominator layout
        """
        if self.layer_type == "input":
            pass
        if self.layer_type == "softmax":
            jacobian_S_Y = self.compute_jacobian_S_Y()
            jacobian_L_Y = self.compute_jacobian_L_Y(jacobian_L_Z, jacobian_S_Y)
            return None, None, jacobian_L_Y
        else:
            jacobian_Z_sum = self.compute_jacobian_Z_sum()
            jacobian_Z_Y = self.compute_jacobian_Z_Y(jacobian_Z_sum)
            jacobian_Z_W = self.compute_jacobian_Z_W(jacobian_Z_sum)
            jacobian_Z_B = self.compute_jacobian_Z_B(jacobian_Z_sum)
            jacobian_L_W = self.compute_jacobian_L_W(jacobian_L_Z, jacobian_Z_W)
            jacobian_L_B = self.compute_jacobian_L_B(jacobian_L_Z, jacobian_Z_B)
            jacobian_L_Y = self.compute_jacobian_L_Y(jacobian_L_Z, jacobian_Z_Y)
            return jacobian_L_W, jacobian_L_B, jacobian_L_Y

    def update_parameters(self, weight_gradient, bias_gradient, learning_rate):
        # Use private learning rate if applicable
        if self.learning_rate:
            self.weights = self.weights - self.learning_rate * weight_gradient
            self.bias = self.bias - self.learning_rate * bias_gradient
        else:
            self.weights = self.weights - learning_rate * weight_gradient
            self.bias = self.bias - learning_rate * bias_gradient
