import numpy as np

from network.layer import Layer
from loss_functions.cross_entropy import cross_entropy
from loss_functions.mse import mse, mse_derivative


class Network:

    def __init__(self, loss_function, learning_rate, weight_reg_rate, weight_reg_type, verbose):
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.weight_reg_rate = weight_reg_rate
        self.weight_reg_type = weight_reg_type
        self.layers = []
        self.verbose = verbose

    def add_layer(self, layer: Layer):
        if isinstance(layer, Layer):
            if len(self.layers) == 0:
                layer.set_previous_layer(None)
            else:
                layer.set_previous_layer(self.layers[-1])
            layer.init_parameters()
            self.layers.append(layer)
        else:
            raise ValueError("Layer must be instance of Layer class")

    # TODO check if this is used
    def get_layers(self):
        return self.layers

    def fit(self, train_features, train_targets, validation_features, validation_targets, minibatch_size=64, num_minibatches=5000):
        # TODO write assumptions about column vector or row vector input
        # Calculate how many iterations is needed
        # TODO ensure that all minibatches is of size 64 and not for example one is 60
        num_iterations = num_minibatches // (len(train_features) // minibatch_size)
        for _ in range(num_iterations):
            # Fetch a minibatch of training cases
            for minibatch_features, minibatch_targets in self.fetch_minibatch(train_features, train_targets, minibatch_size):
                # Send each case through the network
                outputs = self.forward_pass(features=minibatch_features)
                print(outputs)
                # Compute the loss
                loss = self.compute_loss(predicted_outputs=outputs, targets=minibatch_targets)
                print(loss)
                # Backward pass

    def fetch_minibatch(self, features, targets, minibatch_size):
        assert len(features) == len(targets)
        # Create index array
        indices = np.arange(len(features))
        # Shuffle the indexes to select random later
        np.random.shuffle(indices)
        for start_index in range(0, len(features), minibatch_size):
            end_index = min(start_index + minibatch_size, len(features))
            # Select shuffled indices in minibatch sizes
            selected_indices = indices[start_index:end_index]
            # Yield a randomly selected minibatch
            yield features[selected_indices], targets[selected_indices]

    def forward_pass(self, features):
        # TODO add comments about assumed shape of features
        # TODO to transpose or not to transpose
        # Reshape features to column vectors
        input_tensor = features.T
        # Perform forward pass through each layer
        for layer in self.layers:
            output_tensor = layer.forward_pass(input_tensor)
            # Update input to be previous layer output
            input_tensor = output_tensor
        # Return final output transposed back to row wectors
        return output_tensor.T

    def compute_loss(self, predicted_outputs, targets):
        if self.loss_function == "cross_entropy":
            return cross_entropy(predicted_outputs, targets)
        elif self.loss_function == "mse":
            return mse(predicted_outputs, targets)
        else:
            raise ValueError(f"Received unsupported loss function: {self.loss_function}")

    def compute_jacobian_L_Z(self, predicted_outputs, targets):
        """
        Want jacobian to be shape (output_nodes, cases, 1, cases)
        """
        if self.loss_function == "cross_entropy":
            pass
        elif self.loss_function == "mse":
            derivatives = mse_derivative(predicted_outputs, targets)
        else:
            raise ValueError(f"Received unsupported loss function: {self.loss_function}")
        jacobian = np.zeros((self.layers[-1].neurons, targets.shape[1], 1, targets.shape[1]))
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                for k in range(jacobian.shape[2]):
                    for l in range(jacobian.shape[3]):
                        if j == l:
                            jacobian[i, j, k, l] = derivatives[i, j]
        return jacobian
