import numpy as np

from network.layer import Layer
from loss_functions.cross_entropy import cross_entropy


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

    def get_layers(self):
        return self.layers

    def fit(self, train_features, train_targets, validation_features, validation_targets, num_epochs=5, minibatch_size=64):
        if minibatch_size > len(train_features):
            minibatch_size = len(train_features)
        # fetch a minibatch of training cases
        for minibatch_features, minibatch_targets in self.fetch_minibatch(train_features, train_targets, minibatch_size):
            # send each case through the network
            outputs = self.forward_pass(features=minibatch_features)
            print(outputs)
            # compute the loss
            loss = self.compute_loss(
                predicted_outputs=outputs, targets=minibatch_targets)

    def fetch_minibatch(self, features, targets, minibatch_size):
        assert len(features) == len(targets)
        # Create index array
        indices = np.arange(len(features))
        # Shuffle the indexes
        np.random.shuffle(indices)
        for start_index in range(0, len(features), minibatch_size):
            end_index = min(start_index + minibatch_size, len(features))
            # Select shuffled indices in minibatch sizes
            selected_indices = indices[start_index:end_index]
            # Yield a randomly selected minibatch
            yield features[selected_indices], targets[selected_indices]

    def forward_pass(self, features):
        # Reshape features to column vectors using transpose
        input_tensor = features.T
        # Perform forward pass through each layer
        for layer in self.layers:
            output_tensor = layer.forward_pass(input_tensor)
            # Update input to be previous layer output
            input_tensor = output_tensor
        # Return final output from output layer
        return output_tensor.T

    def compute_loss(self, predicted_outputs, targets):
        if self.loss_function == "cross_entropy":
            loss = cross_entropy(predicted_outputs, targets)
            print(loss)
