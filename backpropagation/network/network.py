import numpy as np

from network.layer import Layer
from loss_functions.cross_entropy import cross_entropy, cross_entropy_derivative
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

    def fit(self, train_X, train_y, validation_X, validation_y, minibatch_size=4, num_minibatches=100):
        """
        Assuming X is in row vector shape (cases, features)
        Assuming y is in row vector shape (cases, targets)
        Both X and y is transposed to support the network conventions
        """
        train_loss = []
        validation_loss = []
        minibatch_count = 0

        while minibatch_count < num_minibatches:
            # Fetch a minibatch of training cases
            for minibatch_X, minibatch_y in self.fetch_minibatch(train_X, train_y, minibatch_size):
                Z = self.forward_pass(minibatch_X.T)
                loss = self.compute_loss(Z, minibatch_y.T)
                train_loss.append(np.sum(loss))
                weight_gradients, bias_gradients = self.backward_pass(Z, minibatch_y.T)
                self.update_parameters(weight_gradients, bias_gradients)

                # Perform validation
                Z = self.forward_pass(validation_X.T)
                loss = self.compute_loss(Z, validation_y.T)
                validation_loss.append(np.sum(loss))

                minibatch_count += 1
                if minibatch_count >= num_minibatches:
                    break

        return train_loss, validation_loss

    def predict(self):
        pass

    def fetch_minibatch(self, X, y, minibatch_size):
        assert len(X) == len(y)
        # Create index array
        indices = np.arange(len(X))
        # Shuffle the indexes to select random later
        np.random.shuffle(indices)
        for start_index in range(0, len(X), minibatch_size):
            end_index = min(start_index + minibatch_size, len(X))
            # Select shuffled indices in minibatch sizes
            selected_indices = indices[start_index:end_index]
            # Yield a randomly selected minibatch
            yield X[selected_indices], y[selected_indices]

    def forward_pass(self, X):
        """
        Assuming X is in column vector shape (features, cases)
        Returning output Z in column vector shape (features, cases)
        """
        input_tensor = X
        # Perform forward pass through each layer
        for layer in self.layers:
            output_tensor = layer.forward_pass(input_tensor)
            # Update input to be previous layer output
            input_tensor = output_tensor
        return output_tensor

    def backward_pass(self, Z, y):
        """
        Assuming Z and y is in column vector shapes (features, cases) and (targets, cases)
        """
        jacobian_L_Z = self.compute_jacobian_L_Z(Z, y)
        weight_gradient_stack = []
        bias_gradient_stack = []

        for layer in reversed(self.layers[1:]):
            jacobian_L_W, jacobian_L_B, jacobian_L_Z = layer.backward_pass(jacobian_L_Z)
            weight_gradient = np.squeeze(np.sum(jacobian_L_W, axis=-1))
            bias_gradient = np.squeeze(np.sum(jacobian_L_B, axis=-1)).reshape(1,-1)
            print(weight_gradient)
            print(bias_gradient)
            weight_gradient_stack.insert(0, weight_gradient)
            bias_gradient_stack.insert(0, bias_gradient)

        return weight_gradient_stack, bias_gradient_stack

    def compute_loss(self, X, y):
        if self.loss_function == "cross_entropy":
            return cross_entropy(X, y)
        elif self.loss_function == "mse":
            return mse(X, y)
        else:
            raise ValueError(f"Received unsupported loss function: {self.loss_function}")

    def compute_jacobian_L_Z(self, Z, y):
        """
        Assuming Z and y is in column vector shapes (features, cases) and (targets, cases)
        Computes jacobian with shape (output_nodes, cases, 1, cases) and denominator layout
        """
        if self.loss_function == "cross_entropy":
            derivatives = cross_entropy_derivative(Z, y)
        elif self.loss_function == "mse":
            derivatives = mse_derivative(Z, y)
        else:
            raise ValueError(f"Received unsupported loss function: {self.loss_function}")
        jacobian = np.zeros((self.layers[-1].neurons, Z.shape[1], 1, y.shape[1]))
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                for k in range(jacobian.shape[2]):
                    for l in range(jacobian.shape[3]):
                        if j == l:
                            jacobian[i, j, k, l] = derivatives[i, j]
        return jacobian

    def update_parameters(self, weight_gradients, bias_gradients):
        for i in range(1, len(self.layers)):
            self.layers[i].update_parameters(weight_gradients[i-1], bias_gradients[i-1], self.learning_rate)
