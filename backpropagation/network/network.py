from network.layer import Layer


class Network:

    def __init__(self, loss_function, learning_rate, weight_reg_rate, weight_reg_type, verbose):
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.weight_reg_rate = weight_reg_rate
        self.weight_reg_type = weight_reg_type
        self.layers = []
        self.verbose = verbose

    def init_parameters(self):
        for layer in self.layers:
            layer.init_parameters()

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

    def fit(self, train_features, train_targets, validation_features, validation_targets):
        for features, target in zip(train_features, train_targets):
            self.forward_pass(features, target)

    def forward_pass(self, features, target):
        # Reshape features to column vector
        input_tensor = features.reshape(-1, 1)
        for layer in self.layers:
            output_tensor = layer.forward_pass(input_tensor)
            input_tensor = output_tensor
        print(output_tensor)
