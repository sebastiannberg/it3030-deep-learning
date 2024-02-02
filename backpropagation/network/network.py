from network.layer import Layer


class Network:

    def __init__(self, loss_function, learning_rate, weight_reg_rate, weight_reg_type) -> None:
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.weight_reg_rate = weight_reg_rate
        self.weight_reg_type = weight_reg_type
        self.layers = []

    def add_layer(self, layer: Layer):
        if isinstance(layer, Layer):
            self.layers.append(layer)
        else:
            raise ValueError("Layer must be instance of Layer class")

    def get_layers(self):
        return self.layers
