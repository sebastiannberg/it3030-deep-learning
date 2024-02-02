

class Layer:

    def __init__(self, layer_type, neurons, activation_function, weight_range, bias_range, learning_rate=None):
        self.layer_type = layer_type
        self.neurons = neurons
        self.activation_function = activation_function
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.learning_rate = learning_rate