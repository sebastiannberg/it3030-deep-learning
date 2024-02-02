

class Layer:

    def __init__(self, layer_type, neurons, activation, weight_range=None, bias_range=None, lrate=None):
        self.layer_type = layer_type
        self.neurons = neurons
        self.activation = activation
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.lrate = lrate