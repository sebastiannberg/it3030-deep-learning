import numpy as np

from network.network import Network
from network.layer import Layer

np.random.seed(123)

network = Network(loss_function="mse",
                  learning_rate=0.01,
                  weight_reg_rate=None,
                  weight_reg_type=None,
                  verbose=True)

input_layer = Layer(layer_type="input",
                    neurons=3)

hidden_layer = Layer(layer_type="hidden",
                     neurons=2,
                     activation_function="sigmoid",
                     weight_range=[-0.1, 0.1],
                     bias_range=[0, 1])

output_layer = Layer(layer_type="output",
                     neurons=2,
                     activation_function="sigmoid",
                     weight_range=[-0.1, 0.1],
                     bias_range=[0, 1])

network.add_layer(input_layer)
network.add_layer(hidden_layer)
network.add_layer(output_layer)

# TEST weight/bias matrix shape
assert hidden_layer.weights.shape == (3, 2), f"Hidden layer weights shape {hidden_layer.weights.shape} does not match expected shape (3, 2)"
assert hidden_layer.bias.shape == (1, 2), f"Hidden layer bias shape {hidden_layer.bias.shape} does not match expected shape (1, 2)"
assert output_layer.weights.shape == (2, 2), f"Output layer weights shape {output_layer.weights.shape} does not match expected shape (2, 1)"
assert output_layer.bias.shape == (1, 2), f"Output layer bias shape {output_layer.bias.shape} does not match expected shape (1, 1)"

# TEST Layer class
# Set weights manually
V = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
W = np.array([[0.7, 0.8], [0.9, 1.0]])
hidden_layer._set_weights(V)
output_layer._set_weights(W)
# Set bias to 0
hidden_layer._set_bias(np.array([[0, 0]]))
output_layer._set_bias(np.array([[0, 0]]))
assert np.array_equal(hidden_layer.weights, np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
assert np.array_equal(hidden_layer.bias, [[0, 0]])
assert np.array_equal(output_layer.weights, np.array([[0.7, 0.8], [0.9, 1.0]]))
assert np.array_equal(output_layer.bias, [[0, 0]])
# Forward pass
X = np.array([[0.2, 0.3], [0.8, 0.9], [1.4, 1.5]])
expected_hidden_layer_sum = np.array([[0.96, 1.05], [1.2, 1.32]])
hidden_layer_sum = hidden_layer.compute_sum(X)
assert np.array_equal(hidden_layer_sum, expected_hidden_layer_sum), f"Hidden layer sum is {hidden_layer_sum} but expected {expected_hidden_layer_sum}"
expected_hidden_layer_output = np.array([[0.7231218051, 0.7407748992], [0.7685247835, 0.7891817065]])
hidden_layer_output = hidden_layer.compute_output(hidden_layer_sum)
assert np.allclose(hidden_layer_output, expected_hidden_layer_output), f"Hidden layer output is {hidden_layer_output} but expected {expected_hidden_layer_output}"
hidden_layer._cache_X_sum_output(X, hidden_layer_sum, hidden_layer_output)
expected_output_layer_sum = np.array([[1.197857571, 1.228805965], [1.347022228, 1.381801626]])
output_layer_sum = output_layer.compute_sum(hidden_layer_output)
assert np.allclose(output_layer_sum, expected_output_layer_sum), f"Output layer sum is {output_layer_sum} but expected {expected_output_layer_sum}"
expected_output_layer_output = np.array([[0.7681434381, 0.7736095219], [0.7936423725, 0.7992801934]])
output_layer_output = output_layer.compute_output(output_layer_sum)
assert np.allclose(output_layer_output, expected_output_layer_output), f"Output layer output is {output_layer_output} but expected {expected_output_layer_output}"
output_layer._cache_X_sum_output(hidden_layer_output, output_layer_sum, output_layer_output)
# Backward pass
# TODO double check every jacobian shape, is  i numerator or denominator layout
targets = np.array([[0.9, 1.3], [1.2, 1.5]])
expected_loss = np.array([[0.09125633717, 0.3840475914]])
loss = network.compute_loss(predicted_outputs=output_layer_output, targets=targets)
assert np.allclose(loss, expected_loss), f"Loss is {loss} but expected {expected_loss}"
expected_jacobian_L_Z = np.array([
    [[[-0.1318565619, 0]], [[0, -0.5263904781]]],
    [[[-0.4063576275, 0]], [[0, -0.700719806]]]
])
jacobian_L_Z = network.compute_jacobian_L_Z(predicted_outputs=output_layer_output, targets=targets)
assert np.allclose(jacobian_L_Z, expected_jacobian_L_Z), f"Jacobian_L_Z is {jacobian_L_Z} but expected {expected_jacobian_L_Z}"
expected_jacobian_Z_sum = np.array([
    [
        [[0.1780990966, 0],
         [0, 0]],
        [[0, 0.1751378295],
         [0,0]]
    ],
    [
        [[0, 0],
         [0.1637741571, 0]],
        [[0, 0],
         [0, 0.1604313658]]
    ]
])
jacobian_Z_sum = output_layer.compute_jacobian_Z_sum()
assert np.allclose(jacobian_Z_sum, expected_jacobian_Z_sum), f"\nJacobian_Z_S is\n{jacobian_Z_sum}\nbut expected\n{expected_jacobian_Z_sum}"
expected_jacobian_Z_Y = np.array([
    [
        [[0.1246693676, 0],
         [0.1310193257, 0]],
        [[0, 0.1225964807],
         [0, 0.1283450927]]
    ],
    [
        [[0.1602891869, 0],
         [0.1637741571, 0]],
        [[0, 0.1576240466],
         [0, 0.1604313658]]
    ]
])
jacobian_Z_Y = output_layer.compute_jacobian_Z_Y(jacobian_Z_sum)
assert np.allclose(jacobian_Z_Y, expected_jacobian_Z_Y), f"\nJacobian_Z_Y is\n{jacobian_Z_Y}\nbut expected\n{expected_jacobian_Z_Y}"
expected_jacobian_Z_W = np.array([
    [
        [[0.1287873402, 0.12973708],
         [0, 0]],
        [[0, 0],
         [0.1184286641, 0.1188435289]]
    ],
    [
        [[0.1368735697, 0.1382155712],
         [0, 0]],
        [[0, 0],
         [0.1258644986, 0.1266094991]]
    ]
])
jacobian_Z_W = output_layer.compute_jacobian_Z_W(jacobian_Z_sum)
assert np.allclose(jacobian_Z_W, expected_jacobian_Z_W), f"\nJacobian_Z_W is\n{jacobian_Z_W}\nbut expected\n{expected_jacobian_Z_W}"
expected_jacobian_Z_B = np.array([
    [
        [[0.1780990966, 0.1751378295],
         [0, 0]],
        [[0, 0],
         [0.1637741571, 0.1604313658]]
    ]
])
jacobian_Z_B = output_layer.compute_jacobian_Z_B(jacobian_Z_sum)
assert np.allclose(jacobian_Z_B, expected_jacobian_Z_B), f"\nJacobian_Z_B is\n{jacobian_Z_B}\nbut expected\n{expected_jacobian_Z_B}"
expected_jacobian_L_W = np.array([
    [[[-0.0169814559, -0.06829236357]], [[-0.04812439097, -0.08327601452]]],
    [[[-0.01804767832, -0.0727553606]], [[-0.05114599904, -0.08871778365]]]
])
jacobian_L_W = output_layer.compute_jacobian_L_W(jacobian_L_Z, jacobian_Z_W)
assert np.allclose(jacobian_L_W, expected_jacobian_L_W), f"\nJacobian_L_W is\n{jacobian_L_W}\nbut expected\n{expected_jacobian_L_W}"
expected_jacobian_L_B = np.array([
    [[[-0.02348353456, -0.09219088582]], [[-0.06655087791, -0.1124174356]]]
])
jacobian_L_B = output_layer.compute_jacobian_L_B(jacobian_L_Z, jacobian_Z_B)
assert np.allclose(jacobian_L_B, expected_jacobian_L_B), f"\nJacobian_L_B is\n{jacobian_L_B}\nbut expected\n{expected_jacobian_L_B}"
expected_jacobian_L_Y = np.array([
    [[[-0.06967917652, 0]], [[0, -0.1544675686]]],
    [[[-0.08768605901, 0]], [[0, -0.1953892329]]]
])
jacobian_L_Y = output_layer.compute_jacobian_L_Y(jacobian_L_Z, jacobian_Z_Y)
assert np.allclose(jacobian_L_Y, expected_jacobian_L_Y), f"\nJacobian_L_Y is\n{jacobian_L_Y}\nbut expected\n{expected_jacobian_L_Y}"

# TODO TEST Network class? already testing network.compute_jacobian_L_Z and network.compute_loss
