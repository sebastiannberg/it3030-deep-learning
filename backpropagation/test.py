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
                     neurons=1,
                     activation_function="sigmoid",
                     weight_range=[-0.1, 0.1],
                     bias_range=[0, 1])

network.add_layer(input_layer)
network.add_layer(hidden_layer)
network.add_layer(output_layer)

# Test weight/bias matrix shape
assert hidden_layer.weights.shape == (3, 2), f"Hidden layer weights shape {hidden_layer.weights.shape} does not match expected shape (3, 2)"
assert hidden_layer.bias.shape == (1, 2), f"Hidden layer bias shape {hidden_layer.bias.shape} does not match expected shape (1, 2)"
assert output_layer.weights.shape == (2, 1), f"Output layer weights shape {output_layer.weights.shape} does not match expected shape (2, 1)"
assert output_layer.bias.shape == (1, 1), f"Output layer bias shape {output_layer.bias.shape} does not match expected shape (1, 1)"

# Set weights
V = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
W = np.array([[0.7], [0.8]])
hidden_layer._set_weights(V)
output_layer._set_weights(W)
# Set bias to 0
hidden_layer._set_bias(np.array([[0, 0]]))
output_layer._set_bias(np.array([[0]]))
assert np.array_equal(hidden_layer.weights, np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
assert np.array_equal(hidden_layer.bias, [[0, 0]])
assert np.array_equal(output_layer.weights, np.array([[0.7], [0.8]]))
assert np.array_equal(output_layer.bias, [[0]])

# Forward pass
X = np.array([[0.2], [0.8], [1.4]])
R = np.array([[0.96], [1.2]])
hidden_layer_sum = hidden_layer.compute_sum(X)
assert np.array_equal(hidden_layer_sum, R)
Y = np.array([[0.7231218051], [0.7685247835]])
hidden_layer_output = hidden_layer.compute_output(hidden_layer_sum)
assert np.allclose(hidden_layer_output, Y, atol=0.00001), f"Hidden layer output is {hidden_layer_output} but should be {Y}"
hidden_layer.cache_X_sum_output(X, hidden_layer_sum, hidden_layer_output)
S = np.array([[1.12100509]])
output_layer_sum = output_layer.compute_sum(hidden_layer_output)
assert np.allclose(output_layer_sum, S, atol=0.00001), f"Output layer sum is {output_layer_sum} but should be {S}"
Z = np.array([[0.7541751027]])
output_layer_output = output_layer.compute_output(output_layer_sum)
assert np.allclose(output_layer_output, Z, atol=0.00001), f"Output layer output is {output_layer_output} but should be {Z}"
output_layer.cache_X_sum_output(hidden_layer_output, output_layer_sum, output_layer_output)

# Backward pass
targets = np.array([[0.8923]])
expected_loss = 0.01907848725
loss = network.compute_loss(predicted_outputs=output_layer_output, targets=targets)
assert np.allclose(loss, expected_loss, atol=0.00001), f"Loss is {loss} but expected {expected_loss}"
expected_jacobian_L_Z = np.array([[-0.2762497946]])
jacobian_L_Z = network.compute_jacobian_L_Z(predicted_outputs=output_layer_output, targets=targets)
assert np.allclose(jacobian_L_Z, expected_jacobian_L_Z, atol=0.00001), f"Jacobian_L_Z was {jacobian_L_Z} but expected {expected_jacobian_L_Z}"
expected_jacobian_Z_S = np.array([[0.1853950172]])
jacobian_Z_S = output_layer.compute_jacobian_Z_sum()
assert np.allclose(jacobian_Z_S, expected_jacobian_Z_S, atol=0.00001), f"Jacobian_Z_S was {jacobian_Z_S} but expected {expected_jacobian_Z_S}"
expected_jacobian_Z_W_hat = np.array([[0.1340631795], [0.1424806655]])
jacobian_Z_W_hat = output_layer.compute_jacobian_Z_W_hat(jacobian_Z_S)
assert np.allclose(jacobian_Z_W_hat, expected_jacobian_Z_W_hat, atol=0.00001), f"Jacobian_Z_W_hat was {jacobian_Z_W_hat} but expected {expected_jacobian_Z_W_hat}"
expected_jacobian_L_W = np.array([[-0.0370349258, -0.03936025458]])
jacobian_L_W = output_layer.compute_jacobian_L_W(jacobian_L_Z, jacobian_Z_W_hat)
assert np.allclose(jacobian_L_W, expected_jacobian_L_W, atol=0.00001), f"Jacobian_L_W was {jacobian_L_W} but expected {expected_jacobian_L_W}"
