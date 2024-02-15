import os
import json

from network.layer import Layer
from network.network import Network


class ConfigParser:

    def parse_config_file(self, filename: str):
        """
        Validate the config file and create layer objects and a final
        network object.
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        file_location = os.path.join(parent_directory, "configs", filename)

        with open(file_location) as f:
            self.config = json.load(f)

        self._validate_globals()
        self._validate_input_layer()
        self._validate_hidden_layers()
        self._validate_output_layer()

        globals_config = self.config["GLOBALS"]
        network = Network(loss_function=globals_config["loss_function"],
                          learning_rate=globals_config["learning_rate"],
                          weight_reg_rate=globals_config["weight_reg_rate"],
                          weight_reg_type=globals_config["weight_reg_type"],
                          verbose=globals_config["verbose"])

        input_layer_config = [layer for layer in self.config["LAYERS"] if layer["layer_type"] == "input"][0]
        input_layer = Layer(layer_type=input_layer_config["layer_type"],
                            neurons=input_layer_config["neurons"])
        network.add_layer(input_layer)

        hidden_layers_configs = [layer for layer in self.config["LAYERS"] if layer["layer_type"] == "hidden"]
        for layer_config in hidden_layers_configs:
            hidden_layer = Layer(layer_type=layer_config["layer_type"],
                                 neurons=layer_config["neurons"],
                                 activation_function=layer_config["activation_function"],
                                 weight_range=layer_config["weight_range"],
                                 bias_range=layer_config["bias_range"],
                                 learning_rate=layer_config["learning_rate"])
            network.add_layer(hidden_layer)

        output_layer_config = [
            layer for layer in self.config["LAYERS"] if layer["layer_type"] == "output"][0]
        output_layer = Layer(layer_type=output_layer_config["layer_type"],
                             neurons=output_layer_config["neurons"],
                             activation_function=output_layer_config["activation_function"],
                             weight_range=output_layer_config["weight_range"],
                             bias_range=output_layer_config["bias_range"],
                             learning_rate=output_layer_config["learning_rate"])
        network.add_layer(output_layer)

        return network

    def _validate_globals(self):
        """
        Private function for validating the settings of global network variables in the
        config file.
        """
        try:
            globals_config = self.config.get("GLOBALS", {})
        except:
            raise ValueError("Config missing GLOBALS")

        required = ["loss_function", "learning_rate", "verbose"]
        optional = ["weight_reg_rate", "weight_reg_type"]

        missing_required_keys = [
            param for param in required if param not in globals_config]
        missing_optional_keys = [
            param for param in optional if param not in globals_config]

        if missing_required_keys or missing_optional_keys:
            missing_keys = ", ".join(
                missing_required_keys + missing_optional_keys)
            raise ValueError(
                f"Missing key(s) in config GLOBALS: {missing_keys}")

        required_keys_with_none_values = [
            param for param in required if param in globals_config and globals_config[param] is None]
        if required_keys_with_none_values:
            raise ValueError(
                f"Required key(s) set to None in config GLOBALS: {required_keys_with_none_values}")

        if globals_config["loss_function"] not in ("cross_entropy", "mse"):
            raise ValueError(
                f"In config GLOBALS: loss_function must be cross_entropy or mse")

        if globals_config["weight_reg_type"] is not None and globals_config["weight_reg_type"] not in ("L1", "L2"):
            raise ValueError(
                f"In config GLOBALS: weight_reg_type must be None, L1 or L2")

        unexpected_keys = [key for key in globals_config if key not in required + optional]
        if unexpected_keys:
            raise ValueError(
                f"Unexpected key(s) in config GLOBALS: {', '.join(unexpected_keys)}")

    def _validate_input_layer(self):
        try:
            layers_config = self.config.get("LAYERS", {})
        except:
            raise ValueError("Config missing LAYERS")

        input_layer = [
            layer for layer in layers_config if layer["layer_type"] == "input"]
        if not input_layer or len(input_layer) > 1:
            raise ValueError(
                "Either no input layer or more than one input layer in config")

        required = ["layer_type", "neurons"]

        missing_required_keys = [
            param for param in required if param not in input_layer[0]]
        if missing_required_keys:
            raise ValueError(
                f"Missing key(s) in input layer: {missing_required_keys}")

        required_keys_with_none_values = [
            param for param in required if param in input_layer[0] and input_layer[0][param] is None]
        if required_keys_with_none_values:
            raise ValueError(
                f"Required key(s) is set to None: {required_keys_with_none_values}")

        unexpected_keys = [
            key for key in input_layer[0] if key not in required]
        if unexpected_keys:
            raise ValueError(
                f"Unexpected key(s) in input layer: {', '.join(unexpected_keys)}")

    def _validate_hidden_layers(self):
        try:
            layers_config = self.config.get("LAYERS", {})
        except:
            raise ValueError("Config missing LAYERS")

        hidden_layers = [
            layer for layer in layers_config if layer["layer_type"] == "hidden"]
        if not hidden_layers:
            return

        required = ["layer_type", "neurons",
                    "activation_function", "weight_range", "bias_range"]
        optional = ["learning_rate"]

        for layer in hidden_layers:
            missing_required_keys = [
                param for param in required if param not in layer]
            missing_optional_keys = [
                param for param in optional if param not in layer]
            if missing_required_keys or missing_optional_keys:
                raise ValueError(
                    f"Missing key(s) in hidden layer: {', '.join(missing_required_keys + missing_optional_keys)}")

            required_keys_with_none_values = [
                param for param in required if layer[param] is None]
            if required_keys_with_none_values:
                raise ValueError(
                    f"Required key(s) with None value in hidden layer: {required_keys_with_none_values}")

            if layer["activation_function"] not in ("relu", "identity", "sigmoid"):
                raise ValueError(
                    f"Invalid activation function for hidden layer: {layer['activation_function']}")

            if not isinstance(layer["weight_range"], list) or len(layer["weight_range"]) != 2:
                raise ValueError(
                    f"Invalid weight_range in hidden layer. It should be a list of two integers or floats")

            if not isinstance(layer["bias_range"], list) or len(layer["bias_range"]) != 2:
                raise ValueError(
                    f"Invalid bias_range in hidden layer. It should be a list of two integers or floats")

            unexpected_keys = [
                key for key in layer if key not in required + optional]
            if unexpected_keys:
                raise ValueError(
                    f"Unexpected key(s) in hidden layer: {', '.join(unexpected_keys)}")

    def _validate_output_layer(self):
        # TODO
        try:
            layers_config = self.config.get("LAYERS", {})
        except:
            raise ValueError("Config missing LAYERS")

        output_layer = [
            layer for layer in layers_config if layer["layer_type"] == "output"]
        if not output_layer or len(output_layer) > 1:
            raise ValueError(
                "Either no output layer or more than one output layer in config")

        # TODO

        if output_layer[0]["activation_function"] not in ("softmax", "identity", "relu", "sigmoid"):
            raise ValueError(
                f"Invalid activation function in output layer: {output_layer[0]['activation_function']}")
