import os
import json


class ConfigParser:

    def __init__(self, filename):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        self.file_location = os.path.join(parent_directory, "configs", filename)

    def parse_config_file(self):
        """
        Validate the config file and create layer objects and a final
        network object.
        """
        with open(self.file_location) as f:
            self.config = json.load(f)

        self._validate_globals()
        # validate input layer
        # validate hidden layers
        # validate output layer


        # create the objects
        print("done")
        

        # Rules
        # if output layer is type:softmax then the size of the layer is the same as the previous layer
        # each layer can have its own private learning rate, it must therefore override the global learning rate
        # this means the network object can have the globals, and each layer can have its own private variables
        # optional: bias range
        # if softmax, no weights
        # only output can be softmax
        # error if not one input and one output layer
        # error if input layer has more keywords than layer type and neurons
        # globals, loss function is required
        # globals learning rate is required
        # do not allow softmax for any other layer than output layer
    
    def _validate_globals(self):
        """
        Private function for validating the settings of global network variables in the 
        config file.
        """
        try:
            globals_config = self.config.get("GLOBALS", {})
        except:
            raise ValueError("Config missing GLOBALS")
        
        required = ["loss_function", "learning_rate"]
        optionals = ["weight_reg_rate", "weight_reg_type"]

        missing_required_keys = [param for param in required if param not in globals_config]
        missing_optional_keys = [param for param in optionals if param not in globals_config]
        
        if missing_required_keys or missing_optional_keys:
            missing_keys = ", ".join(missing_required_keys + missing_optional_keys)
            raise ValueError(f"Missing key(s) in config GLOBALS: {missing_keys}")

        required_keys_with_none_values = [param for param in required if param in globals_config and globals_config[param] is None]
        if required_keys_with_none_values:
            raise ValueError(f"Required key(s) set to None in config GLOBALS: {required_keys_with_none_values}")
        
        if globals_config["loss_function"] not in ("cross_entropy", "mse"):
            raise ValueError(f"In config GLOBALS: loss_function is not valid")
        
        if globals_config["weight_reg_type"] is not None and globals_config["weight_reg_type"] not in ("L1", "L2"):
            raise ValueError(f"In config GLOBALS: weight_reg_type must be None, L1 or L2")

    def _validate_input_layer(self):
        # Find all entries in LAYERS dict that has layer_type set to input
        try:
            layers_config = self.config.get("LAYERS", {})
        except:
            raise ValueError("Config missing LAYERS")
        
        input_layer = [layer for layer in layers_config if layer["layer_type"] == "input"]
        if not input_layer or len(input_layer) > 1:
            raise ValueError("Either no input layers or more than one input layer in config")

        required = ["layer_type", "neurons"]
        missing_required_keys = [param for param in required if param not in input_layer[0]]
        if missing_required_keys:
            raise ValueError(f"Missing key(s) in input layer: {missing_required_keys}")
