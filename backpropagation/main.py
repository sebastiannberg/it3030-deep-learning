from utils.config_parser import ConfigParser


# Parse config file
config_parser = ConfigParser()
network = config_parser.parse_config_file("config_1.json")
print(network.get_layers())

# Generate data
