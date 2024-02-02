from utils.config_parser import ConfigParser
from utils.data_generator import DataGenerator
from utils.visualizer import Visualizer

# TODO move verbose to network class, maybe config file
VERBOSE = True

# Parse config file
config_parser = ConfigParser()
network = config_parser.parse_config_file("config_1.json")

# Generate data
data_generator = DataGenerator()
train, validation, test = data_generator.generate_dataset(count=100, train_size=0.7, validation_size=0.2, test_size=0.1, n=50, wr=[
                                                          0.2, 0.5], hr=[0.2, 0.4], noise=0, types=("ring", "frame", "flower", "triangle"), center=True, flatten=False)

# Create visualizer
visualizer = Visualizer()
# View images
visualizer.view_images(train, num_images=3)
