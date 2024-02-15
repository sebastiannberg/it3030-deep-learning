import numpy as np

from utils.config_parser import ConfigParser
from utils.data_generator import DataGenerator
from utils.visualizer import Visualizer


# Parse config file
config_parser = ConfigParser()
network = config_parser.parse_config_file("config_1.json")

# Generate data
data_generator = DataGenerator()
train, validation, test = data_generator.generate_dataset(count=1000,
                                                          train_size=0.7,
                                                          validation_size=0.2,
                                                          test_size=0.1,
                                                          n=50,
                                                          wr=[0.3, 0.5],
                                                          hr=[0.2, 0.6],
                                                          noise=0,
                                                          types=("ring", "frame", "flower", "triangle"),
                                                          center=True,
                                                          flatten=False)

# Preprocess data
train_X = np.sum(train[0], axis=-1)
train_y = train[1]
validation_X = np.sum(validation[0], axis=-1)
validation_y = validation[1]
test_X = np.sum(validation[0], axis=-1)
test_y = test[1]

# View images
visualizer = Visualizer()
# visualizer.view_images(train, num_images=5)

# Train network
train_loss, validation_loss = network.fit(train_X, train_y, validation_X, validation_y, minibatch_size=4, num_minibatches=100)

# Test network using network.predict()

# Plot learning progression
visualizer.plot_learning_progression(train_loss=train_loss, validation_loss=validation_loss)
