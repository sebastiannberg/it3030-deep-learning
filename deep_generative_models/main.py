import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.stacked_mnist import DataMode, StackedMNISTData
from utils.visualization import visualize_dataset


# Initialize your dataset
current_dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_dir_path, "data")
dataset = StackedMNISTData(root=data_path, mode=DataMode.MONO | DataMode.BINARY)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Visualize the dataset
visualize_dataset(data_loader, num_images=16)
