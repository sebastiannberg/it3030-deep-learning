import torch
from torch.utils.data import DataLoader
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.stacked_mnist import StackedMNISTData, DataMode
from utils.verification_net import VerificationNet
from utils.visualization import visualize_generated_examples


current_dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_dir_path, "data")
dataset = StackedMNISTData(root=data_path, mode=DataMode.MONO | DataMode.BINARY)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

generated_examples = torch.zeros((16, 28, 28))
# Visualize Generated Examples
visualize_generated_examples(generated_examples)

# Verification Net
verification_net_path = os.path.join(current_dir_path, "saved_models", "verification_net", "mono")
verification_net = VerificationNet(file_name=verification_net_path)
predictability, accuracy = verification_net.check_predictability(generated_examples)
print(predictability, accuracy)
