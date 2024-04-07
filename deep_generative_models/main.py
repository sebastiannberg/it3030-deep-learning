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

# Generate Examples
generated_examples = torch.zeros((16, 28, 28, 1))

# Visualize Generated Examples
visualize_generated_examples(generated_examples)

# Verification Net
verification_net_path = os.path.join(current_dir_path, "utils", "saved_weights", "verification_net.mono.weights.h5")
net = VerificationNet(file_name=verification_net_path)
cov = net.check_class_coverage(data=generated_examples, tolerance=0.98)
pred, acc = net.check_predictability(data=generated_examples)
if cov != None:
    print(f"Coverage: {100*cov:.2f}%")
if pred != None:
    print(f"Predictability: {100*pred:.2f}%")
if acc != None:
    print(f"Accuracy: {100 * acc:.2f}%")
