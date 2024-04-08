import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from models.autoencoder import Autoencoder
from utils.stacked_mnist import StackedMNISTData, DataMode
from utils.verification_net import VerificationNet
from utils.visualization import visualize_generated_examples

# Setup
TRAIN = True

current_dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_dir_path, "data")
mono_dataset = StackedMNISTData(root=data_path, mode=DataMode.MONO | DataMode.BINARY)
mono_data_loader = DataLoader(mono_dataset, batch_size=16, shuffle=True)
color_dataset = StackedMNISTData(root=data_path, mode=DataMode.COLOR | DataMode.BINARY)
color_data_loader = DataLoader(color_dataset, batch_size=16, shuffle=True)

# Train Model
def train(model, optimizer, loss_function, data_loader, num_epochs):
    print("\033[1;32m" + "="*15 + " Training " + "="*15 + "\033[0m")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = len(data_loader)
        for batch_idx, (images, _) in enumerate(data_loader):
            print(f"\rBatch {batch_idx+1} of {total_batches}", end="")
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print()
        print(f"Epoch {epoch+1}, Total Loss: {total_loss}")
    end_time = time.time()
    minutes, seconds = divmod(end_time-start_time, 60)
    print(f"Total training time: {int(minutes)} minutes and {round(seconds, 2)} seconds")

autoencoder = Autoencoder(channels=1)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
loss_function = nn.MSELoss()
if TRAIN:
    train(autoencoder, optimizer, loss_function, mono_data_loader, num_epochs=3)

# Generate Examples
generated_examples_mono = torch.zeros((16, 28, 28, 1))

# Visualize Generated Examples
visualize_generated_examples(generated_examples_mono)

# Verification Net
verification_net_path = os.path.join(current_dir_path, "utils", "saved_weights", "verification_net.mono.weights.h5")
net = VerificationNet(file_name=verification_net_path)
cov = net.check_class_coverage(data=generated_examples_mono, tolerance=0.98)
pred, acc = net.check_predictability(data=generated_examples_mono)
if cov != None:
    print(f"Coverage: {100*cov:.2f}%")
if pred != None:
    print(f"Predictability: {100*pred:.2f}%")
if acc != None:
    print(f"Accuracy: {100 * acc:.2f}%")
