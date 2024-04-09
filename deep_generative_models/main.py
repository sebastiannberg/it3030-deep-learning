import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from datetime import datetime

from models.autoencoder import Autoencoder
from utils.stacked_mnist import StackedMNISTData, DataMode
from utils.verification_net import VerificationNet
from utils.visualization import visualize_reconstructions, visualize_generated_examples

# Setup
TRAIN = False
MODE = "mono" # Options: "mono", "color"
MODEL_FILENAME = "autoencoder_mono_1712603988" # Set TRAIN to False to load
NUM_EPOCHS = 7
LEARNING_RATE = 0.001
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR_PATH, "data")
mono_dataset = StackedMNISTData(root=DATA_PATH, mode=DataMode.MONO | DataMode.BINARY)
color_dataset = StackedMNISTData(root=DATA_PATH, mode=DataMode.COLOR | DataMode.BINARY)
mono_data_loader = DataLoader(mono_dataset, batch_size=16, shuffle=True)
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

if MODE == "mono":
    autoencoder = Autoencoder(channels=1)
    data_loader = mono_data_loader
elif MODE == "color":
    autoencoder = Autoencoder(channels=3)
    data_loader = color_data_loader
else:
    raise ValueError("MODE must be either 'mono' or 'color'")

optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

if TRAIN:
    train(autoencoder, optimizer, loss_function, data_loader, num_epochs=NUM_EPOCHS)
    torch.save(autoencoder.state_dict(), os.path.join(CURRENT_DIR_PATH, "saved_models", f"autoencoder_{MODE}_{int(datetime.now().timestamp())}"))
else:
    print("Loading model...")
    autoencoder.load_state_dict(torch.load(os.path.join(CURRENT_DIR_PATH, "saved_models", MODEL_FILENAME)))

# Check Accuracy
verification_net_path = os.path.join(CURRENT_DIR_PATH, "utils", "saved_weights", "mono_float_complete.weights.h5")
net = VerificationNet(file_name=verification_net_path)

all_images, all_targets = [], []
for images, targets in data_loader:
    all_images.append(images)
    all_targets.append(targets)
all_images = torch.cat(all_images, dim=0)
all_targets = torch.cat(all_targets, dim=0)
with torch.no_grad():
    all_predictions = autoencoder(all_images)
# visualize_reconstructions(all_images, all_predictions)

if MODE == "mono":
    tolerance = 0.8
    predictability, acc = net.check_predictability(data=all_predictions, correct_labels=all_targets, tolerance=tolerance)
elif MODE == "color":
    tolerance = 0.5
    # Correctly order channels for hundreds, tens, and ones
    all_predictions = torch.flip(all_predictions, (3,))
    predictability, acc = net.check_predictability(data=all_predictions, correct_labels=all_targets, tolerance=tolerance)

if predictability != None:
    print(f"Predictability: {100 * predictability:.2f}%")
if acc != None:
    print(f"Accuracy: {100 * acc:.2f}%")

# Generate Examples
generated_examples_mono = torch.zeros((16, 28, 28, 1))

# Visualize Generated Examples
visualize_generated_examples(generated_examples_mono)
