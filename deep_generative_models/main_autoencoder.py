import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from datetime import datetime

from models.autoencoder import Autoencoder
from utils.stacked_mnist import StackedMNISTData, DataMode
from utils.verification_net import VerificationNet
from utils.visualization import visualize_images, visualize_reconstructions

# SETUP
MODE = "mono" # Options: "mono", "color"
TRAIN = False
MODEL_FILENAME = "mono_1712696920" # Set TRAIN to False to load
TRAIN_ANOMALY = False
ANOMALY_MODEL_FILENAME = "mono_missing_1712752424" # Set TRAIN_ANOMALY to False to load
MONO_ENCODING_DIM = 16
COLOR_ENCODING_DIM = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR_PATH, "data")
mono_dataset = StackedMNISTData(root=DATA_PATH, mode=DataMode.MONO | DataMode.BINARY)
color_dataset = StackedMNISTData(root=DATA_PATH, mode=DataMode.COLOR | DataMode.BINARY)
mono_missing_dataset = StackedMNISTData(root=DATA_PATH, mode=DataMode.MONO | DataMode.BINARY | DataMode.MISSING)
color_missing_dataset = StackedMNISTData(root=DATA_PATH, mode=DataMode.COLOR | DataMode.BINARY | DataMode.MISSING)
mono_data_loader = DataLoader(mono_dataset, batch_size=16, shuffle=True)
color_data_loader = DataLoader(color_dataset, batch_size=16, shuffle=True)
mono_missing_data_loader = DataLoader(mono_missing_dataset, batch_size=16, shuffle=True)
color_missing_data_loader = DataLoader(color_missing_dataset, batch_size=16, shuffle=True)

# TRAIN MODEL
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
    autoencoder = Autoencoder(channels=1, encoding_dim=MONO_ENCODING_DIM)
    data_loader = mono_data_loader
elif MODE == "color":
    autoencoder = Autoencoder(channels=3, encoding_dim=COLOR_ENCODING_DIM)
    data_loader = color_data_loader
else:
    raise ValueError("MODE must be either 'mono' or 'color'")

optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

if TRAIN:
    train(autoencoder, optimizer, loss_function, data_loader, num_epochs=NUM_EPOCHS)
    torch.save(autoencoder.state_dict(), os.path.join(CURRENT_DIR_PATH, "saved_models", "autoencoder", f"{MODE}_{int(datetime.now().timestamp())}"))
else:
    print("Loading model...")
    autoencoder.load_state_dict(torch.load(os.path.join(CURRENT_DIR_PATH, "saved_models", "autoencoder", MODEL_FILENAME)))

# CHECK ACCURACY
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
visualize_reconstructions(all_images, all_predictions)

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

# GENERATE NEW IMAGES
with torch.no_grad():
    x = all_images.permute(0, 3, 1, 2)
    x = autoencoder.encoder(x)
    x = autoencoder.bottleneck(x)
x = x.numpy()
# Mean of each dimension
mean = np.mean(x, axis=0)
# Standard deviation of each dimension
std = np.std(x, axis=0)
# Generate random encodings with mean and std
if MODE == "mono":
    z = np.random.normal(mean, std, (1000, MONO_ENCODING_DIM)).astype(np.float32)
elif MODE == "color":
    z = np.random.normal(mean, std, (1000, COLOR_ENCODING_DIM)).astype(np.float32)
z_tensor = torch.from_numpy(z)
with torch.no_grad():
    generated = autoencoder.expand(z_tensor)
    generated = generated.view(-1, 64, 4, 4)
    generated = autoencoder.decoder(generated)
    generated = generated.permute(0, 2, 3, 1)
if MODE == "mono":
    predictability, _ = net.check_predictability(data=generated, tolerance=0.8)
    coverage = net.check_class_coverage(data=generated, tolerance=0.8)
elif MODE == "color":
    predictability, _ = net.check_predictability(data=generated, tolerance=0.5)
    coverage = net.check_class_coverage(data=generated, tolerance=0.5)
if predictability != None:
    print(f"Predictability: {100 * predictability:.2f}%")
if coverage != None:
    print(f"Coverage: {100 * coverage:.2f}%")
visualize_images(title="Generated Images", images=generated[:16])

# ANOMALY DETECTION
if MODE == "mono":
    anomaly_autoencoder = Autoencoder(channels=1, encoding_dim=MONO_ENCODING_DIM)
    data_loader = mono_missing_data_loader
elif MODE == "color":
    anomaly_autoencoder = Autoencoder(channels=3, encoding_dim=COLOR_ENCODING_DIM)
    data_loader = color_missing_data_loader
else:
    raise ValueError("MODE must be either 'mono' or 'color'")

optimizer = optim.Adam(anomaly_autoencoder.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

if TRAIN_ANOMALY:
    train(anomaly_autoencoder, optimizer, loss_function, data_loader, num_epochs=NUM_EPOCHS)
    torch.save(anomaly_autoencoder.state_dict(), os.path.join(CURRENT_DIR_PATH, "saved_models", "autoencoder", f"{MODE}_missing_{int(datetime.now().timestamp())}"))
else:
    print("Loading model...")
    anomaly_autoencoder.load_state_dict(torch.load(os.path.join(CURRENT_DIR_PATH, "saved_models", "autoencoder", ANOMALY_MODEL_FILENAME)))

print("Calculating reconstruction loss...")
losses = []
with torch.no_grad():
    for image in all_images:
        image = image[None, :, :, :]  # Ensure image has a batch dimension
        reconstruction = anomaly_autoencoder(image)
        reconstruction_loss = loss_function(reconstruction, image).item()
        losses.append((image, reconstruction_loss))
sorted_losses = sorted(losses, key=lambda x: x[1], reverse=True)
print([x[1] for x in sorted_losses[:16]])
most_anomalous_images = torch.cat([x[0] for x in sorted_losses], dim=0)[:16]
visualize_images("Anomaly Detection", most_anomalous_images)
