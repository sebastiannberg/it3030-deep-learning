import torch
from torch.utils.data import DataLoader

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import numpy as np
import os
import time
from datetime import datetime

from models.vae import VAE
from utils.stacked_mnist import StackedMNISTData, DataMode
from utils.verification_net import VerificationNet
from utils.visualization import visualize_images, visualize_reconstructions

# SETUP
MODE = "mono" # Options: "mono", "color"
TRAIN = False
MODEL_FILENAME = "mono_1712842393" # Set TRAIN to False to load
MONO_ENCODING_DIM = 16
COLOR_ENCODING_DIM = 128
NUM_EPOCHS = 3
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
def train(svi, data_loader):
    print("\033[1;32m" + "="*15 + " Training " + "="*15 + "\033[0m")
    start_time = time.time()
    elbo = []
    for epoch in range(NUM_EPOCHS):
        total_epoch_loss = train_one_epoch(svi, data_loader)
        elbo.append(-total_epoch_loss)
        print(f"Epoch {epoch+1}, Total Loss: {total_epoch_loss}")
    end_time = time.time()
    minutes, seconds = divmod(end_time-start_time, 60)
    print(f"Total training time: {int(minutes)} minutes and {round(seconds, 2)} seconds")

def train_one_epoch(svi, data_loader):
    epoch_loss = 0.
    total_batches = len(data_loader)
    for batch_idx, (images, _) in enumerate(data_loader):
        print(f"\rBatch {batch_idx+1} of {total_batches}", end="")
        epoch_loss += svi.step(images)
    print()
    normalizer_train = len(data_loader.dataset)
    total_epoch_loss = epoch_loss / normalizer_train
    return total_epoch_loss

if MODE == "mono":
    vae = VAE(channels=1, encoding_dim=MONO_ENCODING_DIM)
    data_loader = mono_data_loader
elif MODE == "color":
    vae = VAE(channels=3, encoding_dim=COLOR_ENCODING_DIM)
    data_loader = color_data_loader
else:
    raise ValueError("MODE must be either 'mono' or 'color'")

adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

if TRAIN:
    train(svi, data_loader)
    torch.save(vae.state_dict(), os.path.join(CURRENT_DIR_PATH, "saved_models", "vae", f"{MODE}_{int(datetime.now().timestamp())}"))
else:
    print("Loading model...")
    vae.load_state_dict(torch.load(os.path.join(CURRENT_DIR_PATH, "saved_models", "vae", MODEL_FILENAME)))

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
    reconstructions = vae.reconstruct_img(all_images)
visualize_reconstructions(all_images, reconstructions)

if MODE == "mono":
    tolerance = 0.8
    predictability, acc = net.check_predictability(data=reconstructions, correct_labels=all_targets, tolerance=tolerance)
elif MODE == "color":
    tolerance = 0.5
    # Correctly order channels for hundreds, tens, and ones
    reconstructions = torch.flip(reconstructions, (3,))
    predictability, acc = net.check_predictability(data=reconstructions, correct_labels=all_targets, tolerance=tolerance)
if predictability != None:
    print(f"Predictability: {100 * predictability:.2f}%")
if acc != None:
    print(f"Accuracy: {100 * acc:.2f}%")
