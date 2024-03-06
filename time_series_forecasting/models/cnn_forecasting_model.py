import torch
import torch.nn as nn


class CNNForecastingModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=24, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128, 50)
        self.dense2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # x = self.pool(x)
        # x = torch.relu(self.conv2(x))
        # x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.dense1(x))
        return self.dense2(x)
