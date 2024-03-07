import torch
import torch.nn as nn


class CNNForecastingModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64 * 12, 50)
        self.dense2 = nn.Linear(50, 1)

    def forward(self, x):
        # Permute x from [batch_size, sequence_length, num_channels] to [batch_size, num_channels, sequence_length]
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.dense1(x))
        return self.dense2(x)
