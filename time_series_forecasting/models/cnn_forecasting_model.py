import torch
import torch.nn as nn


class CNNForecastingModel(nn.Module):

    def __init__(self, num_features, sequence_length):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        # Each pooling halves sequence length
        sequence_length_after_pooling = sequence_length // (2 * 2)
        self.dense1 = nn.Linear(128 * sequence_length_after_pooling, 50)
        self.dense2 = nn.Linear(50, 1)

    def forward(self, x):
        # Permute x from [batch_size, sequence_length, num_channels] to [batch_size, num_channels, sequence_length]
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        return x
