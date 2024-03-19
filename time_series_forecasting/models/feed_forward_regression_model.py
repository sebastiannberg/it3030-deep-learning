import torch
import torch.nn as nn


class FeedForwardRegressionModel(nn.Module):

    def __init__(self, num_features, sequence_length):
        super().__init__()
        self.flatten = nn.Flatten()
        input_size = num_features * sequence_length
        self.dense1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x
