import torch
import torch.nn as nn


class RNNForecastingModel(nn.Module):

    def __init__(self, num_features, sequence_length):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=32, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(32, 50)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(50, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Feeding the last time step output into the dense layers
        x = lstm_out[:, -1, :]
        x = torch.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x
