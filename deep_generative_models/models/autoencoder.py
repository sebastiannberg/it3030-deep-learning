import torch.nn as nn


class PrintSize(nn.Module):

    def __init__(self, message):
        super().__init__()
        self.message = message

    def forward(self, x):
        print(self.message, x.size())
        return x

class Autoencoder(nn.Module):

    def __init__(self, channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure input x is in the [batch_size, channels, height, width] format
        x = x.permute(0, 3, 1, 2)

        x = self.encoder(x)
        x = self.decoder(x)

        # Back to original shape
        x = x.permute(0, 2, 3, 1)

        return x
