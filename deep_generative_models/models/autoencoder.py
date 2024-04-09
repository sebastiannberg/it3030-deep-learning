import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, channels=1, encoding_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=encoding_dim)
        )

        self.expand = nn.Sequential(
            nn.Linear(in_features=encoding_dim, out_features=1024),
            nn.ReLU()
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
        x = self.bottleneck(x)
        x = self.expand(x)
        # Upsample before decoding
        x = x.view(-1, 64, 4, 4)
        x = self.decoder(x)

        # Back to original shape
        x = x.permute(0, 2, 3, 1)

        return x
