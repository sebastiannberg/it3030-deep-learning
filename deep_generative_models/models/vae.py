import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist


class VAE(nn.Module):

    def __init__(self, channels=1, encoding_dim=16):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.encoder = Encoder(channels, encoding_dim)
        self.decoder = Decoder(channels, encoding_dim)

    # Define the model p(x|z)p(z)
    def model(self, x):
        # Register PyTorch module "decoder" with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # Setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.encoding_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.encoding_dim)))
            # Sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # Decode the latent code z
            img = self.decoder(z)
            # Score against actual images
            pyro.sample("obs", dist.Bernoulli(probs=img).to_event(3), obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        img = self.decoder(z)
        return img

class Encoder(nn.Module):
    def __init__(self, channels, encoding_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_z_loc = nn.Linear(in_features=1024, out_features=encoding_dim)
        self.fc_z_scale = nn.Linear(in_features=1024, out_features=encoding_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        z_loc = self.fc_z_loc(x)
        z_scale = torch.exp(self.fc_z_scale(x))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, channels, encoding_dim):
        super().__init__()
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
        x = self.expand(x)
        # Upsample before decoding
        x = x.view(-1, 64, 4, 4)
        x = self.decoder(x)
        x = x.permute(0, 2, 3, 1)
        return x
