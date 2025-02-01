import torch
import torch.nn as nn
from pyro.contrib.examples.util import get_data_loader

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import AdamW

from mnist import mnist_bar


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.hidden = nn.Linear(784, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.log_var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = torch.tanh(self.hidden(x))
        z_mean = self.mean(hidden)
        z_log_var = self.log_var(hidden)
        return z_mean, z_log_var


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.hidden = nn.Linear(z_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 784)

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = torch.tanh(self.hidden(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        img = torch.sigmoid(self.output(hidden))
        return img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=16, hidden_dim=512, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Normal(loc_img, 1, validate_args=False).to_event(1),
                obs=x.reshape(-1, 784),
            )
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_mean, z_log_var = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_mean, torch.exp(z_log_var)).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


if __name__ == "__main__":
    # clear param store
    pyro.clear_param_store()

    train_loader = get_data_loader("MNIST", "data", batch_size=256)

    test_loader = get_data_loader("MNIST", "data", batch_size=256, is_training_set=False)
    #
    # setup the VAE
    vae = VAE(use_cuda=False)

    # setup the optimizer
    adam_args = {"lr": 0.0001}
    optimizer = AdamW(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    update_loss, mnist = mnist_bar()

    for x in mnist():
        if False:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        loss = svi.step(x)
        update_loss(loss)
