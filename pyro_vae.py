import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import AdamW

from mnist import mnist_bar

latent_dim = 16

def linear_init(in_features, out_features, name):
    pyro.sample(name + ".weight", dist.Normal(0., 0.001).expand([in_features, out_features]).to_event(2))
    pyro.sample(name + ".bias", dist.Normal(0., 0.001).expand([out_features]).to_event(1))

def linear_params(x, in_features, out_features, name):
    weight_mean = pyro.param(name + ".weight.mean", torch.zeros(in_features, out_features))
    weight_std = pyro.param(name + ".weight.std", torch.ones(in_features, out_features) * 0.001)
    weight = pyro.sample(name + ".weight", dist.Normal(weight_mean, weight_std).to_event(2))
    bias_mean = pyro.param(name + ".bias.mean", torch.zeros(out_features))
    bias_std = pyro.param(name + ".bias.std", torch.ones(out_features))
    bias = pyro.sample(name + ".bias", dist.Normal(bias_mean, bias_std).to_event(1))
    return x @ weight + bias

def linear_params_init(in_features, out_features, name):
    weight_mean = pyro.param(name + ".weight.mean", torch.zeros(in_features, out_features))
    weight_std = pyro.param(name + ".weight.std", torch.ones(in_features, out_features) * 0.001)
    pyro.sample(name + ".weight", dist.Normal(weight_mean, weight_std).to_event(2))
    bias_mean = pyro.param(name + ".bias.mean", torch.zeros(out_features))
    bias_std = pyro.param(name + ".bias.std", torch.ones(out_features))
    pyro.sample(name + ".bias", dist.Normal(bias_mean, bias_std).to_event(1))

def linear(x, in_features, out_features, name):
    weight = pyro.sample(name + ".weight", dist.Normal(0., 1.).expand([in_features, out_features]).to_event(2))
    bias = pyro.sample(name + ".bias", dist.Normal(0., 1.).expand([out_features]).to_event(1))
    return x @ weight + bias


def encoder_init():
    linear_init(784, 512, "encoder.hidden")
    linear_init(512, latent_dim, "encoder.mean")
    linear_init(512, latent_dim, "encoder.log_var")

def encoder_params(x):
    x = linear_params(x, 784, 512, "encoder.hidden")
    mean = linear_params(x, 512, latent_dim, "encoder.mean")
    log_var = linear_params(x, 512, latent_dim, "encoder.log_var")
    return mean, log_var

def decoder_params_init():
    linear_init(latent_dim, 512, "decoder.hidden")
    linear_init(512, 784, "decoder.output")

def decoder(z):
    hidden = torch.tanh(linear(z, 784, 512, "decoder.hidden"))
    img = torch.sigmoid(linear(hidden, 512, 784, "decoder.output"))
    return img

def model(x):
    with pyro.plate("data", x.shape[0]):
        encoder_init()
        z_loc = torch.zeros(x.shape[0], latent_dim)
        z_scale = torch.ones(x.shape[0], latent_dim)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        loc_img = decoder(z)
        pyro.sample("obs", dist.Normal(loc_img, 1.).to_event(1), obs=x)

def guide(x):
    with pyro.plate("data", x.shape[0]):
        decoder_params_init()
        mean, log_var = encoder_params(x)
        var = torch.exp(log_var)
        print("guide", mean.shape, var.shape)
        pyro.sample("latent", dist.Normal(mean, var).to_event(1))

optimizer = AdamW({"lr": 1.0e-4})

elbo = JitTrace_ELBO()
svi = SVI(model, guide, optimizer, loss=elbo)

update_loss, mnist = mnist_bar()

for x in mnist():
    if False:
        x = x.cuda()
    loss = svi.step(x)
    update_loss(loss)
