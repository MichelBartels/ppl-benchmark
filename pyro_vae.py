import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import AdamW

from mnist import mnist_bar

latent_dim = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_init(in_features, out_features, name):
    pyro.sample(name + ".weight", dist.Normal(0., 0.001).expand([in_features, out_features]).to_event(2))
    pyro.sample(name + ".bias", dist.Normal(0., 0.001).expand([1, out_features]).to_event(2))

def linear_params(x, in_features, out_features, name):
    weight_mean = pyro.param(name + ".weight.mean", torch.zeros(in_features, out_features, device=device))
    weight_std = pyro.param(name + ".weight.std", torch.ones(in_features, out_features, device=device) * 0.001)
    weight = pyro.sample(name + ".weight", dist.Normal(weight_mean, weight_std).to_event(2))
    bias_mean = pyro.param(name + ".bias.mean", torch.zeros(1, out_features, device=device))
    bias_std = pyro.param(name + ".bias.std", torch.ones(1, out_features, device=device))
    bias = pyro.sample(name + ".bias", dist.Normal(bias_mean, bias_std).to_event(2))
    return x @ weight + bias

def linear_params_init(in_features, out_features, name):
    weight_mean = pyro.param(name + ".weight.mean", torch.zeros(in_features, out_features, device=device))
    weight_std = pyro.param(name + ".weight.std", torch.ones(in_features, out_features, device=device) * 0.001)
    pyro.sample(name + ".weight", dist.Normal(weight_mean, weight_std).to_event(2))
    bias_mean = pyro.param(name + ".bias.mean", torch.zeros(1, out_features, device=device))
    bias_std = pyro.param(name + ".bias.std", torch.ones(1, out_features, device=device))
    pyro.sample(name + ".bias", dist.Normal(bias_mean, bias_std).to_event(2))

def linear(x, in_features, out_features, name):
    weight = pyro.sample(name + ".weight", dist.Normal(0., 1.).expand([in_features, out_features]).to_event(2))
    bias = pyro.sample(name + ".bias", dist.Normal(0., 1.).expand([1, out_features]).to_event(2))
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
    hidden = torch.tanh(linear(z, latent_dim, 512, "decoder.hidden"))
    img = torch.sigmoid(linear(hidden, 512, 784, "decoder.output"))
    return img

def model(x):
    with pyro.plate("data", x.shape[0]):
        encoder_init()
        z_loc = torch.zeros(x.shape[0], latent_dim, device=device)
        z_scale = torch.ones(x.shape[0], latent_dim, device=device)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        z = z.unsqueeze(1)
        loc_img = decoder(z)
        loc_img = loc_img.squeeze(1)
        pyro.sample("obs", dist.Normal(loc_img, 1.).to_event(1), obs=x)

def guide(x):
    with pyro.plate("data", x.shape[0]):
        decoder_params_init()
        x = x.unsqueeze(1)
        mean, log_var = encoder_params(x)
        mean = mean.squeeze(1)
        log_var = log_var.squeeze(1)
        var = torch.exp(log_var)
        pyro.sample("latent", dist.Normal(mean, var).to_event(1))

optimizer = AdamW({"lr": 1.0e-4})

elbo = JitTrace_ELBO()
svi = SVI(model, guide, optimizer, loss=elbo)

update_loss, mnist = mnist_bar()

for x in mnist():
    if torch.cuda.is_available():
        x = x.cuda()
    loss = svi.step(x)
    update_loss(loss)
