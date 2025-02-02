import numpyro
from numpyro.distributions import Normal
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from jax import jit
import jax.numpy as jnp
from jax.random import PRNGKey, split

from mnist import mnist_bar, batch_size

latent_dim = 16

def linear_init(in_features, out_features, name):
    numpyro.sample(name + ".weight", Normal(0., 0.001).expand([in_features, out_features]).to_event(2))
    numpyro.sample(name + ".bias", Normal(0., 0.001).expand([1, out_features]).to_event(2))

def linear_params(x, in_features, out_features, name):
    weight_mean = numpyro.param(name + ".weight.mean", jnp.zeros((in_features, out_features)))
    weight_std = numpyro.param(name + ".weight.std", jnp.ones((in_features, out_features)) * 0.001)
    weight = numpyro.sample(name + ".weight", Normal(weight_mean, weight_std).to_event(2))
    bias_mean = numpyro.param(name + ".bias.mean", jnp.zeros((1, out_features)))
    bias_std = numpyro.param(name + ".bias.std", jnp.ones((1, out_features)))
    bias = numpyro.sample(name + ".bias", Normal(bias_mean, bias_std).to_event(2))
    return x @ weight + bias

def linear_params_init(in_features, out_features, name):
    weight_mean = numpyro.param(name + ".weight.mean", jnp.zeros((in_features, out_features)))
    weight_std = numpyro.param(name + ".weight.std", jnp.ones((in_features, out_features)) * 0.001)
    numpyro.sample(name + ".weight", Normal(weight_mean, weight_std).to_event(2))
    bias_mean = numpyro.param(name + ".bias.mean", jnp.zeros((1, out_features)))
    bias_std = numpyro.param(name + ".bias.std", jnp.ones((1, out_features)))
    numpyro.sample(name + ".bias", Normal(bias_mean, bias_std).to_event(2))

def linear(x, in_features, out_features, name):
    weight = numpyro.sample(name + ".weight", Normal(0., 1.).expand([in_features, out_features]).to_event(2))
    bias = numpyro.sample(name + ".bias", Normal(0., 1.).expand([1, out_features]).to_event(2))
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

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def decoder(z):
    hidden = jnp.tanh(linear(z, latent_dim, 512, "decoder.hidden"))
    img = sigmoid(linear(hidden, 512, 784, "decoder.output"))
    return img

def model(x):
    with numpyro.plate("data", x.shape[0]):
        encoder_init()
        z_loc = jnp.zeros((x.shape[0], latent_dim))
        z_scale = jnp.ones((x.shape[0], latent_dim))
        z = numpyro.sample("latent", Normal(z_loc, z_scale).to_event(1))
        z = jnp.expand_dims(z, 1)
        loc_img = decoder(z)
        loc_img = loc_img.squeeze(1)
        numpyro.sample("obs", Normal(loc_img, 1.).to_event(1), obs=x)

def guide(x):
    with numpyro.plate("data", x.shape[0]):
        decoder_params_init()
        x = jnp.expand_dims(x, 1)
        mean, log_var = encoder_params(x)
        mean = mean.squeeze(1)
        log_var = log_var.squeeze(1)
        var = jnp.exp(log_var)
        numpyro.sample("latent", Normal(mean, var).to_event(1))

optimizer = Adam(1.0e-4)

svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

rng_key = PRNGKey(0)

sample_batch = jnp.zeros((batch_size, 784))

svi_state = svi.init(rng_key, sample_batch)

step = jit(svi.update)

update_loss, mnist = mnist_bar()

for x in mnist():
    x = x.numpy()
    svi_state, loss = step(svi_state, x)
    update_loss(loss)
