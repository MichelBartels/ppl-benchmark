import numpyro
from numpyro.distributions import Normal, Distribution
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from numpyro.distributions.constraints import positive

from jax import jit, debug
from jax.nn import softplus
import jax.numpy as jnp
from jax.random import PRNGKey, normal

from optax import adamw, sgd

latent_dim = 16
hidden_dim_enc = 512
hidden_dim_dec = 512
input_dim = 784

def linear_params(in_features, out_features, name):
    weight_mean = numpyro.param(f"{name}.weight.mean", jnp.zeros((in_features, out_features)))
    weight_std = numpyro.param(f"{name}.weight.std_inv_softplus", jnp.full((in_features, out_features), 0.01), constraint=positive)

    numpyro.sample(f"{name}.weight", Normal(weight_mean, weight_std).to_event(2))

    bias_mean = numpyro.param(f"{name}.bias.mean", jnp.zeros((1, out_features)))
    bias_std = numpyro.param(f"{name}.bias.std_inv_softplus", jnp.full((1, out_features), 0.01), constraint=positive)

    numpyro.sample(f"{name}.bias", Normal(bias_mean, bias_std).to_event(2))

def linear(x, in_features, out_features, name, bayesian=True):
    if bayesian:
        weight = numpyro.sample(f"{name}.weight", Normal(0., 0.01).expand([in_features, out_features]).to_event(2))
        bias = numpyro.sample(f"{name}.bias", Normal(0., 0.01).expand([1, out_features]).to_event(2))
    else:
        weight = numpyro.param(f"{name}.weight", lambda key: normal(key, shape=(in_features, out_features)) * 0.01)
        bias = numpyro.param(f"{name}.bias", lambda key: normal(key, shape=(1, out_features)) * 0.01)
    return x @ weight + bias

def encoder(x):
    hidden = jnp.tanh(linear(x[:, None], input_dim, hidden_dim_enc, "encoder.hidden", bayesian=False))
    mean = linear(hidden, hidden_dim_enc, latent_dim, "encoder.mean", bayesian=False)
    log_var = linear(hidden, hidden_dim_enc, latent_dim, "encoder.log_var", bayesian=False)
    return mean, log_var

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def decoder_generative(z):
    hidden = jnp.tanh(linear(z, latent_dim, hidden_dim_dec, "decoder.hidden"))
    output_mean = sigmoid(linear(hidden, hidden_dim_dec, input_dim, "decoder.output") * 100)
    return output_mean

def decoder_guide():
    linear_params(latent_dim, hidden_dim_dec, "decoder.hidden")
    linear_params(hidden_dim_dec, input_dim, "decoder.output")

class ContinuousBernoulli(Distribution):
    def __init__(self, p):
        self.p = p
        super().__init__(batch_shape=p.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        return self.p

    def log_prob(self, x):
        cut_p = jnp.where(jnp.logical_or(self.p < 0.49, self.p > 0.51), self.p, jnp.full_like(self.p, 0.49))
        log_norm = jnp.log(jnp.abs(2 * jnp.arctanh(1 - 2 * cut_p))) - jnp.log(jnp.abs(1 - 2 * cut_p))
        taylor = jnp.log(2) - 4 / 3 * jnp.pow(self.p - 0.5, 2) + 104 / 45 * jnp.pow(self.p - 0.5, 4)
        c = jnp.where(jnp.logical_or(self.p < 0.49, self.p > 0.51), log_norm, taylor)
        return c + x * jnp.log(self.p) + (1 - x) * jnp.log(1 - self.p)

def model(x, observed):
    batch_size = x.shape[0] if x is not None else 1

    with numpyro.plate("data", batch_size):
        z_loc = jnp.zeros((latent_dim,))
        z_scale = jnp.ones((latent_dim))
        z = numpyro.sample("latent", Normal(z_loc, z_scale).to_event(1))

        loc_img = decoder_generative(z[:, None])[:, 0]
        scale_img = 0.01
        if observed is None:
            numpyro.sample("obs", Normal(loc_img, scale_img).to_event(1))
            #numpyro.sample("obs", ContinuousBernoulli(loc_img).to_event(1))
            return
        numpyro.sample("obs", Normal(loc_img, scale_img).to_event(1), obs=observed)
        #numpyro.sample("obs", ContinuousBernoulli(loc_img).to_event(1), obs=observed)


def guide(x, observed):
    if x is None:
        decoder_guide()
        return
    batch_size = x.shape[0]

    with numpyro.plate("data", batch_size):
        mean_z, inv_std_z = encoder(x)
        std_z = jnp.exp(inv_std_z)

        mean_z = jnp.reshape(mean_z, (batch_size, latent_dim))
        std_z = jnp.reshape(std_z, (batch_size, latent_dim))

        numpyro.sample("latent", Normal(mean_z, std_z).to_event(1))

        decoder_guide()


def prepare_batch(batch):
    batch = jnp.reshape(batch.numpy(), (batch.shape[0], -1)) # Flatten
    return batch


def init(batch_size, learning_rate=1e-3):
    optimizer = adamw(learning_rate=learning_rate)
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())

    rng_key_init = PRNGKey(4)
    dummy_prepared_batch = jnp.zeros((batch_size, input_dim))
    svi_state = svi.init(rng_key_init, dummy_prepared_batch, dummy_prepared_batch)

    update = jit(svi.update)

    def step(prepared_batch):
        nonlocal svi_state
        svi_state, loss = update(svi_state, x=prepared_batch, observed=prepared_batch)
        return loss

    def generate(num_samples):
        params = svi.get_params(svi_state)
        predictive_model = Predictive(model, guide=guide, num_samples=num_samples, params=params)
        generated_samples = predictive_model(PRNGKey(0), x=None, observed=None)
        return generated_samples['obs']

    def reconstruct(prepared_batch):
        params = svi.get_params(svi_state)
        predictive_model = Predictive(model, guide=guide, num_samples=1, params=params)
        reconstructed_samples = predictive_model(PRNGKey(0), x=prepared_batch, observed=None)
        return reconstructed_samples['obs']

    return step, generate, reconstruct
