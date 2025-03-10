import jax
import jax.numpy as jnp

def wrap(f):
    def return_fn(size):
        inner = jax.jit(lambda key: f(key, size))
        key = jax.random.PRNGKey(0)
        def g():
            nonlocal key
            key1, key2 = jax.random.split(key)
            key = key2
            return inner(key1)
        return g
    return return_fn

@wrap
def uniform(key, size):
    return jnp.mean(jax.random.uniform(key, shape=(size,)))

@wrap
def normal(key, size):
    return jnp.mean(jax.random.normal(key, shape=(size,)))
