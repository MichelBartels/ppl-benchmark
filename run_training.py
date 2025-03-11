from time import perf_counter_ns

import numpy as np
from matplotlib import pyplot as plt
import mpl_typst

from benchmark import benchmark_grid
from mnist import one_sample

numpyro = True
if numpyro:
    from numpyro_vae import step_fn, prepare_batch
else:
    from pyro_vae import step_fn, prepare_batch


def training_step(batch_size):
    batch = one_sample(batch_size)
    batch = prepare_batch(batch)
    step = step_fn(batch_size)
    def f():
        return step(batch)
    return f

batch_sizes = range(64, 512 + 1, 64)
label = "Numpyro" if numpyro else "Pyro"
benchmark_grid(training_step, label, batch_sizes, 100, 1000, f"results/{label.lower()}_vae_without_loading.json")
