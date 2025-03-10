import jax_rng
import pytorch_rng
from benchmark import benchmark_grid

sizes = range(2 ** 24, 2 ** 27 + 1, 2 ** 24)

benchmark_grid(jax_rng.uniform, "Jax", sizes, 100, 1000, "rng/results/jax_uniform.json")
benchmark_grid(jax_rng.normal, "Jax", sizes, 100, 1000, "rng/results/jax_normal.json")
benchmark_grid(pytorch_rng.uniform, "PyTorch", sizes, 10, 1000, "rng/results/pytorch_uniform.json")
benchmark_grid(pytorch_rng.normal, "PyTorch", sizes, 10, 1000, "rng/results/pytorch_normal.json")
