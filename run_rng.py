import jax_rng
import pytorch_rng
from benchmark import benchmark_grid

sizes = range(2 ** 26, 2 ** 29 + 1, 2 ** 26)

benchmark_grid(jax_rng.uniform, "Jax", sizes, 100, 1000, "results/jax_uniform.json")
benchmark_grid(jax_rng.normal, "Jax", sizes, 100, 1000, "results/jax_normal.json")
benchmark_grid(pytorch_rng.uniform, "PyTorch", sizes, 10, 1000, "results/pytorch_uniform.json")
benchmark_grid(pytorch_rng.normal, "PyTorch", sizes, 10, 1000, "results/pytorch_normal.json")
