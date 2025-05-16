from benchmark import Result, Grid, Bar, Plot

mine = "Our PPL"

jax_uniform_grid = Grid.load_json("results/jax_uniform.json")
jax_normal_grid = Grid.load_json("results/jax_normal.json")
pytorch_uniform_grid = Grid.load_json("results/pytorch_uniform.json")
pytorch_normal_grid = Grid.load_json("results/pytorch_normal.json")
mine_uniform_grid = Grid.load_json("results/mine_uniform.json", label=mine)
mine_normal_grid = Grid.load_json("results/mine_normal.json", label=mine)

uniform_plot = Plot([jax_uniform_grid, pytorch_uniform_grid, mine_uniform_grid], "Array Size", "Time (ms)", xlim=(0, 6e8), ylim=(0, 20))

normal_plot = Plot([jax_normal_grid, pytorch_normal_grid, mine_normal_grid], "Array Size", "Time (ms)", xlim=(0, 6e8), ylim=(0, 20))

uniform_plot.plot("uniform.svg", interval=0.95)
normal_plot.plot("normal.svg", interval=0.95)

pytorch_mnist_grid = Grid.load_json("results/pytorch_mnist.json")
mine_mnist_grid = Grid.load_json("results/mine_mnist.json", label=mine)

mnist_plot = Plot([pytorch_mnist_grid, mine_mnist_grid], "Batch Size", "Time (ms)")

mnist_plot.plot("mnist.svg", interval=0.95)

jax_usage_grid = Grid.load_json("results/jax_usage.json")
pytorch_usage_grid = Grid.load_json("results/pytorch_usage.json")
mine_usage_grid = Grid.load_json("results/mine_usage.json", label=mine)

usage_plot = Plot([jax_usage_grid, pytorch_usage_grid, mine_usage_grid], "Batch Size", "GPU utilisation in %")
usage_plot.plot("usage.svg", interval=0.95)

numpyro_vae_without_loading_grid = Grid.load_json("results/numpyro_vae_without_loading.json")
pyro_vae_without_loading_grid = Grid.load_json("results/pyro_vae_without_loading.json")
mine_vae_without_loading_grid = Grid.load_json("results/mine_vae_without_loading.json")

vae_without_loading_plot = Plot([numpyro_vae_without_loading_grid, pyro_vae_without_loading_grid, mine_vae_without_loading_grid], "Batch Size", "Time (ms)")
vae_without_loading_plot.plot("vae_without_loading.svg", interval=0.95)

numpyro_vae_with_loading_grid = Grid.load_json("results/numpyro_vae_with_loading.json")
pyro_vae_with_loading_grid = Grid.load_json("results/pyro_vae_with_loading.json")
mine_vae_with_loading_grid = Grid.load_json("results/mine_vae_with_loading.json", label=mine)

vae_with_loading_plot = Plot([numpyro_vae_with_loading_grid, pyro_vae_with_loading_grid, mine_vae_with_loading_grid], "Batch Size", "Time (ms)")
vae_with_loading_plot.plot("vae_with_loading.svg", interval=0.95)

jax_memory_usage = Result.load_json("results/jax_memory_usage.json")
jax_memory_usage = Result(times=jax_memory_usage.times / 1024 ** 3)
pytorch_memory_usage = Result.load_json("results/pytorch_memory_usage.json")
pytorch_memory_usage = Result(times=pytorch_memory_usage.times / 1024 ** 3)
mine_memory_usage = Result.load_json("results/mine_memory_usage.json")
mine_memory_usage = Result(times=mine_memory_usage.times / 1024 ** 3)
memory_usage_bar = Bar({"JAX": jax_memory_usage, "PyTorch": pytorch_memory_usage, mine: mine_memory_usage}, "Memory Usage in GB")
memory_usage_bar.plot("memory_usage.svg", interval=0.95)

jax_gpu_memory_usage = Grid.load_json("results/jax_gpu_memory.json")
jax_gpu_memory_usage = jax_gpu_memory_usage.scale(1 / 1024)
pytorch_gpu_memory_usage = Grid.load_json("results/pytorch_gpu_memory.json")
pytorch_gpu_memory_usage = pytorch_gpu_memory_usage.scale(1 / 1024)
mine_gpu_memory_usage = Grid.load_json("results/mine_gpu_memory.json", label=mine)
mine_gpu_memory_usage = mine_gpu_memory_usage.scale(1 / 1024)
gpu_memory_usage_plot = Plot([jax_gpu_memory_usage, pytorch_gpu_memory_usage, mine_gpu_memory_usage], "Batch Size", "GPU Memory Usage in GB")
gpu_memory_usage_plot.plot("gpu_memory_usage.svg", interval=0.95)
