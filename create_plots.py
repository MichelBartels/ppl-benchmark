from benchmark import Grid, Plot

jax_uniform_grid = Grid.load_json("results/jax_uniform.json")
jax_normal_grid = Grid.load_json("results/jax_normal.json")
pytorch_uniform_grid = Grid.load_json("results/pytorch_uniform.json")
pytorch_normal_grid = Grid.load_json("results/pytorch_normal.json")
mine_uniform_grid = Grid.load_json("results/mine_uniform.json")
mine_normal_grid = Grid.load_json("results/mine_normal.json")

uniform_plot = Plot([jax_uniform_grid, pytorch_uniform_grid, mine_uniform_grid], "Array Size", "Time (ms)", "Uniform Random Number Generation")

normal_plot = Plot([jax_normal_grid, pytorch_normal_grid, mine_normal_grid], "Array Size", "Time (ms)", "Normal Random Number Generation")

uniform_plot.plot("uniform.svg", interval=0.95)
normal_plot.plot("normal.svg", interval=0.95)

pytorch_mnist_grid = Grid.load_json("results/pytorch_mnist.json")
mine_mnist_grid = Grid.load_json("results/mine_mnist.json")

mnist_plot = Plot([pytorch_mnist_grid, mine_mnist_grid], "Batch Size", "Time (ms)", "Preparing a Batch of MNIST Images")

mnist_plot.plot("mnist.svg", interval=0.95)

jax_usage_grid = Grid.load_json("results/jax_usage.json")
pytorch_usage_grid = Grid.load_json("results/pytorch_usage.json")
mine_usage_grid = Grid.load_json("results/mine_usage.json")

usage_plot = Plot([jax_usage_grid, pytorch_usage_grid, mine_usage_grid], "Array Size", "Usage in %", "GPU Usage")
usage_plot.plot("usage.svg", interval=0.95)
