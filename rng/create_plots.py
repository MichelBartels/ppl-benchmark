from benchmark import Grid, Plot

jax_uniform_grid = Grid.load_json("rng/results/jax_uniform.json")
jax_normal_grid = Grid.load_json("rng/results/jax_normal.json")
pytorch_uniform_grid = Grid.load_json("rng/results/pytorch_uniform.json")
pytorch_normal_grid = Grid.load_json("rng/results/pytorch_normal.json")

uniform_plot = Plot([jax_uniform_grid, pytorch_uniform_grid], "Array Size", "Time (ms)", "Uniform Random Number Generation")

normal_plot = Plot([jax_normal_grid, pytorch_normal_grid], "Array Size", "Time (ms)", "Normal Random Number Generation")

uniform_plot.plot("uniform.svg", interval=0.99)
normal_plot.plot("normal.svg", interval=0.99)
