from pytorch_mnist import mnist
from benchmark import benchmark_grid

sizes = range(64, 512 + 1, 64)

benchmark_grid(mnist, "PyTorch", sizes, 10, 1000, "results/pytorch_mnist.json")
