from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import numpy as np
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.MNIST('data/', download=True, transform=transform)

batch_size = 128

def cycle(loader):
    while True:
        for x in loader:
            yield x

def mnist():
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for x, y in cycle(loader):
        x = x.reshape(x.shape[0], -1)
        y = one_hot(y, 10)
        yield x, y

def mnist_bar(total=25000):
    bar = tqdm(zip(range(total), mnist()), total=total)
    update_loss = lambda loss: bar.set_description(f"Loss: {loss:.4f}")
    def gen():
        for _, (x, _) in bar:
            yield x

    return update_loss, gen
