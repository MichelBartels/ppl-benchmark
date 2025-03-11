from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def mnist(batch_size):
    dataset = MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    def f():
        for data, target in dataloader:
            return data, target
    return f
