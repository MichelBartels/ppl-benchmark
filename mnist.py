from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm
import matplotlib
matplotlib.use("kitcat")
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.MNIST('data/', download=True, transform=transform)

def cycle(loader):
    while True:
        for x in loader:
            yield x

def mnist(batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    for x, y in cycle(loader):
        x = x.reshape(x.shape[0], -1)
        y = one_hot(y, 10)
        yield x, y

def mnist_bar(batch_size, total=1_000_000):
    bar = tqdm(zip(range(total), mnist(batch_size)), total=total)
    update_loss = lambda loss: bar.set_description(f"Loss: {loss:.4f}")
    def gen():
        for _, (x, _) in bar:
            yield x

    return update_loss, gen

def one_sample(batch_size):
    sample, _ = next(mnist(batch_size))
    return sample

def show_sample(imgs, w, h, filename='sample.png'):
    imgss = np.array(imgs).reshape(w, h, 28, 28)
    imgs = imgss.reshape(w, -1, 28)
    img = imgs.transpose(1, 0, 2).reshape(h * 28, w * 28)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()
