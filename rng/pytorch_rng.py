import torch

def uniform(size):
    @torch.compile
    def f():
        return torch.mean(torch.rand(size, device="cuda"))
    return lambda: f().item()

def normal(size):
    @torch.compile
    def f():
        return torch.mean(torch.randn(size, device="cuda"))
    return lambda: f().item()
