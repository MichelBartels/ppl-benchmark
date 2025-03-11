import os
import time
import numpy as np
from benchmark import Result, Grid

def gpu_usage():
    cmd = "nvidia-smi --query-gpu=utilization.gpu --format=noheader,nounits,csv"
    return int(os.popen(cmd).read().strip())

def record(n, sleep=0.01):
    usages = []
    for _ in range(n):
        usages.append(gpu_usage())
        time.sleep(sleep)
    return Result(np.array(usages))

def record_grid(values, n, label, sleep=0.01):
    results = {}
    for value in values:
        input(f"Press enter to start recording for value {value}")
        results[value] = record(n, sleep)
    return Grid(results, label)

if __name__ == "__main__":
    batch_sizes = range(64, 256 + 1, 64)
    label = input("Enter label: ")
    grid = record_grid(batch_sizes, 1000, label)
    grid.save_json(f"rng/results/{label.lower()}_usage.json")
