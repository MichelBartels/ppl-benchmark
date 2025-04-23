from time import perf_counter_ns
import numpy as np
from dataclasses import dataclass
import json
from matplotlib import pyplot as plt
import mpl_typst

@dataclass(frozen=True)
class Result:
    times: np.ndarray

    def median(self):
        return np.median(self.times)

    def confidence_interval(self, interval):
        sorted_times = np.sort(self.times)
        num_elements_in_interval = int(len(sorted_times) * interval)
        interval_sizes = sorted_times[num_elements_in_interval - 1:] - sorted_times[:-num_elements_in_interval + 1]
        smallest_interval_start = np.argmin(interval_sizes)
        smallest_interval_end = smallest_interval_start + num_elements_in_interval - 1
        interval = np.array([self.median() - sorted_times[smallest_interval_start], sorted_times[smallest_interval_end] - self.median()])
        return interval

    def to_list(self):
        return self.times.tolist()

    @staticmethod
    def from_list(d):
        return Result(np.array(d))

    def save_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_list(), f)

    @staticmethod
    def load_json(filename):
        with open(filename, "r") as f:
            times = json.load(f)
        return Result.from_list(times)

    def scale(self, factor):
        return Result(times=self.times * factor)

@dataclass(frozen=True)
class Grid:
    results: dict[int, Result]
    label: str

    def xs(self):
        return np.sort(np.array(list(self.results.keys())))

    def ys(self):
        return np.array([self.results[x].median() for x in self.xs()])

    def confidence_intervals(self, interval):
        return np.stack([self.results[x].confidence_interval(interval) for x in self.xs()]).T

    def plot(self, interval=0.95):
        xs = self.xs()
        ys = self.ys()
        confidence_intervals = self.confidence_intervals(interval)
        print(confidence_intervals.shape)
        plt.errorbar(xs, ys, yerr=confidence_intervals, label=self.label, alpha=0.5)

    def to_dict(self):
        return {"label": self.label, "results": {k: v.to_list() for k, v in self.results.items()}}

    @staticmethod
    def from_dict(d):
        return Grid({int(k): Result.from_list(v) for k, v in d["results"].items()}, d["label"])

    def save_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load_json(filename):
        with open(filename, "r") as f:
            grid = json.load(f)
        return Grid.from_dict(grid)

    def scale(self, factor):
        return Grid({k: v.scale(factor) for k, v in self.results.items()}, label=self.label)

@dataclass(frozen=True)
class Plot:
    grids: list[Grid]
    xlabel: str
    ylabel: str

    def plot(self, filename, interval=0.95):
        for grid in self.grids:
            grid.plot(interval=interval)
        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.savefig(filename)
        plt.clf()

@dataclass(frozen=True)
class Bar:
    results: dict[str, Result]
    label: str

    def plot(self, filename, interval=0.95):
        labels = list(self.results.keys())
        ys = np.array([self.results[k].median() for k in labels])
        confidence_intervals = np.stack([self.results[k].confidence_interval(interval) for k in labels]).T
        plt.bar(labels, ys, yerr=confidence_intervals, label=self.results.keys(), alpha=0.5)
        plt.ylabel(self.label)
        plt.savefig(filename)
        plt.clf()

def benchmark(f, n_warmup, n_bench, filename=None):
    for _ in range(n_warmup):
        f()
    times = []
    for _ in range(n_bench):
        start = perf_counter_ns()
        f()
        end = perf_counter_ns()
        times.append(end - start)
    result = Result(np.array(times) / 1e6)
    if filename is not None:
        result.save_json(filename)
    return result

def benchmark_grid(f, label, values, n_warmup, n_bench, filename=None):
    results = {}
    for value in values:
        print(f"Benchmarking with value: {value}")
        results[value] = benchmark(f(value), n_warmup, n_bench)
    grid = Grid(results, label)
    if filename is not None:
        grid.save_json(filename)
    return grid
