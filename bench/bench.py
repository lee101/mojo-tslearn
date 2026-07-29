"""Benchmarks against tslearn on identical NumPy arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojotslearn as mts  # noqa: E402
warnings.filterwarnings("ignore", message="h5py not installed")
import tslearn  # noqa: E402
from tslearn import barycenters as ts_barycenters  # noqa: E402
from tslearn import clustering as ts_clustering  # noqa: E402
from tslearn import metrics as ts_metrics  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


CASES = []


def case(name):
    def decorate(function):
        CASES.append((name, function))
        return function

    return decorate


@case("DTW (2,000 x 3 against 2,000 x 3)")
def _():
    rng = np.random.default_rng(1)
    first = rng.normal(size=(2_000, 3))
    second = rng.normal(size=(2_000, 3))
    return (
        lambda: mts.dtw(first, second),
        lambda: ts_metrics.dtw(first, second),
    )


@case("soft-DTW gamma=1 (1,200 x 2)")
def _():
    rng = np.random.default_rng(2)
    first = rng.normal(size=(1_200, 2))
    second = rng.normal(size=(1_200, 2))
    return (
        lambda: mts.soft_dtw(first, second, gamma=1.0),
        lambda: ts_metrics.soft_dtw(first, second, gamma=1.0),
    )


@case("cdist_dtw (48 x 48, length 80 x 2)")
def _():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(48, 80, 2))
    return (
        lambda: mts.cdist_dtw(values),
        lambda: ts_metrics.cdist_dtw(values),
    )


@case("cdist_soft_dtw (36 x 36, length 64)")
def _():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(36, 64, 1))
    return (
        lambda: mts.cdist_soft_dtw(values, gamma=0.5),
        lambda: ts_metrics.cdist_soft_dtw(values, gamma=0.5),
    )


@case("cdist_soft_dtw GPU (36 x 36, length 64)")
def _():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(36, 64, 1))
    return (
        lambda: mts.cdist_soft_dtw(values, gamma=0.5, device="gpu"),
        lambda: ts_metrics.cdist_soft_dtw(values, gamma=0.5),
    )


@case("DBA (24 series, length 80, 5 epochs)")
def _():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(24, 80, 1))
    return (
        lambda: mts.dtw_barycenter_averaging(values, max_iter=5),
        lambda: ts_barycenters.dtw_barycenter_averaging(values, max_iter=5),
    )


@case("TimeSeriesKMeans DTW (30 x 48, k=2)")
def _():
    rng = np.random.default_rng(6)
    wave = np.sin(np.linspace(0, 2 * np.pi, 48))
    values = np.concatenate(
        [
            wave[None, :, None] + rng.normal(0, 0.15, (15, 48, 1)),
            -wave[None, :, None] + rng.normal(0, 0.15, (15, 48, 1)),
        ]
    )
    init = values[[0, 15]]
    return (
        lambda: mts.TimeSeriesKMeans(
            n_clusters=2,
            metric="dtw",
            init=init,
            n_init=1,
            max_iter=2,
            max_iter_barycenter=5,
        ).fit(values),
        lambda: ts_clustering.TimeSeriesKMeans(
            n_clusters=2,
            metric="dtw",
            init=init,
            n_init=1,
            max_iter=2,
            max_iter_barycenter=5,
        ).fit(values),
    )


def main():
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"tslearn {tslearn.__version__}"
    )
    print()
    print("| benchmark | mojo-tslearn | tslearn | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, build_case in CASES:
        ours, upstream = build_case()
        ours()
        upstream()
        ours_time = timeit(ours)
        upstream_time = timeit(upstream)
        ratio = upstream_time / ours_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1.0
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
