"""Barycenter routines for time-series datasets."""

from __future__ import annotations

import warnings

import numpy as np

from ._lib import addr, lib
from ._utils import dataset, time_series
from .metrics import dtw_path


def _weights(weights, size: int) -> np.ndarray:
    if weights is None or len(weights) != size:
        return np.ones(size, dtype=np.float64)
    return np.ascontiguousarray(weights, dtype=np.float64)


def _initial_average(
    series: list[np.ndarray], barycenter_size: int
) -> np.ndarray:
    feature_count = series[0].shape[1]
    padded = np.full(
        (len(series), max(map(len, series)), feature_count), np.nan, dtype=np.float64
    )
    for i, item in enumerate(series):
        padded[i, : len(item)] = item
    average = np.nanmean(padded, axis=0)
    if len(average) == barycenter_size:
        return np.ascontiguousarray(average)
    old_x = np.linspace(0.0, 1.0, len(average))
    new_x = np.linspace(0.0, 1.0, barycenter_size)
    result = np.empty((barycenter_size, feature_count), dtype=np.float64)
    for feature in range(feature_count):
        result[:, feature] = np.interp(new_x, old_x, average[:, feature])
    return result


def euclidean_barycenter(X, weights=None):
    series = dataset(X)
    if len({item.shape for item in series}) != 1:
        raise ValueError("Euclidean barycenters require equal-sized time series")
    return np.average(np.stack(series), axis=0, weights=_weights(weights, len(series)))


def dtw_barycenter_averaging_one_init(
    X,
    barycenter_size=None,
    init_barycenter=None,
    max_iter=30,
    tol=1e-5,
    weights=None,
    metric_params=None,
    verbose=False,
    n_jobs=None,
):
    del n_jobs
    series = dataset(X)
    dimensions = {item.shape[1] for item in series}
    if len(dimensions) != 1:
        raise ValueError("All input time series must have the same feature size.")
    item_weights = _weights(weights, len(series))
    if init_barycenter is None:
        size = max(map(len, series)) if barycenter_size is None else int(barycenter_size)
        barycenter = _initial_average(series, size)
    else:
        barycenter = time_series(init_barycenter).copy()
        size = len(barycenter)
    params = {} if metric_params is None else dict(metric_params)
    unconstrained = not any(
        params.get(name) is not None
        for name in (
            "global_constraint",
            "sakoe_chiba_radius",
            "itakura_max_slope",
        )
    )
    previous_cost = np.inf
    cost = np.inf
    for iteration in range(int(max_iter)):
        sums = np.zeros_like(barycenter)
        counts = np.zeros(size, dtype=np.float64)
        weighted_cost = 0.0
        if unconstrained:
            max_length = max(map(len, series))
            acc = np.empty((size + 1, max_length + 1), dtype=np.float64)
            for item, weight in zip(series, item_weights):
                weighted_cost += lib().mts_dba_accumulate(
                    addr(barycenter),
                    addr(item),
                    size,
                    len(item),
                    barycenter.shape[1],
                    float(weight),
                    addr(sums),
                    addr(counts),
                    addr(acc),
                )
        else:
            for item, weight in zip(series, item_weights):
                path, distance = dtw_path(barycenter, item, **params)
                weighted_cost += distance * distance * weight
                for center_index, item_index in path:
                    counts[center_index] += weight
                    sums[center_index] += weight * item[item_index]
        cost = float(weighted_cost / item_weights.sum())
        if verbose:
            print(f"[DBA] epoch {iteration + 1}, cost: {cost:.3f}")
        barycenter = sums / counts[:, None]
        if abs(previous_cost - cost) < tol:
            break
        if previous_cost < cost:
            warnings.warn(
                "DBA loss is increasing while it should not be. Stopping optimization.",
                RuntimeWarning,
            )
            break
        previous_cost = cost
    return barycenter, cost


def dtw_barycenter_averaging(
    X,
    barycenter_size=None,
    init_barycenter=None,
    max_iter=30,
    tol=1e-5,
    weights=None,
    metric_params=None,
    verbose=False,
    n_init=1,
    n_jobs=None,
):
    best_cost = np.inf
    best = None
    for attempt in range(int(n_init)):
        if verbose:
            print(f"Attempt {attempt + 1}")
        barycenter, cost = dtw_barycenter_averaging_one_init(
            X,
            barycenter_size=barycenter_size,
            init_barycenter=init_barycenter,
            max_iter=max_iter,
            tol=tol,
            weights=weights,
            metric_params=metric_params,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        if cost < best_cost:
            best_cost = cost
            best = barycenter
    return best


dtw_barycenter_averaging_petitjean = dtw_barycenter_averaging

__all__ = [
    "dtw_barycenter_averaging",
    "dtw_barycenter_averaging_one_init",
    "dtw_barycenter_averaging_petitjean",
    "euclidean_barycenter",
]
