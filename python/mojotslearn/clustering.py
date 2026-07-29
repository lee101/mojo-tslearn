"""Time-series k-means over Euclidean or DTW distances."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state

from ._lib import addr, lib
from ._utils import dataset
from .barycenters import dtw_barycenter_averaging
from .metrics import cdist_dtw


def _dense_dataset(value) -> tuple[np.ndarray, list[np.ndarray]]:
    series = dataset(value)
    dimensions = {item.shape[1] for item in series}
    if len(dimensions) != 1:
        raise ValueError("All input time series must have the same feature size.")
    width = max(map(len, series))
    dense = np.full(
        (len(series), width, series[0].shape[1]), np.nan, dtype=np.float64
    )
    for i, item in enumerate(series):
        dense[i, : len(item)] = item
    return dense, series


def _cdist_euclidean(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    result = np.empty((len(x), len(centers)), dtype=np.float64)
    lib().mts_cdist_euclidean(
        addr(x),
        addr(centers),
        len(x),
        len(centers),
        x.shape[1] * x.shape[2],
        addr(result),
    )
    return result


class TimeSeriesKMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_clusters=3,
        max_iter=50,
        tol=1e-6,
        n_init=1,
        metric="euclidean",
        max_iter_barycenter=100,
        metric_params=None,
        n_jobs=None,
        dtw_inertia=False,
        verbose=0,
        random_state=None,
        init="k-means++",
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.metric = metric
        self.max_iter_barycenter = max_iter_barycenter
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.dtw_inertia = dtw_inertia
        self.verbose = verbose
        self.random_state = random_state
        self.init = init

    def _distances(self, X):
        if self.metric == "euclidean":
            return _cdist_euclidean(X, self.cluster_centers_)
        if self.metric == "dtw":
            params = {} if self.metric_params is None else dict(self.metric_params)
            return cdist_dtw(
                X,
                self.cluster_centers_,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                **params,
            )
        raise ValueError(
            "metric must be 'euclidean' or 'dtw'; soft-DTW barycenter clustering "
            "is not implemented"
        )

    def _seed(self, X, rng):
        if not isinstance(self.init, str):
            centers, _ = _dense_dataset(self.init)
            if centers.shape != (self.n_clusters, X.shape[1], X.shape[2]):
                raise ValueError("Initial centers have the wrong shape")
            return centers.copy()
        if self.init == "random":
            return X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
        if self.init != "k-means++":
            raise ValueError("init must be 'k-means++', 'random', or an array")
        centers = np.empty((self.n_clusters, X.shape[1], X.shape[2]))
        centers[0] = X[rng.randint(len(X))]
        distances = self._distance_to_centers(X, centers[:1]).ravel() ** 2
        potential = distances.sum()
        trials = 2 + int(np.log(self.n_clusters))
        for center_index in range(1, self.n_clusters):
            draws = rng.random_sample(trials) * potential
            candidates = np.searchsorted(np.cumsum(distances), draws)
            candidates = np.minimum(candidates, len(X) - 1)
            candidate_distances = (
                self._distance_to_centers(X, X[candidates]) ** 2
            ).T
            candidate_distances = np.minimum(candidate_distances, distances)
            candidate_potentials = candidate_distances.sum(axis=1)
            best = int(np.argmin(candidate_potentials))
            centers[center_index] = X[candidates[best]]
            distances = candidate_distances[best]
            potential = candidate_potentials[best]
        return centers

    def _distance_to_centers(self, X, centers):
        if self.metric == "euclidean":
            return _cdist_euclidean(X, np.ascontiguousarray(centers))
        params = {} if self.metric_params is None else dict(self.metric_params)
        return cdist_dtw(X, centers, **params)

    def _fit_once(self, X, rng):
        self.cluster_centers_ = self._seed(X, rng)
        old_inertia = np.inf
        for iteration in range(int(self.max_iter)):
            distances = self._distances(X)
            labels = np.argmin(distances, axis=1)
            if len(np.unique(labels)) != self.n_clusters:
                return None
            inertia_distances = (
                cdist_dtw(X, self.cluster_centers_)
                if self.dtw_inertia and self.metric != "dtw"
                else distances
            )
            inertia = float(np.mean(inertia_distances[np.arange(len(X)), labels] ** 2))
            if self.verbose:
                print(f"{inertia:.3f}", end=" --> ")
            for cluster_index in range(self.n_clusters):
                members = X[labels == cluster_index]
                if self.metric == "euclidean":
                    self.cluster_centers_[cluster_index] = np.mean(members, axis=0)
                else:
                    self.cluster_centers_[cluster_index] = dtw_barycenter_averaging(
                    members,
                    init_barycenter=self.cluster_centers_[cluster_index],
                    max_iter=self.max_iter_barycenter,
                    metric_params=self.metric_params,
                )
            if abs(old_inertia - inertia) < self.tol:
                break
            old_inertia = inertia
        if self.verbose:
            print()
        return labels, inertia, iteration + 1

    def fit(self, X, y=None):
        del y
        if self.metric not in {"euclidean", "dtw"}:
            raise ValueError(
                "metric must be 'euclidean' or 'dtw'; soft-DTW barycenter "
                "clustering is not implemented"
            )
        dense, _ = _dense_dataset(X)
        if np.isnan(dense).any():
            raise ValueError(
                "TimeSeriesKMeans currently requires equal-length finite series"
            )
        if self.n_clusters <= 0 or self.n_clusters > len(dense):
            raise ValueError("n_clusters must be between 1 and the number of series")
        rng = check_random_state(self.random_state)
        best = None
        attempts = 0
        while attempts < max(int(self.n_init), 10) and (
            best is None or attempts < self.n_init
        ):
            attempts += 1
            fitted = self._fit_once(dense, rng)
            if fitted is not None and (best is None or fitted[1] < best[1]):
                best = (
                    self.cluster_centers_.copy(),
                    fitted[1],
                    fitted[0].copy(),
                    fitted[2],
                )
        if best is None:
            raise RuntimeError("No initialization produced non-empty clusters")
        self.cluster_centers_, self.inertia_, self.labels_, self.n_iter_ = best
        self._X_fit = dense
        return self

    def transform(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("This TimeSeriesKMeans instance is not fitted")
        dense, _ = _dense_dataset(X)
        if self.metric == "euclidean" and dense.shape[1:] != self.cluster_centers_.shape[1:]:
            raise ValueError("Input series have the wrong shape")
        return self._distances(dense)

    def predict(self, X):
        return np.argmin(self.transform(X), axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_


__all__ = ["TimeSeriesKMeans"]
