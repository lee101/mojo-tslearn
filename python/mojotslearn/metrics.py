"""Dynamic Time Warping metrics with tslearn-compatible call signatures."""

from __future__ import annotations

from collections.abc import Callable
import subprocess
import warnings

import numpy as np

from ._lib import addr, f64, lib
from ._utils import (
    check_numpy_backend,
    check_series_pair,
    constraint_mask,
    dataset,
    time_series,
)


def _scratch(n: int, m: int) -> np.ndarray:
    return np.empty((n + 1, m + 1), dtype=np.float64)


_PARALLEL_DTW_CELLS = 1_000_000
_SOFT_DTW_DIAGONAL_CELLS = 65_536
_GPU_MIN_FREE_MIB = 4_000
_GPU_MAX_BYTES = 2 * 1024**3


def _parallel_cdist_dtw(nx: int, ny: int, sx: int, sy: int, n_jobs) -> bool:
    return (
        n_jobs != 1
        and nx >= 4
        and nx * ny * sx * sy >= _PARALLEL_DTW_CELLS
    )


def _soft_dtw_scratch(n: int, m: int, gamma: float) -> np.ndarray:
    if gamma == 0.0:
        return _scratch(n, m)
    if n * m >= _SOFT_DTW_DIAGONAL_CELLS:
        return np.empty(3 * (min(n, m) + 2), dtype=np.float64)
    return np.empty((2, m + 1), dtype=np.float64)


def _gpu_memory_available(required_bytes: int) -> bool:
    if required_bytes >= _GPU_MAX_BYTES:
        return False
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        free_mib = [int(line) for line in completed.stdout.splitlines()]
        return bool(free_mib) and free_mib[0] >= _GPU_MIN_FREE_MIB
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def _equal_dataset_buffer(value, series: list[np.ndarray]) -> np.ndarray:
    try:
        candidate = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.ascontiguousarray(np.stack(series))
    if (
        candidate.ndim == 3
        and candidate.flags.c_contiguous
        and len(candidate) == len(series)
        and all(item.shape == candidate.shape[1:] for item in series)
    ):
        return candidate
    return np.ascontiguousarray(np.stack(series))


def _path_from_acc(acc: np.ndarray, n: int, m: int) -> list[tuple[int, int]]:
    path_i = np.empty(n + m - 1, dtype=np.int64)
    path_j = np.empty(n + m - 1, dtype=np.int64)
    length = lib().mts_dtw_backtrack(
        addr(acc), n, m, addr(path_i), addr(path_j)
    )
    return list(
        zip(
            path_i[:length][::-1].astype(int).tolist(),
            path_j[:length][::-1].astype(int).tolist(),
        )
    )


def _dtw_native(first: np.ndarray, second: np.ndarray, mask=None, acc=None) -> float:
    n, m = len(first), len(second)
    if acc is None:
        acc = _scratch(n, m)
    mask_array = acc if mask is None else f64(mask)
    return float(
        lib().mts_dtw(
            addr(first),
            addr(second),
            n,
            m,
            first.shape[1],
            addr(mask_array),
            mask is not None,
            addr(acc),
        )
    )


def dtw(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    check_numpy_backend(be)
    first, second = check_series_pair(s1, s2)
    mask = constraint_mask(
        len(first),
        len(second),
        global_constraint,
        sakoe_chiba_radius,
        itakura_max_slope,
    )
    return _dtw_native(first, second, mask)


def dtw_path(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    check_numpy_backend(be)
    first, second = check_series_pair(s1, s2)
    mask = constraint_mask(
        len(first),
        len(second),
        global_constraint,
        sakoe_chiba_radius,
        itakura_max_slope,
    )
    acc = _scratch(len(first), len(second))
    distance = _dtw_native(first, second, mask, acc)
    return _path_from_acc(acc, len(first), len(second)), distance


def cdist_dtw(
    dataset1,
    dataset2=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    del verbose
    check_numpy_backend(be)
    left = dataset(dataset1)
    right = left if dataset2 is None else dataset(dataset2)
    dimensions = {series.shape[1] for series in left + right}
    if len(dimensions) != 1:
        raise ValueError("All input time series must have the same feature size.")
    constrained = (
        global_constraint is not None
        or sakoe_chiba_radius is not None
        or itakura_max_slope is not None
    )
    same_left = len({len(series) for series in left}) == 1
    same_right = len({len(series) for series in right}) == 1
    if same_left and same_right and not constrained:
        x = _equal_dataset_buffer(dataset1, left)
        y = x if dataset2 is None else _equal_dataset_buffer(dataset2, right)
        result = np.empty((len(left), len(right)), dtype=np.float64)
        use_parallel = _parallel_cdist_dtw(
            len(x), len(y), x.shape[1], y.shape[1], n_jobs
        )
        scratch_rows = len(x) if use_parallel else 1
        acc = np.empty(
            (scratch_rows, x.shape[1] + 1, y.shape[1] + 1),
            dtype=np.float64,
        )
        lib().mts_cdist_dtw(
            addr(x),
            addr(y),
            len(x),
            len(y),
            x.shape[1],
            y.shape[1],
            x.shape[2],
            addr(result),
            addr(acc),
            use_parallel,
        )
        return result
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for i, first in enumerate(left):
        for j, second in enumerate(right):
            mask = constraint_mask(
                len(first),
                len(second),
                global_constraint,
                sakoe_chiba_radius,
                itakura_max_slope,
            )
            result[i, j] = _dtw_native(first, second, mask)
    return result


def soft_dtw(
    ts1,
    ts2,
    gamma=1.0,
    be=None,
    compute_with_backend=False,
):
    del compute_with_backend
    check_numpy_backend(be)
    first, second = check_series_pair(ts1, ts2)
    gamma = float(gamma)
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    acc = _soft_dtw_scratch(len(first), len(second), gamma)
    return float(
        lib().mts_soft_dtw(
            addr(first),
            addr(second),
            len(first),
            len(second),
            first.shape[1],
            gamma,
            addr(acc),
        )
    )


def cdist_soft_dtw(
    dataset1,
    dataset2=None,
    gamma=1.0,
    be=None,
    compute_with_backend=False,
    device="cpu",
):
    del compute_with_backend
    check_numpy_backend(be)
    gamma = float(gamma)
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")
    left = dataset(dataset1)
    right = left if dataset2 is None else dataset(dataset2)
    dimensions = {series.shape[1] for series in left + right}
    if len(dimensions) != 1:
        raise ValueError("All input time series must have the same feature size.")
    if len({len(series) for series in left}) == 1 and len(
        {len(series) for series in right}
    ) == 1:
        x = _equal_dataset_buffer(dataset1, left)
        y = x if dataset2 is None else _equal_dataset_buffer(dataset2, right)
        result = np.empty((len(x), len(y)), dtype=np.float64)
        gpu_bytes = (
            x.nbytes
            + y.nbytes
            + result.nbytes
            + len(x) * len(y) * 2 * (y.shape[1] + 1) * 8
        )
        gpu_ready = (
            device == "gpu"
            and gamma > 0.0
            and _gpu_memory_available(gpu_bytes)
        )
        if gpu_ready:
            gpu_succeeded = lib().mts_cdist_soft_dtw_gpu(
                addr(x),
                addr(y),
                len(x),
                len(y),
                x.shape[1],
                y.shape[1],
                x.shape[2],
                gamma,
                addr(result),
            )
            if gpu_succeeded:
                return result
            warnings.warn(
                "The Mojo GPU kernel failed; falling back to the CPU implementation",
                RuntimeWarning,
                stacklevel=2,
            )
        acc = _soft_dtw_scratch(x.shape[1], y.shape[1], gamma)
        lib().mts_cdist_soft_dtw(
            addr(x),
            addr(y),
            len(x),
            len(y),
            x.shape[1],
            y.shape[1],
            x.shape[2],
            gamma,
            addr(result),
            addr(acc),
        )
        return result
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for i, first in enumerate(left):
        for j, second in enumerate(right):
            result[i, j] = soft_dtw(first, second, gamma=gamma)
    return result


def cdist_soft_dtw_normalized(
    dataset1,
    dataset2=None,
    gamma=1.0,
    be=None,
    compute_with_backend=False,
    device="cpu",
):
    check_numpy_backend(be)
    left = dataset(dataset1)
    right = left if dataset2 is None else dataset(dataset2)
    distances = cdist_soft_dtw(
        left,
        None if dataset2 is None else right,
        gamma=gamma,
        compute_with_backend=compute_with_backend,
        device=device,
    )
    left_self = np.array([soft_dtw(x, x, gamma=gamma) for x in left])
    right_self = (
        left_self
        if dataset2 is None
        else np.array([soft_dtw(x, x, gamma=gamma) for x in right])
    )
    distances -= 0.5 * (left_self[:, None] + right_self[None, :])
    return distances


def dtw_path_from_metric(
    s1,
    s2=None,
    metric="euclidean",
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
    **kwds,
):
    check_numpy_backend(be)
    if metric == "precomputed":
        if s2 is not None:
            raise ValueError("s2 must be None when metric='precomputed'")
        costs = f64(s1)
        if costs.ndim != 2:
            raise ValueError("A precomputed cost matrix must be two-dimensional")
    else:
        first, second = check_series_pair(s1, s2)
        if callable(metric):
            costs = np.empty((len(first), len(second)), dtype=np.float64)
            for i, x in enumerate(first):
                for j, y in enumerate(second):
                    costs[i, j] = metric(x, y, **kwds)
        else:
            from scipy.spatial.distance import cdist

            costs = np.ascontiguousarray(cdist(first, second, metric=metric, **kwds))
    n, m = costs.shape
    if n == 0 or m == 0:
        raise ValueError("A precomputed cost matrix cannot have an empty dimension")
    mask = constraint_mask(
        n,
        m,
        global_constraint,
        sakoe_chiba_radius,
        itakura_max_slope,
    )
    acc = _scratch(n, m)
    mask_array = acc if mask is None else mask
    distance = float(
        lib().mts_dtw_from_cost(
            addr(costs), n, m, addr(mask_array), mask is not None, addr(acc)
        )
    )
    return _path_from_acc(acc, n, m), distance


def sakoe_chiba_mask(sz1, sz2, radius=1, be=None):
    check_numpy_backend(be)
    return constraint_mask(
        int(sz1), int(sz2), "sakoe_chiba", sakoe_chiba_radius=radius
    ).astype(bool)


def itakura_mask(sz1, sz2, max_slope=2.0, be=None):
    check_numpy_backend(be)
    return constraint_mask(
        int(sz1), int(sz2), "itakura", itakura_max_slope=max_slope
    ).astype(bool)


def compute_mask(
    s1,
    s2,
    global_constraint=0,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    check_numpy_backend(be)
    n = int(s1) if isinstance(s1, (int, np.integer)) else len(s1)
    m = int(s2) if isinstance(s2, (int, np.integer)) else len(s2)
    names = {0: None, 1: "itakura", 2: "sakoe_chiba"}
    if global_constraint not in names:
        raise ValueError("global_constraint must be 0, 1, or 2")
    mask = constraint_mask(
        n,
        m,
        names[global_constraint],
        sakoe_chiba_radius,
        itakura_max_slope,
    )
    return np.ones((n, m), dtype=bool) if mask is None else mask.astype(bool)


def lb_keogh(
    ts_query,
    ts_candidate=None,
    radius=1,
    envelope_candidate=None,
):
    query = time_series(ts_query)
    if envelope_candidate is None:
        if ts_candidate is None:
            raise ValueError("Either ts_candidate or envelope_candidate is required")
        candidate = time_series(ts_candidate)
        lower = np.empty_like(candidate)
        upper = np.empty_like(candidate)
        radius = int(radius)
        for i in range(len(candidate)):
            window = candidate[max(0, i - radius) : i + radius + 1]
            lower[i] = np.min(window, axis=0)
            upper[i] = np.max(window, axis=0)
    else:
        lower, upper = (time_series(part, remove_nans=False) for part in envelope_candidate)
    if query.shape != lower.shape or lower.shape != upper.shape:
        raise ValueError("Query and envelope must have identical shapes")
    below = np.minimum(query - lower, 0.0)
    above = np.maximum(query - upper, 0.0)
    return float(np.linalg.norm(below + above))


__all__ = [
    "cdist_dtw",
    "cdist_soft_dtw",
    "cdist_soft_dtw_normalized",
    "compute_mask",
    "dtw",
    "dtw_path",
    "dtw_path_from_metric",
    "itakura_mask",
    "lb_keogh",
    "sakoe_chiba_mask",
    "soft_dtw",
]
