from __future__ import annotations

import warnings

import numpy as np


def time_series(value, *, remove_nans: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("A time series must have shape (size,) or (size, features)")
    if remove_nans:
        finite_rows = ~np.all(np.isnan(array), axis=1)
        if finite_rows.any():
            array = array[: np.flatnonzero(finite_rows)[-1] + 1]
        else:
            array = array[:0]
    return np.ascontiguousarray(array)


def dataset(value) -> list[np.ndarray]:
    if isinstance(value, np.ndarray) and value.dtype != object:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            result = [time_series(array)]
        elif array.ndim == 2:
            result = [time_series(row) for row in array]
        elif array.ndim == 3:
            result = [time_series(row) for row in array]
        else:
            raise ValueError("A dataset must have 1, 2, or 3 dimensions")
        if not result or any(len(item) == 0 for item in result):
            raise ValueError("A time-series dataset cannot contain empty series")
        return result
    values = list(value)
    if not values:
        raise ValueError("A time-series dataset cannot be empty")
    if np.isscalar(values[0]):
        return [time_series(values)]
    result = [time_series(row) for row in values]
    if any(len(item) == 0 for item in result):
        raise ValueError("A time-series dataset cannot contain empty series")
    return result


def check_series_pair(s1, s2) -> tuple[np.ndarray, np.ndarray]:
    first, second = time_series(s1), time_series(s2)
    if not len(first) or not len(second):
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )
    if first.shape[1] != second.shape[1]:
        raise ValueError("All input time series must have the same feature size.")
    return first, second


def check_numpy_backend(be) -> None:
    if be is None or be == "numpy":
        return
    name = type(be).__name__.lower()
    if "numpy" not in name:
        raise NotImplementedError("mojotslearn currently supports the NumPy backend only")


def constraint_mask(
    n: int,
    m: int,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
) -> np.ndarray | None:
    aliases = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}
    if global_constraint not in aliases:
        raise ValueError(
            "global_constraint must be None, 'itakura', or 'sakoe_chiba'"
        )
    code = aliases[global_constraint]
    if code == 0 and sakoe_chiba_radius is not None and itakura_max_slope is not None:
        raise RuntimeWarning(
            "global_constraint is not set for DTW, but both sakoe_chiba_radius "
            "and itakura_max_slope are set"
        )
    if code == 2 or (code == 0 and sakoe_chiba_radius is not None):
        radius = 1 if sakoe_chiba_radius is None else int(sakoe_chiba_radius)
        mask = np.zeros((n, m), dtype=np.float64)
        if n > m:
            width = n - m + radius
            for j in range(m):
                lower = max(0, j - radius)
                upper = min(n, j + width) + 1
                mask[lower:upper, j] = 1.0
        else:
            width = m - n + radius
            for i in range(n):
                lower = max(0, i - radius)
                upper = min(m, i + width) + 1
                mask[i, lower:upper] = 1.0
        return mask
    if code == 1 or (code == 0 and itakura_max_slope is not None):
        slope = 2.0 if itakura_max_slope is None else float(itakura_max_slope)
        min_slope = (1.0 / slope) * n / m
        max_slope = slope * n / m
        columns = np.arange(m)
        lower = np.ceil(
            np.maximum(
                np.round(min_slope * columns, 2),
                np.round((n - 1) - max_slope * (m - 1) + max_slope * columns, 2),
            )
        ).astype(int)
        upper = np.floor(
            np.minimum(
                np.round(max_slope * columns, 2),
                np.round((n - 1) - min_slope * (m - 1) + min_slope * columns, 2),
            )
            + 1
        ).astype(int)
        mask = np.zeros((n, m), dtype=np.float64)
        for j in range(m):
            mask[max(0, lower[j]) : min(n, upper[j]), j] = 1.0
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            warnings.warn(
                "'itakura_max_slope' constraint is unfeasible for the provided "
                "time series sizes",
                RuntimeWarning,
            )
        return mask
    return None
