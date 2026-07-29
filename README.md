# mojo-tslearn

`mojo-tslearn` is a focused port of tslearn's compute-heavy time-series
algorithms to Mojo. It keeps tslearn's public names and call signatures for the
covered subset, returns NumPy arrays, and exposes the compiled kernels to Python
through a small `ctypes` layer.

```python
import numpy as np
from mojotslearn.clustering import TimeSeriesKMeans
from mojotslearn.metrics import dtw, dtw_path, soft_dtw

first = np.array([1.0, 2.0, 3.0])
second = np.array([1.0, 2.0, 2.0, 3.0])

assert dtw(first, second) == 0.0
path, distance = dtw_path(first, second)
print(path, distance)
print(soft_dtw(first, second, gamma=1.0))

X = np.array([
    [[0.0], [0.1], [0.0], [0.2]],
    [[0.1], [0.0], [0.2], [0.1]],
    [[3.0], [3.1], [2.9], [3.0]],
    [[3.1], [2.9], [3.0], [3.2]],
])
labels = TimeSeriesKMeans(
    n_clusters=2, metric="dtw", random_state=0
).fit_predict(X)
print(labels)
```

## Coverage

| tslearn module | covered API |
| --- | --- |
| `metrics` | `dtw`, `dtw_path`, `cdist_dtw`, Sakoe-Chiba and Itakura constraints, `soft_dtw`, `cdist_soft_dtw`, `cdist_soft_dtw_normalized`, `dtw_path_from_metric`, `lb_keogh`, and mask helpers |
| `barycenters` | `euclidean_barycenter`, `dtw_barycenter_averaging`, `dtw_barycenter_averaging_one_init`, and the Petitjean alias |
| `clustering` | `TimeSeriesKMeans` with `metric="euclidean"` or `metric="dtw"`; `fit`, `fit_predict`, `predict`, `transform`, and the scikit-learn estimator protocol |

DTW metrics accept univariate or multivariate series, unequal lengths, ragged
datasets, trailing-NaN padding, and the listed constraint parameters.
The pairwise fast path handles equal-length datasets in one native call.

This is deliberately not all of tslearn. Soft-DTW barycenter optimization and
`TimeSeriesKMeans(metric="softdtw")` are not implemented. LCSS, GAK, CTW,
subsequence DTW, limited-warping-length DTW, shapelets, neural-network helpers,
datasets, and model persistence are also out of scope. Clustering currently
requires equal-length finite series, and only the NumPy backend is supported.
Large equal-length pairwise DTW calls run across CPU rows once they exceed one
million dynamic-programming cells; smaller calls stay serial. Passing
`n_jobs=1` forces the serial path.

## Install and verify

The Pixi environment supplies the pinned Mojo nightly, Python dependencies, and
the real `tslearn` package used by the parity suite.

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles the single Mojo unit to
`dist/libmojo-tslearn.so`. Importing the Python package also rebuilds a missing
or stale library. Set `MOJOTSLEARN_LIB=/absolute/path/to/library.so` to use an
already-built shared library.

For a source install outside Pixi:

```bash
python -m pip install -e .
bash build/build.sh
```

That route requires a compatible `mojo` executable on `PATH`.

## Correctness

The test suite contains 37 numerical and behavioural parity tests against
tslearn 0.9.0. It compares exact alignment paths and tie-breaking, random
multivariate distances, both global constraints, pairwise and normalized
soft-DTW, weighted and constrained DBA, estimator attributes, cluster
partitions, centers, inertia, predictions, and transformed distances.
Published tslearn examples are included as fixed test vectors. Dedicated tests
cover empty-buffer rejection at the FFI boundary, the SIMD remainder, the
pairwise serial/parallel threshold, zero-copy dataset reuse, GPU parity on this
machine, unavailable-GPU fallback, and propagation of the clustering
barycenter-iteration setting.

## Benchmarks

Measured on an Intel Xeon E5-2697 v4 at 2.30 GHz (`Linux x86_64`) with Python
3.13.14, NumPy 2.4.6, tslearn 0.9.0, and the pinned Mojo
`1.0.0b3.dev2026072406`. These are best-of-three warm timings from
`pixi run bench`; the Pixi task takes a machine-wide lock.

| benchmark | mojo-tslearn | tslearn | result |
| --- | ---: | ---: | ---: |
| DTW (2,000 x 3 against 2,000 x 3) | 18.94 ms | 41.41 ms | 2.19x faster |
| soft-DTW gamma=1 (1,200 x 2) | 130.47 ms | 117.18 ms | 1.11x slower |
| cdist_dtw (48 x 48, length 80 x 2) | 17.66 ms | 77.71 ms | 4.40x faster |
| cdist_soft_dtw (36 x 36, length 64) | 437.62 ms | 551.27 ms | 1.26x faster |
| cdist_soft_dtw GPU (36 x 36, length 64) | 46.62 ms | 586.38 ms | 12.58x faster |
| DBA (24 series, length 80, 5 epochs) | 5.23 ms | 15.29 ms | 2.92x faster |
| TimeSeriesKMeans DTW (30 x 48, k=2) | 8.66 ms | 72.42 ms | 8.37x faster |

Pairwise DTW benefits from symmetry and thresholded row parallelism. Soft-DTW
keeps only two DP rows, skips one exponential per cell, and reuses the
reciprocal of `gamma`. The table reports the CPU soft-DTW regression as
measured; no result is extrapolated from another run.

Pairwise soft-DTW also has an optional `device="gpu"` path because its
independent matrices and transcendental-heavy cells have enough arithmetic
intensity to amortize transfers. CPU remains the default. The GPU call checks
available device memory and caps allocation before entering native code. An
unavailable GPU falls back to CPU; a native GPU launch failure emits a runtime
warning before falling back.

## How it works

All numerical kernels live in `src/capi.mojo`, one compilation unit so the
fixed Mojo build cost is paid once. The exports use `@export("name")` and the C
ABI. NumPy owns every CPU input, result, and scratch buffer; Python normalizes
inputs to C-contiguous `float64`, validates non-empty dimensions, retains every
array for the duration of the synchronous call, and passes addresses as 64-bit
integers. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside each export. Mojo does
not allocate CPU memory through the FFI. Contiguous `float64` dataset inputs
cross the boundary without a copy. The native C ABI is an implementation
detail; callers should use the Python API, which supplies the required buffer
sizes and dtype.

Time series use C-contiguous row-major `float64` storage:
`(n_series, n_timestamps, n_features)`. A scalar DTW call allocates one
`(n + 1, m + 1)` accumulated-cost matrix. Soft-DTW retains two rows, and
parallel pairwise DTW gives each active row an independent scratch matrix.
Local squared Euclidean distances use the host `float64` SIMD width with a
scalar remainder. DBA backtracks the accumulated matrix and updates weighted
alignment sums in the same native call.

## License

MIT
