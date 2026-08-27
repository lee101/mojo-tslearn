"""Numerical and behavioural parity checks against tslearn."""

import numpy as np
import pytest

ts_metrics = pytest.importorskip("tslearn.metrics")
ts_barycenters = pytest.importorskip("tslearn.barycenters")
ts_clustering = pytest.importorskip("tslearn.clustering")

import mojotslearn as mts
from mojotslearn import metrics as mts_metrics


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([1, 2, 3], [1, 2, 2, 3], 0.0),
        ([1, 2, 3], [1, 2, 2, 3, 4], 1.0),
        ([1, 2, 3], [3, 4, -3], np.sqrt(42.0)),
    ],
)
def test_dtw_published_examples(first, second, expected):
    assert mts.dtw(first, second) == pytest.approx(expected)
    assert mts.dtw(first, second) == pytest.approx(ts_metrics.dtw(first, second))


def test_dtw_multivariate_random():
    rng = np.random.default_rng(4)
    first = rng.normal(size=(29, 4))
    second = rng.normal(size=(23, 4))
    assert mts.dtw(first, second) == pytest.approx(
        ts_metrics.dtw(first, second), abs=1e-12
    )


def test_dtw_simd_tail_features():
    rng = np.random.default_rng(44)
    first = rng.normal(size=(13, 5))
    second = rng.normal(size=(11, 5))
    assert mts.dtw(first, second) == pytest.approx(
        ts_metrics.dtw(first, second), abs=1e-12
    )


def test_dtw_path_and_tie_breaking():
    first = [1, 2, 3]
    second = [1, 2, 2, 3]
    assert mts.dtw_path(first, second) == ts_metrics.dtw_path(first, second)


@pytest.mark.parametrize(
    "params",
    [
        {"global_constraint": "sakoe_chiba"},
        {"sakoe_chiba_radius": 3},
        {"global_constraint": "itakura"},
        {"itakura_max_slope": 3.0},
    ],
)
def test_constrained_dtw(params):
    rng = np.random.default_rng(10)
    first = rng.normal(size=(24, 2))
    second = rng.normal(size=(19, 2))
    assert mts.dtw(first, second, **params) == pytest.approx(
        ts_metrics.dtw(first, second, **params), abs=1e-12
    )
    assert mts.dtw_path(first, second, **params) == ts_metrics.dtw_path(
        first, second, **params
    )


def test_constraint_masks():
    assert np.array_equal(
        mts.sakoe_chiba_mask(7, 3, radius=1),
        ts_metrics.sakoe_chiba_mask(7, 3, radius=1),
    )
    assert np.array_equal(
        mts.itakura_mask(13, 11, max_slope=2.5),
        ts_metrics.itakura_mask(13, 11, max_slope=2.5),
    )
    assert np.array_equal(
        mts.compute_mask(9, 12, global_constraint=2, sakoe_chiba_radius=2),
        ts_metrics.compute_mask(9, 12, global_constraint=2, sakoe_chiba_radius=2),
    )


def test_input_validation_and_nan_suffix():
    assert mts.dtw([1, 2, 3, np.nan], [1, 2, 3]) == 0.0
    with pytest.raises(ValueError, match="feature size"):
        mts.dtw(np.ones((3, 2)), np.ones((3, 3)))
    with pytest.raises(ValueError, match="zero length"):
        mts.dtw([], [1])
    with pytest.raises(ValueError, match="empty series"):
        mts.cdist_dtw(np.empty((2, 0, 1)))


def test_cdist_dtw_cross_and_self():
    rng = np.random.default_rng(2)
    first = rng.normal(size=(8, 17, 2))
    second = rng.normal(size=(6, 13, 2))
    assert np.allclose(
        mts.cdist_dtw(first, second),
        ts_metrics.cdist_dtw(first, second),
        atol=1e-12,
    )
    assert np.allclose(
        mts.cdist_dtw(first), ts_metrics.cdist_dtw(first), atol=1e-12
    )


def test_equal_dataset_buffer_is_zero_copy():
    values = np.arange(48, dtype=np.float64).reshape(3, 8, 2)
    series = mts_metrics.dataset(values)
    assert mts_metrics._equal_dataset_buffer(values, series) is values


def test_cdist_dtw_serial_and_parallel_threshold():
    assert not mts_metrics._parallel_cdist_dtw(4, 4, 32, 32, None)
    assert not mts_metrics._parallel_cdist_dtw(4, 4, 256, 256, 1)
    assert mts_metrics._parallel_cdist_dtw(4, 4, 256, 256, None)
    rng = np.random.default_rng(45)
    first = rng.normal(size=(4, 256, 1))
    second = rng.normal(size=(4, 256, 1))
    parallel = mts.cdist_dtw(first, second)
    serial = mts.cdist_dtw(first, second, n_jobs=1)
    assert np.allclose(parallel, serial, atol=1e-12)


def test_cdist_dtw_ragged_and_constrained():
    first = [[1, 2, 3], [1, 2, 2, 3, 4]]
    second = [[0, 1, 2, 3], [3, 2]]
    params = {"sakoe_chiba_radius": 2}
    assert np.allclose(
        mts.cdist_dtw(first, second, **params),
        ts_metrics.cdist_dtw(first, second, **params),
    )


@pytest.mark.parametrize("gamma", [0.0, 0.01, 0.5, 1.0, 10.0])
def test_soft_dtw(gamma):
    first = np.array([1.0, 2.0, 2.0, 3.0])
    second = np.array([1.0, 2.0, 3.0, 4.0])
    assert mts.soft_dtw(first, second, gamma=gamma) == pytest.approx(
        ts_metrics.soft_dtw(first, second, gamma=gamma), abs=5e-9
    )


def test_soft_dtw_multivariate():
    rng = np.random.default_rng(8)
    first = rng.normal(size=(20, 3))
    second = rng.normal(size=(16, 3))
    assert mts.soft_dtw(first, second, gamma=0.7) == pytest.approx(
        ts_metrics.soft_dtw(first, second, gamma=0.7), abs=5e-9
    )


def test_soft_dtw_diagonal_threshold_and_simd_tail():
    assert mts_metrics._soft_dtw_scratch(255, 257, 0.8).shape == (2, 258)
    assert mts_metrics._soft_dtw_scratch(256, 256, 0.8).shape == (774,)
    rng = np.random.default_rng(49)
    first = rng.normal(size=(300, 5))
    second = rng.normal(size=(221, 5))
    assert mts.soft_dtw(first, second, gamma=0.8) == pytest.approx(
        ts_metrics.soft_dtw(first, second, gamma=0.8), abs=5e-9
    )


def test_cdist_soft_dtw_and_normalized():
    rng = np.random.default_rng(11)
    first = rng.normal(size=(7, 15, 2))
    second = rng.normal(size=(4, 12, 2))
    assert np.allclose(
        mts.cdist_soft_dtw(first, second, gamma=0.8),
        ts_metrics.cdist_soft_dtw(first, second, gamma=0.8),
        atol=5e-9,
    )
    assert np.allclose(
        mts.cdist_soft_dtw_normalized(first, second, gamma=0.8),
        ts_metrics.cdist_soft_dtw_normalized(first, second, gamma=0.8),
        atol=5e-9,
    )
    self_distances = mts.cdist_soft_dtw_normalized(first, gamma=0.8)
    assert np.allclose(np.diag(self_distances), 0.0, atol=1e-12)


def test_cdist_soft_dtw_gpu_and_fallback(monkeypatch):
    rng = np.random.default_rng(46)
    values = rng.normal(size=(8, 32, 2))
    cpu = mts.cdist_soft_dtw(values, gamma=0.8)
    if mts_metrics._gpu_memory_available(2_000_000):
        gpu = mts.cdist_soft_dtw(values, gamma=0.8, device="gpu")
        assert np.allclose(gpu, cpu, atol=5e-9)
    monkeypatch.setattr(mts_metrics, "_gpu_memory_available", lambda _: False)
    fallback = mts.cdist_soft_dtw(values, gamma=0.8, device="gpu")
    assert np.allclose(fallback, cpu, atol=5e-9)


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "cityblock"])
def test_dtw_path_from_metric(metric):
    rng = np.random.default_rng(21)
    first = rng.normal(size=(9, 3))
    second = rng.normal(size=(7, 3))
    ours = mts.dtw_path_from_metric(first, second, metric=metric)
    theirs = ts_metrics.dtw_path_from_metric(first, second, metric=metric)
    assert ours[0] == theirs[0]
    assert ours[1] == pytest.approx(theirs[1], abs=1e-12)


def test_dtw_path_from_precomputed():
    costs = np.array([[0.0, 2.0, 4.0], [1.0, 0.0, 2.0], [3.0, 1.0, 0.0]])
    assert mts.dtw_path_from_metric(
        costs, metric="precomputed"
    ) == ts_metrics.dtw_path_from_metric(costs, metric="precomputed")
    with pytest.raises(ValueError, match="empty dimension"):
        mts.dtw_path_from_metric(np.empty((0, 3)), metric="precomputed")


def test_lb_keogh():
    query = [1.0, 2.0, 3.0, 2.0, 1.0]
    candidate = [0.0, 1.0, 1.0, 1.0, 0.0]
    assert mts.lb_keogh(query, candidate, radius=1) == pytest.approx(
        ts_metrics.lb_keogh(query, candidate, radius=1)
    )


def test_euclidean_barycenter():
    values = [[1, 2, 3, 4], [1, 2, 4, 5]]
    assert np.allclose(
        mts.euclidean_barycenter(values),
        ts_barycenters.euclidean_barycenter(values),
    )


def test_dba_published_example():
    values = [[1, 2, 3, 4], [1, 2, 4, 5]]
    assert np.allclose(
        mts.dtw_barycenter_averaging(values, max_iter=5),
        ts_barycenters.dtw_barycenter_averaging(values, max_iter=5),
    )
    one_init, _ = mts.dtw_barycenter_averaging_one_init(values, max_iter=5)
    assert np.allclose(one_init, mts.dtw_barycenter_averaging_petitjean(
        values, max_iter=5
    ))


def test_dba_weighted_and_resampled():
    values = np.array([[[0.0], [0.0], [0.0]], [[10.0], [10.0], [10.0]]])
    weights = np.array([0.75, 0.25])
    ours = mts.dtw_barycenter_averaging(
        values, barycenter_size=4, max_iter=5, weights=weights
    )
    theirs = ts_barycenters.dtw_barycenter_averaging(
        values, barycenter_size=4, max_iter=5, weights=weights
    )
    assert np.allclose(ours, theirs)


def test_dba_multivariate_and_constrained():
    rng = np.random.default_rng(33)
    values = rng.normal(size=(5, 10, 2))
    params = {"sakoe_chiba_radius": 2}
    ours = mts.dtw_barycenter_averaging(values, max_iter=4, metric_params=params)
    theirs = ts_barycenters.dtw_barycenter_averaging(
        values, max_iter=4, metric_params=params
    )
    assert np.allclose(ours, theirs, atol=1e-12)


def _same_partition(first, second):
    return np.array_equal(
        first[:, None] == first[None, :], second[:, None] == second[None, :]
    )


def test_time_series_kmeans_euclidean_parity():
    rng = np.random.default_rng(40)
    values = np.concatenate(
        [rng.normal(-2, 0.2, (10, 12, 1)), rng.normal(2, 0.2, (10, 12, 1))]
    )
    init = values[[0, 10]].copy()
    ours = mts.TimeSeriesKMeans(
        n_clusters=2, init=init, n_init=1, max_iter=10
    ).fit(values)
    theirs = ts_clustering.TimeSeriesKMeans(
        n_clusters=2, init=init, n_init=1, max_iter=10
    ).fit(values)
    assert _same_partition(ours.labels_, theirs.labels_)
    assert np.allclose(ours.cluster_centers_, theirs.cluster_centers_)
    assert ours.inertia_ == pytest.approx(theirs.inertia_)
    assert np.allclose(ours.transform(values), theirs.transform(values))


def test_time_series_kmeans_dtw_parity():
    rng = np.random.default_rng(41)
    base = np.sin(np.linspace(0, 2 * np.pi, 18))
    values = np.concatenate(
        [
            base[None, :, None] + rng.normal(0, 0.05, (7, 18, 1)),
            -base[None, :, None] + rng.normal(0, 0.05, (7, 18, 1)),
        ]
    )
    init = values[[0, 7]].copy()
    ours = mts.TimeSeriesKMeans(
        n_clusters=2, init=init, metric="dtw", n_init=1, max_iter=5
    ).fit(values)
    theirs = ts_clustering.TimeSeriesKMeans(
        n_clusters=2, init=init, metric="dtw", n_init=1, max_iter=5
    ).fit(values)
    assert _same_partition(ours.labels_, theirs.labels_)
    assert np.allclose(ours.cluster_centers_, theirs.cluster_centers_, atol=1e-10)
    assert ours.inertia_ == pytest.approx(theirs.inertia_, abs=1e-10)
    assert np.array_equal(ours.predict(values), theirs.predict(values))


def test_estimator_protocol_and_softdtw_limit():
    estimator = mts.TimeSeriesKMeans(n_clusters=2, random_state=0)
    assert estimator.get_params()["n_clusters"] == 2
    assert estimator.set_params(max_iter=3) is estimator
    values = np.array([[[0.0]], [[0.1]], [[4.0]], [[4.1]]])
    labels = estimator.fit_predict(values)
    assert labels.shape == (4,)
    with pytest.raises(ValueError, match="not implemented"):
        mts.TimeSeriesKMeans(metric="softdtw").fit(np.ones((4, 5, 1)))


def test_kmeans_honors_max_iter_barycenter(monkeypatch):
    calls = []

    def fake_dba(values, **kwargs):
        calls.append(kwargs["max_iter"])
        return np.mean(values, axis=0)

    monkeypatch.setattr(
        "mojotslearn.clustering.dtw_barycenter_averaging", fake_dba
    )
    values = np.array([[[0.0]], [[0.1]], [[4.0]], [[4.1]]])
    mts.TimeSeriesKMeans(
        n_clusters=2,
        metric="dtw",
        init=values[[0, 2]],
        max_iter=1,
        max_iter_barycenter=7,
    ).fit(values)
    assert calls and set(calls) == {7}
