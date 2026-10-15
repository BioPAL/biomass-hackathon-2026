import numpy as np

from src.evaluate.metrics import agreement_metrics


def test_constant_offset():
    ref = np.linspace(10, 40, 100)
    m = agreement_metrics(ref + 2.0, ref)
    assert m["n"] == 100
    assert np.isclose(m["bias"], 2.0)
    assert np.isclose(m["mae"], 2.0)
    assert np.isclose(m["rmse"], 2.0)
    assert np.isclose(m["r"], 1.0)
    assert np.isclose(m["rel_rmse"], 100 * 2.0 / ref.mean())


def test_nan_ignored_and_minimum_sample():
    ref = np.full(40, 20.0)
    est = ref.copy()
    est[:15] = np.nan
    m = agreement_metrics(est, ref)
    assert m["n"] == 25
    assert np.isnan(m["rmse"])  # fewer than 30 valid pairs


def test_rmse_splits_into_bias_and_spread():
    rng = np.random.default_rng(0)
    ref = rng.uniform(10, 40, 5000)
    est = ref - 3 + rng.normal(0, 4, ref.size)
    m = agreement_metrics(est, ref)
    sd = np.std(est - ref)
    assert np.isclose(m["rmse"] ** 2, m["bias"] ** 2 + sd**2, rtol=1e-6)
