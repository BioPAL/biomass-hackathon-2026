import numpy as np

from src.process.aggregate import block_mean, smooth


def test_block_mean_of_constant():
    out = block_mean(np.full((6, 6), 5.0), 3)
    assert out.shape == (2, 2)
    assert np.allclose(out, 5.0)


def test_block_mean_needs_enough_valid_pixels():
    a = np.full((4, 4), 1.0)
    a[:2, :2] = np.nan
    a[0, 0] = 3.0  # one valid pixel out of four
    out = block_mean(a, 2, min_frac=0.5)
    assert np.isnan(out[0, 0])
    assert np.allclose(out[1, 1], 1.0)


def test_smooth_keeps_constant_and_ignores_gaps():
    a = np.full((9, 9), 7.0)
    a[4, 4] = np.nan
    out = smooth(a, 3)
    assert np.isclose(out[4, 4], 7.0)
    assert smooth(a, 1) is a
