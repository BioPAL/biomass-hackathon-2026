import numpy as np

from biomass_tom_gedi import cells as C


def test_pixels_per_cell_lope():
    assert C.pixels_per_cell(500, 0.000417) == 11


def test_cell_ids_and_blocks():
    assert C.cell_ids(np.array([0, 10, 11, 22]), np.array([0, 10, 11, 23]), 11).tolist() == [0, 0, 100001, 200002]
    b = C.block_ids(np.array([0.05, 0.15]), np.array([11.05, 11.15]))
    assert b[0] != b[1]


def test_group_cells_min_shots_and_mean():
    cell = np.array([1, 1, 1, 2, 2, 3, 1, 1])
    ci = C.group_cells(cell, np.ones(8, bool), min_shots=3)
    assert ci.cells.tolist() == [1]                       # cells 2 and 3 have < 3 shots
    assert ci.n.tolist() == [5]
    vals = np.array([1.0, 2, 3, 100, 100, 100, np.nan, 5])
    assert C.mean_by_cell(vals, ci)[0] == np.mean([1, 2, 3, 5])   # NaN-aware


def test_group_cells_respects_keep_mask():
    cell = np.array([1] * 6)
    keep = np.array([True, True, True, False, False, False])
    assert C.group_cells(cell, keep, min_shots=4).cells.size == 0
    assert C.group_cells(cell, keep, min_shots=3).cells.tolist() == [1]


def test_mean_by_cell_profiles_shape():
    cell = np.repeat([5, 9], 6)
    ci = C.group_cells(cell, np.ones(12, bool), 5)
    P = np.random.default_rng(0).random((12, 7, 3))
    out = C.mean_by_cell(P, ci)
    assert out.shape == (2, 7, 3)
    assert np.allclose(out[1], P[6:].mean(0))
