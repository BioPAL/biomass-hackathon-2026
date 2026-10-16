import numpy as np

from biomass_tom_gedi import filters as FL
from biomass_tom_gedi import modeling as M
from biomass_tom_gedi.experiment import evaluate


def test_spatial_cv_recovers_linear_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 10))
    Y = np.c_[X[:, 0] * 2 + rng.normal(0, 0.1, 300), rng.normal(size=300)]
    groups = np.repeat(np.arange(10), 30)
    pred = M.spatial_cv_predict(X, Y, groups)
    res = M.score(Y, pred, ["signal", "noise"])
    assert res.r_cv[0] > 0.98 and res.R2_cv[0] > 0.95
    assert abs(res.r_cv[1]) < 0.3                         # no signal -> no skill


def test_spatial_cv_too_few_blocks_returns_none():
    X, Y = np.zeros((100, 3)), np.zeros((100, 1))
    assert M.spatial_cv_predict(X, Y, np.repeat([0, 1], 50)) is None


def test_flag_bright_takes_upper_tail_of_population():
    rng = np.random.default_rng(2)
    n, nz = 1000, 81
    z = -60.0 + 2.0 * np.arange(nz)
    P = rng.lognormal(0, 1, (n, nz, 3))
    pols = ["HH", "HV", "VV"]
    feats = FL.bright_features(P, z, pols)
    population = np.ones(n, bool)
    population[:200] = False
    fl = FL.flag_bright(feats, pols, "top_db", 0.10, "hv", population)
    assert not fl[:200].any()                              # never flags shots outside the population
    assert abs(fl.sum() - 80) <= 2                         # 10 % of the 800 population shots
    fl_any = FL.flag_bright(feats, pols, "top_db", 0.10, "any", population)
    assert fl_any.sum() >= fl.sum()


def test_evaluate_on_synthetic_dataset(synthetic_dataset):
    ds = synthetic_dataset
    for level in ("shot", "cell"):
        res, pred, ld = evaluate(ds, ds.slope_ok, level, "baseline")
        assert len(res) == 18 and res.r_cv.max() > 0.9
        assert pred.shape == ld.Y.shape
    cell_ld = ds.level_data(ds.slope_ok, "cell")
    assert (cell_ld.n_shots >= 5).all() and len(cell_ld.groups) == len(cell_ld.Y)
