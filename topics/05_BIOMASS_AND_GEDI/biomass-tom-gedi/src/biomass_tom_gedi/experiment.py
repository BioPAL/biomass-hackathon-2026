"""Evaluate the whole-profile ridge under a given shot selection, and run the bright-profile filter experiment."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import filters as FL
from . import modeling as M
from .dataset import Dataset

log = logging.getLogger(__name__)


def evaluate(ds: Dataset, keep: np.ndarray, level: str, label: str = ""):
    """Ridge spatial-block CV on the selected shots.

    Returns (per-target scores, out-of-fold predictions, LevelData), or None if there are too few samples/blocks.
    """
    ld = ds.level_data(keep, level)
    pred = M.spatial_cv_predict(ld.X, ld.Y, ld.groups)
    if pred is None:
        return None
    res = M.score(ld.Y, pred, list(ds.targets.columns))
    res.insert(0, "n", len(ld.Y))
    res.insert(0, "level", level)
    res.insert(0, "config", label)
    res["max_abs_r_layer"] = M.max_abs_layer_corr(ld.X, ld.Y)
    return res, pred, ld


def run_filter_experiment(ds: Dataset, features=FL.FEATURES, quantiles=(0.01, 0.025, 0.05, 0.10),
                          modes=("hv", "any"), n_random: int = 2, levels=("shot", "cell"),
                          out_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline (slope-kept shots) vs removing the brightest q of profiles, plus random-removal controls."""
    base = ds.slope_ok
    bf = FL.bright_features(ds.P, ds.z, ds.pols)
    rows, frac_rows = [], []

    def run(label: str, keep: np.ndarray, **extra):
        for level in levels:
            out = evaluate(ds, keep, level, label)
            if out is None:
                log.warning("skip %s (%s): too few samples", label, level)
                continue
            res = out[0]
            for k, v in extra.items():
                res[k] = v
            rows.append(res)
            log.info("%-42s %-5s n=%-6d mean r=%.3f R2=%.3f", label, level, res.n.iloc[0], res.r_cv.mean(), res.R2_cv.mean())
        if out_dir is not None and rows:
            pd.concat(rows).to_csv(Path(out_dir) / "results_long.csv", index=False)

    run("baseline", base, kind="real", seed=-1, frac_removed=0.0, mode="none", feature="none", q=np.nan)
    for mode in modes:
        for feat in features:
            for q in quantiles:
                fl = FL.flag_bright(bf, ds.pols, feat, q, mode, base)
                frac = fl.sum() / base.sum()
                label = f"{mode} {feat} worst {100 * q:g}%/pol"
                frac_rows.append(dict(mode=mode, feature=feat, worst_pct_per_pol=100 * q,
                                      n_removed=int(fl.sum()), frac_removed=frac))
                meta = dict(frac_removed=frac, mode=mode, feature=feat, q=q)
                run(label, base & ~fl, kind="real", seed=-1, **meta)
                for s in range(n_random):                       # control: same number of shots removed at random
                    rng = np.random.default_rng(s)
                    rk = base.copy()
                    rk[rng.choice(np.flatnonzero(base), int(fl.sum()), replace=False)] = False
                    run(f"{label} [random{s}]", rk, kind="random", seed=s, **meta)
    results = pd.concat(rows, ignore_index=True)
    frac_tab = pd.DataFrame(frac_rows)
    if out_dir is not None:
        results.to_csv(Path(out_dir) / "results_long.csv", index=False)
        frac_tab.to_csv(Path(out_dir) / "fraction_removed.csv", index=False)
    return results, frac_tab


def summarise_vs_baseline(results: pd.DataFrame) -> pd.DataFrame:
    """One row per config and level: samples left, mean r / R2 and their change vs baseline and vs random removal."""
    out = []
    for level, d in results.groupby("level"):
        mean = d.groupby(["config", "kind", "mode", "feature", "q"], dropna=False).agg(
            n=("n", "first"), r=("r_cv", "mean"), R2=("R2_cv", "mean"), frac_removed=("frac_removed", "first")).reset_index()
        b = mean[mean.config == "baseline"].iloc[0]
        real = mean[mean.kind == "real"].copy()
        rnd = mean[mean.kind == "random"].copy()
        rnd["real_config"] = rnd.config.str.replace(r" \[random\d+\]", "", regex=True)
        rnd_mean = rnd.groupby("real_config").R2.mean()
        real["level"] = level
        real["dr_vs_base"] = real.r - b.r
        real["dR2_vs_base"] = real.R2 - b.R2
        real["dR2_random_control"] = real.config.map(rnd_mean) - b.R2
        out.append(real)
    return pd.concat(out, ignore_index=True).sort_values(["level", "dR2_vs_base"], ascending=[True, False])
