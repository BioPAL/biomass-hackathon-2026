"""Step 05 - correlation of GEDI metrics with TOM radar features, over the 500 m cells (Pearson and Spearman).

Outputs: results/corr_GEDI_vs_TOM_{pearson,spearman}.csv and heatmaps in results/figures/
"""
import logging

import pandas as pd

from .. import cells as C
from .. import features as F
from ..config import Config
from ..dataset import Dataset
from ..plotting import correlation_heatmap

log = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    ds = Dataset.from_config(cfg)
    ci = C.group_cells(ds.cell, ds.slope_ok & ds.targets_ok, ds.min_cell_shots)
    Pc = C.mean_by_cell(ds.P, ci)
    Y = pd.DataFrame({c: C.mean_by_cell(ds.shots[c].values, ci) for c in F.rh_columns(ds.shots)})
    for name, vals in ds.targets.items():
        if name not in Y:
            Y[name] = C.mean_by_cell(vals.values, ci)
    X = F.radar_cell_features(Pc, ds.z, ds.z_ground, ds.pols)
    data = pd.concat([Y, X], axis=1)
    pear = data.corr(method="pearson").loc[Y.columns, X.columns]
    spear = data.corr(method="spearman").loc[Y.columns, X.columns]
    out, figs = cfg.results_dir, cfg.results_dir / "figures"
    figs.mkdir(exist_ok=True)
    pear.round(3).to_csv(out / "corr_GEDI_vs_TOM_pearson.csv")
    spear.round(3).to_csv(out / "corr_GEDI_vs_TOM_spearman.csv")
    correlation_heatmap(pear, f"Pearson r: GEDI metrics vs TOM features, {len(data)} cells", figs / "corr_pearson.png")
    correlation_heatmap(spear, "Spearman rho", figs / "corr_spearman.png")
    top = pear.stack().rename("r").reset_index()
    top.columns = ["GEDI", "TOM", "r"]
    log.info("strongest pairs:\n%s", top.reindex(top.r.abs().sort_values(ascending=False).index).head(10).round(2).to_string(index=False))
