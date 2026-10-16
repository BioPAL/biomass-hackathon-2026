"""Step 06 - ridge regression TOM profile -> GEDI metrics with spatial-block cross-validation.

Runs at 500 m cell level and at shot level on the slope-kept shots. Predictors: the whole HH+HV+VV dB profile.
Outputs: results/ridge_cv_{cell,shot}.csv (r, R2, RMSE per target), results/predictions_{cell,shot}.parquet
"""
import logging

import pandas as pd

from ..config import Config
from ..dataset import Dataset
from ..experiment import evaluate

log = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    ds = Dataset.from_config(cfg)
    for level in cfg.get("ridge_cv", "levels", default=["cell", "shot"]):
        res, pred, ld = evaluate(ds, ds.slope_ok, level, "baseline")
        res.round(4).to_csv(cfg.results_dir / f"ridge_cv_{level}.csv", index=False)
        names = list(ds.targets.columns)
        out = pd.concat([pd.DataFrame({"id": ld.ids}),
                         pd.DataFrame(ld.Y, columns=[f"obs_{n}" for n in names]),
                         pd.DataFrame(pred, columns=[f"pred_{n}" for n in names])], axis=1)
        out.to_parquet(cfg.results_dir / f"predictions_{level}.parquet", index=False)
        log.info("%s level: n=%d, mean r=%.3f, mean R2=%.3f\n%s", level, res.n.iloc[0], res.r_cv.mean(), res.R2_cv.mean(),
                 res[["metric", "r_cv", "R2_cv", "RMSE"]].round(3).to_string(index=False))
