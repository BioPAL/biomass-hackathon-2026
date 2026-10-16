"""Step 08 - observed vs predicted density plot (viridis) from the step-06 predictions.

Config: plots.target (e.g. agbd or rh_98), plots.level (cell or shot). Output: results/figures/obs_vs_pred_<target>_<level>.png
"""
import logging

import pandas as pd

from ..config import Config
from ..plotting import obs_vs_pred_hexbin

log = logging.getLogger(__name__)

UNITS = {"agbd": "Mg/ha", "rh_": "m"}


def run(cfg: Config) -> None:
    target = cfg.get("plots", "target", default="agbd")
    level = cfg.get("plots", "level", default="cell")
    d = pd.read_parquet(cfg.results_dir / f"predictions_{level}.parquet")
    unit = next((u for k, u in UNITS.items() if target.startswith(k)), "")
    figs = cfg.results_dir / "figures"
    figs.mkdir(exist_ok=True)
    out = figs / f"obs_vs_pred_{target}_{level}.png"
    obs_vs_pred_hexbin(d[f"obs_{target}"], d[f"pred_{target}"], f"Observed {target} ({unit})", f"Predicted {target} ({unit})", out)
    log.info("saved %s", out)
