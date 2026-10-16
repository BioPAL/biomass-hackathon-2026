"""Step 03 - slope at every shot from the BIOMASS DTM_L2A, and the slope mask.

Adds `slope_deg` and `slope_ok` (slope < terrain.max_slope_deg; NaN slope -> not kept) to data/interim/shots.parquet.
"""
import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..terrain import DTMSlope

log = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    g = pd.read_parquet(cfg.shots_path)
    dtm = DTMSlope(cfg["paths"]["dtm_product"], smooth_px=float(cfg.get("terrain", "smooth_px", default=3)))
    if "digital_elevation_model" in g:
        dtm.check_orientation(g.latitude.values, g.longitude.values, g["digital_elevation_model"].values)
    max_slope = float(cfg.get("terrain", "max_slope_deg", default=5.0))
    g["slope_deg"] = dtm.slope_at(g.latitude.values, g.longitude.values)
    g["slope_ok"] = (g["slope_deg"] < max_slope).values
    log.info("slope mask (< %g deg): kept %d of %d shots (%.0f%%); no slope value for %d",
             max_slope, g.slope_ok.sum(), len(g), 100 * g.slope_ok.mean(), int(g.slope_deg.isna().sum()))
    g.to_parquet(cfg.shots_path, index=False)
