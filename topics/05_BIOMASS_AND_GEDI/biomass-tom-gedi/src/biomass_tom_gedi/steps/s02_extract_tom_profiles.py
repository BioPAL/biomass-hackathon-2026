"""Step 02 - extract the TOM profile under every GEDI shot.

Reads the cube block by block and averages a (2*half+1)^2 pixel window around each footprint (the tomography has a
150 m averaging window, so half=1 -> 3x3 pixels ~ 140 m). Shots on empty pixels are dropped.
Outputs: data/interim/shots.parquet (GEDI table + row/col), data/interim/tom_profiles.npz (linear power)
"""
import logging

import numpy as np

from .. import gedi
from ..config import Config
from ..tom import TOMProduct, extract_profiles, ground_proxy

log = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    prod = TOMProduct(cfg["paths"]["tom_product"])
    grid = prod.grid
    g = gedi.load_gedi_table(cfg["paths"]["gedi_table"], bbox=grid.bbox, start_time=grid.start_time,
                             max_days=cfg.get("gedi", "max_days"))
    g["row"], g["col"] = grid.row_col(g.latitude.values, g.longitude.values)
    with prod.open_cube() as cube:
        prof = extract_profiles(cube, g.row.values, g.col.values, half=int(cfg.get("tom", "window_half", default=1)))
    ok = np.isfinite(prof).all(axis=(1, 2))
    log.info("%d of %d shots have a valid TOM profile (the rest fall on empty pixels)", ok.sum(), len(g))
    g, prof = g[ok].reset_index(drop=True), prof[ok]
    z_ground = ground_proxy(prof, grid.z, grid.pols)
    log.info("ground proxy Z_GROUND = %.1f m (HH peak of the mean profile)", z_ground)
    g.to_parquet(cfg.shots_path, index=False)
    np.savez_compressed(cfg.profiles_path, profiles=prof, z=grid.z, pols=np.array(grid.pols), z_ground=z_ground,
                        latitude=g.latitude.values, longitude=g.longitude.values)
    log.info("saved %s and %s", cfg.shots_path, cfg.profiles_path)
