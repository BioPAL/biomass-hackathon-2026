"""Step 04 - aggregate the slope-kept shots into ~500 m cells.

Shot metrics are averaged per cell; TOM profiles are averaged in linear power (then converted to dB where used).
Cells with fewer than cells.min_shots shots are dropped.
Outputs: data/interim/cells.parquet, data/interim/cell_profiles.npz, results/cells_500m.csv
"""
import logging

import numpy as np

from ..config import Config
from ..dataset import Dataset

log = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    ds = Dataset.from_config(cfg)
    tab, Pc = ds.cell_table(ds.slope_ok & ds.targets_ok)
    log.info("%d cells with >= %d shots | median %d shots per cell", len(tab), ds.min_cell_shots, int(tab.n_shots.median()))
    tab.to_parquet(cfg.cells_path, index=False)
    tab.to_csv(cfg.results_dir / "cells_500m.csv", index=False)
    np.savez_compressed(cfg.cell_profiles_path, profiles=Pc, z=ds.z, pols=np.array(ds.pols), cell=tab.cell.values,
                        z_ground=ds.z_ground)
