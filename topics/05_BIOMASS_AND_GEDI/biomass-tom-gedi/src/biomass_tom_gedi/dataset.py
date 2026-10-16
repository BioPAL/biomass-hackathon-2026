"""The analysis dataset: shots + TOM profiles + targets, viewable at shot level or 500 m cell level."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import cells as C
from . import features as F
from .config import Config


@dataclass
class LevelData:
    X: np.ndarray            # predictors
    Y: np.ndarray            # targets (n, n_targets)
    groups: np.ndarray       # spatial CV block of each sample
    ids: np.ndarray          # shot indices (shot level) or cell ids (cell level)
    n_shots: np.ndarray | None = None   # shots per cell (cell level)


class Dataset:
    def __init__(self, shots: pd.DataFrame, P: np.ndarray, z: np.ndarray, pols: list[str], z_ground: float,
                 npx: int, feature_cfg: dict | None = None, min_cell_shots: int = 5, block_deg: float = 0.1):
        self.shots, self.P, self.z, self.pols, self.z_ground = shots.reset_index(drop=True), P, z, pols, z_ground
        self.npx, self.min_cell_shots, self.block_deg = npx, min_cell_shots, block_deg
        self.feature_cfg = feature_cfg or {}
        self.targets = F.build_targets(self.shots)
        self.targets_ok = self.targets.notna().all(axis=1).values
        self.cell = C.cell_ids(self.shots.row.values, self.shots.col.values, npx)
        self.block = C.block_ids(self.shots.latitude.values, self.shots.longitude.values, block_deg)
        self.slope_ok = (self.shots["slope_ok"].values.astype(bool) if "slope_ok" in self.shots
                         else np.ones(len(self.shots), bool))

    # ---- loading ------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config) -> "Dataset":
        shots = pd.read_parquet(cfg.shots_path)
        with np.load(cfg.profiles_path, allow_pickle=False) as d:
            P, z, pols = d["profiles"].astype("float64"), d["z"], [str(p) for p in d["pols"]]
            z_ground = float(d["z_ground"])
        if len(shots) != len(P):
            raise ValueError("shots table and profiles are not aligned; rerun steps 02-03")
        return cls(shots, P, z, pols, z_ground,
                   npx=int(cfg.get("cells", "pixels_per_cell", default=11)),
                   feature_cfg=cfg.get("features", default={}),
                   min_cell_shots=int(cfg.get("cells", "min_shots", default=5)),
                   block_deg=float(cfg.get("cv", "block_deg", default=0.1)))

    # ---- features -------------------------------------------------------------
    def features(self, P: np.ndarray) -> np.ndarray:
        mode = self.feature_cfg.get("mode", "profile_db")
        if mode == "profile_db":
            return F.profile_db_features(P, self.z, tuple(self.feature_cfg.get("window_m", (-20, 90))))
        if mode == "ground_relative":
            return F.ground_relative_features(P, self.z, self.z_ground, self.pols)
        raise ValueError(f"unknown features.mode {mode!r} (use 'profile_db' or 'ground_relative')")

    # ---- views ----------------------------------------------------------------
    def level_data(self, keep: np.ndarray, level: str) -> LevelData:
        keep = keep & self.targets_ok
        Yall = self.targets.values
        if level == "shot":
            idx = np.flatnonzero(keep)
            return LevelData(self.features(self.P[idx]), Yall[idx], self.block[idx], idx)
        if level != "cell":
            raise ValueError("level must be 'shot' or 'cell'")
        ci = C.group_cells(self.cell, keep, self.min_cell_shots)
        Pc = C.mean_by_cell(self.P, ci)
        Yc = C.mean_by_cell(Yall, ci)
        lat = C.mean_by_cell(self.shots.latitude.values, ci)
        lon = C.mean_by_cell(self.shots.longitude.values, ci)
        # CV block from the cell centroid (cells straddling a block edge go wholly to one fold)
        return LevelData(self.features(Pc), Yc, C.block_ids(lat, lon, self.block_deg), ci.cells, ci.n)

    def cell_table(self, keep: np.ndarray, extra_cols: tuple[str, ...] = ("rh_98", "agbd", "cover", "slope_deg")):
        """The cell table + mean linear-power profile per cell (notebook section 12c)."""
        ci = C.group_cells(self.cell, keep, self.min_cell_shots)
        tab = pd.DataFrame({
            "cell": ci.cells, "n_shots": ci.n,
            "latitude": C.mean_by_cell(self.shots.latitude.values, ci),
            "longitude": C.mean_by_cell(self.shots.longitude.values, ci),
        })
        tab["block"] = C.block_ids(tab.latitude.values, tab.longitude.values, self.block_deg)
        for col in extra_cols:
            if col in self.shots:
                tab[col] = C.mean_by_cell(self.shots[col].values, ci)
        Pc = C.mean_by_cell(self.P, ci)
        for name, vals in self.targets.items():
            if name not in tab:
                tab[name] = C.mean_by_cell(vals.values, ci)
        tab = pd.concat([tab, F.tom_summary_features(Pc, self.z, self.z_ground, self.pols)], axis=1)
        return tab, Pc
