"""Targets (GEDI metrics) and predictors (TOM profile features)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .tom import db

POLS3 = ("HH", "HV", "VV")


# ---------------------------------------------------------------------------- targets
def build_targets(g: pd.DataFrame) -> pd.DataFrame:
    """The 18 GEDI structure metrics: rh_*, cover, agbd, fhd, pai, PAI/cover above 10/30 m, PAVD layers."""
    T = pd.DataFrame(index=g.index)
    for k in ["rh_25", "rh_50", "rh_75", "rh_90", "rh_98", "rh_100", "cover", "agbd", "fhd_normal", "pai"]:
        T[k] = g[k]
    T["PAI above 10 m"] = g["pai_z_2"]
    T["PAI above 30 m"] = g["pai_z_6"]
    T["cover above 30 m"] = g["cover_z_6"]
    for name, ks in [("PAVD 0–10 m", [0, 1]), ("PAVD 10–20 m", [2, 3]), ("PAVD 20–30 m", [4, 5]),
                     ("PAVD 30–40 m", [6, 7]), ("PAVD 40–60 m", [8, 9, 10, 11])]:
        T[name] = g[[f"pavd_z_{k}" for k in ks]].mean(axis=1)
    return T.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------- predictors
def window_index(z: np.ndarray, window_m: tuple[float, float]) -> np.ndarray:
    return np.where((z >= window_m[0]) & (z <= window_m[1]))[0]


def profile_db_features(P: np.ndarray, z: np.ndarray, window_m: tuple[float, float] = (-20, 90)) -> np.ndarray:
    """Whole profile in dB: (n, nz, npol) linear power -> (n, n_heights_in_window * npol).

    Columns are height-major with polarization fastest. With the Lope cube (2 m step, -20..90 m, 3 pols): 168 features.
    """
    win = window_index(z, window_m)
    return (10 * np.log10(P[:, win, :])).reshape(len(P), -1)


def _at(P, z, z_ground, pols, pol, h):
    """dB of polarization `pol` at height h above the ground proxy."""
    return db(P[:, int(np.argmin(np.abs(z - z_ground - h))), pols.index(pol)])


def ground_relative_features(P: np.ndarray, z: np.ndarray, z_ground: float, pols: list[str],
                             heights: np.ndarray | None = None) -> np.ndarray:
    """Coarser alternative (notebook section 12d): dB at heights -10..80 m (step 4) above the ground proxy, per pol."""
    heights = np.arange(-10, 81, 4) if heights is None else heights
    return np.column_stack([_at(P, z, z_ground, pols, p, h) for p in POLS3 for h in heights])


def tom_summary_features(P: np.ndarray, z: np.ndarray, z_ground: float, pols: list[str]) -> pd.DataFrame:
    """Three simple, broad descriptors of the HV/HH profile (vertical resolution is coarse)."""
    zr = z - z_ground
    sel = (zr >= -10) & (zr <= 80)
    hv = P[:, :, pols.index("HV")]
    return pd.DataFrame({
        "HV centroid (m)": (hv[:, sel] * zr[sel]).sum(1) / hv[:, sel].sum(1),
        "HV(40 m)−HV(10 m) (dB)": _at(P, z, z_ground, pols, "HV", 40) - _at(P, z, z_ground, pols, "HV", 10),
        "HV/HH @ 20 m (dB)": _at(P, z, z_ground, pols, "HV", 20) - _at(P, z, z_ground, pols, "HH", 20),
    }, index=range(len(P)))


def radar_cell_features(Pc: np.ndarray, z: np.ndarray, z_ground: float, pols: list[str]) -> pd.DataFrame:
    """Per-cell radar descriptors used for the correlation tables (notebook section 12d/correlations cell)."""
    zr = z - z_ground
    selw = (zr >= -10) & (zr <= 80)
    X = {}
    for pol in POLS3:
        P = Pc[:, :, pols.index(pol)]
        for h in (0, 10, 20, 30, 40, 50, 60):
            X[f"{pol} @{h} m (dB)"] = _at(Pc, z, z_ground, pols, pol, h)
        w = np.where(selw[None, :], P, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            tot = np.nansum(w, axis=1)
            X[f"{pol} centroid (m)"] = np.nansum(w * zr[None, :], axis=1) / tot
            X[f"{pol} share >30 m"] = np.nansum(np.where(zr[None, :] > 30, w, np.nan), axis=1) / tot
            X[f"{pol} total (dB)"] = 10 * np.log10(tot)
    X["HV(40)-HV(10) (dB)"] = _at(Pc, z, z_ground, pols, "HV", 40) - _at(Pc, z, z_ground, pols, "HV", 10)
    X["HV/HH @20 m (dB)"] = _at(Pc, z, z_ground, pols, "HV", 20) - _at(Pc, z, z_ground, pols, "HH", 20)
    return pd.DataFrame(X)


def rh_columns(g: pd.DataFrame) -> list[str]:
    return sorted([c for c in g.columns if re.fullmatch(r"rh_\d+", c)], key=lambda s: int(s.split("_")[1]))
