"""Filters for TomoSAR profiles that are bright at all heights (the artefact seen in the 3-D curtains).

Per polarization: floor_db = 10th percentile of the profile in dB, top_db = mean dB above 60 m, mean_db = mean dB.
A shot is flagged if the feature is in the upper q tail of the population, for HV only ("hv") or any pol ("any").
"""
from __future__ import annotations

import numpy as np

FEATURES = ("floor_db", "top_db", "mean_db")


def bright_features(P: np.ndarray, z: np.ndarray, pols: list[str], top_from_m: float = 60.0) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for k, pol in enumerate(pols):
        dB = 10 * np.log10(P[:, :, k])
        out[pol] = {
            "floor_db": np.percentile(dB, 10, axis=1),
            "top_db": dB[:, z >= top_from_m].mean(1),
            "mean_db": dB.mean(1),
        }
    return out


def flag_bright(feats: dict, pols: list[str], feature: str, q: float, mode: str, population: np.ndarray) -> np.ndarray:
    """Boolean flag per shot. Thresholds are quantiles of `population` (a boolean mask, e.g. the slope-kept shots)."""
    def one(pol: str) -> np.ndarray:
        v = feats[pol][feature]
        return (v > np.quantile(v[population], 1 - q)) & population

    if mode == "hv":
        return one("HV")
    if mode == "any":
        return np.any([one(p) for p in pols], axis=0)
    raise ValueError("mode must be 'hv' or 'any'")
