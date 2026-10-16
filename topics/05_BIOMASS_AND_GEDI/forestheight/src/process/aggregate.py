"""Spatial aggregation: moving-window means, non-overlapping blocks, GEDI shots per block."""

import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from scipy import ndimage


def smooth(a, k):
    """Mean over a k x k window, ignoring gaps (needs at least half of the window valid)."""
    if a is None or k <= 1:
        return a
    m = np.isfinite(a)
    s = ndimage.uniform_filter(np.where(m, a, 0.0).astype("float64"), k, mode="constant")
    w = ndimage.uniform_filter(m.astype("float64"), k, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        o = s / w
    o[w < 0.5] = np.nan
    return o.astype("float32")


def at_scale(S, k):
    """Copy of a site with every pass AND the LiDAR reference averaged over k x k pixels (same scale for both)."""
    return dict(S, lvis=smooth(S["lvis"], k), gedi=None, scenes=[dict(p, h=smooth(p["h"], k)) for p in S["scenes"]])


def block_mean(a, k, min_frac=0.5):
    """Mean over non-overlapping k x k blocks (NaN where less than 'min_frac' of the block is valid)."""
    R, C = a.shape
    nr, nc = R // k, C // k
    b = np.asarray(a, float)[: nr * k, : nc * k].reshape(nr, k, nc, k)
    v = np.isfinite(b)
    n = v.sum((1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        o = np.where(v, b, 0).sum((1, 3)) / n
    o[n < min_frac * k * k] = np.nan
    return o


def gedi_blocks(S, k, shots=None, min_shots=None):
    """Median GEDI RH98 of the shots falling in each k x k block, and the number of shots."""
    shots = S["gedi_shots"] if shots is None else shots
    min_shots = (1 if k == 1 else 3) if min_shots is None else min_shots
    g = S["grid"]
    nr, nc = g.shape[0] // k, g.shape[1] // k
    r, c = rowcol(g.transform, shots.longitude.to_numpy(), shots.latitude.to_numpy())
    r, c = np.asarray(r) // k, np.asarray(c) // k
    ok = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
    df = (
        pd.DataFrame(dict(r=r[ok], c=c[ok], v=shots.rh_98.to_numpy()[ok]))
        .groupby(["r", "c"])
        .v.agg(["median", "size"])
        .reset_index()
    )
    out = np.full((nr, nc), np.nan)
    cnt = np.zeros((nr, nc), int)
    out[df.r, df.c] = df["median"]
    cnt[df.r, df.c] = df["size"]
    out[cnt < min_shots] = np.nan
    return out, cnt


def best_gedi(shots):
    """Power beams at night: the GEDI shots with the best canopy penetration."""
    return shots[(shots.beam_type == "full") & (shots.solar_elevation < 0)]
