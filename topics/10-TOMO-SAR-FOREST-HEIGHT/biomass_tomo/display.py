"""
Map display in radar geometry at true ground scale.

Images are (azimuth, range[, 3]). x is ground range [km] and y is azimuth [km] with the first line at
the top, so that for a descending, right-looking pass the map is roughly north-up and east-right.
Colour stretches use percentiles so that low-contrast scenes (e.g. closed forest) stay readable.
"""
import numpy as np


def ground_range_km(slant_range_m, incidence_deg):
    """Ground-range coordinate [km] of each range bin. incidence_deg: (range,) or (range, azimuth)."""
    inc = np.asarray(incidence_deg, float)
    if inc.ndim > 1:
        inc = np.nanmean(inc, axis=tuple(range(1, inc.ndim)))
    mid = np.deg2rad(0.5 * (inc[1:] + inc[:-1]))
    return np.r_[0.0, np.cumsum(np.diff(slant_range_m) / np.sin(mid))] / 1e3


def show(ax, img, az_km, gr_km, **kw):
    """imshow at equal ground scale; img is (azimuth, range[, 3])."""
    daz = (az_km[-1] - az_km[0]) / max(len(az_km) - 1, 1)
    dgr = (gr_km[-1] - gr_km[0]) / max(len(gr_km) - 1, 1)
    kw = {"interpolation": "none", **kw}
    ax.set_facecolor("0.15")                      # no-data / NaN (water, masked cells) shown dark, not white
    im = ax.imshow(img, extent=[gr_km[0] - dgr / 2, gr_km[-1] + dgr / 2, az_km[-1] + daz / 2, az_km[0] - daz / 2],
                   origin="upper", aspect="equal", **kw)
    ax.set_xlabel("ground range [km]")
    return im


def limits(a, pct=(2, 98)):
    """Percentile colour limits over the finite values of one or several arrays."""
    arrs = a if isinstance(a, (list, tuple)) else [a]
    v = np.concatenate([np.asarray(x, float)[np.isfinite(x)].ravel() for x in arrs])
    return tuple(np.percentile(v, pct)) if v.size else (0.0, 1.0)


FLOOR = 1e-10      # powers at or below this (-100 dB) are no-data, e.g. pixels clipped by noise subtraction


def to_db(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.where(p > FLOOR, p, np.nan))


def rgb_db(r, g, b, pct=(2, 98), limits_from=None):
    """RGB (…, 3) in [0, 1] from three power images, each stretched in dB between its own percentiles.
    limits_from: optional (r, g, b) arrays to compute the percentiles on (e.g. all heights of a cube),
    so that several images share one colour scale. No-data -> black."""
    src = limits_from or (r, g, b)
    out = []
    for c, s in zip((r, g, b), src):
        lo, hi = limits(to_db(s), pct)
        out.append(np.clip((to_db(c) - lo) / (hi - lo), 0, 1))
    return np.nan_to_num(np.stack(out, axis=-1))


def rgb_uint8(rgb, valid=None):
    """(…, 3) float RGB in [0, 1] -> uint8 with 0 reserved for no-data (valid pixels are 1..255)."""
    out = (1 + np.round(np.clip(rgb, 0, 1) * 254)).astype(np.uint8)
    if valid is not None:
        out[~valid] = 0
    return out
