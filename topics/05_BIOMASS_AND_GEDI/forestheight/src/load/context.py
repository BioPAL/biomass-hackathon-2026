"""Independent context layers on the BIOMASS grid: ESA WorldCover land cover and Copernicus DEM.

Both are public Cloud-Optimised GeoTIFFs read directly over HTTPS (no login).
- ESA WorldCover 10 m 2021 v200: https://esa-worldcover.org (s3://esa-worldcover)
- Copernicus DEM GLO-30: https://registry.opendata.aws/copernicus-dem/
"""

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
COPDEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{tile}_00_DEM/"
    "Copernicus_DSM_COG_10_{tile}_00_DEM.tif"
)
WORLDCOVER_CLASSES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare",
    80: "water",
    90: "wetland",
}


def _ns(lat: int) -> str:
    return f"{'S' if lat < 0 else 'N'}{abs(lat):02d}"


def _ew(lon: int) -> str:
    return f"{'W' if lon < 0 else 'E'}{abs(lon):03d}"


def worldcover_tiles(bbox: list[float]) -> list[str]:
    """3x3 degree WorldCover tiles (named by their south-west corner) covering bbox [W,S,E,N]."""
    w, s, e, n = bbox
    lats = range(int(math.floor(s / 3) * 3), int(math.floor(n / 3) * 3) + 1, 3)
    lons = range(int(math.floor(w / 3) * 3), int(math.floor(e / 3) * 3) + 1, 3)
    return [f"{_ns(la)}{_ew(lo)}" for la in lats for lo in lons]


def copdem_tiles(bbox: list[float]) -> list[str]:
    """1x1 degree Copernicus DEM tiles (named by their south-west corner) covering bbox."""
    w, s, e, n = bbox
    lats = range(int(math.floor(s)), int(math.floor(n)) + 1)
    lons = range(int(math.floor(w)), int(math.floor(e)) + 1)
    return [f"{_ns(la)}_00_{_ew(lo)}" for la in lats for lo in lons]


def _to_grid(url, bbox, shape, transform, crs, resampling, fn=None):
    """Read the bbox window of a remote COG, optionally transform it, and resample onto the target grid."""
    out = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(url) as src:
        win = from_bounds(*bbox, src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        arr = src.read(1, window=win).astype("float32")
        if fn is not None:
            arr = fn(arr, src.res)
        reproject(
            arr,
            out,
            src_transform=src.window_transform(win),
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return out


def _safe(fn, *args, **kw):
    """Run one tile read; a tile that does not exist (e.g. open ocean) yields an all-NaN array."""
    try:
        return fn(*args, **kw)
    except Exception:
        return None


def _merge(arrays):
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        raise RuntimeError("No WorldCover/DEM tile could be read for this area.")
    out = arrays[0].copy()
    for a in arrays[1:]:
        out = np.where(np.isfinite(out), out, a)
    return out


def _slope_deg(dem, res, lat0=0.0):
    """Terrain slope [degrees] from a geographic DEM (res in degrees)."""
    dy = abs(res[1]) * 111320.0
    dx = abs(res[0]) * 111320.0 * math.cos(math.radians(lat0))
    gy, gx = np.gradient(dem, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")


def build_context(bbox, shape, transform, crs, cache: Path | None = None) -> dict:
    """Return land-cover fractions, elevation [m] and slope [deg] on the BIOMASS grid.

    Fractions are the share of 10 m WorldCover pixels of a class inside each BIOMASS pixel
    (average resampling). Elevation and slope are 30 m Copernicus DEM averaged to the grid.
    The result is cached as .npz so the download happens only once.
    """
    if cache is not None and Path(cache).is_file():
        with np.load(cache) as z:
            if all(k in z for k in ("tree_frac", "grass_frac", "water_frac", "dem", "slope")) and z[
                "dem"
            ].shape == tuple(shape):
                return {k: z[k] for k in ("tree_frac", "grass_frac", "water_frac", "dem", "slope")}
    lat0 = 0.5 * (bbox[1] + bbox[3])
    wc = [WORLDCOVER_URL.format(tile=t) for t in worldcover_tiles(bbox)]
    dm = [COPDEM_URL.format(tile=t) for t in copdem_tiles(bbox)]

    def frac(code):
        return _merge(
            [
                _safe(
                    _to_grid,
                    u,
                    bbox,
                    shape,
                    transform,
                    crs,
                    Resampling.average,
                    lambda a, r: (a == code).astype("float32"),
                )
                for u in wc
            ]
        )

    ctx = {
        "tree_frac": frac(10),
        "grass_frac": frac(30),
        "water_frac": frac(80),
        "dem": _merge([_safe(_to_grid, u, bbox, shape, transform, crs, Resampling.average) for u in dm]),
        "slope": _merge(
            [
                _safe(_to_grid, u, bbox, shape, transform, crs, Resampling.average, lambda a, r: _slope_deg(a, r, lat0))
                for u in dm
            ]
        ),
    }
    if cache is not None:
        np.savez_compressed(cache, **ctx)
    return ctx


COVER_LABELS = ["forest", "forest-savanna mosaic", "savanna / grassland", "other"]


def cover_class(tree_frac, grass_frac) -> np.ndarray:
    """0 forest (trees >= 90%), 1 mosaic (50-90% trees), 2 savanna (<50% trees, grass >= 50%), 3 other; NaN if unknown."""
    c = np.full(np.shape(tree_frac), np.nan, dtype="float32")
    ok = np.isfinite(tree_frac) & np.isfinite(grass_frac)
    c[ok & (tree_frac >= 0.9)] = 0
    c[ok & (tree_frac >= 0.5) & (tree_frac < 0.9)] = 1
    c[ok & (tree_frac < 0.5) & (grass_frac >= 0.5)] = 2
    c[ok & (tree_frac < 0.5) & (grass_frac < 0.5)] = 3
    return c


def hillshade(dem, azimuth=315.0, altitude=45.0, res_m=100.0):
    """Simple hillshade (0-1) for map backgrounds."""
    gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmean(dem)), res_m, res_m)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)
