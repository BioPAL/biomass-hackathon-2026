"""Workshop forest-height products (BIOMASS-Coding2026 'fh_gabon_geocoded' folder on MAAP).

One geocoded FH GeoTIFF per acquisition ('tif/<start>_<stop>/FH_inv.tif') and a 100 m mosaic
('tif-mosaic/mosaic_FH_100m.tif'). The files carry no orbit metadata: the orbit direction is derived from
the acquisition time (05 UTC ascending, 17 UTC descending, checked against the official L2A products) and the
footprint from the valid-data mask. The folder path comes from config/sites.json ('workshop_fh_dir').
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

WGS84 = CRS.from_epsg(4326)


def _read(path, out_shape=None):
    with rasterio.open(path) as s:
        a = s.read(1, out_shape=out_shape).astype("float32")
        tr = (
            s.transform
            if out_shape is None
            else s.transform * s.transform.scale(s.width / a.shape[1], s.height / a.shape[0])
        )
        nod, crs = s.nodata, (s.crs or WGS84)
    if nod is not None:
        a[a == nod] = np.nan
    a[a < 0] = np.nan
    return a, tr, crs


def _footprint(a, tr):
    """Four extreme corners (south, east, north, west) of the valid data: a closed lon/lat polygon."""
    rr, cc = np.where(np.isfinite(a))
    lon = tr.c + (cc + 0.5) * tr.a
    lat = tr.f + (rr + 0.5) * tr.e
    idx = [np.argmin(lat), np.argmax(lon), np.argmax(lat), np.argmin(lon)]
    pts = [[float(lon[i]), float(lat[i])] for i in idx]
    return pts + [pts[0]]


def inventory(ws_dir, decim=8) -> pd.DataFrame:
    """One row per acquisition: time, orbit direction, footprint, path."""
    rows = []
    for d in sorted((Path(ws_dir) / "tif").iterdir()):
        f = d / "FH_inv.tif"
        if not f.exists():
            continue
        t0 = pd.Timestamp(d.name[:15])
        with rasterio.open(f) as s:
            shp = (max(s.height // decim, 1), max(s.width // decim, 1))
        a, tr, _ = _read(f, shp)
        rows.append(
            dict(
                scene=d.name,
                time=t0,
                date=t0.normalize(),
                orbit="ascending" if t0.hour < 12 else "descending",
                footprint=_footprint(a, tr),
                path=str(f),
                valid_fraction=float(np.isfinite(a).mean()),
            )
        )
    return pd.DataFrame(rows)


def to_grid(path, grid, resampling=Resampling.nearest):
    a, tr, crs = _read(path)
    out = np.full(grid.shape, np.nan, dtype="float32")
    reproject(
        a,
        out,
        src_transform=tr,
        src_crs=crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return out


def mosaic_on_grid(S, ws_dir):
    return to_grid(Path(ws_dir) / "tif-mosaic" / "mosaic_FH_100m.tif", S["grid"])
