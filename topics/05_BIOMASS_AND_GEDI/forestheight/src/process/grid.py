"""Common north-up lon/lat grid for each site, and aggregation of 25 m LVIS onto it."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin, rowcol
from rasterio.warp import Resampling, reproject


class Grid:
    """A north-up EPSG:4326 grid covering ``bbox`` with square pixels of ``res`` degrees."""

    def __init__(self, bbox, res):
        w, s, e, n = bbox
        self.bbox, self.res = list(bbox), float(res)
        self.shape = (int(math.ceil((n - s) / res)), int(math.ceil((e - w) / res)))
        self.transform = from_origin(w, n, res, res)
        self.crs = rasterio.crs.CRS.from_epsg(4326)
        cols, rows = np.arange(self.shape[1]) + 0.5, np.arange(self.shape[0]) + 0.5
        self.lon, self.lat = np.meshgrid(w + cols * res, n - rows * res)
        self.extent = [w, w + self.shape[1] * res, n - self.shape[0] * res, n]

    def resample(self, path, resampling=Resampling.nearest):
        out = np.full(self.shape, np.nan, dtype="float32")
        with rasterio.open(path) as src:
            a = src.read(1).astype("float32")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            reproject(
                a,
                out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.transform,
                dst_crs=self.crs,
                resampling=resampling,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
        return out

    def points_to_grid(self, lon, lat, values, how="median"):
        """Median of point values falling in each pixel, and the count of points."""
        r, c = rowcol(self.transform, np.asarray(lon), np.asarray(lat))
        r, c = np.asarray(r), np.asarray(c)
        ok = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1]) & np.isfinite(values)
        df = pd.DataFrame({"r": r[ok], "c": c[ok], "v": np.asarray(values)[ok]})
        g = df.groupby(["r", "c"])["v"].agg([how, "size"]).reset_index()
        val = np.full(self.shape, np.nan, dtype="float32")
        cnt = np.zeros(self.shape, dtype="int32")
        val[g.r, g.c] = g[how]
        cnt[g.r, g.c] = g["size"]
        return val, cnt


def native_resolution(site_dir: Path, default=0.000905):
    tifs = sorted((Path(site_dir) / "scenes").glob("*_height.tif"))
    if not tifs:
        return default
    with rasterio.open(tifs[0]) as src:
        return abs(src.transform.a)


def aggregate_lvis_to_biomass(
    lvis: np.ndarray,
    lvis_transform,
    lvis_crs,
    biomass_shape: tuple[int, int],
    biomass_transform,
    biomass_crs,
    min_coverage: float = 0.5,
    lvis_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average 25-m LVIS RH98 to the coarser BIOMASS pixel grid.

    Parameters
    ----------
    lvis : numpy.ndarray
        LVIS gridded mean RH98 [m] at nominal 25 m resolution.
    lvis_transform, biomass_transform : affine.Affine
        Actual georeferencing of source and destination.
    lvis_crs, biomass_crs : rasterio.crs.CRS
        Coordinate systems (typically EPSG:32732 and EPSG:4326 respectively).
    biomass_shape : tuple[int, int]
        Destination (rows, columns).
    min_coverage : float
        Minimum estimated valid LVIS source-pixel coverage fraction [0,1].
    lvis_valid_mask : numpy.ndarray or None
        Optional NASA LVIS valid_data_mask, same grid as LVIS RH98.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Aggregated LVIS RH98 [m] and approximate valid-source coverage [0,1]
        on the BIOMASS grid.

    Notes
    -----
    Reprojection aggregates mapped source pixels; it does not recreate LVIS
    waveform footprint support. Check dates, voids, and valid masks.
    """
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")
    valid = np.isfinite(lvis) & (lvis >= 0)
    if lvis_valid_mask is not None:
        if lvis_valid_mask.shape != lvis.shape:
            raise ValueError("LVIS valid mask and RH98 array must share the same grid")
        valid &= np.isfinite(lvis_valid_mask) & (lvis_valid_mask == 1)
    src = np.where(valid, lvis, np.nan).astype("float32")
    dest = np.full(biomass_shape, np.nan, dtype="float32")
    reproject(
        source=src,
        destination=dest,
        src_transform=lvis_transform,
        src_crs=lvis_crs,
        dst_transform=biomass_transform,
        dst_crs=biomass_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    # Coverage is a spatially averaged 0/1 validity field, not a shot count.
    coverage = np.zeros(biomass_shape, dtype="float32")
    reproject(
        source=valid.astype("float32"),
        destination=coverage,
        src_transform=lvis_transform,
        src_crs=lvis_crs,
        dst_transform=biomass_transform,
        dst_crs=biomass_crs,
        src_nodata=None,
        dst_nodata=0.0,
        resampling=Resampling.average,
    )
    dest[coverage < min_coverage] = np.nan
    return dest, coverage
