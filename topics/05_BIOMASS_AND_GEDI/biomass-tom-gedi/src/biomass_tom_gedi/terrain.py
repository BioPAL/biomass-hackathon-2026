"""Slope from the BIOMASS DTM_L2A height raster.

The TOM cube has no slope correction or shadow/layover flag, so steep terrain is masked from the DTM. The GeoTIFF
geotransform is malformed (row step in the wrong slot), so the axes are rebuilt by hand and the N-S orientation is
checked against GEDI's digital_elevation_model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

from .tom import find_one

log = logging.getLogger(__name__)
M_PER_DEG = 111_320.0


class DTMSlope:
    def __init__(self, dtm_product: str | Path, smooth_px: float = 3.0):
        dtm_product = Path(dtm_product)
        meas = dtm_product / "measurement" if (dtm_product / "measurement").is_dir() else dtm_product
        tif = find_one("*_height.tiff", meas)
        with rasterio.open(tif) as src:
            self.dtm = src.read(1, masked=True).astype("float32").filled(np.nan)
            tr = src.transform
        # malformed geotransform -> build the axes by hand
        self.x0, self.y0 = tr.c, tr.f
        self.dlon, self.dlat = tr.a, (tr.e if tr.e != 0 else tr.d)
        self.south_first = True
        self.smooth_px = smooth_px
        self._smoothed = None
        self._slope = None

    # ---- sampling -----------------------------------------------------------
    def sample(self, arr: np.ndarray, lat, lon, south_first: bool | None = None) -> np.ndarray:
        sf = self.south_first if south_first is None else south_first
        lat, lon = np.asarray(lat), np.asarray(lon)
        r = np.round((lat - self.y0) / self.dlat).astype(int)
        if not sf:
            r = arr.shape[0] - 1 - r
        c = np.round((lon - self.x0) / self.dlon).astype(int)
        inside = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
        out = np.full(len(lat), np.nan, dtype="float32")
        out[inside] = arr[r[inside], c[inside]]
        return out

    def check_orientation(self, lat, lon, gedi_dem) -> dict[bool, float]:
        """Pick the N-S orientation that best correlates with GEDI's DEM."""
        rr = {}
        for flag in (True, False):
            v = self.sample(self.dtm, lat, lon, flag)
            m = np.isfinite(v) & np.isfinite(gedi_dem)
            rr[flag] = float(np.corrcoef(v[m], gedi_dem[m])[0, 1])
        self.south_first = rr[True] >= rr[False]
        log.info("DTM vs GEDI DEM: row0=south r=%.3f | row0=north r=%.3f -> south_first=%s",
                 rr[True], rr[False], self.south_first)
        return rr

    # ---- slope ----------------------------------------------------------------
    @property
    def smoothed(self) -> np.ndarray:
        """DTM Gaussian-smoothed (NaN-aware) to roughly the 150 m radar window."""
        if self._smoothed is None:
            valid = np.isfinite(self.dtm)
            w = gaussian_filter(valid.astype("float32"), self.smooth_px)
            sm = gaussian_filter(np.where(valid, self.dtm, 0).astype("float32"), self.smooth_px) / np.maximum(w, 1e-3)
            sm[~valid] = np.nan
            self._smoothed = sm
        return self._smoothed

    @property
    def slope(self) -> np.ndarray:
        """Slope in degrees on the DTM grid."""
        if self._slope is None:
            gy, gx = np.gradient(self.smoothed, abs(self.dlat) * M_PER_DEG, abs(self.dlon) * M_PER_DEG)
            self._slope = np.degrees(np.arctan(np.hypot(gy, gx))).astype("float32")
        return self._slope

    def slope_at(self, lat, lon) -> np.ndarray:
        return self.sample(self.slope, lat, lon)
