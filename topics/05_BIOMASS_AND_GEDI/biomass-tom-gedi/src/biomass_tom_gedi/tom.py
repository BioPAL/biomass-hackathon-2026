"""BIOMASS TOM L2A product: geographic axes, lazy cube access, profile extraction.

The coordinate variables inside the .nc file are empty, so the axes are rebuilt from the annotation XML.
Row 0 is the southernmost line (latitude increases with the row index).
"""
from __future__ import annotations

import logging
import warnings
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)
FILL_VALUE = -9999


def find_one(pattern: str, folder: Path) -> Path:
    hits = sorted(Path(folder).glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern!r} in {folder}: found {len(hits)} files")
    return hits[0]


def db(x):
    """Linear power -> dB (non-positive values become NaN)."""
    x = np.asarray(x)
    return 10 * np.log10(np.where(x > 0, x, np.nan))


@dataclass
class TOMGrid:
    """Axes of the tomographic cube, from the annotation XML."""

    lat0: float
    lon0: float
    dlat: float
    dlon: float
    nrows: int
    ncols: int
    z0: float
    dz: float
    nz: int
    pols: list[str]
    start_time: str

    @classmethod
    def from_annotation(cls, xml_path: Path) -> "TOMGrid":
        root = ET.parse(xml_path).getroot()

        def txt(tag: str) -> str:
            return root.find(f".//{tag}").text.strip()

        return cls(
            lat0=float(txt("firstLatitude")), lon0=float(txt("firstLongitude")),
            dlat=float(txt("rowStep")), dlon=float(txt("columnStep")),
            nrows=int(txt("numberOfRows")), ncols=int(txt("numberOfColumns")),
            z0=float(txt("minVerticalAxis")), dz=float(txt("verticalAxisStep")),
            nz=int(txt("numberOfVerticalSamples")),
            pols=[e.text.strip() for e in root.findall(".//tomographicCubesPolarizations/polarization")],
            start_time=txt("startTime"),
        )

    @property
    def lat(self) -> np.ndarray:
        return self.lat0 + self.dlat * np.arange(self.nrows)

    @property
    def lon(self) -> np.ndarray:
        return self.lon0 + self.dlon * np.arange(self.ncols)

    @property
    def z(self) -> np.ndarray:
        return self.z0 + self.dz * np.arange(self.nz)

    @property
    def extent(self) -> list[float]:
        """left, right, bottom, top for imshow(origin='lower')."""
        return [self.lon0 - self.dlon / 2, self.lon[-1] + self.dlon / 2,
                self.lat0 - self.dlat / 2, self.lat[-1] + self.dlat / 2]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """min_lon, min_lat, max_lon, max_lat of the pixel centres."""
        return float(self.lon[0]), float(self.lat[0]), float(self.lon[-1]), float(self.lat[-1])

    def row_col(self, lat, lon):
        """Nearest pixel centre, clipped to the scene."""
        r = np.clip(np.round((np.asarray(lat) - self.lat0) / self.dlat).astype(int), 0, self.nrows - 1)
        c = np.clip(np.round((np.asarray(lon) - self.lon0) / self.dlon).astype(int), 0, self.ncols - 1)
        return r, c

    def height_index(self, h: float) -> int:
        return int(round((h - self.z0) / self.dz))

    @property
    def pixel_size_m(self) -> float:
        return abs(self.dlat) * 111_320.0


class TOMProduct:
    """Locates the files of a TOM L2A product directory."""

    def __init__(self, product_dir: str | Path):
        self.dir = Path(product_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"TOM product directory not found: {self.dir}")
        self.cube_nc = find_one("*_geo_tomcubes.nc", self.dir / "measurement")
        self.annot_xml = find_one("*_annot.xml", self.dir / "annotation")
        self.grid = TOMGrid.from_annotation(self.annot_xml)

    @contextmanager
    def open_cube(self):
        """Yield the lazy HDF5 dataset (rows, cols, heights, pols). Never load it whole (~6 GB)."""
        with h5py.File(self.cube_nc, "r") as f:
            cube = f["tomographicCubes"]
            g = self.grid
            if cube.shape != (g.nrows, g.ncols, g.nz, len(g.pols)):
                raise ValueError(f"cube shape {cube.shape} does not match the annotation")
            yield cube


def read_slice(cube, *idx) -> np.ndarray:
    """Read a slice and turn the -9999 fill value into NaN."""
    a = cube[idx].astype("float32")
    a[a == FILL_VALUE] = np.nan
    return a


def extract_profiles(cube, rows: np.ndarray, cols: np.ndarray, half: int = 1) -> np.ndarray:
    """Mean linear-power profile in a (2*half+1)^2 pixel window around each (row, col).

    The cube is read one HDF5 chunk-block at a time (much faster than one read per shot);
    windows are clipped at block edges. Returns (n, nz, npol) float32, NaN where the window is empty.
    """
    n_rows, n_cols, nz, npol = cube.shape
    cr, cc = cube.chunks[0], cube.chunks[1]
    br, bc = rows // cr, cols // cc
    prof = np.full((len(rows), nz, npol), np.nan, dtype="float32")
    order = np.lexsort((bc, br))
    keys = np.stack([br, bc], 1)[order]
    # boundaries between consecutive (br, bc) groups
    cut = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    groups = np.split(order, cut)
    log.info("reading %d chunk-blocks of the cube", len(groups))
    for k, idx in enumerate(groups, 1):
        r0, c0 = int(br[idx[0]]) * cr, int(bc[idx[0]]) * cc
        r1, c1 = min(r0 + cr, n_rows), min(c0 + cc, n_cols)
        blk = read_slice(cube, slice(r0, r1), slice(c0, c1), slice(None), slice(None))
        for i in idx:
            a, b = max(rows[i] - half, r0) - r0, min(rows[i] + half + 1, r1) - r0
            a2, b2 = max(cols[i] - half, c0) - c0, min(cols[i] + half + 1, c1) - c0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prof[i] = np.nanmean(blk[a:b, a2:b2], axis=(0, 1))
        if k % 50 == 0:
            log.info("  %d / %d blocks", k, len(groups))
    return prof


def ground_proxy(profiles: np.ndarray, z: np.ndarray, pols: list[str]) -> float:
    """Scene-wide ground level on the cube's height axis: HH peak of the mean profile (double bounce).

    The vertical reference of the TOM axis is not confirmed to be height above ground; this is a proxy.
    """
    hh = profiles[:, :, pols.index("HH")]
    return float(z[np.argmax(np.nanmean(hh, axis=0))])
