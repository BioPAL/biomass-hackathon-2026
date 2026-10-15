"""LUT annotation of a STA product (``<product>_lut.nc``, STAC asset ``enclosure_nc``).

Reading
    netCDF4 directly on the downloaded LUT file. BPS is not needed: its
    BIOMASSStackProductReader requires the whole product folder, while the
    pipeline only downloads the LUT and range-reads the image COGs.

LUT grid
    The LUT is the L1c image grid decimated by an integer step, LUT node (a, r)
    on image pixel (a * step_az, r * step_rg) (checked against the annotation
    time axes: azimuth nodes every 12 lines, range nodes every 2 samples).

Interpolation onto the pixels
    Port of BPS read_lut_annotation(fit_to_l1c_raster=True): bilinear
    (RectBivariateSpline kx=ky=1) for every layer, skpCalibrationPhaseScreen
    through its sine and cosine (wrapped phase). No-data nodes are set to 0
    before interpolating the phase screens, as BPS does with
    nodata_fill_value=0 (Thesis_v2 data_utils.load_or_extract_data_lat_lon).

Phase correction applied to every product before saving (Thesis_v2)
    s_cal = s * exp(1j * skpCalibrationPhaseScreen) * exp(1j * flatteningPhaseScreen)
"""

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

import netCDF4
import numpy as np
from scipy.ndimage import map_coordinates

from ..core.io import NC_LOCK, download

log = logging.getLogger("flooded_forest.extraction")

LUT_ASSET = "enclosure_nc"

# {layer: netCDF group}
GEOMETRY_LAYERS = {
    "latitude": "geometry",
    "longitude": "geometry",
    "height": "geometry",
    "incidenceAngle": "geometry",
}
PHASE_LAYERS = {
    "flatteningPhaseScreen": "coregistration",
    "waveNumbers": "coregistration",
    "skpCalibrationPhaseScreen": "skpPhaseCalibration",
    "skpCalibrationPhaseScreenQuality": "skpPhaseCalibration",
}
FLAT = "flatteningPhaseScreen"
SKP = "skpCalibrationPhaseScreen"
PHASE_CORRECTION = "s * exp(1j * skpCalibrationPhaseScreen) * exp(1j * flatteningPhaseScreen)"


@contextmanager
def fetched_lut(item, token):
    """Download the LUT of a STAC item into a temporary file and yield its path."""
    with tempfile.TemporaryDirectory(prefix="ff_lut_") as tmp:
        path = Path(tmp) / f"{item.id}_lut.nc"
        download(item.assets[LUT_ASSET].href, token, path)
        yield path


def read_layers(path, layers, lut_slice=None):
    """{name: float64 array} of the {name: group} layers present in the file, NaN = no data.

    lut_slice = (i0, i1, j0, j1) crops the LUT grid.
    """
    out = {}
    with NC_LOCK, netCDF4.Dataset(str(path)) as ds:
        nodata = float(ds.noDataValue)
        for name, group in layers.items():
            if group not in ds.groups or name not in ds[group].variables:
                continue
            var = ds[group][name]
            raw = var[lut_slice[0]:lut_slice[1], lut_slice[2]:lut_slice[3]] if lut_slice else var[:]
            data = np.ma.filled(np.ma.asarray(raw).astype(np.float64), np.nan)
            data[data == nodata] = np.nan
            out[name] = data
    return out


def read_axes(path):
    """LUT time axes: relative azimuth time [s] and slant range time [s]."""
    with NC_LOCK, netCDF4.Dataset(str(path)) as ds:
        return {
            "azimuth_time": np.asarray(ds["relativeAzimuthTime"][:], dtype=np.float64),
            "slant_range_time": np.asarray(ds["slantRangeTime"][:], dtype=np.float64),
        }


def _bilinear(lut, rows, cols, block=1024):
    out = np.empty((rows.size, cols.size), dtype=np.float64)
    for r0 in range(0, rows.size, block):
        rr, cc = np.meshgrid(rows[r0:r0 + block], cols, indexing="ij")
        out[r0:r0 + block] = map_coordinates(lut, [rr, cc], order=1, mode="nearest")
    return out


def to_pixels(lut, shape, step, offset=(0, 0), wrapped=False):
    """Interpolate a LUT crop onto a (rows, cols) pixel window (BPS fit_to_l1c_raster).

    LUT crop node (a, r) sits on window pixel (a * step[0] + offset[0], r * step[1] + offset[1]).
    wrapped=True interpolates a phase through its sine and cosine.
    """
    rows = (np.arange(shape[0]) - offset[0]) / step[0]
    cols = (np.arange(shape[1]) - offset[1]) / step[1]
    if wrapped:
        return np.arctan2(_bilinear(np.sin(lut), rows, cols), _bilinear(np.cos(lut), rows, cols))
    return _bilinear(lut, rows, cols)


def planned_corrections(screens, apply_skp=True, apply_flat=True, product_id=""):
    """Phase corrections that will be applied to a product: subset of ['skp', 'flat']."""
    planned = []
    for key, layer, wanted in (("skp", SKP, apply_skp), ("flat", FLAT, apply_flat)):
        if not wanted:
            continue
        values = screens.get(layer)
        if values is None or not np.isfinite(values).any():
            log.warning("%s: no %s in the LUT, not applied", product_id, layer)
            continue
        planned.append(key)
    return planned


def correction_tag(planned):
    """Tag written into the product tif and stack_info.json, e.g. 'skp+flat' or 'none'."""
    return "+".join(planned) or "none"


def phase_correction(screens, shape, step, planned, offset=(0, 0)):
    """exp(1j * (skp + flat)) on the pixel window, complex64; None when nothing is applied."""
    if not planned:
        return None
    phase = np.zeros(shape, dtype=np.float64)
    if "skp" in planned:
        phase += to_pixels(np.nan_to_num(screens[SKP], nan=0.0), shape, step, offset, wrapped=True)
    if "flat" in planned:
        phase += to_pixels(np.nan_to_num(screens[FLAT], nan=0.0), shape, step, offset)
    return np.exp(1j * phase).astype(np.complex64)


def write_phase_lut(path, stack, screens, tags):
    """phase_lut.nc: flattening, SKP and kz of every product of a stack on the cropped LUT grid.

    screens: {product_id: {layer: (n_az, n_rg)}}, tags: {product_id: correction tag}.
    """
    ids = list(screens)
    i0, i1, j0, j1 = stack["lut_slice"]
    step_az, step_rg = stack["lut_step"]
    w = stack["window"]
    tmp = Path(path).with_suffix(".nc.part")
    with NC_LOCK, netCDF4.Dataset(str(tmp), "w") as ds:
        ds.createDimension("product", len(ids))
        ds.createDimension("azimuth", i1 - i0)
        ds.createDimension("range", j1 - j0)
        ds.description = (
            "Phase screens and wavenumbers of every product, LUT grid cropped to the window. LUT node "
            "(a, r) is window pixel (a*azimuth_step + lut_row_offset, r*range_step + lut_col_offset)."
        )
        ds.phase_correction = PHASE_CORRECTION + " (no-data nodes -> 0 rad)"
        ds.azimuth_step, ds.range_step = step_az, step_rg
        ds.lut_row_offset = i0 * step_az - int(w.row_off)
        ds.lut_col_offset = j0 * step_rg - int(w.col_off)
        ds.lut_slice = [i0, i1, j0, j1]
        pid_var = ds.createVariable("product_id", str, ("product",))
        tag_var = ds.createVariable("phase_calibration", str, ("product",))
        for k, pid in enumerate(ids):
            pid_var[k] = pid
            tag_var[k] = tags[pid]
        for name in PHASE_LAYERS:
            var = ds.createVariable(name, "f8", ("product", "azimuth", "range"), fill_value=np.nan, zlib=True)
            for k, pid in enumerate(ids):
                var[k] = screens[pid].get(name, np.full((i1 - i0, j1 - j0), np.nan))
    tmp.replace(path)
    return path


def load_phase_lut(path, lut_slice):
    """{product_id: {layer: array}} from an existing phase_lut.nc written for the same LUT crop, else {}."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with NC_LOCK, netCDF4.Dataset(str(path)) as ds:
            if [int(v) for v in np.atleast_1d(ds.lut_slice)] != list(lut_slice):
                return {}
            ids = list(ds["product_id"][:])
            layers = {name: np.ma.filled(ds[name][:].astype(np.float64), np.nan)
                      for name in PHASE_LAYERS if name in ds.variables}
    except (OSError, KeyError, AttributeError):
        return {}
    return {pid: {name: data[k] for name, data in layers.items()} for k, pid in enumerate(ids)}


def mean_wavenumbers(path, product_ids):
    """Mean kz [rad/m] of each product (NaN when the product is not in phase_lut.nc)."""
    with NC_LOCK, netCDF4.Dataset(str(path)) as ds:
        ids = list(ds["product_id"][:])
        kz = ds["waveNumbers"]
        return [float(np.nanmean(np.ma.filled(kz[ids.index(pid)].astype(float), np.nan)))
                if pid in ids else float("nan") for pid in product_ids]
