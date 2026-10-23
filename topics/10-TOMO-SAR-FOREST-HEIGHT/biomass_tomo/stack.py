"""
Working with the analysis-ready stack netCDF written by biomass_tomo.l1c.build_stack: crop from an AOI,
open / crop into the course layout, check that the baselines support tomography, and the run manifest
that connects the notebooks.

Layout (shared with the course stack, so both are interchangeable):
    slc_real, slc_imag (polarisation, acquisition, azimuth, range)
    dem_phases, calibration_phases, insar_wavenumbers, ... (acquisition, azimuth, range)
    incidence_angle, dem, latitude, longitude, ... (azimuth, range)
"""
import json
import os
import pathlib
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from . import l1c


# crop from an AOI
def crop_from_aoi(product_dir, aoi_bbox, margin_m=1000.0, step=20):
    """Azimuth/range pixel bounds (az0, az1, rg0, rg1) of the frame that cover a lon/lat
    bounding box plus a margin, found from the /geometry/latitude|longitude LUT."""
    n_az, n_rg = l1c.frame_shape(product_dir)
    grid = l1c.build_grid(product_dir, (0, n_az, 0, n_rg))
    sub = {**grid, "x_ax": grid["x_ax"][::step], "az_time_rel": grid["az_time_rel"][::step]}
    lut = {name: l1c.read_lut_on_grid(product_dir, l1c.LUT_VARIABLES[name], sub)[0]
           for name in ("latitude", "longitude", "incidence_angle")}
    lat, lon = lut["latitude"], lut["longitude"]
    w, s, e, n = aoi_bbox
    inside = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    if not inside.any():
        raise ValueError(f"AOI {aoi_bbox} is not inside the frame of {pathlib.Path(product_dir).name}")
    ia, ir = np.nonzero(inside)
    a = grid["annotation"]
    inc = np.deg2rad(np.nanmean(lut["incidence_angle"][inside]))
    m_az = int(margin_m / a["azimuth_spacing"])
    m_rg = int(margin_m * np.sin(inc) / a["range_spacing"])
    return l1c.crop_bounds(n_az, n_rg, (ia.min() * step - m_az, (ia.max() + 1) * step + m_az,
                                        ir.min() - m_rg, ir.max() + 1 + m_rg))


# open / crop
def open_stack(path):
    return xr.open_dataset(path, decode_timedelta=False)


def reference_index(ds):
    """Coregistration primary: the acquisition whose flattening phase is (near) zero."""
    level = np.abs(ds.dem_phases.isel(azimuth=slice(None, None, 50), range=slice(None, None, 10))).mean(("azimuth", "range"))
    return int(np.argmin(level.values))


def indices_for_aoi(ds, bbox, margin_m=0.0, step=10):
    """(az0, az1, rg0, rg1) indices of the stack (not of the frame) covering a lon/lat bbox."""
    lat = ds.latitude.isel(azimuth=slice(None, None, step)).values
    lon = ds.longitude.isel(azimuth=slice(None, None, step)).values
    w, s, e, n = bbox
    inside = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    if not inside.any():
        return None
    ia, ir = np.nonzero(inside)
    dx = float(ds.azimuth[1] - ds.azimuth[0])
    dr = float(ds.range[1] - ds.range[0])
    inc = np.deg2rad(float(ds.incidence_angle.isel(azimuth=ds.sizes["azimuth"] // 2).mean()))
    m_az, m_rg = int(margin_m / dx), int(margin_m * np.sin(inc) / dr)
    return (max(ia.min() * step - m_az, 0), min((ia.max() + 1) * step + m_az, ds.sizes["azimuth"]),
            max(ir.min() - m_rg, 0), min(ir.max() + 1 + m_rg, ds.sizes["range"]))


def course_layout(da):
    """named dimensions -> (range, azimuth[, acquisition]) as in the ESA course notebooks"""
    return da.transpose("range", "azimuth", ...).values


def load_crop(ds, crop=None, pols=("HH", "HV", "VH", "VV"),
              aux_names=("dem_phases", "calibration_phases", "insar_wavenumbers", "incidence_angle",
                         "range_slope", "dem", "latitude", "longitude")):
    """Load a crop (az0, az1, rg0, rg1 in stack indices; None = all) into course-layout arrays."""
    if crop is not None:
        az0, az1, rg0, rg1 = crop
        sub = ds.isel(azimuth=slice(az0, az1), range=slice(rg0, rg1))
    else:
        sub = ds
    slc = {}
    for pol in pols:
        re_ = course_layout(sub.slc_real.sel(polarisation=pol))
        im_ = course_layout(sub.slc_imag.sel(polarisation=pol))
        s = np.empty(re_.shape, dtype=np.complex64)
        s.real, s.imag = re_, im_
        slc[pol] = s
    aux = {n: course_layout(sub[n]).astype(np.float32) for n in aux_names if n in sub}
    return {"slc": slc, "aux": aux, "r_ax": sub.range.values, "x_ax": sub.azimuth.values,
            "t_ax": sub.temporal_baseline.values, "acquisition": sub.acquisition.values, "sub": sub}


# baseline requirement
def baseline_report(ds, min_acquisitions=3, z_extent=None, step=50):
    """Tomographic capability of the stack from its vertical wavenumbers.
    z_extent: (zmin, zmax) height interval the user wants to image; it must fit inside the
    height of ambiguity for the stack to pass."""
    kz = ds.insar_wavenumbers.isel(azimuth=slice(None, None, step), range=slice(None, None, step)).values
    kz_mean = np.nanmean(kz, axis=(1, 2))
    order = np.sort(kz_mean)
    span = order[-1] - order[0]
    min_step = np.min(np.diff(order)) if order.size > 1 else np.nan
    per_px_span = np.nanmax(kz, axis=0) - np.nanmin(kz, axis=0)
    rep = {
        "n_acquisitions": int(ds.sizes["acquisition"]),
        "cycles": [int(c) for c in ds.acquisition.values],
        "temporal_baseline_days": [float(t) for t in ds.temporal_baseline.values],
        "kz_mean_rad_per_m": [float(k) for k in kz_mean],
        "kz_span": float(span),
        "kz_min_spacing": float(min_step),
        "vertical_resolution_m": float(2 * np.pi / span) if span > 0 else np.inf,
        "vertical_resolution_m_p5_p95": [float(v) for v in np.nanpercentile(2 * np.pi / per_px_span, [5, 95])],
        "height_of_ambiguity_m": float(2 * np.pi / min_step) if min_step > 0 else np.inf,
        "reference_acquisition_index": reference_index(ds),
    }
    problems = []
    if rep["n_acquisitions"] < min_acquisitions:
        problems.append(f"only {rep['n_acquisitions']} acquisitions (< {min_acquisitions})")
    if not np.isfinite(rep["vertical_resolution_m"]) or span <= 0:
        problems.append("no vertical wavenumber spread (zero baselines)")
    if z_extent is not None and np.isfinite(rep["height_of_ambiguity_m"]):
        if (z_extent[1] - z_extent[0]) > rep["height_of_ambiguity_m"]:
            problems.append(f"height range {z_extent} exceeds the height of ambiguity {rep['height_of_ambiguity_m']:.0f} m")
    rep["tomography_ok"] = not problems
    rep["problems"] = problems
    return rep


# run manifest
def write_manifest(path, **content):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content.setdefault("written", datetime.now(timezone.utc).isoformat())

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, pathlib.Path):
            return str(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(content, indent=2, default=default))
    return path


def read_manifest(path):
    return json.loads(pathlib.Path(path).read_text())


def save_netcdf(ds, path, complevel=3):
    """Write a Dataset compressed, via a temporary file renamed into place, so that a reader holding
    the previous version open (e.g. another notebook kernel) does not block or corrupt the write."""
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    ds.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": complevel} for v in ds.data_vars})
    os.replace(tmp, path)
    return path
