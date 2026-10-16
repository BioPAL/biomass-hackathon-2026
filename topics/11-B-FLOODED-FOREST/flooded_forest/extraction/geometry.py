"""Stack geometry, computed ONCE per stack from its reference product's LUT.

All members of a stack are coregistered on the reference radar grid, so the
footprint, the crop window, lat/lon, incidence and the pixel spacings are the
same for every product. Only the phase screens (lut.PHASE_LAYERS) differ.
"""

import netCDF4
import numpy as np
import rasterio
import shapely
from rasterio.control import GroundControlPoint
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

from ..core.io import NC_LOCK

LIGHTSPEED = 299792458.0
EARTH_RADIUS = 6371008.8
GCP_GRID = (12, 6)  # GCPs written into each output GeoTIFF (rows x cols)


def data_footprint(lat, lon, valid):
    """Polygon around the LUT nodes with data: first/last valid range node of each azimuth line."""
    near, far = [], []
    for i in np.nonzero(valid.any(axis=1))[0]:
        js = np.nonzero(valid[i])[0]
        near.append((lon[i, js[0]], lat[i, js[0]]))
        far.append((lon[i, js[-1]], lat[i, js[-1]]))
    polygon = shapely.Polygon(near + far[::-1])
    return polygon if polygon.is_valid else shapely.make_valid(polygon)


def radar_window(stack, area):
    """Pixel window of the image (whole LUT cells) containing `area` -> stack['window'], stack['lut_slice']."""
    lat, lon = stack["lut"]["latitude"], stack["lut"]["longitude"]
    valid = np.isfinite(lat) & np.isfinite(lon)
    inside = np.zeros_like(valid)
    inside[valid] = shapely.contains_xy(area, lon[valid], lat[valid])
    if not inside.any():
        raise RuntimeError(f"Common area falls between LUT nodes of {stack['stack_id']}")

    rows, cols = np.nonzero(inside)
    n_az, n_rg = lat.shape
    i0, i1 = max(rows.min() - 1, 0), min(rows.max() + 1, n_az - 1)
    j0, j1 = max(cols.min() - 1, 0), min(cols.max() + 1, n_rg - 1)
    step_az, step_rg = stack["lut_step"]
    height, width = stack["image_shape"]

    row_off, col_off = i0 * step_az, j0 * step_rg
    row_end, col_end = min(i1 * step_az + 1, height), min(j1 * step_rg + 1, width)
    stack["lut_slice"] = (int(i0), int(i1 + 1), int(j0), int(j1 + 1))
    stack["window"] = Window(col_off, row_off, col_end - col_off, row_end - row_off)


def window_latlon(stack):
    """Bilinear interpolation of the LUT lat/lon onto every pixel of the window, (2, rows, cols) float32."""
    w = stack["window"]
    step_az, step_rg = stack["lut_step"]
    rows = (np.arange(w.row_off, w.row_off + w.height) / step_az).astype(np.float32)
    cols = (np.arange(w.col_off, w.col_off + w.width) / step_rg).astype(np.float32)
    out = np.empty((2, w.height, w.width), dtype=np.float32)
    for r0 in range(0, w.height, 1024):
        rr, cc = np.meshgrid(rows[r0:r0 + 1024], cols, indexing="ij")
        for k, name in enumerate(("latitude", "longitude")):
            out[k, r0:r0 + 1024] = map_coordinates(stack["lut"][name], [rr, cc], order=1, mode="nearest")
    return out


def stack_gcps(stack):
    """A coarse grid of GCPs (relative to the window) taken from the LUT."""
    i0, i1, j0, j1 = stack["lut_slice"]
    step_az, step_rg = stack["lut_step"]
    w = stack["window"]
    lut = stack["lut"]
    gcps = []
    for i in np.linspace(i0, i1 - 1, GCP_GRID[0]).astype(int):
        for j in np.linspace(j0, j1 - 1, GCP_GRID[1]).astype(int):
            lat, lon, h = lut["latitude"][i, j], lut["longitude"][i, j], lut["height"][i, j]
            if np.isfinite(lat) and np.isfinite(lon):
                gcps.append(GroundControlPoint(
                    row=i * step_az - w.row_off, col=j * step_rg - w.col_off,
                    x=float(lon), y=float(lat), z=float(h) if np.isfinite(h) else 0.0,
                ))
    return gcps


def pixel_spacings(stack, latlon):
    """Slant range / ground range / azimuth spacing [m] and mean incidence [deg] of the stack window.

    ground range = slant range / sin(mean incidence) as in BPS main_covariance_estimation_SR,
    azimuth spacing measured on the window lat/lon.
    """
    step_az, step_rg = stack["lut_step"]
    slant = LIGHTSPEED / 2 * float(np.median(np.diff(stack["lut_axes"]["slant_range_time"]))) / step_rg
    i0, i1, j0, j1 = stack["lut_slice"]
    inc = np.deg2rad(np.nanmean(stack["lut"]["incidenceAngle"][i0:i1, j0:j1]))
    lat, lon = np.deg2rad(latlon[:, :, ::8].astype(np.float64))
    d_az = EARTH_RADIUS * np.hypot(np.diff(lat, axis=0), np.diff(lon, axis=0) * np.cos(lat[1:]))
    return {
        "slant_range_m": slant,
        "ground_range_m": float(slant / np.sin(inc)),
        "azimuth_m": float(np.nanmedian(d_az)),
        "incidence_mean_deg": float(np.rad2deg(inc)),
    }


def write_stack_geometry(stack, folder, area):
    """overlap_mask.tif, latlon.tif and geometry_lut.nc of one stack; sets mask_fraction and spacings."""
    w = stack["window"]
    latlon = window_latlon(stack)
    mask = np.zeros((w.height, w.width), dtype=np.uint8)
    for r0 in range(0, w.height, 1024):
        mask[r0:r0 + 1024] = shapely.contains_xy(area, latlon[1, r0:r0 + 1024], latlon[0, r0:r0 + 1024])
    stack["mask_fraction"] = float(mask.mean())
    stack["spacings"] = pixel_spacings(stack, latlon)

    profile = dict(driver="GTiff", width=w.width, height=w.height, tiled=True,
                   blockxsize=512, blockysize=512, compress="deflate")
    gcps = stack_gcps(stack)
    with rasterio.open(folder / "overlap_mask.tif", "w", count=1, dtype="uint8", **profile) as dst:
        dst.write(mask, 1)
        dst.gcps = (gcps, rasterio.CRS.from_epsg(4326))
    with rasterio.open(folder / "latlon.tif", "w", count=2, dtype="float32", **profile) as dst:
        dst.write(latlon)
        dst.gcps = (gcps, rasterio.CRS.from_epsg(4326))
        dst.set_band_description(1, "latitude")
        dst.set_band_description(2, "longitude")
    del latlon, mask

    i0, i1, j0, j1 = stack["lut_slice"]
    step_az, step_rg = stack["lut_step"]
    with NC_LOCK, netCDF4.Dataset(str(folder / "geometry_lut.nc"), "w") as ds:
        ds.createDimension("azimuth", i1 - i0)
        ds.createDimension("range", j1 - j0)
        ds.source_product = stack["reference"].id
        ds.description = (
            "LUT geometry cropped to the window. LUT node (a, r) is crop pixel "
            "(a*azimuth_step + lut_row_offset, r*range_step + lut_col_offset)."
        )
        ds.azimuth_step, ds.range_step = step_az, step_rg
        ds.lut_row_offset = i0 * step_az - w.row_off
        ds.lut_col_offset = j0 * step_rg - w.col_off
        for name, data in stack["lut"].items():
            var = ds.createVariable(name, "f8", ("azimuth", "range"), fill_value=np.nan, zlib=True)
            var[:] = data[i0:i1, j0:j1]
