"""Spatial transforms for regular canopy datasets."""

from __future__ import annotations

import xarray as xr

from tomo_canopy.types import GridSpec


def resample_to_grid(
    dataset: xr.Dataset,
    grid: GridSpec,
    *,
    method: str = "linear",
) -> xr.Dataset:
    """Interpolate a geocoded regular volume onto a target visualization grid."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'")
    missing = [axis for axis in ("x", "y", "z") if axis not in dataset.coords]
    if missing:
        raise ValueError(
            "Regular volume resampling requires x/y/z coordinates; missing "
            + ", ".join(missing)
        )
    source_crs = dataset.attrs.get("crs")
    if source_crs is None:
        raise ValueError("Dataset has no CRS metadata")
    if str(source_crs) != grid.crs:
        raise ValueError(
            "resample_to_grid does not reproject coordinates; source and target "
            "CRS must match"
        )
    if dataset.attrs.get("coordinate_space") == "sar":
        raise ValueError("Native L1D SAR data must be geocoded before grid resampling")

    result = dataset
    for axis in ("x", "y", "z"):
        result = result.interp({axis: getattr(grid, axis)}, method=method)
    result.attrs = {
        **dataset.attrs,
        "crs": grid.crs,
        "resampling_method": method,
        "resampled_to_grid": True,
    }
    return result
