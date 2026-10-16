"""Spatial transforms for regular canopy datasets."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol

import numpy as np
import xarray as xr

from tomo_canopy.types import GridSpec


class CoordinateTransformer(Protocol):
    """Minimal coordinate-transformer interface for dependency injection."""

    def transform(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


def reproject_points(
    dataset: xr.Dataset,
    *,
    target_crs: str,
    transformer: CoordinateTransformer | None = None,
) -> xr.Dataset:
    """Reproject irregular X/Y coordinates while preserving native lon/lat.

    The spatial coordinates must be one-dimensional and share the same native
    observation dimension, such as ``shot`` or ``point``. The original
    longitude and latitude coordinates, when present, are not modified.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset")
    if not isinstance(target_crs, str) or not target_crs.strip():
        raise ValueError("target_crs must be a non-empty CRS identifier")
    if "x" not in dataset.coords or "y" not in dataset.coords:
        raise ValueError("Point reprojection requires x and y coordinates")
    if dataset.x.ndim != 1 or dataset.y.dims != dataset.x.dims:
        raise ValueError("Point x and y coordinates must share one dimension")
    source_crs = dataset.attrs.get("crs")
    if source_crs is None:
        raise ValueError("Dataset has no CRS metadata")
    source_crs = str(source_crs)
    target_crs = target_crs.strip()

    x = np.asarray(dataset.x.data)
    y = np.asarray(dataset.y.data)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Point coordinates contain non-finite values")
    if transformer is None:
        try:
            pyproj = import_module("pyproj")
        except ImportError as error:
            raise ImportError(
                "Point reprojection requires pyproj; install a modality extra "
                "such as `python -m pip install -e '.[lvis]'`"
            ) from error
        transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )

    transformed_x, transformed_y = transformer.transform(x, y)
    transformed_x = np.asarray(transformed_x)
    transformed_y = np.asarray(transformed_y)
    if transformed_x.shape != x.shape or transformed_y.shape != y.shape:
        raise ValueError("Coordinate transformation changed the point-array shape")
    if not np.isfinite(transformed_x).all() or not np.isfinite(
        transformed_y
    ).all():
        raise ValueError("Coordinate transformation produced non-finite values")

    dimension = dataset.x.dims[0]
    result = dataset.assign_coords(
        x=(dimension, transformed_x),
        y=(dimension, transformed_y),
    )
    result.attrs = {
        **dataset.attrs,
        "source_crs": source_crs,
        "crs": target_crs,
        "coordinate_transform": "reprojected",
    }
    return result


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
