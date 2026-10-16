"""Sensor-agnostic points-to-voxels transformation.

Converts any point cloud (TLS, ALS, photogrammetry, UAV lidar, ...) held as an
xarray object with ``x``/``y``/``z`` coordinates into a voxelised volume
(``point_count`` + ``occupancy`` on a regular grid).

The transformation is pure: it depends only on the point data, the voxel size
and (optionally) precomputed bounds. It knows nothing about sensors, file
formats or processors, so a processor simply calls :func:`points_to_voxels`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import xarray as xr

from tomo_canopy.types import GridSpec

#: Axes handled by the transformation, in the order used for grid coordinates.
VOXEL_AXES: tuple[str, str, str] = ("x", "y", "z")


def compute_voxel_edges(
        minimum: Mapping[str, float],
        maximum: Mapping[str, float],
        voxel_size: float,
        axes: Sequence[str] = VOXEL_AXES,
) -> dict[str, np.ndarray]:
    """Voxel edge arrays per axis, aligned to the global voxel-size grid.

    Edges are snapped outward so the grid covers the bounds exactly and is
    reproducible regardless of where the points sit in the world.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    return {
        axis: np.arange(
            np.floor(minimum[axis] / voxel_size) * voxel_size,
            np.ceil(maximum[axis] / voxel_size) * voxel_size + voxel_size,
            voxel_size,
        )
        for axis in axes
    }


def points_to_voxels(
        points: xr.DataArray | xr.Dataset,
        voxel_size: float,
        *,
        crs: str | None = None,
        bounds: xr.DataArray | None = None,
        source: str | None = None,
        count_dtype: np.dtype | type = np.int32,
) -> xr.Dataset:
    """Voxelize a point cloud into a regular-grid volume.

    Parameters
    ----------
    points:
        Point cloud with ``x``, ``y``, ``z`` coordinates (e.g. a mesh with
        face connectivity dropped). Any sensor modality; nothing TLS-specific
        is assumed.
    voxel_size:
        Isotropic voxel edge length in the units of ``crs``.
    crs:
        Coordinate reference system of the points. Defaults to ``points.attrs``
        ``crs`` when available.
    bounds:
        Optional bounds (as produced by a processor's ``bounds()``) with
        ``axis``, ``minimum`` and ``maximum`` variables. When omitted, the
        bounds are derived from the point coordinates themselves.
    source:
        Optional provenance string stored in the output attributes.
    count_dtype:
        Integer dtype for the ``point_count`` variable.

    Returns
    -------
    xr.Dataset
        Dataset with ``point_count`` and ``occupancy`` variables on the
        ``("z", "y", "x")`` dimensions, plus ``crs``, ``voxel_size``,
        ``aggregation`` and (if given) ``source`` attributes.
    """
    if crs is None:
        crs = points.attrs.get("crs")
    if crs is None:
        raise ValueError(
            "No CRS given: pass crs= or set a 'crs' attribute on the points."
        )

    if bounds is not None:
        minimum = dict(zip(bounds.axis.values, bounds.minimum.values))
        maximum = dict(zip(bounds.axis.values, bounds.maximum.values))
    else:
        minimum = {axis: float(points[axis].min()) for axis in VOXEL_AXES}
        maximum = {axis: float(points[axis].max()) for axis in VOXEL_AXES}

    edges = compute_voxel_edges(minimum, maximum, voxel_size)

    point_count, _ = np.histogramdd(
        (points.z.values, points.y.values, points.x.values),
        bins=(edges["z"], edges["y"], edges["x"]),
    )
    occupancy = point_count > 0

    centres = {
        axis: (axis_edges[:-1] + axis_edges[1:]) / 2
        for axis, axis_edges in edges.items()
    }
    grid = GridSpec(x=centres["x"], y=centres["y"], z=centres["z"], crs=crs)

    attrs = {
        "crs": grid.crs,
        "voxel_size": voxel_size,
        "aggregation": "point count per voxel",
    }
    if source is not None:
        attrs["source"] = source
    if bounds is not None:
        attrs["bounds_source"] = "provided"
    else:
        attrs["bounds_source"] = "derived from points"

    return xr.Dataset(
        {
            "point_count": (("z", "y", "x"), point_count.astype(count_dtype)),
            "occupancy": (("z", "y", "x"), occupancy),
        },
        coords=grid.coordinates,
        attrs=attrs,
    )
