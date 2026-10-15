"""Tests for shared spatial and resolution transforms."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tomo_canopy.transforms import (
    match_vertical_resolution,
    reproject_points,
    resample_to_grid,
)
from tomo_canopy.types import GridSpec


def test_match_vertical_resolution_blurs_without_changing_sampling() -> None:
    values = np.zeros((11, 1, 1), dtype=float)
    values[5, 0, 0] = 1.0
    dataset = xr.Dataset(
        {"occupancy": (("z", "y", "x"), values)},
        coords={
            "z": ("z", np.arange(11.0), {"units": "m"}),
            "y": [0.0],
            "x": [0.0],
        },
    )

    result = match_vertical_resolution(dataset, target_resolution=4.0)

    np.testing.assert_array_equal(result.z, dataset.z)
    assert 0.0 < float(result.occupancy.sel(z=5.0).squeeze()) < 1.0
    assert float(result.occupancy.sel(z=4.0).squeeze()) > 0.0
    assert result.attrs["vertical_sampling"] == 1.0
    assert result.attrs["vertical_resolution"] == 4.0


def test_match_vertical_resolution_accounts_for_source_resolution() -> None:
    dataset = xr.Dataset(
        {"value": ("z", [0.0, 1.0, 0.0])},
        coords={"z": [0.0, 1.0, 2.0]},
    )

    unchanged = match_vertical_resolution(
        dataset, target_resolution=20.0, source_resolution=20.0
    )
    xr.testing.assert_identical(unchanged.value, dataset.value)
    with pytest.raises(ValueError, match="cannot be finer"):
        match_vertical_resolution(
            dataset, target_resolution=10.0, source_resolution=20.0
        )


def test_resample_to_grid_preserves_crs_and_target_coordinates() -> None:
    dataset = xr.Dataset(
        {
            "value": (
                ("z", "y", "x"),
                np.arange(8, dtype=float).reshape(2, 2, 2),
            )
        },
        coords={"x": [0.0, 2.0], "y": [0.0, 2.0], "z": [0.0, 2.0]},
        attrs={"crs": "EPSG:32630", "coordinate_space": "geographic"},
    )
    grid = GridSpec(
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0, 1.0, 2.0]),
        z=np.array([0.0, 1.0, 2.0]),
        crs="EPSG:32630",
    )

    result = resample_to_grid(dataset, grid)

    assert result.value.shape == grid.shape
    np.testing.assert_array_equal(result.x, grid.x)
    assert result.attrs["crs"] == grid.crs
    assert result.attrs["resampled_to_grid"] is True


def test_resample_to_grid_rejects_reprojection_and_sar_data() -> None:
    dataset = xr.Dataset(
        {"value": (("z", "y", "x"), np.ones((2, 2, 2)))},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
        attrs={"crs": "EPSG:4326", "coordinate_space": "sar"},
    )
    other_crs = GridSpec(
        x=np.array([0.0]),
        y=np.array([0.0]),
        z=np.array([0.0]),
        crs="EPSG:32630",
    )
    same_crs = GridSpec(
        x=np.array([0.0]),
        y=np.array([0.0]),
        z=np.array([0.0]),
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="CRS must match"):
        resample_to_grid(dataset, other_crs)
    with pytest.raises(ValueError, match="must be geocoded"):
        resample_to_grid(dataset, same_crs)


def test_reproject_points_requires_shared_coordinates() -> None:
    dataset = xr.Dataset(
        coords={"x": ("point", [1.0]), "y": ("other", [2.0])},
        attrs={"crs": "EPSG:4326"},
    )

    with pytest.raises(ValueError, match="share one dimension"):
        reproject_points(dataset, target_crs="EPSG:32632")
