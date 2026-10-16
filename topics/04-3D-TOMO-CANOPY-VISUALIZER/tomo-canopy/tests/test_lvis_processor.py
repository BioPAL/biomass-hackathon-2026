"""Tests for LVIS Level-2 footprint processing."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tomo_canopy.processors import LVISProcessor
from tomo_canopy.transforms import reproject_points
from tomo_canopy.types import DataCube, GeometryType, GridSpec, Modality


@pytest.fixture
def lvis() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "rh": (
                ("shot", "percentile"),
                [[10.0, 20.0, 30.0], [12.0, 22.0, 32.0], [np.nan, 24.0, 34.0]],
            ),
            "ground_elevation": ("shot", [100.0, 110.0, 120.0]),
            "sensitivity": ("shot", [0.99, 0.90, 0.98]),
            "channel_rh": ("shot", [1, 2, 1]),
            "channel_ground": ("shot", [1, 1, 2]),
            "channel_top": ("shot", [1, 1, 1]),
        },
        coords={
            "shot": [0, 1, 2],
            "shot_number": ("shot", [101, 102, 103]),
            "longitude": ("shot", [11.0, 11.1, 11.2]),
            "latitude": ("shot", [0.0, 0.1, 0.2]),
            "x": ("shot", [11.0, 11.1, 11.2]),
            "y": ("shot", [0.0, 0.1, 0.2]),
            "percentile": [50, 75, 98],
        },
        attrs={
            "crs": "EPSG:4326",
            "modality": "lvis",
            "geometry_type": "footprints",
        },
    )


def test_validate_and_bounds(lvis: xr.Dataset) -> None:
    processor = LVISProcessor()

    processor.validate(
        lvis, require_crs=True, require_nonempty=True, check_finite=True
    )
    bounds = processor.bounds(lvis)

    np.testing.assert_allclose(bounds.minimum, [11.0, 0.0])
    np.testing.assert_allclose(bounds.maximum, [11.2, 0.2])
    assert bounds.attrs["crs"] == "EPSG:4326"


def test_validate_rejects_invalid_rh_dimensions(lvis: xr.Dataset) -> None:
    invalid = lvis.drop_vars("rh").assign(
        rh=(("percentile", "shot"), lvis.rh.values.T)
    )

    with pytest.raises(ValueError, match="'rh' must use"):
        LVISProcessor().validate(invalid)


def test_quality_filters_are_explicit_and_recorded(lvis: xr.Dataset) -> None:
    filtered = LVISProcessor().filter_quality(
        lvis,
        minimum_sensitivity=0.95,
        accepted_channel_rh=[1],
        accepted_channel_ground=[1, 2],
        require_finite_rh=True,
    )

    np.testing.assert_array_equal(filtered.shot_number, [101])
    assert filtered.attrs["source_shot_count"] == 3
    assert filtered.attrs["retained_shot_count"] == 1
    assert filtered.attrs["quality_filters"]["minimum_sensitivity"] == 0.95


def test_add_rh_metric_supports_exact_and_nearest(lvis: xr.Dataset) -> None:
    processor = LVISProcessor()

    exact = processor.add_rh_metric(lvis, percentile=98)
    nearest = processor.add_rh_metric(
        lvis, percentile=97, method="nearest", name="rh_nearest"
    )

    np.testing.assert_allclose(exact.rh98, [30.0, 32.0, 34.0])
    np.testing.assert_allclose(nearest.rh_nearest, exact.rh98)
    assert nearest.rh_nearest.attrs["profile_percentile"] == 98.0
    with pytest.raises(ValueError, match="is not available"):
        processor.add_rh_metric(lvis, percentile=97)


def test_add_canopy_top_uses_ground_plus_rh(lvis: xr.Dataset) -> None:
    result = LVISProcessor().add_canopy_top(lvis, percentile=98)

    np.testing.assert_allclose(result.canopy_top_elevation, [130.0, 142.0, 154.0])
    assert result.canopy_top_elevation.attrs["profile_percentile"] == 98.0


def test_sample_shots_is_reproducible(lvis: xr.Dataset) -> None:
    processor = LVISProcessor()

    first = processor.sample_shots(lvis, count=2, seed=42)
    second = processor.sample_shots(lvis, count=2, seed=42)

    xr.testing.assert_identical(first, second)
    assert first.attrs["source_shot_count"] == 3


class OffsetTransformer:
    def transform(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return x + 1000.0, y + 2000.0


def test_reproject_and_integrate_native_footprints(lvis: xr.Dataset) -> None:
    projected = reproject_points(
        lvis,
        target_crs="EPSG:32632",
        transformer=OffsetTransformer(),
    )
    layer = LVISProcessor().to_data_layer(projected)
    grid = GridSpec(
        x=np.array([1000.0]),
        y=np.array([2000.0]),
        z=np.array([0.0]),
        crs="EPSG:32632",
    )
    cube = DataCube(grid=grid)
    cube.add_layer(layer)

    np.testing.assert_allclose(projected.x, lvis.x + 1000.0)
    np.testing.assert_allclose(projected.longitude, lvis.longitude)
    assert projected.attrs["source_crs"] == "EPSG:4326"
    assert layer.modality == Modality.LVIS.value
    assert layer.geometry == GeometryType.FOOTPRINTS.value
    assert cube.has_modality(Modality.LVIS)


def test_cube_rejects_unprojected_lvis_for_projected_grid(lvis: xr.Dataset) -> None:
    layer = LVISProcessor().to_data_layer(lvis)
    grid = GridSpec(
        x=np.array([0.0]),
        y=np.array([0.0]),
        z=np.array([0.0]),
        crs="EPSG:32632",
    )

    with pytest.raises(ValueError, match="does not match grid CRS"):
        DataCube(grid=grid).add_layer(layer)
