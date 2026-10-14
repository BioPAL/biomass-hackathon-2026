"""Tests for BIOMASS tomogram processing operations."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tomo_canopy.processors import TomogramProcessor
from tomo_canopy.types import GeometryType, Modality


@pytest.fixture
def l2a_tomogram() -> xr.Dataset:
    values = np.arange(2 * 3 * 2 * 3, dtype=float).reshape(2, 3, 2, 3)
    values = values + 1j * (values / 2)
    return xr.Dataset(
        {
            "tomographic_cube": (
                ("polarization", "z", "y", "x"),
                values,
            )
        },
        coords={
            "polarization": ["HH", "HV"],
            "z": ("z", [-10.0, 0.0, 10.0], {"units": "m"}),
            "y": [1.0, 2.0],
            "x": [10.0, 11.0, 12.0],
        },
        attrs={
            "product_level": "L2A",
            "coordinate_space": "geographic",
            "crs": "EPSG:4979",
            "geometry_type": "volume",
        },
    )


def test_validate_accepts_l2a_contract(l2a_tomogram: xr.Dataset) -> None:
    TomogramProcessor().validate(l2a_tomogram, require_geocoded=True)


def test_validate_accepts_native_l1d_contract() -> None:
    dataset = xr.Dataset(
        {
            "tomographic_cube": (
                ("polarization", "z", "azimuth_time", "slant_range_time"),
                np.ones((1, 2, 3, 4)),
            )
        },
        coords={
            "polarization": ["HH"],
            "z": [0.0, 10.0],
            "azimuth_time": [0.0, 0.1, 0.2],
            "slant_range_time": [0.01, 0.02, 0.03, 0.04],
        },
        attrs={"product_level": "L1D", "coordinate_space": "sar"},
    )

    TomogramProcessor().validate(dataset)
    with pytest.raises(ValueError, match="geocoded L2A"):
        TomogramProcessor().to_data_layer(dataset)


def test_power_and_decibel_derivations_are_composable(
    l2a_tomogram: xr.Dataset,
) -> None:
    processor = TomogramProcessor()

    power = processor.to_power(l2a_tomogram)
    result = processor.to_db(power, floor_db=-30.0)

    expected = np.abs(l2a_tomogram.tomographic_cube) ** 2
    xr.testing.assert_allclose(power.tomographic_power, expected)
    assert "tomographic_cube" in result
    assert "tomographic_power" in result
    assert "tomographic_power_db" in result
    assert float(result.tomographic_power_db.min()) == -30.0
    assert result.attrs["processing_quantity"] == "power_db"


def test_select_subset_and_slices(l2a_tomogram: xr.Dataset) -> None:
    processor = TomogramProcessor()

    selected = processor.select_polarization(l2a_tomogram, "HV")
    subset = processor.subset(
        selected,
        height_range=(-1.0, 10.0),
        bounds=(10.5, 1.0, 12.0, 2.0),
    )
    horizontal = processor.horizontal_slice(selected, height=1.0)
    vertical = processor.vertical_slice(selected, axis="x", coordinate=10.8)

    assert selected.sizes["polarization"] == 1
    assert subset.sizes == {"polarization": 1, "z": 2, "y": 2, "x": 2}
    assert horizontal.attrs["selected_height"] == 0.0
    assert "z" not in horizontal.dims
    assert vertical.attrs["selected_slice_coordinate"] == 11.0
    assert "x" not in vertical.dims


@pytest.mark.parametrize("statistic", ["mean", "median", "max", "sum"])
def test_vertical_profile_aggregates_horizontal_axes(
    l2a_tomogram: xr.Dataset, statistic: str
) -> None:
    profile = TomogramProcessor().vertical_profile(
        l2a_tomogram, statistic=statistic
    )

    assert profile.tomographic_cube.dims == ("polarization", "z")
    assert profile.attrs["profile_statistic"] == statistic


def test_to_data_layer_returns_tomogram_volume(l2a_tomogram: xr.Dataset) -> None:
    layer = TomogramProcessor().to_data_layer(l2a_tomogram)

    assert layer.modality == Modality.TOMOGRAM.value
    assert layer.geometry == GeometryType.VOLUME.value
    assert layer.crs == "EPSG:4979"
