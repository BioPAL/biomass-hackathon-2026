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


def test_decibel_floor_preserves_missing_cells(l2a_tomogram: xr.Dataset) -> None:
    power = TomogramProcessor().to_power(l2a_tomogram)
    power["tomographic_power"][0, 0, 0, 0] = np.nan
    power["tomographic_power"][0, 0, 0, 1] = 0.0

    result = TomogramProcessor().to_db(power, floor_db=-30.0)

    assert np.isnan(result.tomographic_power_db[0, 0, 0, 0])
    assert float(result.tomographic_power_db[0, 0, 0, 1]) == -30.0


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


def test_prepare_vertical_structure_produces_normalized_bands_and_metrics(
    l2a_tomogram: xr.Dataset,
) -> None:
    result = TomogramProcessor().prepare_vertical_structure(
        l2a_tomogram, polarization="HV", band_size=10.0, upper_height=0.0
    )

    assert result.sizes["polarization"] == 1
    assert result.vertical_contribution.dims == (
        "polarization",
        "height_band",
        "y",
        "x",
    )
    xr.testing.assert_allclose(
        result.vertical_contribution.sum("height_band"),
        xr.ones_like(result.vertical_contribution.isel(height_band=0)),
    )
    assert {
        "scattering_height_50",
        "scattering_height_90",
        "scattering_centroid",
        "dominant_scattering_height",
        "dominant_contribution",
        "vertical_entropy",
        "upper_contribution",
    }.issubset(result)
    assert result.attrs["processing_recipe"] == "tomogram_vertical_structure"


def test_vertical_band_integration_rejects_complex_amplitude(
    l2a_tomogram: xr.Dataset,
) -> None:
    with pytest.raises(ValueError, match="power, not complex"):
        TomogramProcessor().integrate_vertical_bands(
            l2a_tomogram, variable="tomographic_cube"
        )


def test_prepare_vertical_structure_supports_dask(
    l2a_tomogram: xr.Dataset,
) -> None:
    pytest.importorskip("dask.array")
    chunked = l2a_tomogram.chunk({"polarization": 1, "z": 2, "y": 1, "x": 2})

    result = TomogramProcessor().prepare_vertical_structure(
        chunked, band_size=10.0
    )

    assert result.dominant_scattering_height.chunks is not None
    computed = result[
        ["dominant_scattering_height", "dominant_contribution"]
    ].compute()
    assert np.isfinite(computed.dominant_scattering_height).all()
