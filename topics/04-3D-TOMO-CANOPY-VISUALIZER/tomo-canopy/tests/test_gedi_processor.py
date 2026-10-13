"""Tests for GEDI processing operations."""

import numpy as np
import pytest
import xarray as xr

from tomo_canopy.processors import GEDIProcessor


@pytest.fixture
def shots() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "agbd": ("shot_number", [100.0, np.nan, 140.0, 160.0]),
            "rh": (
                ("shot_number", "rh_percentile"),
                [
                    [0.0, 10.0, 20.0],
                    [0.0, 11.0, 21.0],
                    [np.nan, np.nan, np.nan],
                    [0.0, 13.0, 23.0],
                ],
            ),
        },
        coords={
            "shot_number": [10, 11, 12, 13],
            "longitude": ("shot_number", [-1.00, -0.99, -0.98, -0.97]),
            "latitude": ("shot_number", [5.00, 5.01, 5.02, 5.03]),
            "x": ("shot_number", [-1.00, -0.99, -0.98, -0.97]),
            "y": ("shot_number", [5.00, 5.01, 5.02, 5.03]),
            "time": (
                "shot_number",
                np.array(
                    ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                    dtype="datetime64[D]",
                ),
            ),
            "rh_percentile": [0.0, 50.0, 100.0],
        },
        attrs={
            "source": "gedidb",
            "modality": "gedi",
            "geometry_type": "point_cloud",
            "crs": "EPSG:4326",
        },
    )


@pytest.fixture
def processor() -> GEDIProcessor:
    return GEDIProcessor()


def test_validate_and_bounds(processor: GEDIProcessor, shots: xr.Dataset) -> None:
    processor.validate(shots, require_crs=True, check_finite=True)

    bounds = processor.bounds(shots)

    np.testing.assert_allclose(bounds.minimum, [-1.0, 5.0])
    np.testing.assert_allclose(bounds.maximum, [-0.97, 5.03])
    assert bounds.attrs["crs"] == "EPSG:4326"


def test_filter_valid_shots(processor: GEDIProcessor, shots: xr.Dataset) -> None:
    strict = processor.filter_valid(shots, variables=["agbd", "rh"], how="any")
    permissive = processor.filter_valid(shots, variables=["agbd", "rh"], how="all")

    np.testing.assert_array_equal(strict.shot_number, [10, 13])
    np.testing.assert_array_equal(permissive.shot_number, [10, 11, 12, 13])


def test_sample_shots_is_reproducible(
    processor: GEDIProcessor, shots: xr.Dataset
) -> None:
    first = processor.sample_shots(shots, count=2, seed=7)
    second = processor.sample_shots(shots, count=2, seed=7)

    np.testing.assert_array_equal(first.shot_number, second.shot_number)
    assert first.sizes == {"shot_number": 2, "rh_percentile": 3}
    assert first.attrs["source_shot_count"] == 4


def test_add_rh_metric(processor: GEDIProcessor, shots: xr.Dataset) -> None:
    result = processor.add_rh_metric(shots, percentile=98)

    assert "rh98" in result
    np.testing.assert_allclose(result.rh98, [20.0, 21.0, np.nan, 23.0])
    assert result.rh98.attrs["profile_percentile"] == 100.0


def test_reproject_with_injected_transformer(
    processor: GEDIProcessor, shots: xr.Dataset
) -> None:
    class Transformer:
        @staticmethod
        def transform(x, y):
            return np.asarray(x) * 1000, np.asarray(y) * 1000

    projected = processor.reproject(
        shots, target_crs="EPSG:32630", transformer=Transformer()
    )

    np.testing.assert_allclose(projected.x, shots.x * 1000)
    np.testing.assert_allclose(projected.y, shots.y * 1000)
    assert projected.attrs["source_crs"] == "EPSG:4326"
    assert projected.attrs["crs"] == "EPSG:32630"


def test_local_coordinates_and_data_layer(
    processor: GEDIProcessor, shots: xr.Dataset
) -> None:
    local = processor.to_local_coordinates(shots)
    layer = processor.to_data_layer(shots)

    np.testing.assert_allclose([local.x.min(), local.y.min()], [0.0, 0.0])
    assert local.attrs["source_crs"] == "EPSG:4326"
    assert layer.modality == "gedi"
    assert layer.geometry == "point_cloud"
    assert layer.crs == "EPSG:4326"


def test_validate_allows_empty_provider_result(processor: GEDIProcessor) -> None:
    empty = xr.Dataset(
        coords={"shot_number": np.array([], dtype=np.uint64)},
        attrs={"crs": "EPSG:4326"},
    )

    processor.validate(empty)
    with pytest.raises(ValueError, match="contains no shots"):
        processor.validate(empty, require_nonempty=True)
