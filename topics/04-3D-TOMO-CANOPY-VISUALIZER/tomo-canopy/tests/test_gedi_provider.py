"""Tests for the optional gedidb-backed provider."""

import numpy as np
import pytest
import xarray as xr

from tomo_canopy.providers import GEDIProvider


class FakeGEDIDBProvider:
    def __init__(self, result: xr.Dataset | None) -> None:
        self.result = result
        self.calls: list[dict] = []

    def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def make_result() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "agbd": ("shot_number", np.array([101.0, 123.0])),
            "rh": (
                ("shot_number", "rh_percentile"),
                np.array([[2.0, 20.0], [3.0, 25.0]]),
            ),
        },
        coords={
            "shot_number": np.array([10, 11], dtype=np.uint64),
            "longitude": ("shot_number", np.array([-1.0, -0.9])),
            "latitude": ("shot_number", np.array([5.0, 5.1])),
            "time": (
                "shot_number",
                np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
            ),
            "rh_percentile": np.array([0.0, 100.0]),
        },
    )


def test_bounding_box_query_normalizes_dataset() -> None:
    backend = FakeGEDIDBProvider(make_result())
    provider = GEDIProvider(backend=backend)
    geometry = object()

    dataset = provider.get_data(
        variables=["agbd", "rh"],
        geometry=geometry,
        start_time="2018-01-01",
        end_time="2024-07-25",
        quality_filters={"l4_quality_flag": "= 1"},
    )

    assert dataset.sizes == {"shot_number": 2, "rh_percentile": 2}
    np.testing.assert_array_equal(dataset.x, dataset.longitude)
    np.testing.assert_array_equal(dataset.y, dataset.latitude)
    assert dataset.attrs["modality"] == "gedi"
    assert dataset.attrs["geometry_type"] == "point_cloud"
    assert dataset.attrs["crs"] == "EPSG:4326"
    assert backend.calls[0]["geometry"] is geometry
    assert backend.calls[0]["return_type"] == "xarray"
    assert backend.calls[0]["l4_quality_flag"] == "= 1"


def test_open_is_bounding_box_shortcut() -> None:
    backend = FakeGEDIDBProvider(make_result())
    provider = GEDIProvider(backend=backend)

    provider.open(object(), variables=["agbd"])

    assert backend.calls[0]["query_type"] == "bounding_box"
    assert backend.calls[0]["variables"] == ["agbd"]


def test_nearest_query() -> None:
    backend = FakeGEDIDBProvider(make_result())
    provider = GEDIProvider(backend=backend)

    provider.get_data(
        variables=["agbd"],
        query_type="nearest",
        point=(-1.0, 5.0),
        num_shots=1,
        radius=0.1,
    )

    assert backend.calls[0]["point"] == (-1.0, 5.0)
    assert backend.calls[0]["num_shots"] == 1


def test_no_results_returns_empty_dataset() -> None:
    provider = GEDIProvider(backend=FakeGEDIDBProvider(None))

    dataset = provider.get_data(variables=["agbd"], geometry=object())

    assert dataset.sizes == {"shot_number": 0}
    assert dataset.attrs["requested_variables"] == ["agbd"]
    assert dataset.attrs["modality"] == "gedi"


def test_reject_invalid_date_range() -> None:
    provider = GEDIProvider(backend=FakeGEDIDBProvider(None))

    with pytest.raises(ValueError, match="must not be after"):
        provider.get_data(
            variables=["agbd"],
            geometry=object(),
            start_time="2024-01-02",
            end_time="2024-01-01",
        )


def test_reject_variables_as_one_string() -> None:
    provider = GEDIProvider(backend=FakeGEDIDBProvider(None))

    with pytest.raises(TypeError, match="not a string"):
        provider.get_data(variables="agbd", geometry=object())
