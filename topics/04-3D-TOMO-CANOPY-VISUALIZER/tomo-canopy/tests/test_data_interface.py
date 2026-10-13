"""Tests for the shared multimodal data interface."""

import numpy as np
import pytest
import xarray as xr

from tomo_canopy.types import DataCube, DataLayer, GeometryType, GridSpec, Modality


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(
        x=np.array([0.5, 1.5]),
        y=np.array([0.5, 1.5, 2.5]),
        z=np.array([0.25, 0.75]),
        crs="EPSG:32630",
    )


def make_tls_layer(grid: GridSpec) -> DataLayer:
    dataset = xr.Dataset(
        {"occupancy": (("z", "y", "x"), np.ones(grid.shape, dtype=np.float32))},
        coords=grid.coordinates,
        attrs={"crs": grid.crs},
    )
    return DataLayer(
        name="tls_occupancy",
        data=dataset,
        modality=Modality.TLS,
        geometry=GeometryType.VOLUME,
    )


def test_cube_allows_no_modalities(grid: GridSpec) -> None:
    cube = DataCube(grid=grid)

    assert cube.layers == {}
    assert cube.modalities == frozenset()
    assert cube.missing_modalities(
        {Modality.TLS, Modality.ALS, Modality.GEDI, Modality.TOMOGRAM}
    ) == frozenset({"tls", "als", "gedi", "tomogram"})


def test_cube_allows_tls_without_other_modalities(grid: GridSpec) -> None:
    cube = DataCube(grid=grid)
    cube.add_layer(make_tls_layer(grid))

    assert cube.has_modality(Modality.TLS)
    assert not cube.has_modality(Modality.ALS)
    assert cube.missing_modalities(
        {Modality.TLS, Modality.ALS, Modality.GEDI}
    ) == frozenset({"als", "gedi"})
    assert len(cube.layers_for("tls")) == 1
    assert cube.layers_for("als") == ()


def test_require_modalities_reports_only_missing_data(grid: GridSpec) -> None:
    cube = DataCube(grid=grid, layers={"tls_occupancy": make_tls_layer(grid)})

    cube.require_modalities(Modality.TLS)
    with pytest.raises(ValueError, match="als, gedi"):
        cube.require_modalities(Modality.TLS, Modality.GEDI, Modality.ALS)


def test_reject_layer_on_different_grid(grid: GridSpec) -> None:
    shifted = xr.Dataset(
        {"height": (("y", "x"), np.ones((3, 2)))},
        coords={"x": grid.x + 1, "y": grid.y},
        attrs={"crs": grid.crs},
    )
    layer = DataLayer(
        name="als_height",
        data=shifted,
        modality=Modality.ALS,
        geometry=GeometryType.RASTER,
    )

    with pytest.raises(ValueError, match="coordinate 'x'"):
        DataCube(grid=grid).add_layer(layer)


def test_reject_layer_on_different_crs(grid: GridSpec) -> None:
    dataset = xr.Dataset(
        {"height": (("y", "x"), np.ones((3, 2)))},
        coords={"x": grid.x, "y": grid.y},
        attrs={"crs": "EPSG:4326"},
    )
    layer = DataLayer(
        name="als_height",
        data=dataset,
        modality=Modality.ALS,
        geometry=GeometryType.RASTER,
    )

    with pytest.raises(ValueError, match="does not match grid CRS"):
        DataCube(grid=grid).add_layer(layer)


def test_to_xarray_contains_only_present_layers(grid: GridSpec) -> None:
    cube = DataCube(grid=grid, layers={"tls_occupancy": make_tls_layer(grid)})

    dataset = cube.to_xarray()

    assert set(dataset.data_vars) == {"tls_occupancy__occupancy"}
    assert dataset.attrs["modalities"] == ["tls"]
    assert dataset.attrs["crs"] == grid.crs


def test_grid_axes_must_be_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        GridSpec(
            x=np.array([1.0, 0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            crs="EPSG:32630",
        )
