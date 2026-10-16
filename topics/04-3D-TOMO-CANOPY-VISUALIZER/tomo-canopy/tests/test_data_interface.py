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
        {Modality.TLS, Modality.LVIS, Modality.GEDI, Modality.TOMOGRAM}
    ) == frozenset({"tls", "lvis", "gedi", "tomogram"})


def test_cube_allows_tls_without_other_modalities(grid: GridSpec) -> None:
    cube = DataCube(grid=grid)
    cube.add_layer(make_tls_layer(grid))

    assert cube.has_modality(Modality.TLS)
    assert not cube.has_modality(Modality.LVIS)
    assert cube.missing_modalities(
        {Modality.TLS, Modality.LVIS, Modality.GEDI}
    ) == frozenset({"gedi", "lvis"})
    assert len(cube.layers_for("tls")) == 1
    assert cube.layers_for("lvis") == ()


def test_require_modalities_reports_only_missing_data(grid: GridSpec) -> None:
    cube = DataCube(grid=grid, layers={"tls_occupancy": make_tls_layer(grid)})

    cube.require_modalities(Modality.TLS)
    with pytest.raises(ValueError, match="gedi, lvis"):
        cube.require_modalities(Modality.TLS, Modality.GEDI, Modality.LVIS)


def test_reject_layer_on_different_grid(grid: GridSpec) -> None:
    shifted = xr.Dataset(
        {"height": (("y", "x"), np.ones((3, 2)))},
        coords={"x": grid.x + 1, "y": grid.y},
        attrs={"crs": grid.crs},
    )
    layer = DataLayer(
        name="lvis_height",
        data=shifted,
        modality=Modality.LVIS,
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
        name="lvis_height",
        data=dataset,
        modality=Modality.LVIS,
        geometry=GeometryType.RASTER,
    )

    with pytest.raises(ValueError, match="does not match grid CRS"):
        DataCube(grid=grid).add_layer(layer)


def test_cube_accepts_native_tls_point_cloud(grid: GridSpec) -> None:
    dataset = xr.Dataset(
        {"red": ("point", np.array([10, 20], dtype=np.uint8))},
        coords={
            "x": ("point", [0.7, 1.2]),
            "y": ("point", [1.1, 2.1]),
            "z": ("point", [0.3, 0.8]),
        },
        attrs={"crs": grid.crs},
    )
    layer = DataLayer(
        name="tls_points",
        data=dataset,
        modality=Modality.TLS,
        geometry=GeometryType.POINT_CLOUD,
    )

    cube = DataCube(grid=grid)
    cube.add_layer(layer)

    assert cube.layers["tls_points"] is layer


def test_cube_accepts_native_gedi_shots_without_z(grid: GridSpec) -> None:
    dataset = xr.Dataset(
        {"agbd": ("shot_number", [120.0, 140.0])},
        coords={
            "x": ("shot_number", [0.7, 1.2]),
            "y": ("shot_number", [1.1, 2.1]),
        },
        attrs={"crs": grid.crs},
    )
    layer = DataLayer(
        name="gedi_shots",
        data=dataset,
        modality=Modality.GEDI,
        geometry=GeometryType.POINT_CLOUD,
    )

    DataCube(grid=grid).add_layer(layer)


def test_cube_accepts_native_lvis_footprints_without_z(grid: GridSpec) -> None:
    dataset = xr.Dataset(
        {"rh98": ("shot", [32.0, 41.0])},
        coords={
            "x": ("shot", [0.7, 1.2]),
            "y": ("shot", [1.1, 2.1]),
        },
        attrs={"crs": grid.crs},
    )
    layer = DataLayer(
        name="lvis_footprints",
        data=dataset,
        modality=Modality.LVIS,
        geometry=GeometryType.FOOTPRINTS,
    )

    DataCube(grid=grid).add_layer(layer)


def test_cube_accepts_native_triangular_mesh(grid: GridSpec) -> None:
    dataset = xr.Dataset(
        {"face_vertices": (("face", "corner"), [[0, 1, 2]])},
        coords={
            "x": ("vertex", [0.0, 1.0, 0.0]),
            "y": ("vertex", [0.0, 0.0, 1.0]),
            "z": ("vertex", [0.0, 0.0, 0.0]),
        },
        attrs={"crs": grid.crs},
    )
    layer = DataLayer(
        name="tls_mesh",
        data=dataset,
        modality=Modality.TLS,
        geometry=GeometryType.TRIANGULAR_MESH,
    )

    DataCube(grid=grid).add_layer(layer)


def test_reject_point_cloud_with_incompatible_coordinate_dimensions(
    grid: GridSpec,
) -> None:
    dataset = xr.Dataset(
        coords={
            "x": ("point", [0.0, 1.0]),
            "y": ("observation", [0.0, 1.0]),
            "z": ("point", [0.0, 1.0]),
        },
        attrs={"crs": grid.crs},
    )
    layer = DataLayer(
        name="invalid_points",
        data=dataset,
        modality=Modality.TLS,
        geometry=GeometryType.POINT_CLOUD,
    )

    with pytest.raises(ValueError, match="coordinate 'y'"):
        DataCube(grid=grid).add_layer(layer)


def test_volume_requires_all_three_grid_axes(grid: GridSpec) -> None:
    dataset = xr.Dataset(
        {"occupancy": (("y", "x"), np.ones((3, 2)))},
        coords={"x": grid.x, "y": grid.y},
        attrs={"crs": grid.crs},
    )
    layer = DataLayer(
        name="invalid_volume",
        data=dataset,
        modality=Modality.TLS,
        geometry=GeometryType.VOLUME,
    )

    with pytest.raises(ValueError, match="missing coordinate.*z"):
        DataCube(grid=grid).add_layer(layer)


def test_to_xarray_contains_only_present_layers(grid: GridSpec) -> None:
    cube = DataCube(grid=grid, layers={"tls_occupancy": make_tls_layer(grid)})

    dataset = cube.to_xarray()

    assert set(dataset.data_vars) == {"tls_occupancy__occupancy"}
    assert dataset.attrs["modalities"] == ["tls"]
    assert dataset.attrs["crs"] == grid.crs


def test_to_xarray_preserves_native_layer_with_prefixed_coordinates(
    grid: GridSpec,
) -> None:
    point_dataset = xr.Dataset(
        {"intensity": ("point", [0.4, 0.8])},
        coords={
            "x": ("point", [0.7, 1.2]),
            "y": ("point", [1.1, 2.1]),
            "z": ("point", [0.3, 0.8]),
        },
        attrs={"crs": grid.crs},
    )
    point_layer = DataLayer(
        name="tls_points",
        data=point_dataset,
        modality=Modality.TLS,
        geometry=GeometryType.POINT_CLOUD,
    )
    cube = DataCube(
        grid=grid,
        layers={
            "tls_occupancy": make_tls_layer(grid),
            "tls_points": point_layer,
        },
    )

    dataset = cube.to_xarray()

    assert dataset["tls_points__intensity"].dims == ("tls_points__point",)
    assert dataset["tls_points__x"].dims == ("tls_points__point",)
    assert dataset["tls_occupancy__occupancy"].dims == ("z", "y", "x")


def test_get_layer_enforces_optional_contract(grid: GridSpec) -> None:
    cube = DataCube(grid=grid, layers={"tls_occupancy": make_tls_layer(grid)})

    assert cube.get_layer("tls_occupancy", modality=Modality.TLS).name == (
        "tls_occupancy"
    )
    with pytest.raises(ValueError, match="expected 'tomogram'"):
        cube.get_layer("tls_occupancy", modality=Modality.TOMOGRAM)
    with pytest.raises(KeyError, match="available layers: tls_occupancy"):
        cube.get_layer("missing")


def test_grid_axes_must_be_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        GridSpec(
            x=np.array([1.0, 0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            crs="EPSG:32630",
        )
