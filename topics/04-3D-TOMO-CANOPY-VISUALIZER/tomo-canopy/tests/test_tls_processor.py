"""Tests for TLS processing operations."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tomo_canopy.processors import TLSProcessor

TEST_DATA = Path(__file__).parent / "data" / "tls"


@pytest.fixture
def processor() -> TLSProcessor:
    return TLSProcessor()


@pytest.fixture
def mesh(processor: TLSProcessor) -> xr.Dataset:
    return processor.open(TEST_DATA / "tiny_mesh_ascii.ply", crs="EPSG:32630")


@pytest.fixture
def trees(processor: TLSProcessor) -> xr.Dataset:
    return processor.open(TEST_DATA / "tiny_trees.txt", crs="EPSG:32630")


def test_validate_and_bounds(processor: TLSProcessor, mesh: xr.Dataset) -> None:
    processor.validate(mesh, require_crs=True, check_finite=True)

    bounds = processor.bounds(mesh)

    np.testing.assert_allclose(bounds.minimum, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(bounds.maximum, [1.0, 1.0, 0.5])
    assert bounds.attrs["crs"] == "EPSG:32630"


def test_to_local_coordinates(processor: TLSProcessor, mesh: xr.Dataset) -> None:
    local = processor.to_local_coordinates(mesh, origin=(0.5, 0.5, 0.0))

    np.testing.assert_allclose(local.x, [-0.5, 0.5, 0.5, -0.5])
    np.testing.assert_allclose(mesh.x, [0.0, 1.0, 1.0, 0.0])
    assert local.attrs["coordinates_are_local"] is True
    assert local.attrs["coordinate_origin_x"] == 0.5
    assert local.attrs["source_crs"] == "EPSG:32630"
    assert "crs" not in local.attrs


def test_sample_vertices_is_reproducible(
    processor: TLSProcessor, mesh: xr.Dataset
) -> None:
    first = processor.sample_vertices(mesh, count=2, seed=7)
    second = processor.sample_vertices(mesh, count=2, seed=7)

    assert first.sizes == {"point": 2}
    assert "face_vertices" not in first
    assert first.attrs["geometry_type"] == "point_cloud"
    np.testing.assert_array_equal(first.source_vertex_id, second.source_vertex_id)
    np.testing.assert_allclose(first.x, second.x)


def test_tree_metrics(processor: TLSProcessor, trees: xr.Dataset) -> None:
    metrics = processor.tree_metrics(trees)

    np.testing.assert_array_equal(metrics.tree_id, [0, 1])
    np.testing.assert_array_equal(metrics.segment_count, [3, 2])
    np.testing.assert_allclose(metrics.height, [1.5, 1.0])
    np.testing.assert_allclose(metrics.maximum_radius, [0.5, 0.45])
    np.testing.assert_allclose(metrics.x_minimum, [0.0, 2.0])
    np.testing.assert_allclose(metrics.x_maximum, [0.5, 2.0])


def test_validate_rejects_missing_coordinate(processor: TLSProcessor) -> None:
    dataset = xr.Dataset(
        coords={"x": ("point", [0.0]), "y": ("point", [0.0])},
        attrs={"geometry_type": "point_cloud"},
    )

    with pytest.raises(ValueError, match="missing coordinate 'z'"):
        processor.validate(dataset)


def test_tree_metrics_reject_mesh(
    processor: TLSProcessor, mesh: xr.Dataset
) -> None:
    with pytest.raises(ValueError, match="require tree-segment geometry"):
        processor.tree_metrics(mesh)
