"""Tests for the TLS PLY provider."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from tomo_canopy.providers import TLSProvider, open_tls

TEST_DATA = Path(__file__).parent / "data" / "tls"


def test_open_ascii_ply() -> None:
    dataset = open_tls(TEST_DATA / "tiny_mesh_ascii.ply")

    assert dataset.sizes == {"vertex": 4, "face": 2, "corner": 3}
    assert set(dataset.coords) == {"x", "y", "z"}
    np.testing.assert_allclose(dataset.z, [0.0, 0.0, 0.5, 0.5])
    np.testing.assert_array_equal(dataset.face_vertices, [[0, 1, 2], [0, 2, 3]])
    np.testing.assert_array_equal(dataset.red, [192, 192, 192, 192])
    assert dataset.red.dtype == np.dtype("u1")
    assert dataset.attrs["ply_format"] == "ascii"


def test_open_binary_ply_and_select_properties(tmp_path: Path) -> None:
    path = tmp_path / "points.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "end_header\n"
    ).encode("ascii")
    records = struct.pack("<dddBBdddBB", 1, 2, 3, 10, 11, 4, 5, 6, 20, 21)
    path.write_bytes(header + records)

    dataset = TLSProvider().open(path, properties=["red"])

    assert set(dataset.coords) == {"x", "y", "z"}
    assert set(dataset.data_vars) == {"red"}
    np.testing.assert_allclose(dataset.z, [3, 6])
    np.testing.assert_array_equal(dataset.red, [10, 20])
    assert dataset.attrs["ply_format"] == "binary_little_endian"


def test_open_binary_triangle_mesh(tmp_path: Path) -> None:
    path = tmp_path / "mesh.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 1\n"
        "property list int int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    vertices = struct.pack("<fffffffff", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    face = struct.pack("<iiii", 3, 0, 1, 2)
    path.write_bytes(header + vertices + face)

    dataset = open_tls(path, crs="EPSG:32630")

    assert dataset.sizes == {"vertex": 3, "face": 1, "corner": 3}
    np.testing.assert_array_equal(dataset.face_vertices, [[0, 1, 2]])
    assert dataset.attrs["geometry_type"] == "triangular_mesh"
    assert dataset.attrs["crs"] == "EPSG:32630"


def test_open_tree_segments() -> None:
    dataset = TLSProvider().open_trees(
        TEST_DATA / "tiny_trees.txt", crs="EPSG:32630"
    )

    assert dataset.sizes == {"segment": 5}
    np.testing.assert_array_equal(dataset.tree_id, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(dataset.segment_id, [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(dataset.parent_id, [-1, 0, 1, -1, 0])
    np.testing.assert_allclose(dataset.radius, [0.5, 0.4, 0.2, 0.45, 0.3])
    assert dataset.attrs["geometry_type"] == "tree_segments"


def test_reject_unknown_property(tmp_path: Path) -> None:
    path = tmp_path / "points.ply"
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "end_header\n"
        "1.0\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="Unknown PLY vertex properties: blue"):
        open_tls(path, properties=["blue"])
