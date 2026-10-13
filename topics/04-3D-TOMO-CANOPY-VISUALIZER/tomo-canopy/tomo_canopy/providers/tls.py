"""Terrestrial laser scanning (TLS) data providers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import xarray as xr

_PLY_TYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass(frozen=True)
class _PlyHeader:
    """Information needed to read the vertex element of a PLY file."""

    data_format: str
    vertex_count: int
    vertex_properties: tuple[tuple[str, str], ...]
    face_count: int
    face_property: tuple[str, str, str] | None
    element_order: tuple[str, ...]
    data_offset: int


class TLSProvider:
    """Open TLS point clouds as :class:`xarray.Dataset` objects.

    PLY vertex properties become one-dimensional variables over the ``point``
    dimension. Properties named ``x``, ``y``, and ``z`` are represented as
    coordinates. Binary PLY vertex blocks are memory-mapped by default, which
    keeps large point clouds lazy and avoids an eager copy into memory.

    Mesh faces are intentionally not exposed: this provider models the TLS
    vertices as a point cloud.
    """

    def open(
        self,
        source: str | Path,
        *,
        properties: Iterable[str] | None = None,
        load: bool = False,
        crs: str | None = None,
    ) -> xr.Dataset:
        """Open a supported TLS file.

        Parameters
        ----------
        source:
            Path to an ASCII/binary PLY file or a tree-structure TXT file.
        properties:
            Optional vertex properties to include. Spatial coordinates present
            in the file are always included.
        load:
            Load all arrays into memory. By default, binary data remains backed
            by a NumPy memory map.
        crs:
            Coordinate reference system identifier, such as ``"EPSG:32630"``.
            PLY and tree text files do not carry this metadata themselves.
        """
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".ply":
            return self.open_ply(
                path, properties=properties, load=load, crs=crs
            )
        if path.suffix.lower() == ".txt":
            if properties is not None:
                raise ValueError("Property selection is not supported for tree files")
            return self.open_trees(path, load=load, crs=crs)
        raise ValueError(f"Expected a .ply or .txt file, got {path}")

    def open_ply(
        self,
        source: str | Path,
        *,
        properties: Iterable[str] | None = None,
        load: bool = False,
        crs: str | None = None,
    ) -> xr.Dataset:
        """Open PLY vertices and triangular face connectivity."""
        path = Path(source).expanduser()
        if path.suffix.lower() != ".ply":
            raise ValueError(f"Expected a .ply file, got {path}")
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as stream:
            header = _read_ply_header(stream)

        data = _read_vertices(path, header)
        faces = _read_faces(path, header) if header.face_count else None
        available = tuple(name for name, _ in header.vertex_properties)
        selected = _select_properties(available, properties)
        point_dimension = "vertex" if faces is not None else "point"

        coords: dict[str, tuple[str, np.ndarray]] = {}
        data_vars: dict[str, tuple[str, np.ndarray]] = {}
        for name in selected:
            target = coords if name in {"x", "y", "z"} else data_vars
            target[name] = (point_dimension, data[name])
        if faces is not None:
            data_vars["face_vertices"] = (("face", "corner"), faces)

        geometry_type = "triangular_mesh" if faces is not None else "point_cloud"
        attributes = {
            "source": str(path.resolve()),
            "source_format": "PLY",
            "ply_format": header.data_format,
            "geometry_type": geometry_type,
        }
        if crs is not None:
            attributes["crs"] = crs
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attributes,
        )
        return dataset.load() if load else dataset

    def open_trees(
        self,
        source: str | Path,
        *,
        load: bool = False,
        crs: str | None = None,
    ) -> xr.Dataset:
        """Open a segment-based TLS tree model from its text representation."""
        path = Path(source).expanduser()
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Expected a .txt tree file, got {path}")
        if not path.is_file():
            raise FileNotFoundError(path)

        arrays = _read_tree_segments(path)
        attributes = {
            "source": str(path.resolve()),
            "source_format": "TLS tree text",
            "geometry_type": "tree_segments",
        }
        if crs is not None:
            attributes["crs"] = crs
        dataset = xr.Dataset(
            data_vars={
                "radius": ("segment", arrays["radius"]),
                "parent_id": ("segment", arrays["parent_id"]),
                "section_id": ("segment", arrays["section_id"]),
                "tree_id": ("segment", arrays["tree_id"]),
                "segment_id": ("segment", arrays["segment_id"]),
            },
            coords={
                name: ("segment", arrays[name]) for name in ("x", "y", "z")
            },
            attrs=attributes,
        )
        return dataset.load() if load else dataset


def open_tls(
    source: str | Path,
    *,
    properties: Iterable[str] | None = None,
    load: bool = False,
    crs: str | None = None,
) -> xr.Dataset:
    """Open a supported TLS file as an :class:`xarray.Dataset`."""
    return TLSProvider().open(
        source, properties=properties, load=load, crs=crs
    )


def _read_ply_header(stream: BinaryIO) -> _PlyHeader:
    first_line = stream.readline()
    if first_line.rstrip(b"\r\n") != b"ply":
        raise ValueError("Not a PLY file: missing 'ply' signature")

    data_format: str | None = None
    current_element: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    face_count = 0
    face_property: tuple[str, str, str] | None = None
    element_order: list[str] = []

    while True:
        raw_line = stream.readline()
        if not raw_line:
            raise ValueError("Invalid PLY file: missing end_header")
        try:
            line = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("Invalid PLY file: header is not ASCII") from error

        fields = line.split()
        if not fields or fields[0] in {"comment", "obj_info"}:
            continue
        if fields[0] == "format":
            if len(fields) != 3 or fields[2] != "1.0":
                raise ValueError(f"Unsupported PLY format declaration: {line}")
            data_format = fields[1]
        elif fields[0] == "element":
            if len(fields) != 3:
                raise ValueError(f"Invalid PLY element declaration: {line}")
            current_element = fields[1]
            element_order.append(current_element)
            if current_element == "vertex":
                vertex_count = int(fields[2])
            elif current_element == "face":
                face_count = int(fields[2])
        elif fields[0] == "property" and current_element == "vertex":
            if len(fields) != 3:
                raise ValueError("List properties on PLY vertices are not supported")
            type_name, name = fields[1], fields[2]
            if type_name not in _PLY_TYPES:
                raise ValueError(f"Unsupported PLY property type: {type_name}")
            vertex_properties.append((name, type_name))
        elif fields[0] == "property" and current_element == "face":
            if len(fields) != 5 or fields[1] != "list":
                raise ValueError("Only list properties on PLY faces are supported")
            count_type, index_type, name = fields[2], fields[3], fields[4]
            if count_type not in _PLY_TYPES or index_type not in _PLY_TYPES:
                raise ValueError(f"Unsupported PLY face property types: {line}")
            if face_property is not None:
                raise ValueError("Multiple PLY face properties are not supported")
            face_property = (name, count_type, index_type)
        elif fields[0] == "end_header":
            break

    supported_formats = {"ascii", "binary_little_endian", "binary_big_endian"}
    if data_format not in supported_formats:
        raise ValueError(f"Unsupported PLY data format: {data_format}")
    if vertex_count is None:
        raise ValueError("PLY file does not declare a vertex element")
    if not vertex_properties:
        raise ValueError("PLY vertex element has no scalar properties")
    if face_count and face_property is None:
        raise ValueError("PLY face element has no index-list property")
    if face_count:
        vertex_position = element_order.index("vertex")
        face_position = element_order.index("face")
        if face_position != vertex_position + 1:
            raise ValueError("PLY face element must immediately follow vertices")

    return _PlyHeader(
        data_format=data_format,
        vertex_count=vertex_count,
        vertex_properties=tuple(vertex_properties),
        face_count=face_count,
        face_property=face_property,
        element_order=tuple(element_order),
        data_offset=stream.tell(),
    )


def _vertex_dtype(header: _PlyHeader) -> np.dtype:
    byte_order = {
        "ascii": "=",
        "binary_little_endian": "<",
        "binary_big_endian": ">",
    }[header.data_format]
    return np.dtype(
        [
            (name, byte_order + _PLY_TYPES[type_name])
            for name, type_name in header.vertex_properties
        ]
    )


def _read_vertices(path: Path, header: _PlyHeader) -> np.ndarray:
    dtype = _vertex_dtype(header)
    if header.data_format == "ascii":
        with path.open("rb") as stream:
            stream.seek(header.data_offset)
            vertices = np.loadtxt(stream, dtype=dtype, max_rows=header.vertex_count)
        return np.atleast_1d(vertices)

    expected_end = header.data_offset + header.vertex_count * dtype.itemsize
    if path.stat().st_size < expected_end:
        raise ValueError("PLY file is truncated inside the vertex data")
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=header.data_offset,
        shape=(header.vertex_count,),
    )


def _read_faces(path: Path, header: _PlyHeader) -> np.ndarray:
    """Read triangular face connectivity from a PLY file."""
    if header.face_property is None:
        raise ValueError("PLY file does not define face connectivity")
    _, count_type, index_type = header.face_property
    index_dtype = np.dtype(_PLY_TYPES[index_type])

    if header.data_format == "ascii":
        return _read_ascii_faces(path, header, index_dtype)

    byte_order = "<" if header.data_format == "binary_little_endian" else ">"
    count_dtype = np.dtype(byte_order + _PLY_TYPES[count_type])
    index_dtype = np.dtype(byte_order + _PLY_TYPES[index_type])
    face_dtype = np.dtype(
        [("_vertex_count", count_dtype), ("indices", index_dtype, (3,))]
    )
    face_offset = header.data_offset + header.vertex_count * _vertex_dtype(
        header
    ).itemsize
    expected_end = face_offset + header.face_count * face_dtype.itemsize
    if path.stat().st_size < expected_end:
        raise ValueError("PLY file is truncated inside the face data")
    records = np.memmap(
        path,
        mode="r",
        dtype=face_dtype,
        offset=face_offset,
        shape=(header.face_count,),
    )
    if not np.all(records["_vertex_count"] == 3):
        raise ValueError("Only triangular PLY faces are currently supported")
    return records["indices"]


def _read_ascii_faces(
    path: Path, header: _PlyHeader, index_dtype: np.dtype
) -> np.ndarray:
    faces = np.empty((header.face_count, 3), dtype=index_dtype)
    with path.open("rb") as stream:
        stream.seek(header.data_offset)
        for _ in range(header.vertex_count):
            if not stream.readline():
                raise ValueError("PLY file is truncated inside the vertex data")
        for face_index in range(header.face_count):
            fields = stream.readline().split()
            if not fields:
                raise ValueError("PLY file is truncated inside the face data")
            if int(fields[0]) != 3 or len(fields) != 4:
                raise ValueError("Only triangular PLY faces are currently supported")
            faces[face_index] = fields[1:]
    return faces


def _read_tree_segments(path: Path) -> dict[str, np.ndarray]:
    segment_count = 0
    tree_count = 0
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().strip()
        columns = stream.readline().replace(" ", "").strip()
        if not header.startswith("# Tree file"):
            raise ValueError("Unrecognised TLS tree text header")
        if columns != "x,y,z,radius,parent_id,section_id":
            raise ValueError(f"Unsupported TLS tree columns: {columns}")
        for line in stream:
            value_count = line.count(",") + 1
            if value_count % 6:
                raise ValueError(f"Tree {tree_count} has an incomplete segment")
            segment_count += value_count // 6
            tree_count += 1

    arrays = {
        name: np.empty(segment_count, dtype=dtype)
        for name, dtype in {
            "x": "f8",
            "y": "f8",
            "z": "f8",
            "radius": "f8",
            "parent_id": "i8",
            "section_id": "i8",
            "tree_id": "i4",
            "segment_id": "i4",
        }.items()
    }
    position = 0
    with path.open(encoding="utf-8") as stream:
        next(stream)
        next(stream)
        for tree_id, line in enumerate(stream):
            values = np.fromstring(line, sep=",").reshape(-1, 6)
            count = len(values)
            selection = slice(position, position + count)
            for column_index, name in enumerate(
                ("x", "y", "z", "radius", "parent_id", "section_id")
            ):
                arrays[name][selection] = values[:, column_index]
            arrays["tree_id"][selection] = tree_id
            arrays["segment_id"][selection] = np.arange(count, dtype="i4")
            position += count
    return arrays


def _select_properties(
    available: tuple[str, ...], requested: Iterable[str] | None
) -> tuple[str, ...]:
    if requested is None:
        return available

    requested_names = tuple(dict.fromkeys(requested))
    unknown = sorted(set(requested_names).difference(available))
    if unknown:
        raise ValueError(f"Unknown PLY vertex properties: {', '.join(unknown)}")

    coordinates = tuple(name for name in ("x", "y", "z") if name in available)
    return tuple(dict.fromkeys((*coordinates, *requested_names)))

