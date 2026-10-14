"""Shared data contracts for providers, processors, cubes, and visualisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TypeAlias

import numpy as np
import xarray as xr

Timestamp: TypeAlias = str | datetime | np.datetime64


class Modality(str, Enum):
    """Known observation modalities."""

    TLS = "tls"
    ALS = "als"
    GEDI = "gedi"
    TOMOGRAM = "tomogram"
    OTHER = "other"


class GeometryType(str, Enum):
    """Spatial representation carried by a data layer."""

    POINT_CLOUD = "point_cloud"
    TRIANGULAR_MESH = "triangular_mesh"
    TREE_SEGMENTS = "tree_segments"
    RASTER = "raster"
    VOLUME = "volume"
    WAVEFORM = "waveform"


def _label(value: str | Enum) -> str:
    label = value.value if isinstance(value, Enum) else value
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Interface labels must be non-empty strings")
    return label.strip()


@dataclass(frozen=True)
class DataLayer:
    """A named, self-describing dataset from one observation modality.

    A layer may use sensor-native geometry or a rectilinear grid. Layers in a
    :class:`DataCube` share a CRS, while only raster and volume layers must be
    aligned with the cube's common grid.
    """

    name: str
    data: xr.Dataset
    modality: Modality | str
    geometry: GeometryType | str
    crs: str | None = None
    timestamp: Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _label(self.name))
        object.__setattr__(self, "modality", _label(self.modality))
        object.__setattr__(self, "geometry", _label(self.geometry))
        if not isinstance(self.data, xr.Dataset):
            raise TypeError("DataLayer.data must be an xarray.Dataset")

        dataset_crs = self.data.attrs.get("crs")
        if self.crs is not None:
            normalised_crs = _label(self.crs)
            if dataset_crs is not None and str(dataset_crs) != normalised_crs:
                raise ValueError(
                    f"Layer CRS {normalised_crs!r} conflicts with dataset CRS "
                    f"{dataset_crs!r}"
                )
            object.__setattr__(self, "crs", normalised_crs)
        elif dataset_crs is not None:
            object.__setattr__(self, "crs", _label(str(dataset_crs)))


@dataclass(frozen=True)
class GridSpec:
    """Coordinates and CRS of a shared rectilinear visualization grid."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    crs: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "crs", _label(self.crs))
        for axis in ("x", "y", "z"):
            values = np.asarray(getattr(self, axis), dtype=np.float64)
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"Grid axis {axis!r} must be non-empty and 1D")
            if not np.isfinite(values).all():
                raise ValueError(f"Grid axis {axis!r} contains non-finite values")
            if values.size > 1 and not np.all(np.diff(values) > 0):
                raise ValueError(f"Grid axis {axis!r} must be strictly increasing")
            values = values.copy()
            values.flags.writeable = False
            object.__setattr__(self, axis, values)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape in Xarray's conventional ``(z, y, x)`` order."""
        return (self.z.size, self.y.size, self.x.size)

    @property
    def coordinates(self) -> dict[str, np.ndarray]:
        """Return coordinate arrays suitable for an Xarray constructor."""
        return {"x": self.x, "y": self.y, "z": self.z}

    def validate_dataset(self, dataset: xr.Dataset) -> None:
        """Check that a gridded dataset uses a compatible subset of axes."""
        present_axes = [axis for axis in ("x", "y", "z") if axis in dataset.coords]
        if not present_axes:
            raise ValueError("Cube layers must contain at least one spatial grid axis")

        for axis in present_axes:
            coordinate = dataset.coords[axis]
            if coordinate.dims != (axis,):
                raise ValueError(
                    f"Grid coordinate {axis!r} must use the ({axis!r},) dimension"
                )
            expected = getattr(self, axis)
            actual = np.asarray(coordinate.data)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f"Layer coordinate {axis!r} does not match the cube grid"
                )


@dataclass
class DataCube:
    """Collection of native and gridded layers in one spatial reference.

    Modalities are deliberately optional. A cube can contain TLS alone, any
    other subset of sources, or initially no layers at all. Raster and volume
    layers must align with ``grid``; point clouds, meshes, tree segments, and
    waveforms retain their native geometry and only need to share its CRS.
    """

    grid: GridSpec
    layers: dict[str, DataLayer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, GridSpec):
            raise TypeError("DataCube.grid must be a GridSpec")
        initial_layers = dict(self.layers)
        self.layers = {}
        for name, layer in initial_layers.items():
            if name != layer.name:
                raise ValueError(
                    f"Layer mapping key {name!r} does not match layer name "
                    f"{layer.name!r}"
                )
            self.add_layer(layer)

    def add_layer(self, layer: DataLayer, *, replace: bool = False) -> None:
        """Register a layer after geometry-aware spatial validation."""
        if not isinstance(layer, DataLayer):
            raise TypeError("layer must be a DataLayer")
        if layer.name in self.layers and not replace:
            raise ValueError(f"Layer {layer.name!r} is already present")
        if layer.crs is None:
            raise ValueError(f"Layer {layer.name!r} has no CRS metadata")
        if layer.crs != self.grid.crs:
            raise ValueError(
                f"Layer CRS {layer.crs!r} does not match grid CRS {self.grid.crs!r}"
            )
        self._validate_geometry(layer)
        self.layers[layer.name] = layer

    def _validate_geometry(self, layer: DataLayer) -> None:
        geometry = layer.geometry
        if geometry == GeometryType.RASTER.value:
            _require_grid_axes(layer.data, ("x", "y"))
            self.grid.validate_dataset(layer.data)
        elif geometry == GeometryType.VOLUME.value:
            _require_grid_axes(layer.data, ("x", "y", "z"))
            self.grid.validate_dataset(layer.data)
        elif geometry == GeometryType.POINT_CLOUD.value:
            _validate_point_geometry(layer.data)
        elif geometry == GeometryType.TRIANGULAR_MESH.value:
            _validate_mesh_geometry(layer.data)
        elif geometry == GeometryType.TREE_SEGMENTS.value:
            _validate_shared_coordinates(layer.data, ("x", "y", "z"), "segment")
        elif geometry == GeometryType.WAVEFORM.value:
            _validate_point_geometry(layer.data)
        else:
            supported = ", ".join(member.value for member in GeometryType)
            raise ValueError(
                f"Unsupported layer geometry {geometry!r}; expected {supported}"
            )

    def remove_layer(self, name: str) -> DataLayer:
        """Remove and return a layer."""
        try:
            return self.layers.pop(name)
        except KeyError as error:
            raise KeyError(f"Layer {name!r} is not present") from error

    def has_modality(self, modality: Modality | str) -> bool:
        """Return whether at least one layer exists for *modality*."""
        expected = _label(modality)
        return any(layer.modality == expected for layer in self.layers.values())

    def layers_for(self, modality: Modality | str) -> tuple[DataLayer, ...]:
        """Return all layers belonging to a modality."""
        expected = _label(modality)
        return tuple(
            layer for layer in self.layers.values() if layer.modality == expected
        )

    @property
    def modalities(self) -> frozenset[str]:
        """Modalities currently represented in the cube."""
        return frozenset(layer.modality for layer in self.layers.values())

    def missing_modalities(
        self, expected: set[Modality | str] | frozenset[Modality | str]
    ) -> frozenset[str]:
        """Return expected modalities for which no layer is present."""
        expected_labels = frozenset(_label(modality) for modality in expected)
        return expected_labels.difference(self.modalities)

    def require_modalities(self, *modalities: Modality | str) -> None:
        """Raise a clear error if an operation requires unavailable modalities."""
        missing = self.missing_modalities(frozenset(modalities))
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"Missing required modalities: {missing_names}")

    def to_xarray(self) -> xr.Dataset:
        """Combine available layers into one visualization-ready dataset.

        Variables are prefixed with their layer name. Native dimensions and
        coordinates are prefixed as well, preventing their irregular ``x``,
        ``y``, and ``z`` coordinates from colliding with the common grid axes.
        """
        datasets: list[xr.Dataset] = [xr.Dataset(coords=self.grid.coordinates)]
        for layer_name, layer in self.layers.items():
            native_geometry = layer.geometry not in {
                GeometryType.RASTER.value,
                GeometryType.VOLUME.value,
            }
            rename: dict[str, str] = {
                name: f"{layer_name}__{name}" for name in layer.data.data_vars
            }
            if native_geometry:
                rename.update(
                    {
                        name: f"{layer_name}__{name}"
                        for name in (*layer.data.dims, *layer.data.coords)
                        if name not in rename
                    }
                )
            else:
                rename.update(
                    {
                        name: f"{layer_name}__{name}"
                        for name in layer.data.dims
                        if name not in {"x", "y", "z"} and name not in rename
                    }
                )

            output = layer.data.rename(rename).copy(deep=False)
            for variable in output.data_vars.values():
                variable.attrs = {
                    **variable.attrs,
                    "layer": layer_name,
                    "modality": layer.modality,
                    "geometry": layer.geometry,
                }
            datasets.append(output)

        result = xr.merge(datasets, compat="no_conflicts", join="exact")
        result.attrs = {
            "crs": self.grid.crs,
            "modalities": sorted(self.modalities),
        }
        return result


def _require_grid_axes(dataset: xr.Dataset, axes: tuple[str, ...]) -> None:
    missing = [axis for axis in axes if axis not in dataset.coords]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Gridded layer is missing coordinate(s): {names}")


def _validate_shared_coordinates(
    dataset: xr.Dataset,
    coordinates: tuple[str, ...],
    expected_dimension: str | None = None,
) -> str:
    missing = [name for name in coordinates if name not in dataset.coords]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Native layer is missing coordinate(s): {names}")

    first = dataset.coords[coordinates[0]]
    if first.ndim != 1 or first.size == 0:
        raise ValueError("Native spatial coordinates must be non-empty and 1D")
    dimension = first.dims[0]
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"Native coordinates must use the {expected_dimension!r} dimension"
        )
    for name in coordinates:
        coordinate = dataset.coords[name]
        if coordinate.dims != (dimension,):
            raise ValueError(
                f"Native coordinate {name!r} must use the ({dimension!r},) dimension"
            )
        if not np.issubdtype(coordinate.dtype, np.number):
            raise ValueError(f"Native coordinate {name!r} must be numeric")
    return dimension


def _validate_point_geometry(dataset: xr.Dataset) -> None:
    coordinates = ("x", "y", "z") if "z" in dataset.coords else ("x", "y")
    _validate_shared_coordinates(dataset, coordinates)


def _validate_mesh_geometry(dataset: xr.Dataset) -> None:
    vertex_dimension = _validate_shared_coordinates(
        dataset, ("x", "y", "z"), "vertex"
    )
    if "face_vertices" not in dataset:
        raise ValueError("Triangular mesh is missing 'face_vertices'")
    faces = dataset["face_vertices"]
    if faces.dims != ("face", "corner"):
        raise ValueError("'face_vertices' must use ('face', 'corner') dimensions")
    if faces.sizes["corner"] != 3:
        raise ValueError("Triangular meshes must contain three indices per face")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("'face_vertices' must contain integer indices")
    if dataset.sizes[vertex_dimension] == 0:
        raise ValueError("Triangular mesh contains no vertices")
