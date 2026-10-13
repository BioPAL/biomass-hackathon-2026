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

    A layer may still use its sensor-native geometry. Only layers registered in
    a :class:`DataCube` must be aligned with that cube's common grid.
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
    """Collection of zero or more layers aligned to one spatial grid.

    Modalities are deliberately optional. A cube can contain TLS alone, any
    other subset of sources, or initially no layers at all.
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
        """Register a present modality layer after grid and CRS validation."""
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
        self.grid.validate_dataset(layer.data)
        self.layers[layer.name] = layer

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

        Variables are prefixed with their layer name, avoiding collisions when
        different modalities expose variables with the same name.
        """
        data_variables: dict[str, xr.DataArray] = {}
        for layer_name, layer in self.layers.items():
            for variable_name, variable in layer.data.data_vars.items():
                output_name = f"{layer_name}__{variable_name}"
                output = variable.copy(deep=False)
                output.attrs = {
                    **variable.attrs,
                    "layer": layer_name,
                    "modality": layer.modality,
                    "geometry": layer.geometry,
                }
                data_variables[output_name] = output

        return xr.Dataset(
            data_vars=data_variables,
            coords=self.grid.coordinates,
            attrs={
                "crs": self.grid.crs,
                "modalities": sorted(self.modalities),
            },
        )
