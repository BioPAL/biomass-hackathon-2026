"""Helpers for assembling :class:`~tomo_canopy.types.DataCube` objects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from tomo_canopy.types import DataCube, DataLayer, GeometryType, GridSpec


@dataclass
class DataCubeBuilder:
    """Incrementally assemble a cube while preserving native geometries."""

    grid: GridSpec
    _layers: dict[str, DataLayer] = field(default_factory=dict, init=False)

    def add(self, layer: DataLayer, *, replace: bool = False) -> DataCubeBuilder:
        """Validate and stage one layer, returning the builder for chaining."""
        trial = DataCube(grid=self.grid, layers=dict(self._layers))
        trial.add_layer(layer, replace=replace)
        self._layers = dict(trial.layers)
        return self

    def build(self) -> DataCube:
        """Return an independent cube containing all staged layers."""
        return DataCube(grid=self.grid, layers=dict(self._layers))


def cube_from_layers(
    layers: Iterable[DataLayer], *, grid: GridSpec | None = None
) -> DataCube:
    """Build a cube, inferring its grid from the first volume when possible."""
    selected = tuple(layers)
    if grid is None:
        volume = next(
            (
                layer
                for layer in selected
                if layer.geometry == GeometryType.VOLUME.value
                and all(axis in layer.data.coords for axis in ("x", "y", "z"))
            ),
            None,
        )
        if volume is None or volume.crs is None:
            raise ValueError("grid is required when no georeferenced volume is present")
        grid = GridSpec(
            x=volume.data.x.values,
            y=volume.data.y.values,
            z=volume.data.z.values,
            crs=volume.crs,
        )
    builder = DataCubeBuilder(grid)
    for layer in selected:
        builder.add(layer)
    return builder.build()
