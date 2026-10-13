"""Processing operations for terrestrial laser scanning datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from tomo_canopy.providers import TLSProvider

_GEOMETRY_DIMENSIONS = {
    "point_cloud": "point",
    "triangular_mesh": "vertex",
    "tree_segments": "segment",
}


@dataclass
class TLSProcessor:
    """Prepare provider datasets for analysis and downstream cube generation.

    The processor operates on Xarray datasets and delegates source-file I/O to
    :class:`~tomo_canopy.providers.TLSProvider`. Methods return new datasets and
    do not mutate their inputs.
    """

    provider: TLSProvider = field(default_factory=TLSProvider)

    def open(
        self,
        source: str | Path,
        *,
        crs: str | None = None,
        load: bool = False,
    ) -> xr.Dataset:
        """Open and structurally validate a supported TLS source."""
        dataset = self.provider.open(source, crs=crs, load=load)
        self.validate(dataset)
        return dataset

    def validate(
        self,
        dataset: xr.Dataset,
        *,
        require_crs: bool = False,
        check_finite: bool = False,
    ) -> None:
        """Validate the common TLS schema.

        ``check_finite=False`` avoids scanning large memory-mapped arrays during
        routine structural validation. Enable it before numerical processing
        when coordinates may contain invalid values.
        """
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("TLS input must be an xarray.Dataset")

        geometry_type = dataset.attrs.get("geometry_type")
        if geometry_type not in _GEOMETRY_DIMENSIONS:
            supported = ", ".join(sorted(_GEOMETRY_DIMENSIONS))
            raise ValueError(
                f"Unsupported TLS geometry_type {geometry_type!r}; expected {supported}"
            )

        geometry_dimension = _GEOMETRY_DIMENSIONS[geometry_type]
        if geometry_dimension not in dataset.dims:
            raise ValueError(
                f"TLS {geometry_type} is missing the {geometry_dimension!r} dimension"
            )
        if dataset.sizes[geometry_dimension] == 0:
            raise ValueError(f"TLS {geometry_type} contains no geometry elements")
        for coordinate in ("x", "y", "z"):
            if coordinate not in dataset.coords:
                raise ValueError(f"TLS dataset is missing coordinate {coordinate!r}")
            if dataset[coordinate].dims != (geometry_dimension,):
                raise ValueError(
                    f"TLS coordinate {coordinate!r} must use only the "
                    f"{geometry_dimension!r} dimension"
                )
            if check_finite and not bool(np.isfinite(dataset[coordinate]).all()):
                raise ValueError(
                    f"TLS coordinate {coordinate!r} contains non-finite values"
                )

        if geometry_type == "triangular_mesh":
            self._validate_mesh(dataset)
        elif geometry_type == "tree_segments":
            self._validate_trees(dataset)

        if require_crs and not dataset.attrs.get("crs"):
            raise ValueError("TLS dataset has no CRS metadata")

    def bounds(self, dataset: xr.Dataset) -> xr.Dataset:
        """Return minimum and maximum spatial coordinates."""
        self.validate(dataset, check_finite=True)
        result = xr.Dataset(
            data_vars={
                "minimum": (
                    "axis",
                    [float(dataset[axis].min()) for axis in ("x", "y", "z")],
                ),
                "maximum": (
                    "axis",
                    [float(dataset[axis].max()) for axis in ("x", "y", "z")],
                ),
            },
            coords={"axis": ["x", "y", "z"]},
            attrs={"geometry_type": dataset.attrs["geometry_type"]},
        )
        if "crs" in dataset.attrs:
            result.attrs["crs"] = dataset.attrs["crs"]
        return result

    def to_local_coordinates(
        self,
        dataset: xr.Dataset,
        *,
        origin: tuple[float, float, float] | None = None,
    ) -> xr.Dataset:
        """Translate spatial coordinates to a local origin.

        If no origin is supplied, the minimum coordinate on each axis is used.
        The source CRS and translation are retained as metadata so global
        coordinates can be reconstructed.
        """
        self.validate(dataset, check_finite=True)
        if dataset.attrs.get("coordinates_are_local"):
            raise ValueError("TLS dataset already uses local coordinates")

        if origin is None:
            spatial_bounds = self.bounds(dataset)
            origin = tuple(float(value) for value in spatial_bounds.minimum.data)
        if len(origin) != 3 or not np.all(np.isfinite(origin)):
            raise ValueError("origin must contain three finite coordinates")

        result = dataset.assign_coords(
            x=dataset.x - origin[0],
            y=dataset.y - origin[1],
            z=dataset.z - origin[2],
        )
        result.attrs = dict(dataset.attrs)
        source_crs = result.attrs.pop("crs", None)
        if source_crs is not None:
            result.attrs["source_crs"] = source_crs
        result.attrs.update(
            {
                "coordinates_are_local": True,
                "coordinate_origin_x": float(origin[0]),
                "coordinate_origin_y": float(origin[1]),
                "coordinate_origin_z": float(origin[2]),
            }
        )
        return result

    def sample_vertices(
        self,
        dataset: xr.Dataset,
        *,
        count: int,
        seed: int | None = 0,
    ) -> xr.Dataset:
        """Select a reproducible subset of mesh vertices or point-cloud points.

        Sampling mesh vertices produces a point cloud; face connectivity is
        intentionally removed because a vertex subset is no longer a valid
        representation of the original surface.
        """
        self.validate(dataset)
        geometry_type = dataset.attrs["geometry_type"]
        if geometry_type not in {"triangular_mesh", "point_cloud"}:
            raise ValueError("Vertex sampling requires a mesh or point cloud")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")

        source_dimension = _GEOMETRY_DIMENSIONS[geometry_type]
        source_size = dataset.sizes[source_dimension]
        sample_count = min(count, source_size)
        if sample_count == source_size:
            indices = np.arange(source_size, dtype=np.int64)
        else:
            generator = np.random.default_rng(seed)
            indices = np.sort(
                generator.choice(source_size, size=sample_count, replace=False)
            )

        points = dataset
        if "face_vertices" in points:
            points = points.drop_vars("face_vertices")
        points = points.isel({source_dimension: indices})
        if source_dimension != "point":
            points = points.rename({source_dimension: "point"})
        source_id_name = (
            "source_vertex_id"
            if geometry_type == "triangular_mesh"
            else "source_point_id"
        )
        points = points.assign_coords({source_id_name: ("point", indices)})
        points.attrs = dict(dataset.attrs)
        points.attrs.update(
            {
                "geometry_type": "point_cloud",
                "sampled_from_geometry": geometry_type,
                "sample_seed": seed,
                "source_element_count": source_size,
            }
        )
        return points

    def tree_metrics(self, dataset: xr.Dataset) -> xr.Dataset:
        """Calculate basic structural metrics for every reconstructed tree."""
        self.validate(dataset, check_finite=True)
        if dataset.attrs["geometry_type"] != "tree_segments":
            raise ValueError("Tree metrics require tree-segment geometry")

        tree_ids = np.asarray(dataset.tree_id.data)
        unique_ids, inverse, counts = np.unique(
            tree_ids, return_inverse=True, return_counts=True
        )
        metrics: dict[str, np.ndarray] = {}
        for axis in ("x", "y", "z"):
            values = np.asarray(dataset[axis].data)
            minimum = np.full(len(unique_ids), np.inf)
            maximum = np.full(len(unique_ids), -np.inf)
            np.minimum.at(minimum, inverse, values)
            np.maximum.at(maximum, inverse, values)
            metrics[f"{axis}_minimum"] = minimum
            metrics[f"{axis}_maximum"] = maximum

        radius = np.asarray(dataset.radius.data)
        maximum_radius = np.full(len(unique_ids), -np.inf)
        np.maximum.at(maximum_radius, inverse, radius)

        result = xr.Dataset(
            data_vars={
                "segment_count": ("tree", counts.astype(np.int64)),
                "height": (
                    "tree",
                    metrics["z_maximum"] - metrics["z_minimum"],
                ),
                "maximum_radius": ("tree", maximum_radius),
                **{name: ("tree", values) for name, values in metrics.items()},
            },
            coords={"tree_id": ("tree", unique_ids)},
            attrs={"source_geometry_type": "tree_segments"},
        )
        for attribute in ("source", "crs", "source_crs"):
            if attribute in dataset.attrs:
                result.attrs[attribute] = dataset.attrs[attribute]
        return result

    @staticmethod
    def _validate_mesh(dataset: xr.Dataset) -> None:
        if "face_vertices" not in dataset:
            raise ValueError("Triangular mesh is missing 'face_vertices'")
        if dataset.face_vertices.dims != ("face", "corner"):
            raise ValueError("'face_vertices' must use ('face', 'corner') dimensions")
        if dataset.sizes["corner"] != 3:
            raise ValueError("TLS meshes must contain triangular faces")
        if not np.issubdtype(dataset.face_vertices.dtype, np.integer):
            raise ValueError("'face_vertices' must contain integer indices")

    @staticmethod
    def _validate_trees(dataset: xr.Dataset) -> None:
        required = {"radius", "parent_id", "section_id", "tree_id", "segment_id"}
        missing = sorted(required.difference(dataset.data_vars))
        if missing:
            raise ValueError(f"Tree dataset is missing variables: {', '.join(missing)}")
        for name in required:
            if dataset[name].dims != ("segment",):
                raise ValueError(
                    f"Tree variable {name!r} must use the 'segment' dimension"
                )
