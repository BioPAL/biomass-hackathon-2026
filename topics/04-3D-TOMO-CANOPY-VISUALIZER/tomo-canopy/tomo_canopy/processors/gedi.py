"""Processing operations for GEDI shot and profile datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol

import numpy as np
import xarray as xr

from tomo_canopy.providers import GEDIProvider
from tomo_canopy.types import DataLayer, GeometryType, Modality


class CoordinateTransformer(Protocol):
    """Minimal coordinate-transformer interface used for dependency injection."""

    def transform(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class GEDIProcessor:
    """Prepare GEDI shots for analysis and downstream cube generation."""

    provider: GEDIProvider = field(default_factory=GEDIProvider)

    def open(
        self,
        source: Any,
        *,
        variables: Iterable[str],
        start_time: str | None = None,
        end_time: str | None = None,
        quality_filters: Mapping[str, Any] | None = None,
    ) -> xr.Dataset:
        """Query a region through the provider and validate its shot schema."""
        dataset = self.provider.open(
            source,
            variables=variables,
            start_time=start_time,
            end_time=end_time,
            quality_filters=quality_filters,
        )
        self.validate(dataset)
        return dataset

    def validate(
        self,
        dataset: xr.Dataset,
        *,
        require_crs: bool = False,
        require_nonempty: bool = False,
        check_finite: bool = False,
    ) -> None:
        """Validate the normalized GEDI shot schema."""
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("GEDI input must be an xarray.Dataset")
        if "shot_number" not in dataset.dims:
            raise ValueError("GEDI dataset is missing the 'shot_number' dimension")

        shot_count = dataset.sizes["shot_number"]
        if require_nonempty and shot_count == 0:
            raise ValueError("GEDI dataset contains no shots")
        if require_crs and not dataset.attrs.get("crs"):
            raise ValueError("GEDI dataset has no CRS metadata")
        if shot_count == 0:
            return

        for coordinate in ("longitude", "latitude", "x", "y"):
            if coordinate not in dataset.coords:
                raise ValueError(f"GEDI dataset is missing coordinate {coordinate!r}")
            if dataset[coordinate].dims != ("shot_number",):
                raise ValueError(
                    f"GEDI coordinate {coordinate!r} must use 'shot_number'"
                )
            if check_finite and not bool(np.isfinite(dataset[coordinate]).all()):
                raise ValueError(
                    f"GEDI coordinate {coordinate!r} contains non-finite values"
                )

        longitude = np.asarray(dataset.longitude.data)
        latitude = np.asarray(dataset.latitude.data)
        if np.any((longitude < -180) | (longitude > 180)):
            raise ValueError("GEDI longitude values fall outside WGS84 bounds")
        if np.any((latitude < -90) | (latitude > 90)):
            raise ValueError("GEDI latitude values fall outside WGS84 bounds")
        for name, variable in dataset.data_vars.items():
            if "shot_number" not in variable.dims:
                raise ValueError(
                    f"GEDI variable {name!r} does not use the shot dimension"
                )

    def bounds(self, dataset: xr.Dataset) -> xr.Dataset:
        """Return the X/Y extent in the dataset's current CRS."""
        self.validate(dataset, require_nonempty=True, check_finite=True)
        result = xr.Dataset(
            data_vars={
                "minimum": (
                    "axis",
                    [float(dataset.x.min()), float(dataset.y.min())],
                ),
                "maximum": (
                    "axis",
                    [float(dataset.x.max()), float(dataset.y.max())],
                ),
            },
            coords={"axis": ["x", "y"]},
        )
        if "crs" in dataset.attrs:
            result.attrs["crs"] = dataset.attrs["crs"]
        return result

    def filter_valid(
        self,
        dataset: xr.Dataset,
        *,
        variables: Iterable[str] | None = None,
        how: str = "any",
    ) -> xr.Dataset:
        """Remove shots lacking requested measurements.

        ``how="any"`` removes a shot when any requested variable is entirely
        missing; ``how="all"`` removes it only when every requested variable is
        entirely missing. Profile variables are reduced across profile axes.
        """
        self.validate(dataset)
        if how not in {"any", "all"}:
            raise ValueError("how must be 'any' or 'all'")
        selected = (
            tuple(variables) if variables is not None else tuple(dataset.data_vars)
        )
        if not selected:
            raise ValueError("At least one variable is required for filtering")
        missing = sorted(set(selected).difference(dataset.data_vars))
        if missing:
            raise ValueError(f"Unknown GEDI variables: {', '.join(missing)}")

        valid_by_variable: list[np.ndarray] = []
        for name in selected:
            variable = dataset[name]
            valid = variable.notnull()
            profile_dimensions = [
                dimension for dimension in variable.dims if dimension != "shot_number"
            ]
            if profile_dimensions:
                valid = valid.any(profile_dimensions)
            valid_by_variable.append(np.asarray(valid.data, dtype=bool))

        stacked = np.stack(valid_by_variable)
        keep = stacked.all(axis=0) if how == "any" else stacked.any(axis=0)
        result = dataset.isel(shot_number=np.flatnonzero(keep))
        result.attrs = dict(dataset.attrs)
        result.attrs["validity_filter"] = how
        result.attrs["validity_variables"] = list(selected)
        return result

    def sample_shots(
        self,
        dataset: xr.Dataset,
        *,
        count: int,
        seed: int | None = 0,
    ) -> xr.Dataset:
        """Select a reproducible subset of GEDI shots."""
        self.validate(dataset)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        shot_count = dataset.sizes["shot_number"]
        sample_count = min(count, shot_count)
        if sample_count == shot_count:
            indices = np.arange(shot_count, dtype=np.int64)
        else:
            generator = np.random.default_rng(seed)
            indices = np.sort(
                generator.choice(shot_count, size=sample_count, replace=False)
            )
        result = dataset.isel(shot_number=indices)
        result.attrs = dict(dataset.attrs)
        result.attrs.update(
            sample_seed=seed,
            source_shot_count=shot_count,
        )
        return result

    def add_rh_metric(
        self,
        dataset: xr.Dataset,
        *,
        percentile: float = 98,
        variable: str = "rh",
        name: str | None = None,
    ) -> xr.Dataset:
        """Extract the nearest available RH percentile as a scalar shot variable."""
        self.validate(dataset)
        if variable not in dataset:
            raise ValueError(f"GEDI dataset has no {variable!r} variable")
        profile_dimensions = [
            dimension
            for dimension in dataset[variable].dims
            if dimension != "shot_number"
        ]
        if len(profile_dimensions) != 1:
            raise ValueError(f"{variable!r} must have exactly one profile dimension")
        if not np.isfinite(percentile):
            raise ValueError("percentile must be finite")

        profile_dimension = profile_dimensions[0]
        labels = np.asarray(dataset[profile_dimension].data, dtype=float)
        if labels.size == 0 or not np.isfinite(labels).all():
            raise ValueError(f"{variable!r} has invalid profile labels")
        label_index = int(np.abs(labels - percentile).argmin())
        selected_label = float(labels[label_index])
        output_name = name or f"{variable}{percentile:g}"
        metric = dataset[variable].isel({profile_dimension: label_index}, drop=True)
        metric = metric.copy(deep=False)
        metric.attrs = {
            **dataset[variable].attrs,
            "profile_percentile": selected_label,
            "derived_from": variable,
        }
        result = dataset.assign({output_name: metric})
        result.attrs = dict(dataset.attrs)
        return result

    def reproject(
        self,
        dataset: xr.Dataset,
        *,
        target_crs: str,
        transformer: CoordinateTransformer | None = None,
    ) -> xr.Dataset:
        """Transform X/Y shot coordinates into a projected target CRS."""
        self.validate(dataset, require_crs=True, check_finite=True)
        if not target_crs:
            raise ValueError("target_crs must be a non-empty CRS identifier")
        source_crs = str(dataset.attrs["crs"])
        if transformer is None:
            try:
                pyproj = import_module("pyproj")
            except ImportError as error:
                raise ImportError(
                    "GEDI reprojection requires pyproj; install the GEDI extra"
                ) from error
            transformer = pyproj.Transformer.from_crs(
                source_crs, target_crs, always_xy=True
            )

        transformed_x, transformed_y = transformer.transform(
            np.asarray(dataset.x.data), np.asarray(dataset.y.data)
        )
        if not np.isfinite(transformed_x).all() or not np.isfinite(
            transformed_y
        ).all():
            raise ValueError("Coordinate transformation produced non-finite values")
        result = dataset.assign_coords(
            x=("shot_number", transformed_x),
            y=("shot_number", transformed_y),
        )
        result.attrs = dict(dataset.attrs)
        result.attrs.update(source_crs=source_crs, crs=target_crs)
        return result

    def to_local_coordinates(
        self,
        dataset: xr.Dataset,
        *,
        origin: tuple[float, float] | None = None,
    ) -> xr.Dataset:
        """Translate X/Y coordinates while retaining the source CRS and origin."""
        self.validate(dataset, require_nonempty=True, check_finite=True)
        if dataset.attrs.get("coordinates_are_local"):
            raise ValueError("GEDI dataset already uses local coordinates")
        if origin is None:
            spatial_bounds = self.bounds(dataset)
            origin = tuple(float(value) for value in spatial_bounds.minimum.data)
        if len(origin) != 2 or not np.isfinite(origin).all():
            raise ValueError("origin must contain two finite coordinates")

        result = dataset.assign_coords(
            x=dataset.x - origin[0],
            y=dataset.y - origin[1],
        )
        result.attrs = dict(dataset.attrs)
        source_crs = result.attrs.pop("crs", None)
        if source_crs is not None:
            result.attrs["source_crs"] = source_crs
        result.attrs.update(
            coordinates_are_local=True,
            coordinate_origin_x=float(origin[0]),
            coordinate_origin_y=float(origin[1]),
        )
        return result

    def to_data_layer(
        self,
        dataset: xr.Dataset,
        *,
        name: str = "gedi_shots",
    ) -> DataLayer:
        """Wrap processed native shots in the shared data interface."""
        self.validate(dataset)
        return DataLayer(
            name=name,
            data=dataset,
            modality=Modality.GEDI,
            geometry=GeometryType.POINT_CLOUD,
        )
