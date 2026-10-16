"""Processing operations for NASA LVIS Level-2 footprint metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from tomo_canopy.providers import LVISProvider
from tomo_canopy.types import DataLayer, GeometryType, Modality


@dataclass
class LVISProcessor:
    """Prepare LVIS footprints for analysis and multimodal integration."""

    provider: LVISProvider = field(default_factory=LVISProvider)

    def open(
        self,
        source: str | Path,
        *,
        variables: Iterable[str] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        max_shots: int | None = None,
    ) -> xr.Dataset:
        """Open a Level-2 TXT product and validate its footprint schema."""
        dataset = self.provider.open(
            source,
            variables=variables,
            bounds=bounds,
            max_shots=max_shots,
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
        """Validate the normalized LVIS Level-2 footprint contract."""
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("LVIS input must be an xarray.Dataset")
        if "shot" not in dataset.dims:
            raise ValueError("LVIS dataset is missing the 'shot' dimension")
        shot_count = dataset.sizes["shot"]
        if require_nonempty and shot_count == 0:
            raise ValueError("LVIS dataset contains no footprints")
        if require_crs and not dataset.attrs.get("crs"):
            raise ValueError("LVIS dataset has no CRS metadata")
        if dataset.attrs.get("modality") not in {None, Modality.LVIS.value}:
            raise ValueError("LVIS dataset has incompatible modality metadata")
        if dataset.attrs.get("geometry_type") not in {
            None,
            GeometryType.FOOTPRINTS.value,
        }:
            raise ValueError("LVIS dataset has incompatible geometry metadata")

        for coordinate in (
            "shot_number",
            "longitude",
            "latitude",
            "x",
            "y",
        ):
            if coordinate not in dataset.coords:
                raise ValueError(f"LVIS dataset is missing coordinate {coordinate!r}")
            if dataset[coordinate].dims != ("shot",):
                raise ValueError(
                    f"LVIS coordinate {coordinate!r} must use the 'shot' dimension"
                )
            if (
                check_finite
                and coordinate != "shot_number"
                and not bool(np.isfinite(dataset[coordinate]).all())
            ):
                raise ValueError(
                    f"LVIS coordinate {coordinate!r} contains non-finite values"
                )

        if "rh" in dataset:
            if "percentile" not in dataset.coords:
                raise ValueError("LVIS RH data is missing the 'percentile' coordinate")
            if dataset.rh.dims != ("shot", "percentile"):
                raise ValueError("LVIS 'rh' must use ('shot', 'percentile') dimensions")
            percentiles = np.asarray(dataset.percentile.data, dtype=float)
            if (
                percentiles.size == 0
                or not np.isfinite(percentiles).all()
                or (percentiles.size > 1 and not np.all(np.diff(percentiles) > 0))
            ):
                raise ValueError("LVIS percentiles must be finite and increasing")

        for name, variable in dataset.data_vars.items():
            if "shot" not in variable.dims:
                raise ValueError(
                    f"LVIS variable {name!r} does not use the 'shot' dimension"
                )

    def filter_quality(
        self,
        dataset: xr.Dataset,
        *,
        minimum_sensitivity: float | None = None,
        accepted_channel_rh: Iterable[int] | None = None,
        accepted_channel_ground: Iterable[int] | None = None,
        accepted_channel_top: Iterable[int] | None = None,
        require_finite_rh: bool = False,
    ) -> xr.Dataset:
        """Filter footprints using only explicitly requested quality criteria."""
        self.validate(dataset)
        keep = np.ones(dataset.sizes["shot"], dtype=bool)
        filters: dict[str, object] = {}

        if minimum_sensitivity is not None:
            if not np.isfinite(minimum_sensitivity):
                raise ValueError("minimum_sensitivity must be finite")
            _require_variable(dataset, "sensitivity")
            keep &= np.asarray(dataset.sensitivity.data) >= minimum_sensitivity
            filters["minimum_sensitivity"] = float(minimum_sensitivity)

        channel_filters = {
            "channel_rh": accepted_channel_rh,
            "channel_ground": accepted_channel_ground,
            "channel_top": accepted_channel_top,
        }
        for name, accepted in channel_filters.items():
            if accepted is None:
                continue
            values = tuple(dict.fromkeys(int(value) for value in accepted))
            if not values:
                raise ValueError(f"Accepted values for {name!r} cannot be empty")
            _require_variable(dataset, name)
            keep &= np.isin(np.asarray(dataset[name].data), values)
            filters[f"accepted_{name}"] = list(values)

        if require_finite_rh:
            _require_variable(dataset, "rh")
            keep &= np.isfinite(np.asarray(dataset.rh.data)).all(axis=1)
            filters["require_finite_rh"] = True

        result = dataset.isel(shot=np.flatnonzero(keep))
        result.attrs = dict(dataset.attrs)
        result.attrs.update(
            quality_filters=filters,
            source_shot_count=int(dataset.sizes["shot"]),
            retained_shot_count=int(keep.sum()),
        )
        return result

    def add_rh_metric(
        self,
        dataset: xr.Dataset,
        *,
        percentile: float = 98,
        name: str | None = None,
        method: str = "exact",
    ) -> xr.Dataset:
        """Add one RH percentile as a scalar footprint variable."""
        self.validate(dataset)
        _require_variable(dataset, "rh")
        if not np.isfinite(percentile):
            raise ValueError("percentile must be finite")
        if method not in {"exact", "nearest"}:
            raise ValueError("method must be 'exact' or 'nearest'")
        try:
            metric = dataset.rh.sel(
                percentile=float(percentile),
                method=None if method == "exact" else "nearest",
            )
        except KeyError as error:
            raise ValueError(
                f"LVIS percentile {percentile:g} is not available"
            ) from error
        selected = float(metric.percentile)
        metric = metric.drop_vars("percentile")
        output_name = name or f"rh{selected:g}"
        metric.name = output_name
        metric.attrs = {
            **dataset.rh.attrs,
            "profile_percentile": selected,
            "derived_from": "rh",
        }
        result = dataset.assign({output_name: metric})
        result.attrs = dict(dataset.attrs)
        return result

    def add_canopy_top(
        self,
        dataset: xr.Dataset,
        *,
        percentile: float = 98,
        name: str = "canopy_top_elevation",
        method: str = "exact",
    ) -> xr.Dataset:
        """Add ground elevation plus an RH percentile as canopy-top elevation."""
        self.validate(dataset)
        _require_variable(dataset, "ground_elevation")
        with_metric = self.add_rh_metric(
            dataset,
            percentile=percentile,
            name="_selected_rh",
            method=method,
        )
        selected = float(with_metric._selected_rh.attrs["profile_percentile"])
        canopy_top = with_metric.ground_elevation + with_metric._selected_rh
        canopy_top.name = name
        canopy_top.attrs = {
            "long_name": f"ground elevation plus RH{selected:g}",
            "units": dataset.ground_elevation.attrs.get("units", "m"),
            "derived_from": ["ground_elevation", "rh"],
            "profile_percentile": selected,
        }
        result = with_metric.drop_vars("_selected_rh").assign({name: canopy_top})
        result.attrs = dict(dataset.attrs)
        return result

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

    def sample_shots(
        self,
        dataset: xr.Dataset,
        *,
        count: int,
        seed: int | None = 0,
    ) -> xr.Dataset:
        """Select a reproducible subset of LVIS footprints."""
        self.validate(dataset)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        shot_count = dataset.sizes["shot"]
        sample_count = min(count, shot_count)
        if sample_count == shot_count:
            indices = np.arange(shot_count, dtype=np.int64)
        else:
            generator = np.random.default_rng(seed)
            indices = np.sort(
                generator.choice(shot_count, size=sample_count, replace=False)
            )
        result = dataset.isel(shot=indices)
        result.attrs = dict(dataset.attrs)
        result.attrs.update(sample_seed=seed, source_shot_count=shot_count)
        return result

    def to_data_layer(
        self,
        dataset: xr.Dataset,
        *,
        name: str = "lvis_footprints",
    ) -> DataLayer:
        """Wrap native LVIS footprints in the shared data interface."""
        self.validate(dataset, require_crs=True)
        return DataLayer(
            name=name,
            data=dataset,
            modality=Modality.LVIS,
            geometry=GeometryType.FOOTPRINTS,
        )


def _require_variable(dataset: xr.Dataset, name: str) -> None:
    if name not in dataset:
        raise ValueError(f"LVIS dataset has no {name!r} variable")
