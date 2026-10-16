"""Processing operations for BIOMASS tomographic datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from tomo_canopy.providers import TomogramProvider
from tomo_canopy.types import DataLayer, GeometryType, Modality


@dataclass
class TomogramProcessor:
    """Prepare native TOM L1D or geocoded TOM L2A datasets for analysis."""

    provider: TomogramProvider = field(default_factory=TomogramProvider)

    def open(
        self,
        source: str | Path,
        *,
        expected_level: str | None = None,
        polarization: str | Iterable[str] | None = None,
        height_range: tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        chunks: str | Mapping[str, int] | None = None,
        include_auxiliary: bool = False,
        load: bool = False,
    ) -> xr.Dataset:
        """Open a product through the provider and validate its normalized schema."""
        dataset = self.provider.open(
            source,
            expected_level=expected_level,
            polarization=polarization,
            height_range=height_range,
            bounds=bounds,
            chunks=chunks,
            include_auxiliary=include_auxiliary,
            load=load,
        )
        self.validate(dataset)
        return dataset

    def validate(
        self,
        dataset: xr.Dataset,
        *,
        require_geocoded: bool = False,
        check_finite_coordinates: bool = False,
    ) -> None:
        """Validate the provider's normalized L1D/L2A tomogram contract."""
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("Tomogram input must be an xarray.Dataset")
        if "tomographic_cube" not in dataset:
            raise ValueError("Tomogram dataset is missing 'tomographic_cube'")
        if dataset.attrs.get("product_level") not in {"L1D", "L2A"}:
            raise ValueError("Tomogram product_level must be 'L1D' or 'L2A'")
        coordinate_space = dataset.attrs.get("coordinate_space")
        if coordinate_space not in {"sar", "geographic"}:
            raise ValueError("Tomogram coordinate_space must be 'sar' or 'geographic'")
        if require_geocoded and coordinate_space != "geographic":
            raise ValueError("This operation requires a geocoded L2A tomogram")

        horizontal = (
            ("x", "y")
            if coordinate_space == "geographic"
            else ("azimuth_time", "slant_range_time")
        )
        expected = (
            ("polarization", "z", "y", "x")
            if coordinate_space == "geographic"
            else ("polarization", "z", "azimuth_time", "slant_range_time")
        )
        cube = dataset.tomographic_cube
        if cube.dims != expected:
            raise ValueError(
                f"'tomographic_cube' must use dimensions {expected!r}, "
                f"found {cube.dims!r}"
            )
        for coordinate in ("polarization", "z", *horizontal):
            if coordinate not in dataset.coords:
                raise ValueError(f"Tomogram is missing coordinate {coordinate!r}")
            if dataset[coordinate].dims != (coordinate,):
                raise ValueError(
                    f"Tomogram coordinate {coordinate!r} must be one-dimensional"
                )
            if dataset.sizes[coordinate] == 0:
                raise ValueError(f"Tomogram coordinate {coordinate!r} is empty")
            if (
                check_finite_coordinates
                and coordinate != "polarization"
                and not bool(np.isfinite(dataset[coordinate]).all())
            ):
                raise ValueError(
                    f"Tomogram coordinate {coordinate!r} contains non-finite values"
                )

        z = np.asarray(dataset.z.data, dtype=float)
        if not np.isfinite(z).all() or (z.size > 1 and not np.all(np.diff(z) > 0)):
            raise ValueError("Tomogram z coordinates must be finite and increasing")

    def select_polarization(
        self,
        dataset: xr.Dataset,
        polarization: str | Iterable[str],
    ) -> xr.Dataset:
        """Select one or more available polarizations while retaining the axis."""
        self.validate(dataset)
        selected = (
            (polarization,) if isinstance(polarization, str) else tuple(polarization)
        )
        if not selected:
            raise ValueError("At least one polarization must be selected")
        available = {str(value) for value in dataset.polarization.values}
        unknown = sorted(set(selected).difference(available))
        if unknown:
            raise ValueError(f"Unknown polarization(s): {', '.join(unknown)}")
        return dataset.sel(polarization=list(dict.fromkeys(selected)))

    def subset(
        self,
        dataset: xr.Dataset,
        *,
        height_range: tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> xr.Dataset:
        """Subset an open tomogram by height and, for L2A, geographic bounds."""
        self.validate(dataset)
        result = dataset
        if height_range is not None:
            minimum, maximum = _interval(height_range, "height_range")
            result = result.sel(z=slice(minimum, maximum))
        if bounds is not None:
            self.validate(dataset, require_geocoded=True)
            if len(bounds) != 4 or not np.isfinite(bounds).all():
                raise ValueError("bounds must contain four finite values")
            minimum_x, minimum_y, maximum_x, maximum_y = map(float, bounds)
            if minimum_x > maximum_x or minimum_y > maximum_y:
                raise ValueError("bounds minima must not exceed maxima")
            result = result.sel(
                x=slice(minimum_x, maximum_x),
                y=slice(minimum_y, maximum_y),
            )
        result.attrs = dict(dataset.attrs)
        return result

    def to_power(
        self,
        dataset: xr.Dataset,
        *,
        variable: str = "tomographic_cube",
        name: str = "tomographic_power",
    ) -> xr.Dataset:
        """Convert complex amplitude (or real amplitude) to linear power."""
        self.validate(dataset)
        source = _variable(dataset, variable)
        power = np.abs(source) ** 2
        power.name = name
        power.attrs = {
            **source.attrs,
            "long_name": "BIOMASS tomographic power",
            "quantity": "power",
            "derived_from": variable,
        }
        result = dataset.assign({name: power})
        result.attrs = {**dataset.attrs, "processing_quantity": "power"}
        return result

    def to_db(
        self,
        dataset: xr.Dataset,
        *,
        variable: str = "tomographic_power",
        name: str | None = None,
        floor_db: float | None = None,
        reference: float = 1.0,
    ) -> xr.Dataset:
        """Convert a non-negative power variable to decibels."""
        source = _variable(dataset, variable)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("reference must be a positive finite value")
        if floor_db is not None and not np.isfinite(floor_db):
            raise ValueError("floor_db must be finite")
        if np.issubdtype(source.dtype, np.complexfloating):
            raise ValueError("Convert complex amplitude to power before decibels")
        # Preserve lazy backends: only perform an eager value scan when the
        # array is already resident in memory.
        if isinstance(source.data, np.ndarray) and np.any(source.data < 0):
            raise ValueError("Power values must be non-negative")

        with np.errstate(divide="ignore", invalid="ignore"):
            values = 10.0 * np.log10(source / reference)
        if floor_db is not None:
            # ``clip`` maps valid zero power (-inf dB) to the display floor,
            # while deliberately preserving missing source cells as NaN.
            values = values.clip(min=floor_db)
        output_name = name or f"{variable}_db"
        values.name = output_name
        values.attrs = {
            **source.attrs,
            "long_name": "BIOMASS tomographic power in decibels",
            "units": "dB",
            "quantity": "power_db",
            "derived_from": variable,
            "decibel_reference": float(reference),
        }
        result = dataset.assign({output_name: values})
        result.attrs = {**dataset.attrs, "processing_quantity": "power_db"}
        if floor_db is not None:
            result.attrs["decibel_floor"] = float(floor_db)
        return result

    def horizontal_slice(
        self,
        dataset: xr.Dataset,
        *,
        height: float,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Extract a horizontal plane at a requested relative height."""
        self.validate(dataset)
        if not np.isfinite(height):
            raise ValueError("height must be finite")
        result = dataset.sel(z=float(height), method=_selection_method(method))
        result.attrs = {
            **dataset.attrs,
            "slice_orientation": "horizontal",
            "requested_height": float(height),
            "selected_height": float(result.z),
        }
        return result

    def vertical_slice(
        self,
        dataset: xr.Dataset,
        *,
        axis: str,
        coordinate: float,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Extract a vertical plane along one horizontal coordinate."""
        self.validate(dataset)
        horizontal = (
            {"x", "y"}
            if dataset.attrs["coordinate_space"] == "geographic"
            else {"azimuth_time", "slant_range_time"}
        )
        if axis not in horizontal:
            expected = ", ".join(sorted(horizontal))
            raise ValueError(f"axis must be one of: {expected}")
        if not np.isfinite(coordinate):
            raise ValueError("coordinate must be finite")
        result = dataset.sel(
            {axis: float(coordinate)}, method=_selection_method(method)
        )
        result.attrs = {
            **dataset.attrs,
            "slice_orientation": "vertical",
            "slice_axis": axis,
            "requested_slice_coordinate": float(coordinate),
            "selected_slice_coordinate": float(result[axis]),
        }
        return result

    def vertical_profile(
        self,
        dataset: xr.Dataset,
        *,
        statistic: str = "mean",
    ) -> xr.Dataset:
        """Aggregate horizontal dimensions into a profile over relative height."""
        self.validate(dataset)
        horizontal = (
            ("x", "y")
            if dataset.attrs["coordinate_space"] == "geographic"
            else ("azimuth_time", "slant_range_time")
        )
        if statistic not in {"mean", "median", "max", "sum"}:
            raise ValueError("statistic must be 'mean', 'median', 'max', or 'sum'")
        result = getattr(dataset, statistic)(dim=horizontal, keep_attrs=True)
        result.attrs = {
            **dataset.attrs,
            "profile_statistic": statistic,
            "profile_dimensions": list(horizontal),
        }
        return result

    def integrate_vertical_bands(
        self,
        dataset: xr.Dataset,
        *,
        variable: str = "tomographic_power",
        band_size: float = 20.0,
        name: str = "vertical_contribution",
    ) -> xr.Dataset:
        """Integrate power into height bands and normalize within each column.

        Integration is performed on the supplied power variable before any
        display smoothing. The resulting contribution sums to one along
        ``height_band`` for every column with positive finite power.
        """
        self.validate(dataset)
        source = _variable(dataset, variable)
        if "z" not in source.dims:
            raise ValueError(f"Variable {variable!r} has no 'z' dimension")
        if np.issubdtype(source.dtype, np.complexfloating):
            raise ValueError("Vertical bands require power, not complex amplitude")
        if not np.isfinite(band_size) or band_size <= 0:
            raise ValueError("band_size must be a positive finite value")

        z = np.asarray(dataset.z.data, dtype=float)
        start = np.floor(z.min() / band_size) * band_size - band_size / 2
        stop = np.ceil(z.max() / band_size) * band_size + 1.5 * band_size
        edges = np.arange(start, stop, band_size, dtype=float)
        centers = (edges[:-1] + edges[1:]) / 2
        bands = source.groupby_bins(
            "z", edges, labels=centers, right=False, include_lowest=True
        ).sum(dim="z", skipna=True)
        bands = bands.rename({"z_bins": "height_band"}).assign_coords(
            height_band=centers
        )
        total = bands.sum("height_band", skipna=True)
        contribution = bands / total.where(total > 0)
        contribution.name = name
        contribution.attrs = {
            "long_name": "normalized vertical scattering contribution",
            "quantity": "fraction_of_column_power",
            "units": "1",
            "derived_from": variable,
            "vertical_band_size": float(band_size),
            "normalization": "sum over height_band equals one",
        }
        total_name = f"{variable}_column_total"
        total = total.rename(total_name)
        total.attrs = {
            "long_name": "vertically integrated tomographic power",
            "derived_from": variable,
        }
        result = dataset.assign({name: contribution, total_name: total})
        result.attrs = {
            **dataset.attrs,
            "vertical_structure_band_size": float(band_size),
        }
        return result

    def add_vertical_metrics(
        self,
        dataset: xr.Dataset,
        *,
        contribution: str = "vertical_contribution",
        upper_height: float = 30.0,
    ) -> xr.Dataset:
        """Derive column metrics from normalized height-band contributions."""
        values = _variable(dataset, contribution)
        if "height_band" not in values.dims:
            raise ValueError(
                f"Variable {contribution!r} has no 'height_band' dimension"
            )
        if not np.isfinite(upper_height):
            raise ValueError("upper_height must be finite")
        height = dataset.height_band
        cdf = values.fillna(0).cumsum("height_band")

        def quantile_height(quantile: float) -> xr.DataArray:
            return height.where(cdf >= quantile).min("height_band", skipna=True)

        centroid = (values * height).sum("height_band", skipna=True)
        # ``idxmax`` returns the height coordinate directly and remains lazy.
        # Using a Dask-backed ``argmax`` result as an ``isel`` indexer is not
        # supported by Xarray's vectorized indexing.
        dominant_height = values.idxmax("height_band", skipna=True)
        dominant_contribution = values.max("height_band", skipna=True)
        entropy_terms = xr.where(values > 0, values * np.log(values), 0.0)
        entropy = -entropy_terms.sum("height_band", skipna=True) / np.log(
            values.sizes["height_band"]
        )
        valid = values.notnull().any("height_band")
        metrics = {
            "scattering_height_50": quantile_height(0.50).where(valid),
            "scattering_height_90": quantile_height(0.90).where(valid),
            "scattering_centroid": centroid.where(valid),
            "dominant_scattering_height": dominant_height.where(valid),
            "dominant_contribution": dominant_contribution.where(valid),
            "vertical_entropy": entropy.where(valid),
            "upper_contribution": values.where(height >= upper_height, 0).sum(
                "height_band", skipna=True
            ).where(valid),
        }
        for name, metric in metrics.items():
            metric.attrs["derived_from"] = contribution
        metrics["scattering_height_50"].attrs.update(units="m", quantile=0.5)
        metrics["scattering_height_90"].attrs.update(units="m", quantile=0.9)
        metrics["scattering_centroid"].attrs.update(units="m")
        metrics["dominant_scattering_height"].attrs.update(units="m")
        metrics["dominant_contribution"].attrs.update(units="1")
        metrics["vertical_entropy"].attrs.update(units="1")
        metrics["upper_contribution"].attrs.update(
            units="1", lower_height=float(upper_height)
        )
        result = dataset.assign(metrics)
        result.attrs = {**dataset.attrs, "upper_contribution_height": upper_height}
        return result

    def prepare_vertical_structure(
        self,
        dataset: xr.Dataset,
        *,
        polarization: str | Iterable[str] | None = None,
        band_size: float = 20.0,
        upper_height: float = 30.0,
    ) -> xr.Dataset:
        """Create the explicit TOM variables consumed by visualisation tools."""
        selected = (
            dataset
            if polarization is None
            else self.select_polarization(dataset, polarization)
        )
        powered = self.to_power(selected)
        banded = self.integrate_vertical_bands(powered, band_size=band_size)
        result = self.add_vertical_metrics(
            banded, upper_height=upper_height
        )
        result.attrs = {
            **result.attrs,
            "processing_recipe": "tomogram_vertical_structure",
        }
        return result

    def to_data_layer(
        self,
        dataset: xr.Dataset,
        *,
        name: str = "biomass_tomogram",
    ) -> DataLayer:
        """Wrap a geocoded L2A volume in the shared data interface."""
        self.validate(dataset, require_geocoded=True)
        return DataLayer(
            name=name,
            data=dataset,
            modality=Modality.TOMOGRAM,
            geometry=GeometryType.VOLUME,
        )


def _variable(dataset: xr.Dataset, name: str) -> xr.DataArray:
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Tomogram input must be an xarray.Dataset")
    if name not in dataset:
        raise ValueError(f"Dataset has no {name!r} variable")
    return dataset[name]


def _interval(value: tuple[float, float], name: str) -> tuple[float, float]:
    if len(value) != 2 or not np.isfinite(value).all() or value[0] > value[1]:
        raise ValueError(f"{name} must contain two increasing finite values")
    return float(value[0]), float(value[1])


def _selection_method(method: str) -> str | None:
    if method not in {"nearest", "exact"}:
        raise ValueError("method must be 'nearest' or 'exact'")
    return None if method == "exact" else method
