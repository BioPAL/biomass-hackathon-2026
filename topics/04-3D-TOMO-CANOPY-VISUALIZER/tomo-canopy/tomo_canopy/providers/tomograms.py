"""Providers for BIOMASS L1D and L2A tomographic cube products."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class TomogramProduct:
    """Paths and essential metadata discovered for one BIOMASS TOM product."""

    root: Path
    product_id: str
    product_level: str
    measurement: Path
    annotation: Path
    stac: Path
    lut: Path | None
    shape: tuple[int, int, int, int]
    bbox: tuple[float, float, float, float]
    crs: str
    polarizations: tuple[str, ...]
    vertical_minimum: float
    vertical_step: float
    vertical_samples: int
    acquisition_start: str | None
    acquisition_end: str | None

    @property
    def vertical_maximum(self) -> float:
        """Return the coordinate of the last vertical sample."""
        return self.vertical_minimum + self.vertical_step * (
            self.vertical_samples - 1
        )


class TomogramProvider:
    """Open BIOMASS TOM L1D and L2A products as lazy Xarray datasets.

    A complete product directory (or its STAC/measurement file) is required
    because the measurement NetCDF does not carry all coordinate information.
    L1D remains in native SAR coordinates. L2A is normalized to a regular
    geographic ``x/y/z`` volume after its reconstructed axes are checked
    against the STAC bounding box.
    """

    def inspect(self, source: str | Path) -> TomogramProduct:
        """Discover product assets and parse metadata without opening the cube."""
        root, stac_path = _find_product(source)
        stac = json.loads(stac_path.read_text(encoding="utf-8"))
        assets = stac.get("assets")
        if not isinstance(assets, dict):
            raise ValueError(f"STAC item has no asset mapping: {stac_path}")

        measurement = _asset_path(root, assets, required_role="data")
        annotation = _asset_path(
            root, assets, required_roles={"annotation", "metadata"}, suffix=".xml"
        )
        lut = _asset_path(
            root,
            assets,
            required_role="lut",
            suffix=".nc",
            required=False,
        )
        properties = stac.get("properties", {})
        level = _normalise_level(properties.get("processing:level"))
        shape_values = tuple(int(value) for value in properties.get("proj:shape", ()))
        if len(shape_values) != 4:
            raise ValueError("Tomogram STAC proj:shape must contain four dimensions")
        shape = (shape_values[0], shape_values[1], shape_values[2], shape_values[3])
        bbox_values = tuple(float(value) for value in stac.get("bbox", ()))
        if len(bbox_values) != 4:
            raise ValueError("Tomogram STAC bbox must contain four values")
        bbox = (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3])

        xml_root = ET.parse(annotation).getroot()
        polarizations = _polarizations(xml_root)
        vertical_minimum = _required_float(xml_root, "minVerticalAxis")
        vertical_step = _required_float(xml_root, "verticalAxisStep")
        vertical_samples = _required_int(xml_root, "numberOfVerticalSamples")
        crs = _required_text(xml_root, "code")
        if len(polarizations) != shape[3]:
            raise ValueError(
                "Polarization metadata does not match the measurement shape"
            )
        if vertical_samples != shape[2]:
            raise ValueError("Vertical metadata does not match the measurement shape")

        return TomogramProduct(
            root=root,
            product_id=str(stac.get("id", root.name)),
            product_level=level,
            measurement=measurement,
            annotation=annotation,
            stac=stac_path,
            lut=lut,
            shape=shape,
            bbox=bbox,
            crs=crs,
            polarizations=polarizations,
            vertical_minimum=vertical_minimum,
            vertical_step=vertical_step,
            vertical_samples=vertical_samples,
            acquisition_start=properties.get("start_datetime"),
            acquisition_end=properties.get("end_datetime"),
        )

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
        """Open and normalize a tomographic product.

        ``bounds`` uses ``(minimum_x, minimum_y, maximum_x, maximum_y)`` and is
        available only for geocoded L2A products. Data remains lazy unless
        ``load=True``. Passing chunks requires the optional Dask dependency.
        """
        product = self.inspect(source)
        if expected_level is not None:
            expected = _normalise_level(expected_level)
            if product.product_level != expected:
                raise ValueError(
                    f"Expected {expected}, found {product.product_level} product"
                )
        selected_polarizations = _select_polarizations(
            polarization, product.polarizations
        )
        selected_height = _validate_interval(height_range, "height_range")
        selected_bounds = _validate_bounds(bounds)
        if selected_bounds is not None and product.product_level != "L2A":
            raise ValueError("Geographic bounds are supported only for L2A")

        dataset = _open_netcdf(product.measurement, chunks=chunks)
        if "tomographicCubes" not in dataset:
            dataset.close()
            raise ValueError("Measurement has no 'tomographicCubes' variable")
        if dataset["tomographicCubes"].shape != product.shape:
            dataset.close()
            raise ValueError("Measurement shape does not match the STAC metadata")
        dataset = dataset[["tomographicCubes"]]

        if product.product_level == "L2A":
            result = self._normalise_l2a(dataset, product)
        else:
            result = self._normalise_l1d(dataset, product, chunks=chunks)

        result = result.sel(polarization=list(selected_polarizations))
        if selected_height is not None:
            result = result.sel(z=slice(*selected_height))
        if selected_bounds is not None:
            minimum_x, minimum_y, maximum_x, maximum_y = selected_bounds
            result = result.sel(
                x=slice(minimum_x, maximum_x), y=slice(minimum_y, maximum_y)
            )

        if include_auxiliary and product.lut is not None:
            auxiliary = self._open_auxiliary(product, chunks=chunks)
            result = xr.merge([result, auxiliary], compat="no_conflicts")

        result.attrs.update(_product_attributes(product))
        return result.load() if load else result

    @staticmethod
    def _normalise_l2a(
        dataset: xr.Dataset, product: TomogramProduct
    ) -> xr.Dataset:
        xml_root = ET.parse(product.annotation).getroot()
        first_axis_start = _required_float(xml_root, "firstLatitude")
        second_axis_start = _required_float(xml_root, "firstLongitude")
        first_axis_step = _required_float(xml_root, "rowStep")
        second_axis_step = _required_float(xml_root, "columnStep")
        first_axis = first_axis_start + np.arange(product.shape[0]) * first_axis_step
        second_axis = (
            second_axis_start + np.arange(product.shape[1]) * second_axis_step
        )
        x, y, convention = _resolve_l2a_axes(
            first_axis, second_axis, product.bbox
        )
        result = dataset.rename(
            {
                "tomographicCubes": "tomographic_cube",
                "relativeAzimuthTime": "x",
                "slantRangeTime": "y",
                "tomographicHeight": "z",
                "polarizations": "polarization",
            }
        ).assign_coords(
            x=("x", x),
            y=("y", y),
            z=("z", _vertical_axis(product)),
            polarization=("polarization", np.asarray(product.polarizations)),
        )
        result = result.transpose("polarization", "z", "y", "x")
        result.x.attrs.update(standard_name="longitude", units="degrees_east")
        result.y.attrs.update(standard_name="latitude", units="degrees_north")
        result.z.attrs.update(
            long_name="relative tomographic height", units="m", positive="up"
        )
        result.tomographic_cube.attrs.update(
            long_name="BIOMASS tomographic cube values"
        )
        result.attrs["coordinate_normalization"] = convention
        return result

    @staticmethod
    def _normalise_l1d(
        dataset: xr.Dataset,
        product: TomogramProduct,
        *,
        chunks: str | Mapping[str, int] | None,
    ) -> xr.Dataset:
        if product.lut is None:
            dataset.close()
            raise ValueError("L1D coordinate reconstruction requires the LUT asset")
        coordinates = _open_netcdf(product.lut, chunks=chunks, group="tomography")
        required = ("tomoRelativeAzimuthTime", "tomoSlantRangeTime")
        if any(name not in coordinates for name in required):
            coordinates.close()
            dataset.close()
            raise ValueError("L1D LUT is missing native SAR coordinate arrays")
        azimuth = np.asarray(coordinates[required[0]].values)
        slant_range = np.asarray(coordinates[required[1]].values)
        coordinates.close()
        if azimuth.size != product.shape[0] or slant_range.size != product.shape[1]:
            dataset.close()
            raise ValueError("L1D LUT coordinates do not match the measurement shape")

        result = dataset.rename(
            {
                "tomographicCubes": "tomographic_cube",
                "relativeAzimuthTime": "azimuth_time",
                "slantRangeTime": "slant_range_time",
                "tomographicHeight": "z",
                "polarizations": "polarization",
            }
        ).assign_coords(
            azimuth_time=("azimuth_time", azimuth),
            slant_range_time=("slant_range_time", slant_range),
            z=("z", _vertical_axis(product)),
            polarization=("polarization", np.asarray(product.polarizations)),
        )
        result = result.transpose(
            "polarization", "z", "azimuth_time", "slant_range_time"
        )
        result.azimuth_time.attrs.update(
            long_name="relative azimuth time", units="s"
        )
        result.slant_range_time.attrs.update(
            long_name="two-way slant range time", units="s"
        )
        result.z.attrs.update(
            long_name="relative tomographic height", units="m", positive="up"
        )
        result.tomographic_cube.attrs.update(
            long_name="BIOMASS tomographic cube values"
        )
        return result

    @staticmethod
    def _open_auxiliary(
        product: TomogramProduct,
        *,
        chunks: str | Mapping[str, int] | None,
    ) -> xr.Dataset:
        if product.lut is None:
            return xr.Dataset()
        group = "geometry" if product.product_level == "L1D" else "tomography"
        auxiliary = _open_netcdf(product.lut, chunks=chunks, group=group)
        rename = {
            name: f"auxiliary_{name}"
            for name in (*auxiliary.dims, *auxiliary.data_vars)
        }
        return auxiliary.rename(rename)


def open_tomogram(source: str | Path, **kwargs: Any) -> xr.Dataset:
    """Open a BIOMASS tomographic product with :class:`TomogramProvider`."""
    return TomogramProvider().open(source, **kwargs)


def _find_product(source: str | Path) -> tuple[Path, Path]:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    candidates: list[Path] = []
    if path.is_dir():
        candidates.append(path)
    else:
        if path.name.endswith("_stac.json"):
            return path.parent, path
        candidates.extend(path.parents[:3])
    for root in candidates:
        stac_files = sorted(root.glob("*_stac.json"))
        if len(stac_files) == 1:
            return root, stac_files[0]
        if len(stac_files) > 1:
            raise ValueError(f"Multiple STAC items found in {root}")
    raise ValueError(f"Could not find a TOM STAC item for {path}")


def _asset_path(
    root: Path,
    assets: Mapping[str, Any],
    *,
    required_role: str | None = None,
    required_roles: set[str] | None = None,
    suffix: str | None = None,
    required: bool = True,
) -> Path | None:
    matches: list[Path] = []
    for asset in assets.values():
        if not isinstance(asset, dict) or "href" not in asset:
            continue
        roles = set(asset.get("roles", ()))
        if required_role is not None and required_role not in roles:
            continue
        if required_roles is not None and not required_roles.issubset(roles):
            continue
        candidate = (root / asset["href"]).resolve()
        if suffix is not None and candidate.suffix.lower() != suffix:
            continue
        matches.append(candidate)
    if len(matches) != 1:
        description = required_role or "+".join(sorted(required_roles or ()))
        if not required and not matches:
            return None
        raise ValueError(f"Expected one {description} asset, found {len(matches)}")
    if not matches[0].is_file():
        raise FileNotFoundError(matches[0])
    return matches[0]


def _open_netcdf(
    path: Path,
    *,
    chunks: str | Mapping[str, int] | None,
    group: str | None = None,
) -> xr.Dataset:
    try:
        import_module("netCDF4")
    except ImportError as error:
        raise ImportError(
            "Tomogram support requires netCDF4; install tomo-canopy with "
            "`python -m pip install -e '.[tomogram]'`"
        ) from error
    if chunks is not None:
        try:
            import_module("dask.array")
        except ImportError as error:
            raise ImportError(
                "Chunked tomogram access requires dask; install the tomogram extra"
            ) from error
    return xr.open_dataset(
        path,
        engine="netcdf4",
        group=group,
        chunks=chunks,
        mask_and_scale=True,
        cache=False,
    )


def _normalise_level(value: Any) -> str:
    level = str(value or "").upper()
    if level not in {"L1D", "L2A"}:
        raise ValueError(f"Unsupported tomogram product level: {value!r}")
    return level


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _texts(root: ET.Element, name: str) -> list[str]:
    return [
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element) == name and (element.text or "").strip()
    ]


def _required_text(root: ET.Element, name: str) -> str:
    values = _texts(root, name)
    if not values:
        raise ValueError(f"Annotation is missing {name!r}")
    return values[0]


def _required_float(root: ET.Element, name: str) -> float:
    return float(_required_text(root, name))


def _required_int(root: ET.Element, name: str) -> int:
    return int(_required_text(root, name))


def _polarizations(root: ET.Element) -> tuple[str, ...]:
    for element in root.iter():
        if _local_name(element) == "tomographicCubesPolarizations":
            values = tuple(
                (child.text or "").strip()
                for child in element
                if _local_name(child) == "polarization"
            )
            if values:
                return values
    raise ValueError("Annotation has no tomographic polarization list")


def _vertical_axis(product: TomogramProduct) -> np.ndarray:
    return product.vertical_minimum + np.arange(product.vertical_samples) * (
        product.vertical_step
    )


def _resolve_l2a_axes(
    first_axis: np.ndarray,
    second_axis: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, str]:
    def error(x: np.ndarray, y: np.ndarray) -> float:
        extent = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
        return float(np.max(np.abs(np.asarray(extent) - np.asarray(bbox))))

    metadata_error = error(second_axis, first_axis)
    swapped_error = error(first_axis, second_axis)
    tolerance = max(
        abs(float(np.diff(first_axis[:2])[0])) if first_axis.size > 1 else 0.0,
        abs(float(np.diff(second_axis[:2])[0])) if second_axis.size > 1 else 0.0,
        1e-8,
    ) * 1.5
    if swapped_error <= tolerance and swapped_error < metadata_error:
        convention = "XML latitude/longitude fields swapped to match STAC"
        return first_axis, second_axis, convention
    if metadata_error <= tolerance:
        return second_axis, first_axis, "XML latitude/longitude field semantics"
    raise ValueError(
        "Reconstructed L2A coordinates do not match the STAC bbox "
        f"{bbox!r}"
    )


def _select_polarizations(
    selected: str | Iterable[str] | None,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    if selected is None:
        return available
    values = (selected,) if isinstance(selected, str) else tuple(selected)
    if not values:
        raise ValueError("At least one polarization must be selected")
    unknown = sorted(set(values).difference(available))
    if unknown:
        raise ValueError(f"Unknown polarization(s): {', '.join(unknown)}")
    return tuple(dict.fromkeys(values))


def _validate_interval(
    value: tuple[float, float] | None, name: str
) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2 or not np.isfinite(value).all() or value[0] > value[1]:
        raise ValueError(f"{name} must contain two increasing finite values")
    return float(value[0]), float(value[1])


def _validate_bounds(
    value: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if len(value) != 4 or not np.isfinite(value).all():
        raise ValueError("bounds must contain four finite values")
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, value)
    if minimum_x > maximum_x or minimum_y > maximum_y:
        raise ValueError("bounds minima must not exceed maxima")
    return minimum_x, minimum_y, maximum_x, maximum_y


def _product_attributes(product: TomogramProduct) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "source": str(product.measurement),
        "product_root": str(product.root),
        "product_id": product.product_id,
        "product_level": product.product_level,
        "modality": "tomogram",
        "geometry_type": "volume",
        "crs": product.crs,
        "vertical_reference": "relative_tomographic_height",
        "vertical_units": "m",
        "stac_bbox": product.bbox,
        "coordinate_space": (
            "geographic" if product.product_level == "L2A" else "sar"
        ),
    }
    if product.acquisition_start is not None:
        attributes["acquisition_start"] = product.acquisition_start
    if product.acquisition_end is not None:
        attributes["acquisition_end"] = product.acquisition_end
    return attributes

