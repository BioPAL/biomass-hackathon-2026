"""Providers for NASA LVIS Level-2 footprint metrics."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

_PRODUCT_PATTERN = re.compile(
    r"^LVISC2_(?P<campaign>.+?)(?P<year>\d{4})_"
    r"(?P<month>\d{2})(?P<day>\d{2})_(?P<release>R\d+)_"
    r"(?P<segment>\d+)$",
    re.IGNORECASE,
)

_RH_PERCENTILES = (
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    96,
    97,
    98,
    99,
    100,
)

_TXT_COLUMNS = (
    "LFID",
    "SHOTNUMBER",
    "TIME",
    "GLON",
    "GLAT",
    "ZG",
    "HLON",
    "HLAT",
    "ZH",
    "TLON",
    "TLAT",
    "ZT",
    *tuple(f"RH{percentile}" for percentile in _RH_PERCENTILES),
    "AZIMUTH",
    "INCIDENTANGLE",
    "RANGE",
    "COMPLEXITY",
    "SENSITIVITY",
    "CHANNEL_ZT",
    "CHANNEL_ZG",
    "CHANNEL_RH",
)

_TXT_VARIABLES = {
    "file_id": ("LFID",),
    "time": ("TIME",),
    "ground_elevation": ("ZG",),
    "high_longitude": ("HLON",),
    "high_latitude": ("HLAT",),
    "high_elevation": ("ZH",),
    "top_longitude": ("TLON",),
    "top_latitude": ("TLAT",),
    "top_elevation": ("ZT",),
    "rh": tuple(f"RH{percentile}" for percentile in _RH_PERCENTILES),
    "azimuth": ("AZIMUTH",),
    "incident_angle": ("INCIDENTANGLE",),
    "range": ("RANGE",),
    "complexity": ("COMPLEXITY",),
    "sensitivity": ("SENSITIVITY",),
    "channel_top": ("CHANNEL_ZT",),
    "channel_ground": ("CHANNEL_ZG",),
    "channel_rh": ("CHANNEL_RH",),
}

_SCALAR_TXT_NAMES = {
    "file_id": "LFID",
    "time": "TIME",
    "ground_elevation": "ZG",
    "high_longitude": "HLON",
    "high_latitude": "HLAT",
    "high_elevation": "ZH",
    "top_longitude": "TLON",
    "top_latitude": "TLAT",
    "top_elevation": "ZT",
    "azimuth": "AZIMUTH",
    "incident_angle": "INCIDENTANGLE",
    "range": "RANGE",
    "complexity": "COMPLEXITY",
    "sensitivity": "SENSITIVITY",
    "channel_top": "CHANNEL_ZT",
    "channel_ground": "CHANNEL_ZG",
    "channel_rh": "CHANNEL_RH",
}


@dataclass(frozen=True)
class LVISProduct:
    """Metadata inferred from one LVIS Level-2 source."""

    source: Path
    product_id: str
    source_format: str
    campaign: str | None
    acquisition_date: str | None
    release: str | None
    segment: str | None
    sensor: str = "LVIS-C"
    product_level: str = "L2"
    crs: str = "EPSG:4326"


class LVISProvider:
    """Read complete LVIS-C Level-2 TXT records as footprint datasets."""

    def inspect(self, source: str | Path) -> LVISProduct:
        """Inspect filename metadata without reading footprint records."""
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix != ".txt":
            raise ValueError(f"Expected an LVIS Level-2 .TXT product, got {path}")

        match = _PRODUCT_PATTERN.match(path.stem)
        campaign = acquisition_date = release = segment = None
        if match is not None:
            values = match.groupdict()
            campaign = f"{values['campaign']}{values['year']}"
            acquisition_date = (
                f"{values['year']}-{values['month']}-{values['day']}"
            )
            release = values["release"].upper()
            segment = values["segment"]
        return LVISProduct(
            source=path,
            product_id=path.stem,
            source_format="ASCII TXT",
            campaign=campaign,
            acquisition_date=acquisition_date,
            release=release,
            segment=segment,
        )

    def open(
        self,
        source: str | Path,
        *,
        variables: Iterable[str] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        max_shots: int | None = None,
    ) -> xr.Dataset:
        """Open one LVIS Level-2 file as a shot-indexed Xarray dataset.

        ``bounds`` uses WGS84 ``(minimum_x, minimum_y, maximum_x, maximum_y)``.
        Selecting variables reduces I/O, which is especially useful for the
        complete ASCII records. Spatial filtering of TXT files still requires
        scanning their coordinate columns.
        """
        product = self.inspect(source)
        selected_bounds = _validate_bounds(bounds)
        selected_limit = _validate_limit(max_shots)
        return self._open_txt(
            product,
            variables=variables,
            bounds=selected_bounds,
            max_shots=selected_limit,
        )

    def _open_txt(
        self,
        product: LVISProduct,
        *,
        variables: Iterable[str] | None,
        bounds: tuple[float, float, float, float] | None,
        max_shots: int | None,
    ) -> xr.Dataset:
        pandas = _optional_import("pandas", "LVIS support", "lvis")
        selected = _select_variables(variables, tuple(_TXT_VARIABLES))
        columns = {"SHOTNUMBER", "GLON", "GLAT"}
        for name in selected:
            columns.update(_TXT_VARIABLES[name])

        frames: list[Any] = []
        retained = 0
        chunk_size = (
            min(100_000, max_shots)
            if max_shots is not None and bounds is None
            else 100_000
        )
        reader = pandas.read_csv(
            product.source,
            sep=r"\s+",
            comment="#",
            header=None,
            names=_TXT_COLUMNS,
            usecols=lambda name: name in columns,
            chunksize=chunk_size,
        )
        for frame in reader:
            if bounds is not None:
                minimum_x, minimum_y, maximum_x, maximum_y = bounds
                frame = frame.loc[
                    frame["GLON"].between(minimum_x, maximum_x)
                    & frame["GLAT"].between(minimum_y, maximum_y)
                ]
            if max_shots is not None:
                frame = frame.head(max_shots - retained)
            if not frame.empty:
                frames.append(frame)
                retained += len(frame)
            if max_shots is not None and retained >= max_shots:
                break

        if frames:
            frame = pandas.concat(frames, ignore_index=True)
        else:
            frame = pandas.DataFrame(columns=sorted(columns))
        return _txt_to_xarray(frame, product, selected)

def open_lvis(source: str | Path, **kwargs: Any) -> xr.Dataset:
    """Open an LVIS Level-2 product with :class:`LVISProvider`."""
    return LVISProvider().open(source, **kwargs)


def _txt_to_xarray(
    frame: Any,
    product: LVISProduct,
    selected: tuple[str, ...],
) -> xr.Dataset:
    shot_count = len(frame)
    coordinates: dict[str, Any] = {
        "shot": np.arange(shot_count, dtype=np.int64),
        "shot_number": ("shot", frame["SHOTNUMBER"].to_numpy(dtype=np.int64)),
        "longitude": ("shot", frame["GLON"].to_numpy(dtype=np.float64)),
        "latitude": ("shot", frame["GLAT"].to_numpy(dtype=np.float64)),
        "x": ("shot", frame["GLON"].to_numpy(dtype=np.float64)),
        "y": ("shot", frame["GLAT"].to_numpy(dtype=np.float64)),
    }
    data_vars: dict[str, Any] = {}
    for name in selected:
        if name == "rh":
            coordinates["percentile"] = np.asarray(_RH_PERCENTILES, dtype=np.int16)
            data_vars["rh"] = (
                ("shot", "percentile"),
                frame[list(_TXT_VARIABLES["rh"])].to_numpy(dtype=np.float32),
            )
        else:
            column = _SCALAR_TXT_NAMES[name]
            dtype = np.int64 if name == "file_id" else _variable_dtype(name)
            data_vars[name] = ("shot", frame[column].to_numpy(dtype=dtype))
    return _dataset(data_vars, coordinates, product, selected)


def _dataset(
    data_vars: dict[str, Any],
    coordinates: dict[str, Any],
    product: LVISProduct,
    selected: tuple[str, ...],
) -> xr.Dataset:
    result = xr.Dataset(data_vars=data_vars, coords=coordinates)
    result.longitude.attrs.update(standard_name="longitude", units="degrees_east")
    result.latitude.attrs.update(standard_name="latitude", units="degrees_north")
    if "rh" in result:
        result.rh.attrs.update(
            long_name="relative height percentile metric", units="m"
        )
    for name in ("ground_elevation", "high_elevation", "top_elevation"):
        if name in result:
            result[name].attrs["units"] = "m"
    result.attrs = {
        "source": str(product.source),
        "product_id": product.product_id,
        "source_format": product.source_format,
        "product_level": product.product_level,
        "sensor": product.sensor,
        "modality": "lvis",
        "geometry_type": "footprints",
        "crs": product.crs,
        "selected_variables": list(selected),
    }
    for name in ("campaign", "acquisition_date", "release", "segment"):
        value = getattr(product, name)
        if value is not None:
            result.attrs[name] = value
    return result


def _select_variables(
    variables: Iterable[str] | None,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    if variables is None:
        return available
    if isinstance(variables, str):
        raise TypeError("variables must be an iterable of names, not a string")
    selected = tuple(dict.fromkeys(variables))
    if not selected:
        raise ValueError("At least one variable must be selected")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(
            "Variables unavailable in this LVIS format: " + ", ".join(unknown)
        )
    return selected


def _validate_bounds(
    bounds: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if bounds is None:
        return None
    if len(bounds) != 4 or not np.isfinite(bounds).all():
        raise ValueError("bounds must contain four finite values")
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, bounds)
    if minimum_x > maximum_x or minimum_y > maximum_y:
        raise ValueError("bounds minima must not exceed maxima")
    return minimum_x, minimum_y, maximum_x, maximum_y


def _validate_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_shots must be a positive integer")
    return value


def _variable_dtype(name: str) -> Any:
    return np.int16 if name.startswith("channel_") else np.float32


def _optional_import(module: str, feature: str, extra: str) -> Any:
    try:
        return import_module(module)
    except ImportError as error:
        raise ImportError(
            f"{feature} requires {module}; install tomo-canopy with "
            f"`python -m pip install -e '.[{extra}]'`"
        ) from error
