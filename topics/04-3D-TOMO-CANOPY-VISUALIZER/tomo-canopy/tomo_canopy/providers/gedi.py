"""GEDI data access backed by the optional :mod:`gedidb` package."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

DEFAULT_S3_BUCKET = "dog-ext.gedidb.gedi-l2-l4-v002.0"
DEFAULT_S3_URL = "https://s3.gfz-potsdam.de"


class GEDIProvider:
    """Query GEDI observations through ``gedidb`` and normalize metadata.

    ``gedidb`` is imported only when the backend is first needed. Tests and
    custom deployments can inject a compatible backend with the ``backend``
    argument without importing the optional dependency.
    """

    def __init__(
        self,
        *,
        storage_type: str = "s3",
        s3_bucket: str | None = DEFAULT_S3_BUCKET,
        url: str | None = DEFAULT_S3_URL,
        local_path: str | Path = "./",
        region: str = "eu-central-1",
        credentials: Mapping[str, Any] | None = None,
        config_overrides: Mapping[str, str] | None = None,
        backend: Any | None = None,
    ) -> None:
        """Configure a GEDI data source.

        Parameters mirror :class:`gedidb.GEDIProvider`. The public GFZ S3
        endpoint is used by default; use ``storage_type="local"`` and
        ``local_path=...`` for a local TileDB store.
        """
        if storage_type not in {"s3", "local"}:
            raise ValueError("storage_type must be 's3' or 'local'")
        if storage_type == "s3" and not s3_bucket:
            raise ValueError("s3_bucket is required for S3 storage")

        self.storage_type = storage_type
        self.s3_bucket = s3_bucket
        self.url = url
        self.local_path = str(local_path)
        self.region = region
        self.credentials = dict(credentials) if credentials is not None else None
        self.config_overrides = (
            dict(config_overrides) if config_overrides is not None else None
        )
        self._backend = backend

    @property
    def backend(self) -> Any:
        """Return the lazily constructed ``gedidb`` provider."""
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    def open(
        self,
        source: Any,
        *,
        variables: Iterable[str],
        start_time: str | None = None,
        end_time: str | None = None,
        quality_filters: Mapping[str, Any] | None = None,
    ) -> xr.Dataset:
        """Query GEDI shots intersecting a GeoDataFrame or GeoJSON source."""
        return self.get_data(
            variables=variables,
            geometry=source,
            query_type="bounding_box",
            start_time=start_time,
            end_time=end_time,
            quality_filters=quality_filters,
        )

    def get_data(
        self,
        *,
        variables: Iterable[str],
        geometry: Any | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        query_type: str = "bounding_box",
        point: tuple[float, float] | None = None,
        num_shots: int | None = None,
        radius: float | None = None,
        use_polygon_filter: bool | str = "auto",
        quality_filters: Mapping[str, Any] | None = None,
        **filters: Any,
    ) -> xr.Dataset:
        """Query GEDI data and return a normalized Xarray dataset.

        Additional keyword arguments and ``quality_filters`` are forwarded to
        ``gedidb`` as variable filters. A path-like geometry is loaded with
        GeoPandas; an existing GeoDataFrame is forwarded directly.
        """
        selected_variables = _validate_variables(variables)
        _validate_dates(start_time, end_time)
        if query_type not in {"bounding_box", "nearest"}:
            raise ValueError("query_type must be 'bounding_box' or 'nearest'")

        query: dict[str, Any] = {
            "variables": list(selected_variables),
            "start_time": start_time,
            "end_time": end_time,
            "return_type": "xarray",
            "query_type": query_type,
        }
        if query_type == "bounding_box":
            if geometry is None:
                raise ValueError("geometry is required for a bounding-box query")
            query["geometry"] = _load_geometry(geometry)
            query["use_polygon_filter"] = use_polygon_filter
        else:
            _validate_nearest(point, num_shots, radius)
            query.update(point=point, num_shots=num_shots, radius=radius)

        forwarded_filters = dict(quality_filters or {})
        duplicate_filters = forwarded_filters.keys() & filters.keys()
        if duplicate_filters:
            names = ", ".join(sorted(duplicate_filters))
            raise ValueError(f"Quality filters supplied twice: {names}")
        forwarded_filters.update(filters)

        result = self.backend.get_data(**query, **forwarded_filters)
        return self._normalize(
            result,
            variables=selected_variables,
            query_type=query_type,
            start_time=start_time,
            end_time=end_time,
        )

    def _create_backend(self) -> Any:
        try:
            gedidb = import_module("gedidb")
        except ImportError as error:
            raise ImportError(
                "GEDI support requires gedidb; install tomo-canopy with "
                "`python -m pip install -e '.[gedi]'`"
            ) from error

        options: dict[str, Any] = {
            "storage_type": self.storage_type,
            "region": self.region,
        }
        if self.storage_type == "s3":
            options.update(s3_bucket=self.s3_bucket, url=self.url)
        else:
            options["local_path"] = self.local_path
        if self.credentials is not None:
            options["credentials"] = self.credentials
        if self.config_overrides is not None:
            options["config_overrides"] = self.config_overrides
        return gedidb.GEDIProvider(**options)

    def _normalize(
        self,
        dataset: xr.Dataset | None,
        *,
        variables: tuple[str, ...],
        query_type: str,
        start_time: str | None,
        end_time: str | None,
    ) -> xr.Dataset:
        if dataset is None:
            dataset = xr.Dataset(
                coords={"shot_number": np.array([], dtype=np.uint64)}
            )
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("gedidb must return an xarray.Dataset or None")

        result = dataset.copy(deep=False)
        if result.sizes.get("shot_number", 0):
            for coordinate in ("longitude", "latitude"):
                if coordinate not in result.coords:
                    raise ValueError(
                        f"gedidb result is missing required coordinate {coordinate!r}"
                    )
                if result[coordinate].dims != ("shot_number",):
                    raise ValueError(
                        f"GEDI coordinate {coordinate!r} must use 'shot_number'"
                    )
            result = result.assign_coords(
                x=("shot_number", result.longitude.data),
                y=("shot_number", result.latitude.data),
            )

        result.attrs = {
            **dataset.attrs,
            "source": "gedidb",
            "modality": "gedi",
            "geometry_type": "point_cloud",
            "crs": "EPSG:4326",
            "storage_type": self.storage_type,
            "query_type": query_type,
            "requested_variables": list(variables),
        }
        if self.s3_bucket is not None and self.storage_type == "s3":
            result.attrs["s3_bucket"] = self.s3_bucket
        if start_time is not None:
            result.attrs["query_start_time"] = start_time
        if end_time is not None:
            result.attrs["query_end_time"] = end_time
        return result


def _load_geometry(geometry: Any) -> Any:
    if not isinstance(geometry, (str, Path)):
        return geometry
    try:
        geopandas = import_module("geopandas")
    except ImportError as error:
        raise ImportError(
            "Loading a geometry path requires geopandas; install the GEDI extra"
        ) from error
    return geopandas.read_file(geometry)


def _validate_variables(variables: Iterable[str]) -> tuple[str, ...]:
    if isinstance(variables, str):
        raise TypeError("variables must be an iterable of variable names, not a string")
    selected = tuple(dict.fromkeys(variables))
    if not selected or any(not isinstance(name, str) or not name for name in selected):
        raise ValueError("variables must contain at least one non-empty name")
    return selected


def _validate_dates(start_time: str | None, end_time: str | None) -> None:
    try:
        start = np.datetime64(start_time) if start_time is not None else None
        end = np.datetime64(end_time) if end_time is not None else None
    except ValueError as error:
        raise ValueError("start_time and end_time must be valid dates") from error
    if start is not None and end is not None and start > end:
        raise ValueError("start_time must not be after end_time")


def _validate_nearest(
    point: tuple[float, float] | None,
    num_shots: int | None,
    radius: float | None,
) -> None:
    if (
        not isinstance(point, tuple)
        or len(point) != 2
        or not np.isfinite(point).all()
    ):
        raise ValueError("point must contain finite longitude and latitude")
    if not (-180 <= point[0] <= 180 and -90 <= point[1] <= 90):
        raise ValueError("point must contain valid WGS84 longitude and latitude")
    if isinstance(num_shots, bool) or not isinstance(num_shots, int) or num_shots < 1:
        raise ValueError("num_shots must be a positive integer")
    if radius is None or not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
