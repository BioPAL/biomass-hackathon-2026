"""Search parameters, validated and translated to what the STAC API expects."""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

from .exceptions import QueryError


@dataclass(frozen=True)
class SearchQuery:
    collection: str
    product_type: Optional[str] = None
    date_start: str = "2024-07-31"
    date_end: str = "2100-01-01"
    bbox: Optional[Sequence[float]] = None      # [min_lon, min_lat, max_lon, max_lat]
    aoi_file: Optional[str] = None              # geojson/shp/kml/gpkg; overrides bbox
    extra_filters: dict = field(default_factory=dict)  # e.g. {"sat:orbit_state": "DESCENDING"}
    max_items: Optional[int] = 10               # None = all (mind the size)
    max_cloud_cover: Optional[float] = None     # percent, for optical products (Sentinel-2)

    def __post_init__(self):
        try:
            start, end = date.fromisoformat(self.date_start), date.fromisoformat(self.date_end)
        except ValueError as e:
            raise QueryError(f"Dates must be YYYY-MM-DD: {e}") from e
        if start > end:
            raise QueryError(f"date_start {self.date_start} is after date_end {self.date_end}")
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise QueryError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
            min_lon, min_lat, max_lon, max_lat = self.bbox
            if min_lon >= max_lon or min_lat >= max_lat:
                raise QueryError(f"bbox is not ordered min < max: {list(self.bbox)}")
        if self.max_items is not None and self.max_items < 1:
            raise QueryError("max_items must be >= 1 or None")

    def datetime_range(self) -> list:
        return [f"{self.date_start}T00:00:00Z", f"{self.date_end}T23:59:59Z"]

    def cql_filter(self) -> Optional[str]:
        parts = []
        if self.product_type:
            parts.append(f"product:type='{self.product_type}'")
        for key, value in self.extra_filters.items():
            parts.append(f"{key}='{value}'" if isinstance(value, str) else f"{key}={str(value).lower()}")
        if self.max_cloud_cover is not None:
            parts.append(f"eo:cloud_cover<={self.max_cloud_cover}")
        return " AND ".join(parts) or None

    def resolved_bbox(self) -> Optional[list]:
        """Bounding box to search with: the AOI file's extent if given, else ``bbox``."""
        if self.aoi_file:
            import geopandas as gpd
            return list(gpd.read_file(self.aoi_file).to_crs("EPSG:4326").total_bounds)
        return list(self.bbox) if self.bbox is not None else None
