"""GEDI footprints (L2A RH metrics, L2B PAVD profiles) for an area of interest, from the public GFZ gediDB.

Endpoint and API checked against the official gediDB documentation and the 2026.4.30 release source.
No token is needed for the public GFZ database. The L2B variables are PAVD profiles, not L1B waveforms.
Can also be run as a script: ``python -m src.load.gedi --bbox W S E N --start ... --end ... --output ...``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

GEDIDB_ENDPOINT = "https://s3.gfz-potsdam.de"
GEDIDB_BUCKET = "dog-ext.gedidb.gedi-l2-l4-v002.0"
GEDIDB_REGION = "eu-central-1"
GEDIDB_DOCS = "https://gedidb.readthedocs.io/en/latest/user/tiledb_database.html"
_CORE = ["rh:98", "rh:50", "quality_flag", "degrade_flag", "sensitivity"]
_OPTIONAL = [
    "pavd_z",
    "l2_quality_flag",
    "l2a_quality_flag",
    "l2b_quality_flag",
    "beam_type",
    "beam_name",
    "solar_elevation",
    "elev_lowestmode",
    "cover",
    "dz",
]


def cube_bbox(cube: dict[str, Any] | Any) -> list[float]:
    """WGS84 bounds [west, south, east, north] of the valid lon/lat coordinates of a raster cube."""

    def field(names: tuple[str, ...]) -> Any:
        for name in names:
            if isinstance(cube, dict) and name in cube:
                return cube[name]
            if hasattr(cube, name):
                return getattr(cube, name)
        raise ValueError(f"The cube has none of the coordinate fields {names}.")

    latitude = np.asarray(field(("lat", "latitude")), dtype=float)
    longitude = np.asarray(field(("lon", "longitude")), dtype=float)
    if latitude.shape != longitude.shape or latitude.size == 0:
        raise ValueError("Latitude and longitude must share the same non-empty grid.")
    valid = np.isfinite(latitude) & np.isfinite(longitude) & (np.abs(latitude) <= 90) & (np.abs(longitude) <= 180)
    if not np.any(valid):
        raise ValueError("No valid WGS84 coordinate in the cube.")
    return [
        float(longitude[valid].min()),
        float(latitude[valid].min()),
        float(longitude[valid].max()),
        float(latitude[valid].max()),
    ]


def _validate_query(
    bbox: Iterable[float], start_time: str, end_time: str, sensitivity_min: float, max_area_deg2: float
) -> list[float]:
    vals = [float(v) for v in bbox]
    if len(vals) != 4 or not all(np.isfinite(v) for v in vals):
        raise ValueError("bbox must be [west, south, east, north] in WGS84 degrees.")
    w, s, e, n = vals
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError("Invalid or degenerate WGS84 bbox.")
    if date.fromisoformat(start_time) > date.fromisoformat(end_time):
        raise ValueError("start_time must not be after end_time (YYYY-MM-DD).")
    if not (0 <= sensitivity_min <= 1):
        raise ValueError("sensitivity_min must be between 0 and 1.")
    if max_area_deg2 <= 0 or (e - w) * (n - s) > max_area_deg2:
        raise ValueError("AOI too large for one query: shrink it or raise max_area_deg2.")
    return vals


def fetch_gedi(
    bbox: Iterable[float],
    output_path: str | Path,
    start_time: str,
    end_time: str,
    sensitivity_min: float = 0.95,
    include_profiles: bool = True,
    full_power_beams_only: bool = False,
    max_area_deg2: float = 4.0,
    overwrite: bool = False,
    provider: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download the requested GEDI shots inside a bounding box and write them to a new Parquet file.

    The query keeps quality_flag == 1, degrade_flag == 0 and sensitivity >= ``sensitivity_min``. gediDB applies
    its own filtering at ingestion, so this is not a copy of the raw granules. The bounding box is only a
    pre-filter: comparisons must still use the valid pixels of the BIOMASS grid.

    Parameters
    ----------
    bbox : iterable of float
        [west, south, east, north] in WGS84 degrees.
    output_path : str or Path
        Parquet file to create; a ``.query.json`` provenance file is written next to it.
    start_time, end_time : str
        Date window, ``YYYY-MM-DD`` (inclusive).
    sensitivity_min : float
        Minimum beam sensitivity (0-1).
    include_profiles : bool
        Also request the L2B PAVD profile (``pavd_z``, 30 bins of 5 m, m2/m3).
    full_power_beams_only : bool
        Keep only the full-power beams.
    max_area_deg2 : float
        Safety limit on the bbox area [deg2].
    overwrite : bool
        Replace existing output files.
    provider : object, optional
        An open ``gedidb.GEDIProvider`` (re-use, or offline tests).

    Returns
    -------
    (pandas.DataFrame, dict)
        The shots (rh98 and rh50 [m], quality fields, time) and the provenance record. An empty result is also
        recorded; missing shots are never replaced with synthetic data.
    """
    box_values = _validate_query(bbox, start_time, end_time, sensitivity_min, max_area_deg2)
    output = Path(output_path)
    audit_path = output.with_suffix(".query.json")
    if not overwrite and (output.exists() or audit_path.exists()):
        raise FileExistsError(f"Output already exists: {output}. Use a new name or overwrite=True.")
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError as exc:
        raise ImportError("Install the optional GEDI dependencies: pip install -r requirements.txt") from exc
    own_provider = provider is None
    if own_provider:
        try:
            import gedidb as gdb
        except ImportError as exc:
            raise ImportError("Install gediDB: pip install -r requirements.txt") from exc
        provider = gdb.GEDIProvider(
            storage_type="s3", s3_bucket=GEDIDB_BUCKET, url=GEDIDB_ENDPOINT, region=GEDIDB_REGION
        )
    geometry = gpd.GeoDataFrame(geometry=[box(*box_values)], crs="EPSG:4326")
    filters: dict[str, str] = {
        "quality_flag": "== 1",
        "degrade_flag": "== 0",
        "sensitivity": f">= {sensitivity_min} and <= 1.0",
    }
    if full_power_beams_only:
        filters["beam_type"] = "== 'full'"
    try:
        available = provider.get_available_variables()
        if not isinstance(available, pd.DataFrame):
            raise TypeError("get_available_variables() did not return a table.")
        names = set(available.index.astype(str))
        missing = {v.split(":")[0] for v in _CORE} - names
        if missing:
            raise ValueError(f"Database lacks the core variables {sorted(missing)}. Check schema/version.")
        if full_power_beams_only and "beam_type" not in names:
            raise ValueError("The database does not expose beam_type for the requested filter.")
        requested_optional = [v for v in _OPTIONAL if include_profiles or v != "pavd_z"]
        variables = _CORE + [v for v in requested_optional if v in names]
        result = provider.get_data(
            variables=variables,
            query_type="bounding_box",
            geometry=geometry,
            start_time=start_time,
            end_time=end_time,
            return_type="dataframe",
            **filters,
        )
    finally:
        if own_provider and hasattr(provider, "close"):
            provider.close()
    if result is None or (isinstance(result, pd.DataFrame) and result.empty):
        df = pd.DataFrame(
            {
                "shot_number": pd.Series(dtype="uint64"),
                "latitude": pd.Series(dtype="float64"),
                "longitude": pd.Series(dtype="float64"),
                "time": pd.Series(dtype="datetime64[ns, UTC]"),
                "rh98": pd.Series(dtype="float64"),
                "rh50": pd.Series(dtype="float64"),
                "quality_flag": pd.Series(dtype="uint8"),
                "degrade_flag": pd.Series(dtype="uint8"),
                "sensitivity": pd.Series(dtype="float64"),
            }
        )
    else:
        if not isinstance(result, pd.DataFrame):
            raise TypeError("gediDB did not return a dataframe.")
        df = result.rename(columns={"rh_p98": "rh98", "rh_p50": "rh50"}).copy()
        required_output = {
            "shot_number",
            "latitude",
            "longitude",
            "time",
            "rh98",
            "rh50",
            "quality_flag",
            "degrade_flag",
            "sensitivity",
        }
        missing_output = required_output - set(df.columns)
        if missing_output:
            raise ValueError(f"gediDB output lacks {sorted(missing_output)}; alternative names are not guessed.")
        if pd.api.types.is_float_dtype(df["shot_number"]):
            raise ValueError("shot_number is float: not converted, to avoid precision loss.")
        shot_ids = pd.to_numeric(df["shot_number"], errors="raise")
        if pd.api.types.is_float_dtype(shot_ids) or shot_ids.isna().any() or (shot_ids < 0).any():
            raise ValueError("shot_number cannot be represented unambiguously as uint64.")
        df["shot_number"] = shot_ids.astype("uint64")
        df["time"] = pd.to_datetime(df["time"], errors="raise", utc=True)
        accepted_quality = (
            (pd.to_numeric(df["quality_flag"], errors="raise") == 1)
            & (pd.to_numeric(df["degrade_flag"], errors="raise") == 0)
            & pd.to_numeric(df["sensitivity"], errors="raise").between(sensitivity_min, 1.0)
        )
        if not accepted_quality.all():
            raise ValueError("Provider ignored the quality filters: output not saved.")
        w, s, e, n = box_values
        inside = df["longitude"].between(w, e) & df["latitude"].between(s, n)
        if not inside.all():
            raise ValueError("Provider returned shots outside the AOI: output not saved.")
        # gediDB stores days, not sub-day granule times.
        if not df["time"].dt.date.between(date.fromisoformat(start_time), date.fromisoformat(end_time)).all():
            raise ValueError("Provider returned dates outside the requested window.")
        if "pavd_z" in df and not all(np.asarray(v).shape == (30,) for v in df["pavd_z"]):
            raise ValueError("pavd_z does not have the documented 30 bins: check schema and vertical axis.")
    try:
        package_version = version("gedidb")
    except PackageNotFoundError:
        package_version = "provider injected / package unavailable locally"
    audit = {
        "source": "Public GFZ gediDB GEDI V2 database",
        "endpoint": GEDIDB_ENDPOINT,
        "bucket": GEDIDB_BUCKET,
        "docs": GEDIDB_DOCS,
        "gedidb_version": package_version,
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "bbox": box_values,
        "start_time": start_time,
        "end_time": end_time,
        "requested_variables": variables,
        "missing_optional_variables": sorted(set(requested_optional) - names),
        "quality_filters": filters,
        "n_returned": len(df),
        "columns": list(df.columns),
        "n_duplicate_shot_ids": int(df["shot_number"].duplicated().sum()),
        "rh_units": "m",
        "pavd_z_units": "m2/m3",
        "pavd_height_bin_labels_m": list(range(0, 150, 5)) if "pavd_z" in df else None,
        "time_note": "gediDB time coordinates are days; no sub-day timestamp is inferred",
        "coverage_note": "Bounding-box query; use valid cube support and common spatial support for comparison",
        "ingestion_note": "gediDB ingestion quality filtering and product joins limit the available population",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return df, audit


def fetch_gedi_for_cube(
    cube: dict[str, Any] | Any, output_path: str | Path, start_time: str, end_time: str, **kwargs: Any
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Same as ``fetch_gedi`` with the bbox taken from the cube's own lon/lat (see ``cube_bbox``)."""
    return fetch_gedi(cube_bbox(cube), output_path, start_time, end_time, **kwargs)


def read_gedi(path: str | Path) -> pd.DataFrame:
    """Load GEDI footprint heights and quality fields from a Parquet file.

    Parameters
    ----------
    path : str or Path
        Path to local Parquet containing lon/lat [degrees] and RH98 [m].

    Returns
    -------
    pandas.DataFrame
        Unified columns longitude, latitude, rh_98 [m], quality_flag,
        degrade_flag, sensitivity, and possibly time.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GEDI file not found: {p}.")
    df = pd.read_parquet(p)
    aliases = {
        "longitude": ["longitude", "lon", "Longitude", "lon_lowestmode"],
        "latitude": ["latitude", "lat", "Latitude", "lat_lowestmode"],
        "rh_98": ["rh_98", "RH98", "rh98"],
    }
    for target, candidates in aliases.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise ValueError(f"GEDI column {target} missing; available columns: {list(df.columns)[:30]}")
        if found != target:
            df = df.rename(columns={found: target})
        df[target] = pd.to_numeric(df[target], errors="coerce")
    for col in ["quality_flag", "degrade_flag", "sensitivity"]:
        if col not in df:
            df[col] = np.nan
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the public gediDB; save Parquet and provenance.")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sensitivity", type=float, default=0.95)
    parser.add_argument("--no-profiles", action="store_true")
    parser.add_argument("--full-power-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _, audit = fetch_gedi(
        args.bbox,
        args.output,
        args.start,
        args.end,
        sensitivity_min=args.sensitivity,
        include_profiles=not args.no_profiles,
        full_power_beams_only=args.full_power_only,
        overwrite=args.overwrite,
    )
    print(f"Saved {audit['n_returned']} GEDI shots to {args.output}")


if __name__ == "__main__":
    main()
