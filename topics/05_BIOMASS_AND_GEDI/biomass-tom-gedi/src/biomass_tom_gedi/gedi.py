"""GEDI shots: download from gediDB with the MAAP quality filters, and load/clean a shot table.

Server-side filters run inside TileDB (before download); the three filters TileDB cannot express
(|elev - DEM|, degrade_flag whitelist, day/night) are applied client-side in pandas.
gediDB: Besnard, Dombrowski & Holcomb, https://doi.org/10.5281/zenodo.13885229
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SERVER_FILTERS = {
    "quality_flag": "== 1",
    "l2a_quality_flag": "== 1",
    "l2b_quality_flag": "== 1",
    "algorithmrun_flag": "== 1",          # MAAP name: l2b_algorithm_run_flag
    "l4_quality_flag": "== 1",
    "l2_quality_flag": "== 1",
    "wsci_quality_flag": "== 1",
    "elev_lowestmode": ">= 0",
    "sensitivity": "> 0.95",
    "surface_flag": "== 1",
    "stale_return_flag": "== 0",
    "landsat_treecover": "> 10",
    "landsat_water_persistence": "< 10",
    "urban_proportion": "< 50",
    "leaf_off_flag": "== 0",
}

# tens digit = attitude degradation (accepted 0,1,2,3,4,6); ones digit = trajectory degradation (accepted 0,3,8)
DEGRADE_OK = [0, 3, 8, 10, 13, 18, 20, 23, 28, 30, 33, 38, 40, 43, 48, 60, 63, 68]
POWER_BEAMS = ["BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]   # full-power beams (usable day or night)

RH_PERCENTILES = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 98, 100]
PROFILES = ["pavd_z", "pai_z", "cover_z"]                        # 30 height bins of 5 m each

SCALARS = [
    "agbd", "agbd_se", "agbd_t", "agbd_t_se", "agbd_pi_lower", "agbd_pi_upper", "predict_stratum",
    "cover", "pai", "fhd_normal", "omega", "pgap_theta",
    "energy_total", "num_detectedmodes", "rx_maxamp", "sd_corrected", "zcross", "toploc", "selected_algorithm",
    "elev_lowestmode", "elev_highestreturn_a1", "elev_highestreturn_a2",
    "wsci", "wsci_xy", "wsci_z",
    "wsci_pi_lower", "wsci_pi_upper", "wsci_xy_pi_lower", "wsci_xy_pi_upper", "wsci_z_pi_lower", "wsci_z_pi_upper",
    "rhog", "rhov", "rv", "rg",
    "digital_elevation_model", "digital_elevation_model_srtm",
    "landsat_treecover", "modis_treecover", "pft_class", "region_class",
    "urban_proportion", "landsat_water_persistence",
    "leaf_off_flag", "leaf_off_doy", "leaf_on_cycle", "leaf_on_doy",
    "sensitivity", "sensitivity_a1", "sensitivity_a2",
    "degrade_flag", "surface_flag", "solar_elevation",
    "l2_quality_flag", "l4_quality_flag", "l2b_quality_flag", "wsci_quality_flag",
    "beam_name", "quality_flag", "l2a_quality_flag", "algorithmrun_flag", "stale_return_flag",
]


def apply_client_filters(df: pd.DataFrame) -> pd.DataFrame:
    beam = df["beam_name"].str.lstrip("/")                     # gedidb stores "/BEAM0101"
    keep = (
        ((df["elev_lowestmode"] - df["digital_elevation_model"]).abs() < 150)
        & df["degrade_flag"].isin(DEGRADE_OK)
        & (beam.isin(POWER_BEAMS) | (df["solar_elevation"] < 0))   # power beam OR night
    )
    return df[keep]


def download(bbox: tuple[float, float, float, float], start_time: str, end_time: str,
             server_filters: dict | None = None) -> pd.DataFrame:
    """Query gediDB for the bounding box (min_lon, min_lat, max_lon, max_lat); needs `pip install gedidb geopandas`."""
    try:
        import geopandas as gpd
        import gedidb as gdb
        from shapely.geometry import box
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("step 01 needs the optional dependencies: pip install 'biomass-tom-gedi[gedidb]'") from e

    roi = gpd.GeoDataFrame([{"name": "ROI", "geometry": box(*bbox)}], crs="EPSG:4326")
    provider = gdb.GEDIProvider(storage_type="s3", s3_bucket="dog-ext.gedidb.gedi-l2-l4-v002.0",
                                url="https://s3.gfz-potsdam.de")
    variables = SCALARS + [f"rh:{p}" for p in RH_PERCENTILES] + PROFILES
    raw = provider.get_data(variables=variables, query_type="bounding_box", geometry=roi,
                            start_time=start_time, end_time=end_time, return_type="dataframe",
                            **(server_filters or SERVER_FILTERS))
    if raw is None or len(raw) == 0:
        raise RuntimeError("No shots passed the server-side filters in this ROI/time window.")
    log.info("%d shots passed the server-side filters", len(raw))

    df = apply_client_filters(raw).copy()
    log.info("%d shots after client-side filters", len(df))
    for var in PROFILES:                                       # expand 30-bin profiles -> var_0 .. var_29
        arr = np.vstack(df[var].to_numpy())
        df = pd.concat([df.drop(columns=var),
                        pd.DataFrame(arr, index=df.index, columns=[f"{var}_{i}" for i in range(arr.shape[1])])], axis=1)
    df = df.rename(columns={f"rh_p{p}": f"rh_{p}" for p in RH_PERCENTILES} | {"algorithmrun_flag": "l2b_algorithm_run_flag"})
    df["beam_name"] = df["beam_name"].str.lstrip("/")
    return df.reset_index(drop=True)


def load_gedi_table(path: str | Path, bbox: tuple[float, float, float, float] | None = None,
                    start_time: str | None = None, max_days: int | None = None) -> pd.DataFrame:
    """Read a GEDI shot table (.parquet/.csv), normalise column names, crop to the TOM bbox.

    Needs `latitude` and `longitude`; the along-track views also need `shot_number` and `beam_name`.
    `max_days`: keep shots within +-max_days of `start_time` (the BIOMASS acquisition date).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GEDI table not found: {path} (run step 01 or point paths.gedi_table at your own file)")
    g = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    g.columns = [c.strip().lower().replace(" ", "_") for c in g.columns]
    g = g.rename(columns={"lat_lowestmode": "latitude", "lon_lowestmode": "longitude", "rh98": "rh_98"})
    if not {"latitude", "longitude"} <= set(g.columns):
        raise KeyError("The GEDI table needs 'latitude' and 'longitude' columns")
    log.info("%d GEDI shots in %s", len(g), path.name)
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        g = g[g.latitude.between(min_lat, max_lat) & g.longitude.between(min_lon, max_lon)]
        log.info("%d inside the TOM bounding box", len(g))
    if max_days is not None and "time" in g.columns and start_time:
        t = pd.to_datetime(g["time"], utc=True).dt.tz_localize(None)
        g = g[(t - pd.Timestamp(start_time[:10])).abs() <= pd.Timedelta(days=max_days)]
        log.info("%d within +-%d days of the BIOMASS acquisition", len(g), max_days)
    return g.reset_index(drop=True)
