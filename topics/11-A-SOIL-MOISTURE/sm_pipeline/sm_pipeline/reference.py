"""Step 5: reference data at the overpass times: ERA5-Land hourly (Open-Meteo), ISMN station files, SMOS later."""
import json, urllib.request
import pandas as pd, numpy as np
OMH = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={s}&end_date={e}"
       "&hourly=precipitation,temperature_2m,dew_point_2m,snow_depth,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,"
       "soil_moisture_28_to_100cm,soil_moisture_100_to_255cm&models={model}&timezone=UTC")
# Open-Meteo archive models with soil moisture. NOTE: without an explicit model the API returns 'best_match', which for
# recent dates is the ECMWF IFS analysis (9 km), not ERA5-Land. Always request models explicitly and keep them apart.
MODELS = {"era5_land": "ERA5-Land 9 km (ECMWF reanalysis, ~5 d latency)", "era5": "ERA5 31 km", "ecmwf_ifs": "ECMWF IFS analysis 9 km (operational)"}

def _get_json(url, tries=4):
    import time
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r: return json.loads(r.read())
        except Exception as e: last = e; time.sleep(5 * (i + 1))
    raise last

def model_hourly(lat, lon, start, end, model="era5_land"):
    j = _get_json(OMH.format(lat=lat, lon=lon, s=start, e=end, model=model))
    d = pd.DataFrame(j["hourly"]); d["time"] = pd.to_datetime(d["time"], utc=True); d = d.set_index("time")
    d.attrs.update(model=model, grid_lat=j["latitude"], grid_lon=j["longitude"], grid_elev=j["elevation"]); return d

def era5land_hourly(lat, lon, start, end):
    return model_hourly(lat, lon, start, end, "era5_land")

def all_model_sources(lat, lon, start, end):
    """dict model -> hourly DataFrame for every Open-Meteo model that has soil moisture (ERA5-Land, ERA5, ECMWF IFS)."""
    out = {}
    for m in MODELS:
        try: out[m] = model_hourly(lat, lon, start, end, m)
        except Exception as e:
            hint = " (Open-Meteo quota of this IP used up for today; resets 00:00 UTC; the run continues without models)" if "429" in str(e) else ""
            print("model", m, "failed:", str(e)[:60] + hint)
    return out

ISMN_DIR_DEFAULT = "~/shared-buckets/3482590a-d411-4ac1-aebb-95a343d33b3a/ISMN"
ISMN_GEOJSON_DEFAULT = ISMN_DIR_DEFAULT + "/ismn_locations.geojson"
# ISMN quality flags (ISMN documentation): G good; C01 SM<0, C02 SM>0.60, C03 SM>saturation; D01 soil T<0, D02 air T<0,
# D03 GLDAS T<0, D04 snow, D05 spike, D06/D07 jumps, D08-D10 plateaus; M missing. For analysis use G only.
SENSOR_RE = r"^(?P<instrument>.+?)_(?P<variable>soil_moisture|soil_temperature|air_temperature|precipitation|snow_depth|snow_water_equivalent|soil_suction|surface_temperature)_(?P<depth_from>-?\d+\.\d+)_(?P<depth_to>-?\d+\.\d+)$"

def read_ismn_export(path):
    """ISMN data-viewer CSV export (long format: one row per sensor and hour; columns network, station, sensor, date_time,
    <variable>, <variable>_flag, <variable>_orig_flag). Returns a tidy DataFrame
    [network, station, sensor, variable, depth_from, depth_to, date_time (UTC), value, flag]."""
    d = pd.read_csv(path, index_col=0, low_memory=False); d["date_time"] = pd.to_datetime(d["date_time"], utc=True)
    parts = d["sensor"].str.extract(SENSOR_RE); d = pd.concat([d, parts], axis=1)
    d["depth_from"] = d["depth_from"].astype(float); d["depth_to"] = d["depth_to"].astype(float)
    val = np.full(len(d), np.nan); flag = np.full(len(d), None, dtype=object)
    for v in d["variable"].dropna().unique():
        m = (d["variable"] == v).values
        if v in d: val[m] = pd.to_numeric(d.loc[m, v], errors="coerce").values
        if f"{v}_flag" in d: flag[m] = d.loc[m, f"{v}_flag"].values
    d["value"], d["flag"] = val, flag
    return d[["network", "station", "sensor", "variable", "depth_from", "depth_to", "date_time", "value", "flag"]]

def ismn_station_file(ismn_dir, network, station):
    """Path of <network>+<station>.csv anywhere below ismn_dir (e.g. ISMN/temporal/ROMPS+Maydantal.csv), else None (point data are optional)."""
    import os, pathlib
    root = pathlib.Path(os.path.expanduser(ismn_dir))
    if not root.exists(): return None
    hits = sorted(root.rglob(f"{network}+{station}.csv"))
    return hits[0] if hits else None

def ismn_series(path, good_only=True):
    """dict sensor_label -> Series (UTC index) for every sensor in an export file, e.g. 'soil_moisture_0.1m', 'precipitation_-2.0m'.
    good_only keeps flag 'G' values; the unfiltered series is returned too under '<label>_allflags'."""
    d = read_ismn_export(path); out = {}
    for (v, z0, z1), g in d.groupby(["variable", "depth_from", "depth_to"]):
        lab = f"{v}_{z0:g}m" if z0 == z1 else f"{v}_{z0:g}-{z1:g}m"
        g = g.sort_values("date_time").drop_duplicates("date_time")
        out[lab + "_allflags"] = pd.Series(g["value"].values, index=g["date_time"], name=lab + "_allflags")
        gg = g[g["flag"] == "G"] if good_only else g
        out[lab] = pd.Series(gg["value"].values, index=gg["date_time"], name=lab)
    return out

def read_ismn_file(path):
    """Backwards-compatible single-series reader: returns the shallowest good-flag soil-moisture series of an export file."""
    ser = ismn_series(path); keys = sorted([k for k in ser if k.startswith("soil_moisture") and not k.endswith("_allflags")])
    return ser[keys[0]] if keys else next(iter(ser.values()))

def at_overpass(series, times, tol="1h", window_h=None):
    """Value of a time series at each overpass time (nearest within tol) or the mean over +-window_h hours."""
    out = []
    for t in pd.to_datetime(times, utc=True):
        if window_h: out.append(series[(series.index >= t - pd.Timedelta(hours=window_h)) & (series.index <= t + pd.Timedelta(hours=window_h))].mean())
        else:
            idx = series.index.get_indexer([t], method="nearest", tolerance=pd.Timedelta(tol)); out.append(series.iloc[idx[0]] if idx[0] >= 0 else np.nan)
    return np.array(out, dtype=float)

def rain_between(precip_hourly, times):
    """Rain sum between consecutive overpass times (first element = NaN)."""
    t = pd.to_datetime(times, utc=True); out = [np.nan]
    for a, b in zip(t[:-1], t[1:]): out.append(precip_hourly[(precip_hourly.index > a) & (precip_hourly.index <= b)].sum(min_count=1))
    return np.array(out)

# --- SMOS L2 soil moisture (MIR_SMUDP2 NetCDF half-orbit files, MAAP collection SMOS_Open_V7) ---------------------------
def smos_at_point(nc_path, lat, lon, radius_km=25.0, max_dqx=0.07, max_rfi_prob=0.1):
    """Mean SMOS L2 soil moisture of the ISEA grid points within radius_km of (lat, lon), after quality filtering.
    Returns dict(sm, n, dqx, time) or None if no valid point. Variable names follow the L2 SM UDP NetCDF (v700);
    adjust if the file differs (print(ds) once on MAAP). Download the file with the MAAP token (smos-diss.eo.esa.int)."""
    import xarray as xr, math
    ds = xr.open_dataset(nc_path)
    la = ds[[v for v in ds.data_vars if v.lower() in ("latitude", "lat")][0]].values
    lo = ds[[v for v in ds.data_vars if v.lower() in ("longitude", "lon")][0]].values
    d = np.hypot(la - lat, (lo - lon) * math.cos(math.radians(lat))) * 111.32
    sel = d <= radius_km
    if not sel.any(): return None
    sm = ds["Soil_Moisture"].values[sel]; ok = np.isfinite(sm) & (sm > 0)
    if "Soil_Moisture_DQX" in ds: ok &= np.nan_to_num(ds["Soil_Moisture_DQX"].values[sel], nan=9) <= max_dqx
    if "RFI_Prob" in ds: ok &= np.nan_to_num(ds["RFI_Prob"].values[sel], nan=1) <= max_rfi_prob
    if not ok.any(): return dict(sm=np.nan, n=0, note="all points flagged")
    return dict(sm=float(np.mean(sm[ok])), n=int(ok.sum()), dqx=float(np.nanmean(ds["Soil_Moisture_DQX"].values[sel][ok])) if "Soil_Moisture_DQX" in ds else np.nan)

# --- reference tiers -------------------------------------------------------------------------------------------------
TIERS = {
    1: "in-situ station inside the footprint (ISMN) -> ground truth, used to TEST the method (sensitivity, RMSE)",
    2: "observation-based satellite products at the point (SMOS L2 on MAAP; SMAP L3/L4 with Earthdata login) -> independent but coarse (25-40 km)",
    3: "model soil moisture (ERA5-Land, ERA5, ECMWF IFS) -> everywhere, consistency check only (direction, timing), never calibration target",
}
def tier_of(stations_in_footprint, smos_available):
    if len(stations_in_footprint): return 1
    return 2 if smos_available else 3
