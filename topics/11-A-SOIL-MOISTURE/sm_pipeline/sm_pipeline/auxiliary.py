"""Automatic auxiliary/reference sources for a stack footprint:
stations inside the footprint (ISMN export geojson), SMOS L2 items in the time window (MAAP STAC), WorldCover land cover and
Copernicus DEM slope at points and per block, Sentinel-1 RTC gamma0 series (Planetary Computer). All public, no token."""
import json, math, pathlib
import numpy as np, pandas as pd
STAC = "https://catalog.maap.eo.esa.int/catalogue/collections/{col}/items"
WCNAMES = {10: "tree", 20: "shrub", 30: "grassland", 40: "cropland", 50: "built", 60: "bare", 70: "snow_ice", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss_lichen"}

from .catalog import _get

def footprint(product_id, collection="BiomassLevel1c"):
    return _get(STAC.format(col=collection) + "/" + product_id)["geometry"]

def _point_in_ring(lon, lat, ring):
    inside = False; n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][:2]; x2, y2 = ring[(i + 1) % n][:2]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1: inside = not inside
    return inside

def stations_in_footprint(geom, ismn_geojson, active_after=None):
    """ISMN stations (from the ismn_locations.geojson export) inside a footprint; optional filter on data end date."""
    rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    g = json.load(open(ismn_geojson)); rows = []
    for f in g["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]; p = f["properties"]
        if any(_point_in_ring(lon, lat, r) for r in rings) and (active_after is None or p["end"] >= active_after):
            rows.append(dict(station=p["station"], network=p["network"], lat=lat, lon=lon, start=p["start"], end=p["end"]))
    return pd.DataFrame(rows)

def smos_items(lat, lon, t0, t1, product_type="MIR_SMUDP2", halfwidth=0.05):
    """SMOS L2 soil-moisture products on MAAP covering the point in [t0, t1] (ISO dates) -> DataFrame(id, datetime, href)."""
    url = STAC.format(col="SMOS_Open_V7") + f"?limit=200&bbox={lon-halfwidth},{lat-halfwidth},{lon+halfwidth},{lat+halfwidth}&datetime={t0}T00:00:00Z/{t1}T23:59:59Z&productType={product_type}"
    rows = [dict(id=f["id"], datetime=f["properties"]["start_datetime"][:19], href=next((a["href"] for a in f["assets"].values() if "data" in a.get("roles", [])), None))
            for f in _get(url).get("features", []) if product_type in f["id"]]
    return pd.DataFrame(rows).drop_duplicates("id").sort_values("datetime").reset_index(drop=True) if rows else pd.DataFrame(columns=["id", "datetime", "href"])

def _wc_tile(lat, lon):
    la, lo = math.floor(lat / 3) * 3, math.floor(lon / 3) * 3
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"

def _dem_url(lat, lon):
    la, lo = math.floor(lat), math.floor(lon)
    n = f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{n}/{n}.tif"

SOILGRIDS = "https://rest.isric.org/soilgrids/v2.0/properties/query"


def soilgrids_point(lat, lon, cache=None, timeout=45, tries=2):
    """Soil context at a point from ISRIC SoilGrids 2.0 (250 m, CC BY 4.0; Poggio et al. 2021, SOIL 7, 217-240): mean clay,
    sand, silt [%], soil organic carbon [g/kg] and bulk density [kg/dm3] for 0-5 cm and 5-15 cm. One request per point,
    cached as JSON (ISRIC asks for moderate use). Returns {} over water / outside the map or if the service fails."""
    import json, pathlib, requests
    if cache is not None and pathlib.Path(cache).exists():
        return json.load(open(cache))
    q = [("lon", lon), ("lat", lat)] + [("property", p) for p in ("clay", "sand", "silt", "soc", "bdod")] + \
        [("depth", d) for d in ("0-5cm", "5-15cm")] + [("value", "mean")]
    for k in range(tries):                        # the ISRIC service is sometimes slow: one retry, then give up (the run continues)
        try:
            r = requests.get(SOILGRIDS, params=q, timeout=timeout); r.raise_for_status(); break
        except requests.RequestException:
            if k == tries - 1: raise
    names = {"clay": "clay_pct", "sand": "sand_pct", "silt": "silt_pct", "soc": "soc_g_kg", "bdod": "bulk_density_kg_dm3"}
    out = {}
    for L in r.json()["properties"]["layers"]:
        f = float(L["unit_measure"]["d_factor"])
        for d in L["depths"]:
            v = d["values"].get("mean")
            if v is not None: out[f"{names[L['name']]}_{d['label'].replace('cm', '').replace('-', '_')}cm"] = round(v / f, 2)
    if cache is not None: json.dump(out, open(cache, "w"), indent=1)
    return out


def terrain_landcover(lat, lon, half_m=500):
    """WorldCover 2021 class fractions (%) and Copernicus DEM 30 m slope statistics in a square window (public AWS COGs)."""
    import rasterio; from rasterio.windows import from_bounds
    dlat = half_m / 111320; dlon = half_m / (111320 * math.cos(math.radians(lat))); b = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    with rasterio.open(f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{_wc_tile(lat, lon)}_Map.tif") as src:
        a = src.read(1, window=from_bounds(*b, src.transform))
    u, c = np.unique(a, return_counts=True); lc = {WCNAMES.get(int(k), str(k)): round(float(v) / a.size * 100, 1) for k, v in zip(u, c)}
    from rasterio.merge import merge
    tiles = sorted({_dem_url(la, lo) for la in (b[1], b[3]) for lo in (b[0], b[2])})   # all tiles touched by the window
    srcs = [rasterio.open(t) for t in tiles]
    try: mos, tr = merge(srcs, bounds=b); z = mos[0].astype(float)
    finally: [s_.close() for s_ in srcs]
    dx = abs(tr.a) * 111320 * math.cos(math.radians(lat)); dy = abs(tr.e) * 111320
    gy, gx = np.gradient(z, dy, dx); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    return dict(landcover_pct=lc, elev_mean_m=float(z.mean()), elev_range_m=float(np.ptp(z)), slope_mean_deg=float(sl.mean()),
                slope_p90_deg=float(np.percentile(sl, 90)), slope_frac_gt5_pct=float((sl > 5).mean() * 100))


PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"


def _bbox(geom):
    """(west, south, east, north, rings) of a GeoJSON Polygon/MultiPolygon footprint."""
    rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    xs = [x for r in rings for x, _ in r]; ys = [y for r in rings for _, y in r]
    return min(xs), min(ys), max(xs), max(ys), rings


def sentinel1_rtc_series(lat, lon, t0, t1, half_m=300, min_valid_pct=80, max_scenes=60, log=print):
    """Sentinel-1 RTC (radiometrically terrain-corrected gamma0, 10 m, VV/VH) in the same square window around the point:
    one row per scene (datetime, orbit state, relative orbit, platform, gamma0 VV/VH [dB], valid %). Scenes covering less than
    min_valid_pct of the window (scene edges) are flagged `partial` and excluded from the statistics. Returns an empty DataFrame
    (and logs why) when the service is not reachable."""
    import requests, rasterio
    from rasterio.windows import from_bounds; from rasterio.warp import transform as rtransform
    cols = ["datetime", "orbit", "rel_orbit", "platform", "gamma0_db_VV", "gamma0_db_VH", "valid_pct", "partial"]
    try:
        body = {"collections": ["sentinel-1-rtc"], "intersects": {"type": "Point", "coordinates": [lon, lat]}, "datetime": f"{t0}T00:00:00Z/{t1}T23:59:59Z", "limit": max_scenes}
        feats = requests.post(PC_STAC, json=body, timeout=60).json().get("features", [])
        tok = requests.get(PC_TOKEN, timeout=30).json()["token"]
    except Exception as ex:
        log(f"[3] Sentinel-1 RTC not reachable ({str(ex)[:60]}) -> no C-band comparison"); return pd.DataFrame(columns=cols)
    rows = []
    for f in sorted(feats, key=lambda f: f["properties"]["datetime"]):
        row = dict(datetime=f["properties"]["datetime"][:19], orbit=f["properties"].get("sat:orbit_state"), rel_orbit=f["properties"].get("sat:relative_orbit"), platform=f["properties"].get("platform"), valid_pct=np.nan)
        for pol in ("vv", "vh"):
            a = f["assets"].get(pol); key = f"gamma0_db_{pol.upper()}"
            if not a: row[key] = np.nan; continue
            try:
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                    with rasterio.open("/vsicurl/" + a["href"] + "?" + tok) as src:
                        xs, ys = rtransform("EPSG:4326", src.crs, [lon], [lat]); x, y = xs[0], ys[0]
                        arr = src.read(1, window=from_bounds(x - half_m, y - half_m, x + half_m, y + half_m, src.transform), boundless=True, fill_value=np.nan).astype(float)
                arr[arr <= 0] = np.nan; row[key] = float(10 * np.log10(np.nanmean(arr))) if np.isfinite(arr).any() else np.nan
                if pol == "vv": row["valid_pct"] = float(100 * np.isfinite(arr).mean())
            except Exception as ex:
                row[key] = np.nan; row["error"] = str(ex)[:80]
        row["partial"] = not (np.isfinite(row["valid_pct"]) and row["valid_pct"] >= min_valid_pct)
        rows.append(row)
    df = pd.DataFrame(rows, columns=cols + (["error"] if any("error" in r for r in rows) else []))
    if len(df): log(f"[3] Sentinel-1 RTC: {len(df)} scenes, {int((~df.partial).sum())} with >= {min_valid_pct} % window coverage; orbits {sorted(df.rel_orbit.dropna().unique().tolist())}")
    return df


# ---- Sentinel-2 L2A NDVI / NDMI over the analysis window (adapted from the team notebook "Biomass L1C Station Ancillary") ----
S2_SEARCH = "https://earth-search.aws.element84.com/v1/search"       # Earth Search (Element 84), public, no account
S2_COLLECTION = "sentinel-2-c1-l2a"                                    # Collection 1: harmonised reflectance = DN * scale + offset
S2_BANDS = {"red": "B04", "nir": "B08", "nir08": "B8A", "swir16": "B11"}
S2_SCL_CLEAR = (4, 5, 6, 7)                                            # vegetation, bare soil, water, unclassified
S2_GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.vrt",
                   GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE")


def sentinel2_items(lat, lon, t0, t1, max_cloud=90):
    """Sentinel-2 L2A items covering the point between the ISO dates t0 and t1 (tile cloud cover <= max_cloud %), all pages."""
    import requests
    body = {"collections": [S2_COLLECTION], "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "datetime": f"{t0}T00:00:00Z/{t1}T23:59:59Z", "limit": 100, "query": {"eo:cloud_cover": {"lte": max_cloud}}}
    items, url = [], S2_SEARCH
    for _ in range(20):
        r = requests.post(url, json=body, timeout=90); r.raise_for_status(); js = r.json(); feats = js.get("features", [])
        items += feats
        nxt = next((l for l in js.get("links", []) if l.get("rel") == "next"), None)
        if not feats or len(feats) < body["limit"] or nxt is None: break
        url, body = nxt.get("href", S2_SEARCH), {**body, **(nxt.get("body") or {})}
    return items


def _s2_scene(item, lat, lon, box_m):
    """Box statistics of one Sentinel-2 L2A item: clear / cloud / snow fractions (SCL), NDVI and NDMI (mean, median, std)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds
    p = item["properties"]; epsg = int(str(p.get("proj:code") or p.get("proj:epsg")).split(":")[-1])
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [float(lon)], [float(lat)]); x, y = xs[0], ys[0]
    n = max(int(round(box_m / 10)), 1)

    def read_box(href, bounds):
        with rasterio.open(href) as src:
            return src.read(1, window=from_bounds(*bounds, transform=src.transform), out_shape=(n, n), resampling=Resampling.nearest, boundless=True, fill_value=0)

    with rasterio.open(item["assets"]["red"]["href"]) as src:            # snap the box to the 10 m grid of the tile
        left, top = src.transform.c, src.transform.f
    x0 = left + round((x - box_m / 2 - left) / 10) * 10; y1 = top - round((top - (y + box_m / 2)) / 10) * 10
    bounds = (x0, y1 - n * 10, x0 + n * 10, y1)
    row = dict(s2_id=item["id"], datetime=str(pd.Timestamp(p["datetime"])), tile=p.get("grid:code"), platform=p.get("platform"), scene_cloud_cover=p.get("eo:cloud_cover"))
    scl = read_box(item["assets"]["scl"]["href"], bounds)
    row.update(data_fraction=float((scl > 0).mean()), clear_fraction=float(np.isin(scl, S2_SCL_CLEAR).mean()),
               cloud_fraction=float(np.isin(scl, [3, 8, 9, 10]).mean()), snow_fraction=float((scl == 11).mean()))
    clear = np.isin(scl, S2_SCL_CLEAR)
    if not clear.any(): return row
    refl = {}
    for k in S2_BANDS:
        dn = read_box(item["assets"][k]["href"], bounds).astype(float); clear &= dn > 0
        rb = (item["assets"][k].get("raster:bands") or [{}])[0]; refl[k] = dn * rb.get("scale", 1e-4) + rb.get("offset", -0.1)
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = dict(ndvi=(refl["nir"] - refl["red"]) / (refl["nir"] + refl["red"]), ndmi=(refl["nir08"] - refl["swir16"]) / (refl["nir08"] + refl["swir16"]))
    row["n_clear_pixels"] = int(clear.sum())
    for name, v in idx.items():
        v = v[clear & np.isfinite(v)]
        if len(v): row.update({f"{name}_mean": float(v.mean()), f"{name}_median": float(np.median(v)), f"{name}_std": float(v.std())})
    return row


def sentinel2_series(lat, lon, times, box_m=600.0, margin_days=30, max_cloud=90, min_clear=0.8, workers=8, cache=None, log=print):
    """Sentinel-2 L2A NDVI = (B08 - B04) / (B08 + B04) and NDMI = (B8A - B11) / (B8A + B11) (Gao 1996) over a north-aligned box of
    box_m around the point, for every scene from margin_days before the first to margin_days after the last BIOMASS acquisition.
    Indices per pixel, then mean / median / std over the clear pixels of the box (SCL classes 4-7); a scene is 'clear' if
    >= min_clear of the box is clear. A tile overlap (same date twice) keeps the scene with the larger data / clear fraction.
    Public data (Earth Search on AWS, no account); only the pixels of the box are read. Cached as CSV (cache)."""
    if cache is not None and pathlib.Path(cache).exists():
        try: return pd.read_csv(cache)
        except Exception: pass
    from concurrent.futures import ThreadPoolExecutor
    import rasterio
    t = pd.to_datetime(pd.Series(times), utc=True)
    t0, t1 = (t.min() - pd.Timedelta(days=margin_days)).date(), (t.max() + pd.Timedelta(days=margin_days)).date()
    items = sentinel2_items(lat, lon, t0, t1, max_cloud)

    def safe(it):
        try:
            with rasterio.Env(**S2_GDAL_ENV): return _s2_scene(it, lat, lon, box_m)
        except Exception as ex: return dict(s2_id=it["id"], datetime=str(pd.Timestamp(it["properties"]["datetime"])), error=repr(ex)[:120])

    with ThreadPoolExecutor(workers) as ex: rows = list(ex.map(safe, items))
    d = pd.DataFrame(rows)
    if not len(d): log(f"[3] Sentinel-2: no scene {t0} .. {t1}"); return d
    d["date"] = pd.to_datetime(d["datetime"], utc=True).dt.date.astype(str)
    if "error" in d: nerr = int(d["error"].notna().sum()); d = d[d["error"].isna()].drop(columns="error")
    else: nerr = 0
    for c in ("data_fraction", "clear_fraction"):
        if c not in d: d[c] = 0.0
    d = d.sort_values(["data_fraction", "clear_fraction"], ascending=False).drop_duplicates("date").sort_values("datetime")
    d = d[d["data_fraction"] > 0].reset_index(drop=True); d["clear"] = d["clear_fraction"] >= min_clear
    log(f"[3] Sentinel-2 {t0} .. {t1}: {len(items)} items, {len(d)} dates with data, {int(d['clear'].sum())} clear (>= {min_clear:.0%} of the {box_m:.0f} m box)"
        + (f", {nerr} read errors" if nerr else ""))
    if cache is not None: d.to_csv(cache, index=False)
    return d


def s2_at_overpasses(s2, times, max_days=15):
    """Per BIOMASS acquisition: nearest clear scene within +-max_days (value and time difference S2 - BIOMASS [days]) and the
    linear interpolation between the clear scenes before and after (both within max_days)."""
    t = pd.to_datetime(pd.Series(times), utc=True); T = len(t); nan = [None] * T
    out = dict(s2_ndvi_nearest=list(nan), s2_ndmi_nearest=list(nan), s2_dt_days=list(nan), s2_ndvi_interp=list(nan), s2_ndmi_interp=list(nan))
    if s2 is None or not len(s2) or "ndvi_mean" not in s2: return out
    c = s2[s2["clear"].astype(bool) & s2["ndvi_mean"].notna()].copy()
    if not len(c): return out
    ct = pd.to_datetime(c["datetime"], utc=True).reset_index(drop=True); c = c.reset_index(drop=True)
    for i, tt in enumerate(t):
        dd = (ct - tt).dt.total_seconds() / 86400                       # S2 - BIOMASS [days]
        k = int(dd.abs().idxmin())
        if abs(dd[k]) <= max_days:
            out["s2_ndvi_nearest"][i], out["s2_ndmi_nearest"][i], out["s2_dt_days"][i] = float(c.at[k, "ndvi_mean"]), float(c.at[k, "ndmi_mean"]), round(float(dd[k]), 2)
        b, f = dd[dd <= 0], dd[dd > 0]
        if len(b) and len(f) and -b.max() <= max_days and f.min() <= max_days:
            kb, kf = int(b.idxmax()), int(f.idxmin()); w = -dd[kb] / (dd[kf] - dd[kb])
            for v in ("ndvi", "ndmi"): out[f"s2_{v}_interp"][i] = float((1 - w) * c.at[kb, f"{v}_mean"] + w * c.at[kf, f"{v}_mean"])
    return out
