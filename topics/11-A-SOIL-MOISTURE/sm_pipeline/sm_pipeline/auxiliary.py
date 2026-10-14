"""Automatic auxiliary/reference sources for a stack footprint:
stations inside the footprint (ISMN export geojson), SMOS L2 items in the time window (MAAP STAC), ERA5-Land at points,
WorldCover land cover and Copernicus DEM slope at points. All public, no token."""
import json, math, pathlib, urllib.request
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

def sentinel1_items(lat, lon, t0, t1, halfwidth=0.05):
    url = STAC.format(col="sentinel-1-grd") + f"?limit=200&bbox={lon-halfwidth},{lat-halfwidth},{lon+halfwidth},{lat+halfwidth}&datetime={t0}T00:00:00Z/{t1}T23:59:59Z"
    return pd.DataFrame([dict(id=f["id"], datetime=f["properties"]["start_datetime"][:19]) for f in _get(url).get("features", [])])

def _wc_tile(lat, lon):
    la, lo = math.floor(lat / 3) * 3, math.floor(lon / 3) * 3
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"

def _dem_url(lat, lon):
    la, lo = math.floor(lat), math.floor(lon)
    n = f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{n}/{n}.tif"

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


# ---- candidate analysis windows inside a footprint (transparent, criteria printed and saved) ------------------------------
CRITERIA_FILE = pathlib.Path(__file__).resolve().parents[1] / "criteria.json"
_DEFAULT_CRITERIA = dict(min_grass_crop_pct=90, max_tree_pct=5, max_built_pct=1, max_water_pct=1, max_slope_mean_deg=5, max_slope_gt5_pct=20,
                         rank_by=["slope_mean_deg", "tree_pct"], n_best=5, min_separation_km=5)


def load_criteria(path=None):
    """Criteria from criteria.json (package root) or a user file; keys starting with '_' are documentation. Missing keys -> defaults."""
    f = pathlib.Path(path) if path else CRITERIA_FILE
    crit = dict(_DEFAULT_CRITERIA)
    if f.exists():
        try: crit.update({k: v for k, v in json.load(open(f)).items() if not k.startswith("_")})
        except Exception as ex: print(f"[cand] criteria file {f} not readable ({str(ex)[:60]}) -> defaults")
    crit["_source"] = str(f) if f.exists() else "defaults"
    return crit


CANDIDATE_CRITERIA = load_criteria()


def _bbox(geom):
    rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    xs = [x for r in rings for x, y in (q[:2] for q in r)]; ys = [y for r in rings for x, y in (q[:2] for q in r)]
    return min(xs), min(ys), max(xs), max(ys), rings


def candidate_windows(geom, block_m=600, sample_m=50, dem_m=90, criteria=None, log=print):
    """Rate every block_m x block_m block of a footprint as an analysis window: WorldCover 2021 class fractions (sampled every
    sample_m) and Copernicus DEM 30 m slope (sampled every dem_m). Returns a DataFrame with one row per block (lat, lon, class
    percentages, slope_mean_deg, slope_gt5_pct, suitable, rank). Nothing is chosen silently: `suitable` is the AND of the
    criteria in CANDIDATE_CRITERIA (printed), `rank` orders suitable blocks by slope, then tree fraction."""
    import rasterio; from rasterio.windows import from_bounds; from rasterio.enums import Resampling
    crit = dict(criteria or CANDIDATE_CRITERIA); w, s, e, n, rings = _bbox(geom); lat0 = 0.5 * (s + n)
    dlat = block_m / 111320; dlon = block_m / (111320 * math.cos(math.radians(lat0)))
    nrow, ncol = int(math.ceil((n - s) / dlat)), int(math.ceil((e - w) / dlon))
    log(f"[cand] footprint {w:.2f}..{e:.2f} E, {s:.2f}..{n:.2f} N -> {nrow} x {ncol} blocks of {block_m} m; criteria from {crit.get('_source', 'defaults')}: "
        + ", ".join(f"{k} {v}" for k, v in crit.items() if not k.startswith("_")))
    # land cover: per WorldCover tile, read the bbox part decimated to sample_m, accumulate class counts per block
    counts = {}; nsamp = np.zeros((nrow, ncol))
    for la in range(int(math.floor(s / 3) * 3), int(math.floor(n / 3) * 3) + 1, 3):
        for lo in range(int(math.floor(w / 3) * 3), int(math.floor(e / 3) * 3) + 1, 3):
            url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{_wc_tile(la + 1, lo + 1)}_Map.tif"
            bb = (max(w, lo), max(s, la), min(e, lo + 3), min(n, la + 3))
            if bb[0] >= bb[2] or bb[1] >= bb[3]: continue
            try:
                with rasterio.open(url) as src:
                    win = from_bounds(*bb, src.transform); dec = max(int(round(sample_m / (abs(src.transform.a) * 111320 * math.cos(math.radians(lat0))))), 1)
                    oh, ow = max(int(win.height // dec), 1), max(int(win.width // dec), 1)
                    a = src.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.nearest)
            except Exception as ex:
                log(f"[cand] WorldCover tile {la},{lo} not read ({str(ex)[:60]})"); continue
            lats = bb[3] - (np.arange(oh) + 0.5) * (bb[3] - bb[1]) / oh; lons = bb[0] + (np.arange(ow) + 0.5) * (bb[2] - bb[0]) / ow
            I = np.clip(((n - lats) / dlat).astype(int), 0, nrow - 1)[:, None] * np.ones((1, ow), int); J = np.clip(((lons - w) / dlon).astype(int), 0, ncol - 1)[None, :] * np.ones((oh, 1), int)
            flat = (I * ncol + J).ravel()
            for cls in np.unique(a):
                counts[int(cls)] = counts.get(int(cls), np.zeros(nrow * ncol)) + np.bincount(flat[a.ravel() == cls], minlength=nrow * ncol)
            nsamp += np.bincount(flat, minlength=nrow * ncol).reshape(nrow, ncol)
    # slope from the DEM tiles touched, decimated to dem_m
    sl_sum = np.zeros(nrow * ncol); sl_gt5 = np.zeros(nrow * ncol); sl_n = np.zeros(nrow * ncol)
    for la in range(int(math.floor(s)), int(math.floor(n)) + 1):
        for lo in range(int(math.floor(w)), int(math.floor(e)) + 1):
            bb = (max(w, lo), max(s, la), min(e, lo + 1), min(n, la + 1))
            if bb[0] >= bb[2] or bb[1] >= bb[3]: continue
            try:
                with rasterio.open(_dem_url(la + 0.5, lo + 0.5)) as src:
                    win = from_bounds(*bb, src.transform); dec = max(int(round(dem_m / (abs(src.transform.e) * 111320))), 1)
                    oh, ow = max(int(win.height // dec), 2), max(int(win.width // dec), 2)
                    z = src.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.average).astype(float)
            except Exception as ex:
                log(f"[cand] DEM tile {la},{lo} not read ({str(ex)[:60]})"); continue
            dy = (bb[3] - bb[1]) / oh * 111320; dx = (bb[2] - bb[0]) / ow * 111320 * math.cos(math.radians(lat0))
            gy, gx = np.gradient(z, dy, dx); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
            lats = bb[3] - (np.arange(oh) + 0.5) * (bb[3] - bb[1]) / oh; lons = bb[0] + (np.arange(ow) + 0.5) * (bb[2] - bb[0]) / ow
            I = np.clip(((n - lats) / dlat).astype(int), 0, nrow - 1)[:, None] * np.ones((1, ow), int); J = np.clip(((lons - w) / dlon).astype(int), 0, ncol - 1)[None, :] * np.ones((oh, 1), int)
            flat = (I * ncol + J).ravel()
            sl_sum += np.bincount(flat, weights=sl.ravel(), minlength=nrow * ncol); sl_gt5 += np.bincount(flat, weights=(sl > 5).ravel(), minlength=nrow * ncol); sl_n += np.bincount(flat, minlength=nrow * ncol)
    tot = np.maximum(nsamp.ravel(), 1)
    df = pd.DataFrame({"block_row": np.repeat(np.arange(nrow), ncol), "block_col": np.tile(np.arange(ncol), nrow)})
    df["lat"] = n - (df.block_row + 0.5) * dlat; df["lon"] = w + (df.block_col + 0.5) * dlon
    for cls, name in WCNAMES.items(): df[f"{name}_pct"] = 100 * counts.get(cls, np.zeros(nrow * ncol)) / tot
    df["n_samples"] = nsamp.ravel(); df["slope_mean_deg"] = np.where(sl_n > 0, sl_sum / np.maximum(sl_n, 1), np.nan); df["slope_gt5_pct"] = np.where(sl_n > 0, 100 * sl_gt5 / np.maximum(sl_n, 1), np.nan)
    df["in_footprint"] = [any(_point_in_ring(lo, la, r) for r in rings) for la, lo in zip(df.lat, df.lon)]
    df = df[df.in_footprint & (df.n_samples >= 0.5 * (block_m / sample_m) ** 2)].copy()
    df["suitable"] = ((df.grassland_pct + df.cropland_pct >= crit["min_grass_crop_pct"]) & (df.tree_pct <= crit["max_tree_pct"]) & (df.built_pct <= crit["max_built_pct"])
                      & (df.water_pct <= crit["max_water_pct"]) & (df.slope_mean_deg <= crit["max_slope_mean_deg"]) & (df.slope_gt5_pct <= crit["max_slope_gt5_pct"]))
    df["rank"] = np.nan
    suit = df[df.suitable].sort_values([c for c in crit.get("rank_by", ["slope_mean_deg", "tree_pct"]) if c in df]); df.loc[suit.index, "rank"] = np.arange(1, len(suit) + 1)
    log(f"[cand] {len(df)} blocks inside the footprint, {int(df.suitable.sum())} suitable ({100 * df.suitable.mean():.0f} %); "
        f"land cover of the footprint: " + ", ".join(f"{k} {v:.0f} %" for k, v in sorted({nm: df[f'{nm}_pct'].mean() for nm in WCNAMES.values()}.items(), key=lambda x: -x[1])[:4]))
    return df.reset_index(drop=True)


def pick_spread(cands, n=None, min_sep_km=None, criteria=None):
    """The n best-ranked suitable blocks that are at least min_sep_km apart (greedy), so that the chosen windows sample
    different model grid cells / different parts of the tile."""
    crit = criteria or CANDIDATE_CRITERIA; n = n or crit.get("n_best", 5); min_sep_km = min_sep_km if min_sep_km is not None else crit.get("min_separation_km", 5)
    out = []
    for _, r in cands[cands.suitable].sort_values("rank").iterrows():
        if all(math.hypot((r.lat - q.lat) * 111.32, (r.lon - q.lon) * 111.32 * math.cos(math.radians(r.lat))) >= min_sep_km for q in out): out.append(r)
        if len(out) >= n: break
    return pd.DataFrame(out).reset_index(drop=True) if out else cands.iloc[0:0]
