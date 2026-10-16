#!/usr/bin/env python
"""Land cover and terrain per L1C tile (information for the dashboard, not a selection criterion).

    python tile_landcover.py [--res-m 100] [--threads 16]        # -> tile_landcover.csv (resumes from an existing file)

For every tile (bounding box of its stacks in l1c_stack_index.csv): ESA WorldCover 2021 class fractions [% of the LAND area;
water_pct and land_pct in % of the box] and Copernicus DEM
elevation and slope, read from the public AWS COGs at --res-m resolution (GDAL uses the internal overviews, so one read per
COG tile; WorldCover sampled with nearest neighbour = unbiased class fractions; DEM averaged, slope from the averaged DEM =
smoother than the 30 m slope used per analysis window). Missing COG tiles (open ocean) count as water / sea level.
Tiles crossing the antimeridian are skipped (NaN).
"""
import argparse, math, os, pathlib, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
# process-wide GDAL options (rasterio.Env is thread-local; worker threads would otherwise list the whole S3 bucket on every open)
for k, v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                 VSI_CACHE="TRUE", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES").items():
    os.environ.setdefault(k, v)
WC = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif"
WCN = {10: "tree", 20: "shrub", 30: "grassland", 40: "cropland", 50: "built", 60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def _dem(la, lo):
    n = f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{n}/{n}.tif"


def _shape(l, b, r, t, res_m):
    lat = 0.5 * (b + t)
    return max(2, int(round((t - b) * 111320 / res_m))), max(2, int(round((r - l) * 111320 * math.cos(math.radians(lat)) / res_m)))


def tile_stats(w, s, e, n, res_m=100):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from rasterio.errors import RasterioIOError
    if e - w > 180: return {}
    counts = {}
    for la in range(math.floor(s / 3) * 3, math.ceil(n), 3):
        for lo in range(math.floor(w / 3) * 3, math.ceil(e), 3):
            l, b, r, t = max(w, lo), max(s, la), min(e, lo + 3), min(n, la + 3)
            if r <= l or t <= b: continue
            shp = _shape(l, b, r, t, res_m); tid = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
            try:
                with rasterio.open(WC.format(t=tid)) as src:
                    a = src.read(1, window=from_bounds(l, b, r, t, src.transform), out_shape=shp, resampling=Resampling.nearest)
                u, c = np.unique(a, return_counts=True)
                for k, v in zip(u, c): counts[int(k)] = counts.get(int(k), 0) + int(v)
            except RasterioIOError:
                counts[80] = counts.get(80, 0) + shp[0] * shp[1]          # no WorldCover tile = open water
    tot = sum(counts.values()) or 1; water = counts.pop(80, 0) + counts.pop(0, 0)          # 0 = no data = open sea
    land = sum(counts.values())
    out = {f"{WCN[k]}_pct": 0.0 for k in WCN if k != 80}
    for k, v in counts.items():
        if land: out[f"{WCN.get(k, 'other')}_pct"] = out.get(f"{WCN.get(k, 'other')}_pct", 0.0) + 100.0 * v / land
    out.update(water_pct=100.0 * water / tot, land_pct=100.0 * land / tot)
    sl_all, z_all = [], []
    for la in range(math.floor(s), math.ceil(n)):
        for lo in range(math.floor(w), math.ceil(e)):
            l, b, r, t = max(w, lo), max(s, la), min(e, lo + 1), min(n, la + 1)
            if r <= l or t <= b: continue
            shp = _shape(l, b, r, t, res_m)
            try:
                with rasterio.open(_dem(la, lo)) as src:
                    z = src.read(1, window=from_bounds(l, b, r, t, src.transform), out_shape=shp, resampling=Resampling.average).astype(float)
            except RasterioIOError:
                continue                                                  # ocean: no DEM tile
            if z.shape[0] < 3 or z.shape[1] < 3: continue
            dy = (t - b) * 111320 / z.shape[0]; dx = (r - l) * 111320 * math.cos(math.radians(0.5 * (b + t))) / z.shape[1]
            gy, gx = np.gradient(z, dy, dx); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
            land = z > 0.5                                                # ignore sea-level pixels for slope / elevation
            if land.any(): sl_all.append(sl[land].ravel()); z_all.append(z[land].ravel())
    if sl_all:
        sl = np.concatenate(sl_all); z = np.concatenate(z_all)
        out.update(slope_mean_deg=float(sl.mean()), slope_p90_deg=float(np.percentile(sl, 90)), slope_gt5_pct=float((sl > 5).mean() * 100),
                   elev_mean_m=float(z.mean()), elev_p90_m=float(np.percentile(z, 90)))
    return out


def main(a):
    st = pd.read_csv(a.index); st = st[st.n_acq >= 6].copy(); st["tile"] = st["stack"].str.extract(r"(T\d+_F\d+)")[0]
    T = st.groupby("tile").agg(w=("w", "min"), s=("s", "min"), e=("e", "max"), n=("n", "max")).reset_index()
    out = pathlib.Path(a.out); done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["tile"])
    todo = T[~T.tile.isin(done.tile)]
    for extra in a.skip or []:                                            # tiles already done in other part files
        if pathlib.Path(extra).exists(): todo = todo[~todo.tile.isin(pd.read_csv(extra).tile)]
    k, n = (int(x) for x in a.part.split("/")); todo = todo.iloc[k - 1::n]; rows = done.to_dict("records"); t0 = time.time()
    print(f"[landcover] {len(T)} tiles, {len(todo)} to do, {a.res_m} m, {a.threads} threads", flush=True)
    import rasterio
    with rasterio.Env():
        with ProcessPoolExecutor(a.threads) as ex:
            fut = {ex.submit(tile_stats, r.w, r.s, r.e, r.n, a.res_m): r.tile for r in todo.itertuples()}
            for k, f in enumerate(as_completed(fut), 1):
                try: rec = f.result()
                except Exception as e: rec = {"error": str(e)[:80]}
                rows.append(dict(tile=fut[f], res_m=a.res_m, **rec))
                if k % 50 == 0 or k == len(fut):
                    pd.DataFrame(rows).round(2).to_csv(out, index=False); print(f"[landcover] {k}/{len(fut)} tiles, {round(time.time() - t0)} s", flush=True)
    print(f"[landcover] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(HERE / "l1c_stack_index.csv")); ap.add_argument("--out", default="tile_landcover.csv")
    ap.add_argument("--res-m", type=float, default=100); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--part", default="1/1", help="k/n: every n-th remaining tile starting at k (parallel runs into separate --out files)")
    ap.add_argument("--skip", nargs="*", help="other output files whose tiles are skipped")
    main(ap.parse_args())
