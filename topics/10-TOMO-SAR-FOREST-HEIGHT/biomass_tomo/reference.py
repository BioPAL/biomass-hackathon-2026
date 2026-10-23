"""
Independent reference data in radar geometry: ALS canopy height models (pooled over the
multilook footprint of every coherence-grid pixel, as in the Capon height notebook) and
any other map raster (e.g. an AGBD map) sampled at the SAR geolocation.
"""
import importlib.util
import os
import pathlib
import sqlite3
import sys
import warnings

import numpy as np


def fix_proj_data():
    """Point PROJ to the newest proj.db available (avoids conda/pip PROJ conflicts)."""
    import rasterio
    import rasterio._env

    def version(folder):
        db = pathlib.Path(folder) / "proj.db"
        if not db.exists():
            return None
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
                meta = dict(con.execute("SELECT key, value FROM metadata").fetchall())
            return int(meta["DATABASE.LAYOUT.VERSION.MAJOR"]), int(meta["DATABASE.LAYOUT.VERSION.MINOR"])
        except Exception:
            return None
    cands = [pathlib.Path(sys.prefix) / "share" / "proj", pathlib.Path(rasterio.__file__).parent / "proj_data"]
    if importlib.util.find_spec("pyproj"):
        import pyproj
        cands.append(pathlib.Path(pyproj.__file__).parent / "proj_dir" / "share" / "proj")
    found = {str(c): version(c) for c in cands if version(c)}
    if not found:
        return None
    best = max(found, key=lambda c: found[c])
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = best
    rasterio._env.set_proj_data_search_path(best)
    if importlib.util.find_spec("pyproj"):
        import pyproj
        pyproj.datadir.set_data_dir(best)
    return best


def raster_bbox_lonlat(path):
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def _read_band(src, band):
    if isinstance(band, str):
        names = [d for d in src.descriptions]
        idx = names.index(band) + 1 if band in names else 1
    else:
        idx = int(band)
    return src.read(idx, masked=True).astype(np.float32).filled(np.nan)


def chm_to_footprints(chm_path, lat, lon, window_r, window_x, ds_r, ds_x, dr, dx, incidence_deg,
                      band="chm_lspikefree", block_m=5.0, min_coverage=0.8, percentiles=(50, 75, 90, 95, 99)):
    """
    Pool a CHM over the multilook footprint of every coherence-grid pixel.
    lat, lon: native-resolution geolocation of the crop (range, azimuth).
    Returns dict of (Nr_coh, Nx_coh) maps: als_mean, als_top, als_pXX, plus 'native_top'
    (range, azimuth) and 'n_blocks'. Empty dict if the CHM does not overlap the crop.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from scipy.spatial import cKDTree

    nr, nx = lat.shape
    nr_coh, nx_coh = len(range(0, nr, ds_r)), len(range(0, nx, ds_x))
    theta = np.deg2rad(np.nanmean(incidence_deg))
    with rasterio.open(chm_path) as src:
        chm = _read_band(src, band)
        crs, tr, res = src.crs, src.transform, src.res[0]
    step = max(int(round(block_m / res)), 1)
    ny, nxb = (chm.shape[0] // step) * step, (chm.shape[1] // step) * step
    blocks = chm[:ny, :nxb].reshape(ny // step, step, nxb // step, step).transpose(0, 2, 1, 3).reshape(ny // step, nxb // step, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ok = np.isfinite(blocks).mean(axis=2) >= 0.5
        bmean, bmax = np.nanmean(blocks, axis=2)[ok], np.nanmax(blocks, axis=2)[ok]
    br, bc = np.nonzero(ok)
    east_b, north_b = rasterio.transform.xy(tr, br * step + step / 2 - 0.5, bc * step + step / 2 - 0.5, offset="center")
    east_b, north_b = np.asarray(east_b), np.asarray(north_b)
    del blocks, chm

    lon_b, lat_b = warp_transform(crs, "EPSG:4326", [east_b.min(), east_b.max()], [north_b.min(), north_b.max()])
    tol = 0.01
    near = (lat >= min(lat_b) - tol) & (lat <= max(lat_b) + tol) & (lon >= min(lon_b) - tol) & (lon <= max(lon_b) + tol)
    if not near.any():
        return {}
    r_nat, x_nat = np.nonzero(near)
    east_s, north_s = warp_transform("EPSG:4326", crs, lon[near], lat[near])
    dist, nearest = cKDTree(np.column_stack([east_s, north_s])).query(np.column_stack([east_b, north_b]))
    ground_dr = dr / np.sin(theta)
    keep = dist <= np.hypot(ground_dr, dx)
    blk_r, blk_x = r_nat[nearest[keep]], x_nat[nearest[keep]]
    blk_mean, blk_max = bmean[keep], bmax[keep]

    native_top = np.full(nr * nx, np.nan, dtype=np.float32)
    np.fmax.at(native_top, blk_r * nx + blk_x, blk_max)

    half_r, half_x = window_r // 2, window_x // 2
    mp, mb = [], []
    for di in range(-(half_r // ds_r) - 1, half_r // ds_r + 2):
        i = np.round(blk_r / ds_r).astype(int) + di
        for dj in range(-(half_x // ds_x) - 1, half_x // ds_x + 2):
            j = np.round(blk_x / ds_x).astype(int) + dj
            inside = ((i >= 0) & (i < nr_coh) & (j >= 0) & (j < nx_coh)
                      & (np.abs(blk_r - i * ds_r) <= half_r) & (np.abs(blk_x - j * ds_x) <= half_x))
            mp.append((i * nx_coh + j)[inside])
            mb.append(np.flatnonzero(inside))
    mp, mb = np.concatenate(mp), np.concatenate(mb)
    expected = (window_r * ground_dr) * (window_x * dx) / (step * res) ** 2
    count = np.bincount(mp, minlength=nr_coh * nx_coh)
    covered = count >= min_coverage * expected

    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, vals in {"als_mean": blk_mean, "als_top": blk_max}.items():
            total = np.bincount(mp, weights=vals[mb], minlength=nr_coh * nx_coh)
            out[name] = np.where(covered, total / count, np.nan).reshape(nr_coh, nx_coh).astype(np.float32)
    order = np.argsort(mp, kind="stable")
    ps, vs = mp[order], blk_mean[mb][order]
    starts = np.flatnonzero(np.r_[True, np.diff(ps) != 0])
    pct = {q: np.full(nr_coh * nx_coh, np.nan, dtype=np.float32) for q in percentiles}
    for pixel, group in zip(ps[starts], np.split(vs, starts[1:])):
        if covered[pixel]:
            for q, v in zip(percentiles, np.percentile(group, percentiles)):
                pct[q][pixel] = v
    for q in percentiles:
        out[f"als_p{q}"] = pct[q].reshape(nr_coh, nx_coh)
    out["native_top"] = native_top.reshape(nr, nx)
    out["n_blocks"] = count.reshape(nr_coh, nx_coh)
    return out


def sample_raster(path, lat, lon, band=1, resampling="nearest"):
    """Values of a map raster (any CRS) at the given lat/lon (e.g. an AGBD map on the coherence grid)."""
    import rasterio
    from rasterio.warp import transform as warp_transform
    with rasterio.open(path) as src:
        data = _read_band(src, band)
        east, north = warp_transform("EPSG:4326", src.crs, lon.ravel(), lat.ravel())
        rows, cols = rasterio.transform.rowcol(src.transform, east, north)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    out = np.full(lat.size, np.nan, dtype=np.float32)
    out[inside] = data[rows[inside], cols[inside]]
    return out.reshape(lat.shape)


def compare(x, y):
    """Agreement statistics between reference x and estimate y where both are finite."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    xo, yo = x[ok].astype(np.float64), y[ok].astype(np.float64)
    slope, c0 = np.polyfit(xo - xo.mean(), yo, 1)               # centred: well conditioned
    intercept = c0 - slope * xo.mean()
    return {"n": int(ok.sum()), "r": float(np.corrcoef(xo, yo)[0, 1]), "bias": float(np.mean(yo - xo)),
            "rmse": float(np.sqrt(np.mean((yo - xo) ** 2))), "mae": float(np.mean(np.abs(yo - xo))), "slope": float(slope),
            "intercept": float(intercept), "ok": ok}


def to_geotiff(values, lat, lon, path, res_deg=None, max_gap=1.5, rgb=False):
    """Write maps in radar geometry to a regular lon/lat GeoTIFF (EPSG:4326) by nearest neighbour.
    values: dict name -> 2-D array (same shape as lat/lon), one band each. Grid cells farther than
    max_gap grid spacings from any radar pixel are no-data. res_deg defaults to the radar grid spacing.
    rgb=True: three uint8 bands (R, G, B; 0 = no-data, e.g. from display.rgb_uint8) written as an RGB image."""
    import rasterio
    from rasterio.transform import from_origin
    from scipy.spatial import cKDTree

    ok = np.isfinite(lat) & np.isfinite(lon)
    pts = np.column_stack([lon[ok], lat[ok]])
    if res_deg is None:                       # coarser of the two grid spacings, so that no holes appear
        step = lambda axis: np.nanmedian(np.hypot(np.diff(lon, axis=axis), np.diff(lat, axis=axis)))
        res_deg = float(max(step(0), step(1)))
    w, e, s, n = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    gx = np.arange(w, e + res_deg, res_deg)
    gy = np.arange(n, s - res_deg, -res_deg)
    GX, GY = np.meshgrid(gx, gy)
    dist, idx = cKDTree(pts).query(np.column_stack([GX.ravel(), GY.ravel()]))
    far = dist > max_gap * res_deg
    dtype, nodata = ("uint8", 0) if rgb else ("float32", np.nan)
    bands = []
    for v in values.values():
        b = v[ok][idx].astype(dtype)
        b[far] = nodata
        bands.append(b.reshape(GX.shape))
    profile = dict(driver="GTiff", height=GX.shape[0], width=GX.shape[1], count=len(bands), dtype=dtype,
                   crs="EPSG:4326", transform=from_origin(w - res_deg / 2, n + res_deg / 2, res_deg, res_deg),
                   nodata=nodata, compress="deflate")
    if rgb:
        profile["photometric"] = "RGB"
    with rasterio.open(path, "w", **profile) as dst:
        for k, (name, b) in enumerate(zip(values, bands), 1):
            dst.write(b, k)
            dst.set_band_description(k, name)
    return path, res_deg


def map_to_grid(path, lat, lon, band=1, invalid=(101, 102, 103), nonforest=0, pool=(3, 3)):
    """Pool a gridded reference map (e.g. a 30 m canopy-height map, any CRS) onto a radar grid.

    Every valid map pixel in the area is assigned to its nearest radar cell (lat/lon of the cells);
    sums are then pooled over `pool` cells (about the multilook footprint) before averaging.
    invalid: map codes that are not heights (GLAD convention: 101 water, 102 snow/ice, 103 no data).
    nonforest: code for 'no trees' (0), kept as height 0 but also counted separately.
    Returns maps on the radar grid: ref_mean, ref_max, ref_std, ref_count, ref_invalid_frac, ref_nonforest_frac."""
    import rasterio
    from rasterio.warp import transform as warp_transform
    from scipy.ndimage import maximum_filter, uniform_filter
    from scipy.spatial import cKDTree

    ok = np.isfinite(lat) & np.isfinite(lon)
    with rasterio.open(path) as src:
        crs = src.crs
        e, n = warp_transform("EPSG:4326", crs, lon[ok], lat[ok])
        e, n = np.asarray(e), np.asarray(n)
        win = rasterio.windows.from_bounds(e.min(), n.min(), e.max(), n.max(), src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(band, window=win).astype(np.float32)
        tr = src.window_transform(win)
        nodata = src.nodata
    rows, cols = np.indices(data.shape)
    px, py = rasterio.transform.xy(tr, rows.ravel(), cols.ravel(), offset="center")
    px, py, v = np.asarray(px), np.asarray(py), data.ravel()
    del rows, cols, data

    # radar cell spacing in map units, to reject map pixels outside the radar grid
    E = np.full(lat.shape, np.nan); N = np.full(lat.shape, np.nan)
    E[ok], N[ok] = e, n
    step = max(np.nanmedian(np.hypot(np.diff(E, axis=0), np.diff(N, axis=0))),
               np.nanmedian(np.hypot(np.diff(E, axis=1), np.diff(N, axis=1))))
    dist, idx = cKDTree(np.column_stack([e, n])).query(np.column_stack([px, py]), distance_upper_bound=step)
    inside = np.isfinite(dist)
    cell = np.flatnonzero(ok.ravel())[idx[inside]]
    v = v[inside]
    bad = np.isin(v, invalid) | (~np.isfinite(v)) | ((v == nodata) if nodata is not None else False)
    size = lat.size
    count_all = np.bincount(cell, minlength=size).astype(np.float64)
    count_bad = np.bincount(cell[bad], minlength=size).astype(np.float64)
    good = ~bad
    count = np.bincount(cell[good], minlength=size).astype(np.float64)
    s1 = np.bincount(cell[good], weights=v[good], minlength=size)
    s2 = np.bincount(cell[good], weights=v[good] ** 2, minlength=size)
    nf = np.bincount(cell[good & (v == nonforest)], minlength=size).astype(np.float64)
    vmax = np.full(size, -np.inf)
    np.maximum.at(vmax, cell[good], v[good])

    shape = lat.shape
    pooled = lambda a: uniform_filter(a.reshape(shape), pool, mode="constant") * pool[0] * pool[1]
    C, CA, CB, NF = (np.rint(pooled(a)) for a in (count, count_all, count_bad, nf))   # pooled counts are integers
    S1, S2 = pooled(s1), pooled(s2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S1 / C
        std = np.sqrt(np.clip(S2 / C - mean ** 2, 0, None))
        out = {"ref_mean": mean, "ref_std": std,
               "ref_max": maximum_filter(vmax.reshape(shape), pool, mode="constant", cval=-np.inf),
               "ref_count": C, "ref_invalid_frac": CB / CA, "ref_nonforest_frac": NF / C}
    empty = C == 0
    for k in out:
        out[k] = np.where(empty | ~np.isfinite(out[k]), np.nan, out[k]).astype(np.float32)
    out["ref_count"] = np.where(empty, 0, C).astype(np.float32)
    return out
