"""Access to BIOMASS L1C (STA) assets on MAAP: token-authorised windowed reads of the amplitude/phase COGs,
LUT NetCDF access (geolocation, radiometry, denoising, flattening, interferometry) and annotation flags.

Conventions (BPS L1 Product Format Description BIO-BPS-L1PFD-ARE-010076 v1.6.1, LUT ADS / Tab. 88):
- COG amplitude = sqrt(beta0) (calibrated, linear), phase in rad; nodata -9999; bands HH, HV, VH, VV (or HH, XX, VV).
- LUT = image grid subsampled (e.g. 1729 x 782 nodes for a 20741 x 1563 image); groups /geometry, /radiometry,
  /denoising, /coregistration, /interferometry, /skpPhaseCalibration; fill value -9999.
- sigma0 = beta0 x sigmaNought (conversion factor). Denoising maps are noise POWERS in beta0 units:
  sigma0_denoised = sigma0 - denoising x sigmaNought.
- Flattening: DEM phase compensation with the '+' convention, CoSCS x exp(+j dphi_flat), dphi_flat = flatteningPhaseScreen [rad]
  (all zeros for the coregistration primary). Annotation flag skpPhaseCorrectionFlag reports whether the product phases were
  already corrected; run_stack.py therefore applies the screen only if it increases the coherence (auto mode).
"""
import math, os, pathlib, re, tempfile
import numpy as np, requests, rasterio
from rasterio.transform import from_gcps
from rasterio.windows import Window

POLS = ["HH", "HV", "VH", "VV"]
FILL = -9999.0


def _vsi(url):
    return url if url.startswith("/vsi") else "/vsicurl/" + url


def _header_file(tok):
    d = os.path.expanduser("~/my-private-bucket"); d = d if os.path.isdir(d) else tempfile.gettempdir()
    f = os.path.join(d, ".gdal_http_headers")
    if not (os.path.exists(f) and tok in open(f).read()):
        open(f, "w").write(f"Authorization: Bearer {tok}\n"); os.chmod(f, 0o600)
    return f


def gdal_env(tok, use_header_file=False):
    """GDAL >= 3.6 reads GDAL_HTTP_HEADERS; older GDAL needs a header file. Never set both (duplicate Authorization -> HTTP 400)."""
    opts = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif,.nc", GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="5")
    major, minor = [int(x) for x in rasterio.__gdal_version__.split(".")[:2]]
    if use_header_file or (major, minor) < (3, 6): opts["GDAL_HTTP_HEADER_FILE"] = _header_file(tok)
    else: opts["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {tok}"
    return rasterio.Env(**opts)


def check_token(tok, url):
    """Range-request the first kB of an asset (independent of GDAL). Returns (HTTP status, note)."""
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}", "Range": "bytes=0-1023"}, timeout=60)
        note = {200: "ok (full response)", 206: "ok (range)", 401: "token rejected/expired -> renew token.txt in the MAAP portal",
                403: "forbidden -> token valid but no access to this asset"}.get(r.status_code, r.reason)
        return r.status_code, note
    except Exception as e:
        return -1, str(e)[:100]


def download(url, dest, tok):
    """Download once (cached at dest)."""
    dest = pathlib.Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 0):
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, stream=True, timeout=600); r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for c in r.iter_content(1 << 20): f.write(c)
        os.replace(tmp, dest)
    return dest


def pixel_of(src, lat, lon):
    """(row, col, m_per_row, m_per_col) of lat/lon from an affine fit to the COG GCPs. Coarse in mountains (no height term);
    use lut_locate() for the station position, this function only for the pixel spacing."""
    gcps, _ = src.gcps
    if not gcps: raise ValueError("COG has no GCPs")
    T = from_gcps(gcps); col, row = ~T * (lon, lat)
    la = np.array([g.y for g in gcps]); lo = np.array([g.x for g in gcps]); cc = np.array([g.col for g in gcps]); rr = np.array([g.row for g in gcps])
    m_col = 111320 * math.cos(math.radians(lat)) * abs(np.polyfit(cc, lo, 1)[0]) if np.ptp(cc) else 50.0
    m_row = 111320 * abs(np.polyfit(rr, la, 1)[0]) if np.ptp(rr) else 8.5
    return int(row), int(col), float(m_row), float(m_col)


def image_info(url, tok, lat, lon):
    """Size, nodata and pixel spacing of the amplitude COG, plus the (coarse) GCP position of lat/lon."""
    with gdal_env(tok), rasterio.open(_vsi(url)) as src:
        row, col, m_row, m_col = pixel_of(src, lat, lon)
        return dict(height=src.height, width=src.width, count=src.count, dtype=src.dtypes[0], nodata=src.nodata,
                    n_gcps=len(src.gcps[0]), row_gcp=row, col_gcp=col, m_row=m_row, m_col=m_col)


def window_from_center(row, col, half_m, m_row, m_col):
    """Window of +-half_m around (row, col); may extend beyond the image (reads are boundless, outside = NaN)."""
    hr, hc = max(int(half_m / m_row), 1), max(int(half_m / m_col), 1)
    return Window(col - hc, row - hr, 2 * hc, 2 * hr)


def read_slc_window(abs_url, phase_url, win, tok):
    """Complex SLC sqrt(beta0) exp(j phase) for all bands in the window -> array [pol, rows, cols] (NaN = nodata), band names."""
    with gdal_env(tok):
        with rasterio.open(_vsi(abs_url)) as a:
            amp = a.read(window=win, boundless=True, fill_value=np.nan).astype("float32"); nd = a.nodata
            names = [d or POLS[i] for i, d in enumerate(a.descriptions)]
        with rasterio.open(_vsi(phase_url)) as p:
            ph = p.read(window=win, boundless=True, fill_value=np.nan).astype("float32"); ndp = p.nodata
    bad = ~np.isfinite(amp) | (amp <= 0) | ~np.isfinite(ph)
    if nd is not None: bad |= np.isclose(amp, nd)
    if ndp is not None: bad |= np.isclose(ph, ndp)
    slc = np.where(bad, np.nan, amp) * np.exp(1j * np.where(bad, 0.0, ph))
    return slc.astype(np.complex64), names


# ---- LUT ---------------------------------------------------------------------------------------------------------------
def lut_groups(path):
    """[(group_path, {variable: shape})] of a LUT NetCDF."""
    import netCDF4
    out = []
    def walk(g, prefix):
        if g.variables: out.append((prefix or "/", {k: tuple(v.shape) for k, v in g.variables.items()}))
        for name, sub in g.groups.items(): walk(sub, f"{prefix}/{name}")
    with netCDF4.Dataset(path) as nc: walk(nc, "")
    return out


def lut_field(path, group, var):
    """Full LUT field as float64 with the fill value replaced by NaN."""
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        v = nc.groups[group].variables[var] if group else nc.variables[var]
        v.set_auto_mask(False); A = np.array(v[:], dtype="float64")
    A[np.isclose(A, FILL)] = np.nan
    return A


def lut_value_at(path, group, var, i, j):
    try: return float(lut_field(path, group, var)[i, j])
    except Exception: return float("nan")


def _grid_coords(win, img_rows, img_cols, R, C):
    sr, sc = img_rows / R, img_cols / C
    rr = np.arange(win.row_off, win.row_off + win.height) / sr
    cc = np.arange(win.col_off, win.col_off + win.width) / sc
    return np.meshgrid(rr, cc, indexing="ij")


def lut_field_window(path, group, var, win, img_rows, img_cols, order=1, phase=False):
    """LUT field interpolated onto the image window (bilinear). phase=True: an unwrapped (smooth) field is interpolated
    linearly; a wrapped field (|values| <= pi with jumps > pi between neighbours) via its sine and cosine."""
    from scipy.ndimage import map_coordinates
    A = lut_field(path, group, var); R, C = A.shape
    RR, CC = _grid_coords(win, img_rows, img_cols, R, C)
    def interp(F):
        ok = np.isfinite(F); Ff = np.where(ok, F, np.nanmean(F) if ok.any() else 0.0)
        v = map_coordinates(Ff, [RR, CC], order=order, mode="nearest")
        m = map_coordinates(ok.astype(float), [RR, CC], order=0, mode="nearest") > 0.5
        return np.where(m, v, np.nan)
    if phase:
        fin = A[np.isfinite(A)]
        wrapped = fin.size and np.max(np.abs(fin)) <= np.pi + 1e-3 and np.nanmax(np.abs(np.diff(A, axis=0))) > np.pi
        if wrapped: return np.arctan2(interp(np.sin(A)), interp(np.cos(A)))
    return interp(A)


def lut_locate(path, lat, lon, img_rows, img_cols):
    """Point -> image pixel via the LUT geometry grid (accurate, includes terrain height). LUT node (i, j) <-> image (i*sr, j*sc)."""
    la, lo = lut_field(path, "geometry", "latitude"), lut_field(path, "geometry", "longitude")
    d = np.hypot(la - lat, (lo - lon) * math.cos(math.radians(lat)))
    if not np.isfinite(d).any(): return dict(row=-1, col=-1, lut_i=-1, lut_j=-1, offset_m=float("nan"))
    i, j = np.unravel_index(np.nanargmin(d), d.shape); sr, sc = img_rows / la.shape[0], img_cols / la.shape[1]
    out = dict(row=int(round(i * sr)), col=int(round(j * sc)), lut_i=int(i), lut_j=int(j), offset_m=float(d[i, j] * 111320), scale_r=sr, scale_c=sc)
    for key, (g, v) in dict(height=("geometry", "height"), incidence=("geometry", "incidenceAngle"), az_slope=("geometry", "azimuthTerrainSlope"),
                            rg_slope=("geometry", "rangeTerrainSlope"), sigma0_factor=("radiometry", "sigmaNought"), gamma0_factor=("radiometry", "gammaNought")).items():
        out[key] = lut_value_at(path, g, v, i, j)
    return out


def lut_window_arrays(path, win, img_rows, img_cols):
    """Geometry and radiometry LUT fields on the image window (bilinear)."""
    out = {}
    for g, vs in (("geometry", ["incidenceAngle", "height", "azimuthTerrainSlope", "rangeTerrainSlope", "latitude", "longitude"]),
                  ("radiometry", ["sigmaNought", "gammaNought"])):
        for v in vs:
            try: out[v] = lut_field_window(path, g, v, win, img_rows, img_cols)
            except Exception: pass
    return out


def annotation_flags(url, tok, keys=("skpPhaseCorrectionFlag", "skpPhaseCorrectionFlatteningOnlyFlag", "skpPhaseCalibrationFlag")):
    """Read processing flags from an annotation XML (small file)."""
    r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, timeout=120); r.raise_for_status(); t = r.text
    out = {}
    for k in keys:
        m = re.search(rf"<{k}>([^<]*)</{k}>", t); out[k] = m.group(1).strip() if m else None
    return out
