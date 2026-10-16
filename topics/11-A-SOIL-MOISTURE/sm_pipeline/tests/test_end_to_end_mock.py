"""End-to-end test of run_stack.py with SIMULATED BIOMASS data (no token; internet only for the reference sources).

The SAR reader, LUT download and image metadata are replaced by a simulator. It produces a 7-date complex stack whose
coherence matrix follows the De Zan et al. (2014) soil-volume model (Mironov dielectric model) for a prescribed
soil-moisture sequence, with backscatter changing by SENS dB per m3/m3, temporal decorrelation, speckle, and a
topographic phase kz_i x (h - h0) that the pipeline must remove with the LUT flattening screen (BPS '+' convention:
the data carry exp(-j phi_flat)). A synthetic ISMN export with the same sequence is the station reference.

Checks: the run completes and writes every output; flattening is chosen automatically and raises the coherence;
the recovered HH/VV sensitivities match the simulated ones; the closure-phase forward check correlates (r > 0.6);
a run with an empty ISMN folder falls back to tier 3 (model reference only).
    python tests/test_end_to_end_mock.py        (2-4 min)
"""
import sys, pathlib, json, re, tempfile
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import run_stack
from sm_pipeline import io_l1c, catalog, estimators

STACK, LAT, LON, NAME = "S2_T009_F118_G01_M04", 41.99651, 70.63916, "Maydantal"
H, W, R_LUT, C_LUT, I0, J0 = 20741, 1563, 1729, 782, 1414, 262
M_SEQ = np.array([0.30, 0.26, 0.22, 0.19, 0.16, 0.14, 0.13])
SENS = {"HH": 20.0, "HV": 12.0, "VH": 12.0, "VV": 25.0}; BASE = {"HH": -7.0, "HV": -17.0, "VH": -17.0, "VV": -8.0}
KZ = None; DATES = None; PRIM = None; THETA = 27.9 - 1.2


def lut_grids():
    ii, jj = np.meshgrid(np.arange(R_LUT), np.arange(C_LUT), indexing="ij")
    lat = LAT - 0.0006 * (ii - I0) + 0.0001 * (jj - J0); lon = LON + 0.0005 * (jj - J0) + 0.00005 * (ii - I0)
    h = 1452.0 + 20.0 * (ii - I0) + 15.0 * (jj - J0)
    return ii, jj, lat, lon, h


def write_lut(path, k):
    import netCDF4
    ii, jj, lat, lon, h = lut_grids(); inc = 27.9 + 0.002 * (jj - J0)
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("relativeAzimuthTime", R_LUT); nc.createDimension("slantRangeTime", C_LUT)
        def put(g, name, arr):
            grp = nc.groups[g] if g in nc.groups else nc.createGroup(g)
            v = grp.createVariable(name, "f4", ("relativeAzimuthTime", "slantRangeTime"), fill_value=-9999.0); v[:] = arr.astype("float32")
        put("geometry", "latitude", lat); put("geometry", "longitude", lon); put("geometry", "height", h); put("geometry", "incidenceAngle", inc)
        put("geometry", "azimuthTerrainSlope", np.full_like(h, -10.5)); put("geometry", "rangeTerrainSlope", np.full_like(h, 1.2))
        put("radiometry", "sigmaNought", np.sin(np.radians(inc - 1.2))); put("radiometry", "gammaNought", np.tan(np.radians(inc - 1.2)))
        for p in ("HH", "HV", "VH", "VV"): put("denoising", f"denoising{p}", np.full_like(h, 1e-4))
        put("coregistration", "flatteningPhaseScreen", KZ[k] * (h - 1452.0)); put("coregistration", "waveNumbers", np.full_like(h, KZ[k]))
        put("interferometry", "coherence", np.full_like(h, 0.8)); put("interferometry", "interferogram", np.zeros_like(h))


def date_index(url):
    for k, d in enumerate(DATES):
        if d in url: return k
    raise KeyError(url)


_CACHE = {}
def simulate(win):
    key = (win.row_off, win.col_off, win.height, win.width)
    if key in _CACHE: return _CACHE[key]
    rng = np.random.default_rng(1)
    rows = np.arange(win.row_off, win.row_off + win.height) / (H / R_LUT); cols = np.arange(win.col_off, win.col_off + win.width) / (W / C_LUT)
    RR, CC = np.meshgrid(rows, cols, indexing="ij"); h = 1452.0 + 20.0 * (RR - I0) + 15.0 * (CC - J0)
    T = len(M_SEQ); t = pd.to_datetime([f"{d[:8]}T{d[9:]}" for d in DATES]); dt = np.abs((t.values[:, None] - t.values[None, :]) / np.timedelta64(1, "D"))
    eps = estimators.make_eps("mironov", 0.2, 0.4)
    G = np.array([[estimators.coherence_model(M_SEQ[i], M_SEQ[j], THETA, eps) if i != j else 1.0 + 0j for j in range(T)] for i in range(T)]) * np.exp(-dt / 90.0)
    out = []
    for p in ("HH", "HV", "VH", "VV"):
        I = 10 ** ((BASE[p] + SENS[p] * (M_SEQ - M_SEQ.mean())) / 10) / np.sin(np.radians(THETA))   # beta0 power
        L = np.linalg.cholesky(np.sqrt(np.outer(I, I)) * G + 1e-9 * I.max() * np.eye(T))
        z = (rng.standard_normal((T,) + h.shape) + 1j * rng.standard_normal((T,) + h.shape)) / np.sqrt(2)
        s = np.einsum("ij,jrc->irc", L, z) * np.exp(-1j * KZ[:, None, None] * (h - 1452.0))
        out.append(s)
    _CACHE[key] = np.stack(out, axis=1).astype(np.complex64)   # [t, pol, rows, cols]
    return _CACHE[key]


def fake_read(abs_url, phase_url, win, tok):
    return simulate(win)[date_index(abs_url)], ["HH", "HV", "VH", "VV"]


def fake_download(url, dest, tok):
    dest = pathlib.Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0: return dest
    if url.endswith(".nc"): write_lut(dest, date_index(url))
    elif url.endswith(".png"):
        from PIL import Image; Image.fromarray((np.random.default_rng(0).random((400, 150, 3)) * 255).astype("uint8")).save(dest)
    else: dest.write_bytes(b"x")
    return dest


def write_ismn(folder, times):
    t = pd.date_range(times.min() - pd.Timedelta(days=10), times.max() + pd.Timedelta(days=3), freq="1h", tz="UTC")
    x = ((t - times.min()) / pd.Timedelta(seconds=1)).to_numpy(); xp = ((pd.DatetimeIndex(times) - times.min()) / pd.Timedelta(seconds=1)).to_numpy()
    rows = []
    for z, seq in ((0.1, M_SEQ), (0.2, 0.5 * (M_SEQ + M_SEQ.mean()))):
        v = np.interp(x, xp, seq)
        rows += [dict(network="ROMPS", station=NAME, sensor=f"Sim_1_1_soil_moisture_{z:.6f}_{z:.6f}", date_time=tt.strftime("%Y-%m-%d %H:%M:%S"),
                      soil_moisture=round(float(vv), 4), soil_moisture_flag="G", soil_moisture_orig_flag="M") for tt, vv in zip(t, v)]
    pd.DataFrame(rows).to_csv(folder / f"ROMPS+{NAME}.csv")


def main():
    global KZ, DATES, PRIM
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="smpipe_test_")); ismn_dir = tmp / "ISMN" / "temporal"; ismn_dir.mkdir(parents=True)
    geo = tmp / "ismn_locations.geojson"                 # the simulated station as the only ISMN location
    geo.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [LON, LAT]},
                   "properties": {"station": NAME, "network": "ROMPS", "start": "2025-01-01", "end": "2026-12-31"}}]}))
    args = run_stack.parser().parse_args(["--stack", STACK, "--lat", str(LAT), "--lon", str(LON), "--name", NAME, "--only-point", "--ismn", str(geo),
                                          "--ismn-dir", str(tmp / "ISMN"), "--out", str(tmp / "runs"), "--half-m", "300", "--no-s2"])
    tab, times = run_stack.products(args)
    DATES = [re.search(r"_(\d{8}T\d{6})_", pid).group(1) for pid in tab["pid"]]; PRIM = int(np.argmax(tab["primary"].astype(bool).values))
    KZ = 0.04 * (np.arange(len(tab)) - PRIM)          # rad/m, 0 at the primary
    write_ismn(ismn_dir, times)
    run_stack.products = lambda a: (tab, times)
    catalog.token = lambda *a, **k: "simulated"
    io_l1c.check_token = lambda tok, url: (206, "ok (simulated)")
    io_l1c.image_info = lambda url, tok, lat, lon: dict(height=H, width=W, count=4, dtype="float32", nodata=-9999.0, n_gcps=60, row_gcp=16964, col_gcp=496, m_row=6.6, m_col=40.0)
    io_l1c.download = fake_download; io_l1c.read_slc_window = fake_read
    io_l1c.annotation_flags = lambda url, tok, keys=None: {"skpPhaseCorrectionFlag": "false (simulated)"}
    run_stack.main(args)
    out = tmp / "runs" / STACK; pdir = out / NAME
    need = ["products.csv", "tier.json", "reference_at_overpass.json", "reference_sources.png", "metrics.csv", "observables_long.csv", "closure_phases.csv"]
    need_p = ["biomass_vs_reference.png", "sar_observables.png", "coherence_pairs.png", "window_images.png", "sigma0_vs_reference.png", "closure_check.png",
              "quicklooks_pauli.png", "backscatter_raw.csv", "slc_window.npz"]
    missing = [f for f in need if not (out / f).exists()] + [f for f in need_p if not (pdir / f).exists()]
    assert not missing, f"missing outputs: {missing}"
    tier = json.load(open(out / "tier.json")); assert tier["tier"] == 1, tier
    flat = str(np.load(pdir / "slc_window.npz")["flattening"]); assert flat.startswith("applied"), flat
    M = pd.read_csv(out / "metrics.csv")
    res = {}
    for pol in ("HH", "VV"):
        row = M[(M.reference == "ISMN 0.1m") & (M.observable == f"sigma0 {pol}")].iloc[0]; res[pol] = row.value
        assert abs(row.value - SENS[pol]) / SENS[pol] < 0.25, f"{pol} sensitivity {row.value:.1f} vs simulated {SENS[pol]}"
    cc = M[M.observable == "closure phase HH"].iloc[0]
    assert cc.r > 0.6, f"closure forward check r = {cc.r:.2f}"
    print(f"PASS run 1: tier 1, flattening '{flat}', sensitivity HH {res['HH']:.1f} (sim 20), VV {res['VV']:.1f} (sim 25), closure check HH r = {cc.r:.2f}")
    # run 2: same stack without a station file -> tier 3, reference only
    empty = tmp / "ISMN_empty"; empty.mkdir()
    a2 = run_stack.parser().parse_args(["--stack", STACK, "--lat", str(LAT), "--lon", str(LON), "--name", NAME, "--only-point", "--ismn", str(geo),
                                        "--ismn-dir", str(empty), "--out", str(tmp / "runs2"), "--no-sar", "--no-s2"])
    run_stack.main(a2); t2 = json.load(open(tmp / "runs2" / STACK / "tier.json"))
    assert t2["tier"] == 3, t2
    print(f"PASS run 2: no station file -> tier {t2['tier']}")
    print("outputs kept in", tmp)


if __name__ == "__main__":
    main()
