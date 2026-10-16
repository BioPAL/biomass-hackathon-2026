"""Rebuild the stage-6 outputs (figures, tables, metrics) of an existing run folder WITHOUT data access, from the saved
SLC window (slc_window.npz), the LUT files and the reference JSON. Use it to iterate on figures or estimators offline,
or to re-plot runs after a code update.

    python tests/replot_from_npz.py ~/data/runs/S2_T009_F118_G01_M04 [--point Maydantal] [--look-m 100] [--no-quicklooks]

The saved SLC is the window AFTER flattening (if it was applied); denoising is not re-applied (the raw noise maps are not
stored) unless the LUT files are present and --denoise is given. Quicklooks are redrawn from the cached PNGs if present.
"""
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import run_stack as RS                                                  # noqa: E402
from sm_pipeline import io_l1c                                           # noqa: E402
from rasterio.windows import Window                                      # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--point"); ap.add_argument("--look-m", type=float, default=100)
ap.add_argument("--no-quicklooks", action="store_true"); ap.add_argument("--dielectric", default="mironov"); ap.add_argument("--clay", type=float, default=None)
ap.add_argument("--sand", type=float, default=None); ap.add_argument("--min-sm-range", type=float, default=0.05); ap.add_argument("--half-m", type=float, default=None, help="default: from the saved window"); ap.add_argument("--no-s1", action="store_true", help="no effect (Sentinel-1 was dropped); kept for old commands")
ap.add_argument("--no-s2", action="store_true", help="do not fetch a missing Sentinel-2 NDVI / NDMI series"); ap.add_argument("--image-look-m", type=float, default=50)
a = ap.parse_args(); run = pathlib.Path(a.run_dir).expanduser(); a.stack = run.name; HALF_ARG = a.half_m
if a.half_m is None: a.half_m = 300.0
(run / "sentinel1_items.csv").unlink(missing_ok=True)  # listing of the dropped GRD step
_tj = run / "tier.json"                                                    # older runs: tier 2 was set from the SMOS LISTING only
if _tj.exists():
    _t = json.load(open(_tj))
    if _t.get("tier") == 2:
        from sm_pipeline.reference import TIERS
        _t.update(tier=3, tier_text=TIERS[3]); json.dump(_t, open(_tj, "w"), indent=1); print("tier.json: 2 -> 3 (SMOS values are not read)")
tab = pd.read_csv(run / "products.csv"); refs = json.load(open(run / "reference_at_overpass.json")); tab["datetime"] = tab["datetime"].astype(str)
times = pd.to_datetime(tab["datetime"], utc=True)
i_prim = int(np.argmax(tab["primary"].astype(bool).values)) if tab["primary"].astype(bool).any() else 0
obs_rows, clos_rows, met_rows = [], [], []
for rec in refs:
    st = rec["station"]; pdir = run / st
    if "aux_soil" not in rec:                                   # older runs: add the SoilGrids soil context (one cached request)
        try: rec["aux_soil"] = RS.auxiliary.soilgrids_point(rec["lat"], rec["lon"], cache=run / f"soilgrids_{st}.json")
        except Exception as ex: rec["aux_soil"] = {}; print(f"{st}: SoilGrids query failed ({str(ex)[:60]})")
    if a.point and st != a.point: continue
    if not (pdir / "slc_window.npz").exists(): print(f"{st}: no slc_window.npz (SAR stage skipped)"); continue
    z = np.load(pdir / "slc_window.npz", allow_pickle=False); slc = z["slc"].astype(np.complex128); names = list(z["names"])
    if HALF_ARG is None: a.half_m = round(float(z["window"][3]) * float(z["m_row"]) / 2 / 10) * 10      # window half width of that run
    if not a.no_s2 and rec.get("s2_box_m") != 2 * a.half_m: RS.add_sentinel2(rec, times, run, 2 * a.half_m)   # older runs / other window
    col_off, row_off, width, height = [int(v) for v in z["window"]]; win = Window(col_off, row_off, width, height); m_row, m_col = float(z["m_row"]), float(z["m_col"])
    T = slc.shape[0]; nr, nc = max(int(round(a.look_m / m_row)), 1), max(int(round(a.look_m / m_col)), 1)
    luts = sorted((run / "lut").glob("*_lut.nc")); lut_p = luts[i_prim] if len(luts) == T else None
    info_old = json.load(open(pdir / "sar_info.json")) if (pdir / "sar_info.json").exists() else {}
    # image size from the LUT grid scale (LUT node (i, j) <-> image (12 i, 2 j)) if a LUT is present
    if lut_p is not None:
        if "image_shape" in z.files: H, W = [int(v) for v in z["image_shape"]]
        else: la = io_l1c.lut_field(lut_p, "geometry", "latitude"); H, W = la.shape[0] * 12, la.shape[1] * 2   # older runs: approximate
        loc = io_l1c.lut_locate(lut_p, rec["lat"], rec["lon"], H, W); lutw = io_l1c.lut_window_arrays(lut_p, win, H, W)
        factor = lutw.get("sigmaNought"); factor = np.where(np.isfinite(factor), factor, np.nan)
        lut_coh = [io_l1c.lut_value_at(f, "interferometry", "coherence", loc["lut_i"], loc["lut_j"]) for f in luts]
        inc_t = [io_l1c.lut_value_at(f, "geometry", "incidenceAngle", loc["lut_i"], loc["lut_j"]) for f in luts]   # per acquisition
    else:
        loc = dict(row=row_off + height // 2, col=col_off + width // 2, offset_m=np.nan, height=np.nan, incidence=np.nan, az_slope=np.nan, rg_slope=np.nan); lutw = {}
        factor = np.ones(slc.shape[-2:]); lut_coh = [np.nan] * T; inc_t = [np.nan] * T; H = W = None
    th_loc = (loc["incidence"] - (loc["rg_slope"] if np.isfinite(loc.get("rg_slope", np.nan)) else 0.0)) if np.isfinite(loc.get("incidence", np.nan)) else 35.0
    der = RS.derive_observables(slc, names, factor, None, nr, nc, i_prim, rec, a, th_loc, m_row=m_row, m_col=m_col)
    sar = dict(names=names, slc=slc, win=win, m_row=m_row, m_col=m_col, nr=nr, nc=nc, i_prim=i_prim, loc=loc, th_loc=th_loc, valid_frac=float(np.isfinite(slc).mean()),
               flat_note=str(z["flattening"]), noise_note=info_old.get("denoising", "not re-applied (replot)"), nsr=info_old.get("nsr_db", {}), lut_coh=lut_coh, inc_t=inc_t,
               lut_itf=[np.nan] * T, pipe_itf_hh=[float(np.angle(np.nanmean(der["G"][i, i_prim, 0]))) for i in range(T)], lutw=lutw, H_img=H, W_img=W, **der)
    sar["rc"] = tuple(float(v) for v in z["point_rc"]) if "point_rc" in z.files else (height / 2, width / 2)   # window is centred on the point by construction
    a.no_quicklooks = a.no_quicklooks or not (pdir / "quicklooks").exists()
    if not a.no_quicklooks:   # reuse cached quicklook PNGs, no token needed
        io_l1c.download = lambda url, dst, tok: dst if pathlib.Path(dst).exists() else None; RS.catalog.token = lambda *x, **k: None
    o, c = RS.point_outputs(rec, sar, tab, a, pdir); obs_rows += o; clos_rows += c; met_rows += RS.metrics_for_point(rec, sar, times, a)
    print(f"{st}: figures rebuilt in {pdir}")
if met_rows:
    json.dump(refs, open(run / "reference_at_overpass.json", "w"), indent=1, default=RS._jsonable)
    done = {r["station"] for r in met_rows}
    def merge(new_rows, name):      # keep the rows of points that were not re-plotted
        new = pd.DataFrame(new_rows); f = run / name
        if f.exists():
            try: old = pd.read_csv(f)
            except (pd.errors.EmptyDataError, pd.errors.ParserError): old = pd.DataFrame()
            new = pd.concat([old[~old["station"].isin(done)], new], ignore_index=True) if "station" in old else new
        new.to_csv(f, index=False); return new
    merge(obs_rows, "observables_long.csv"); merge(clos_rows, "closure_phases.csv"); M = merge(met_rows, "metrics.csv")
    key = M[M["observable"].str.startswith(("sigma0 HH", "sigma0 VV", "closure phase"), na=False)]
    pd.set_option("display.width", 220); print(key[["station", "reference", "observable", "value", "r", "r_diff", "ref_range", "obs_range"]].round(3).to_string(index=False))
