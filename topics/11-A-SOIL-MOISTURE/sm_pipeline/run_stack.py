"""Soil-moisture pipeline for ONE BIOMASS L1C stack (tile): gather all references, compute BIOMASS observables, compare, plot.

    python run_stack.py --stack S2_T009_F118_G01_M04 --lat 41.99651 --lon 70.63916 --name Maydantal --half-m 300
    python run_stack.py --stack <key>                                   # point = stack centre; ISMN stations in the footprint are added
    python run_stack.py --stack <key> --lat .. --lon .. --only-point    # only the given point (or the station at it)
    python run_stack.py --stack <key> ... --no-sar                      # reference part only (no token)

Stages (output in ~/data/runs/<stack>/, one subfolder per point; run.log is written there as well)
 1 catalogue  products of the stack: times, coregistration primary, asset URLs (public MAAP STAC)
 2 points     the given point and every ISMN station inside the footprint with data in the window
 3 reference  per point: ERA5-Land, ERA5, ECMWF IFS (Open-Meteo, explicit models) at the overpass hour; ISMN export file
              (flag G) if present; Copernicus DEM slope and WorldCover land cover; SMOS L2 listing;
              reference tier: 1 usable station, 2 satellite SM listed, 3 models only
 4 SAR        per point (token): LUT geolocation; window of amplitude+phase COGs of all acquisitions; flattening with the
              LUT screen ('+' convention, applied if it raises coherence); sigma0 = beta0 x sigmaNought, denoised with the
              LUT noise maps; multilook; coherence matrix; closure phases (circular window mean); PyPolSARpro decompositions (own H/alpha without it)
 5 estimators closure-phase forward check (De Zan 2014 model + Mironov dielectric model driven by the reference
              sequence); closure-phase inversion (experimental)
 6 output     metrics.csv, observables_long.csv, closure_phases.csv, <point>/backscatter_raw.csv, figures
"""
import argparse, json, os, pathlib, sys, traceback, warnings
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sm_pipeline import deps  # noqa: E402
deps.ensure()                 # install missing packages (pip --user) before the modules that need them are imported
from sm_pipeline import catalog, auxiliary, reference, io_l1c, observables, estimators, evaluate, plots  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
MODEL_VARS = ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "temperature_2m",
              "dew_point_2m", "snow_depth", "soil_temperature_0_to_7cm"]
TIERS = reference.TIERS


LOGFILE = [None]


def log(*args):
    print(*args, flush=True)
    if LOGFILE[0] is not None:
        with open(LOGFILE[0], "a") as fh: print(*args, file=fh)


def _jsonable(o):
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


# ---- stage 1: catalogue --------------------------------------------------------------------------------------------
def products(a):
    if a.lat is None and not a.assets_csv:
        gi = pd.read_csv(HERE / "l1c_stack_index.csv"); g = gi[gi["stack"] == a.stack]
        if g.empty: sys.exit(f"[1] stack {a.stack} not in l1c_stack_index.csv")
        g = g.iloc[0]; a.centre = (float((g.s + g.n) / 2), float((g.w + g.e) / 2))
        a.lat, a.lon = a.centre; a.name = a.name or f"centre_{a.stack}"
        log(f"[1] no point given -> stack centre {a.lat:.4f}, {a.lon:.4f}")
    if a.assets_csv:
        df = pd.read_csv(a.assets_csv); pids = df.loc[df["stack"] == a.stack, "pid"].tolist()
    else:
        st = catalog.stacks_at(a.lat, a.lon, a.collection)
        if a.stack not in st: sys.exit(f"[1] stack {a.stack} does not cover {a.lat:.4f}, {a.lon:.4f}; stacks there: {sorted(st)}")
        pids = list(st[a.stack].values())
    tab = catalog.stack_table(pids, a.collection)
    times = pd.to_datetime(tab["datetime"], utc=True)
    log(f"[1] {a.stack}: {len(tab)} products {tab['datetime'].iloc[0]} .. {tab['datetime'].iloc[-1]} UTC, "
        f"primary {tab.loc[tab['primary'].astype(bool), 'datetime'].str[:10].tolist()}")
    return tab, times


# ---- stage 2: points -----------------------------------------------------------------------------------------------
def points(a, tab, t0):
    cols = ["station", "network", "lat", "lon"]; pts = pd.DataFrame(columns=cols)
    gj = os.path.expanduser(a.ismn) if a.ismn else ""
    if gj and os.path.exists(gj):
        st = auxiliary.stations_in_footprint(auxiliary.footprint(tab["pid"].iloc[0], a.collection), gj, active_after=str(t0.date()))
        if len(st): pts = st[cols].copy()
    else:
        log(f"[2] ISMN station list not found ({a.ismn}) -> no station lookup")
    if a.lat is not None:
        near = pts[((pts["lat"].astype(float) - a.lat).abs() < 2e-3) & ((pts["lon"].astype(float) - a.lon).abs() < 2e-3)] if len(pts) else pts
        user = pd.DataFrame([dict(station=a.name or f"P_{a.lat:.4f}_{a.lon:.4f}", network="user", lat=a.lat, lon=a.lon)])
        if a.only_point: pts = near if len(near) else user
        elif not len(near): pts = pd.concat([pts, user], ignore_index=True)
    if not len(pts):
        geom = auxiliary.footprint(tab["pid"].iloc[0], a.collection); w, s_, e, n, _ = auxiliary._bbox(geom)
        pts = pd.DataFrame([dict(station=f"centre_{a.stack}", network="centre", lat=0.5 * (s_ + n), lon=0.5 * (w + e))])
        log("[2] no station and no point given -> stack centre")
    pts = pts.reset_index(drop=True); pts["lat"] = pts["lat"].astype(float); pts["lon"] = pts["lon"].astype(float)
    log(f"[2] points: {[f'{s} ({n})' for s, n in zip(pts.station, pts.network)]}")
    return pts


# ---- stage 3: reference --------------------------------------------------------------------------------------------
def add_sentinel2(rec, times, out, box_m):
    """Sentinel-2 L2A NDVI / NDMI over the analysis window (cached as <out>/s2_<point>_<box>m.csv) and their values at the
    overpasses (nearest clear scene within 15 days, linear interpolation between clear scenes)."""
    f = f"s2_{rec['station']}_{box_m:.0f}m.csv"
    try:
        s2 = auxiliary.sentinel2_series(rec["lat"], rec["lon"], times, box_m=box_m, cache=pathlib.Path(out) / f, log=log)
        rec.update(auxiliary.s2_at_overpasses(s2, times)); rec.update(s2_file=f, s2_box_m=box_m, s2_n_dates=int(len(s2)),
                                                                     s2_n_clear=int(s2["clear"].sum()) if len(s2) and "clear" in s2 else 0)
    except Exception as ex:
        log(f"[3] {rec['station']}: Sentinel-2 not read ({type(ex).__name__}: {str(ex)[:60]})")


def reference_for_point(p, times, a, out):
    t0, t1 = times.min(), times.max()
    s, e = (t0 - pd.Timedelta(days=10)).date().isoformat(), (t1 + pd.Timedelta(days=3)).date().isoformat()
    rec = dict(station=str(p["station"]), network=str(p["network"]), lat=float(p["lat"]), lon=float(p["lon"]))
    models = {}
    for m in reference.MODELS:            # re-use model series already saved for this point (a re-run costs no Open-Meteo quota)
        f = out / f"{m}_{rec['station']}.csv"
        if f.exists():
            try:
                d = pd.read_csv(f, index_col=0, parse_dates=True)
                if d.index.tz is None: d.index = d.index.tz_localize("UTC")
                if len(d) and d.index.min() <= pd.Timestamp(s, tz="UTC") and d.index.max() >= pd.Timestamp(e, tz="UTC") - pd.Timedelta(hours=23):
                    models[m] = d; log(f"[3] {rec['station']}: {m} from cache {f.name}"); continue
            except Exception: pass
        try: models[m] = reference.model_hourly(rec["lat"], rec["lon"], s, e, m)
        except Exception as ex:
            hint = " (Open-Meteo quota of this IP used up for today; resets 00:00 UTC; the run continues without this model)" if "429" in str(ex) else ""
            log(f"[3] {rec['station']}: model {m} failed: {str(ex)[:60]}{hint}")
    rec["sm_sources"] = list(models)
    for m, d in models.items():
        d.to_csv(out / f"{m}_{rec['station']}.csv")
        for v in MODEL_VARS:
            if v in d: rec[f"{m}_{v}"] = reference.at_overpass(pd.to_numeric(d[v], errors="coerce"), times).tolist()
        rec[f"{m}_rain_between_mm"] = (reference.rain_between(pd.to_numeric(d["precipitation"], errors="coerce"), times).tolist()
                                       if "precipitation" in d else [np.nan] * len(times))
        w = d[(d.index >= t0 - pd.Timedelta(hours=1)) & (d.index <= t1 + pd.Timedelta(hours=1))]
        rec[f"{m}_tmin_window_C"] = float(pd.to_numeric(w["temperature_2m"], errors="coerce").min()) if "temperature_2m" in w else np.nan
        rec[f"{m}_snow_depth_max_window_m"] = float(pd.to_numeric(w["snow_depth"], errors="coerce").max()) if "snow_depth" in w else np.nan
    try: rec.update({f"aux_{k}": v for k, v in auxiliary.terrain_landcover(rec["lat"], rec["lon"], a.half_m).items()})
    except Exception as ex: rec["aux_error"] = str(ex)[:100]
    try:
        rec["aux_soil"] = auxiliary.soilgrids_point(rec["lat"], rec["lon"], cache=out / f"soilgrids_{rec['station']}.json")
        log(f"[3] {rec['station']}: soil (SoilGrids 2.0) {rec['aux_soil'] or 'no data (water / outside the map)'}")
    except Exception as ex:
        rec["aux_soil"] = {}; log(f"[3] {rec['station']}: SoilGrids query failed ({str(ex)[:60]}) -> default texture")
    if not getattr(a, "no_s2", False): add_sentinel2(rec, times, out, 2 * a.half_m)
    rec["has_ismn"] = False
    f = reference.ismn_station_file(a.ismn_dir, rec["network"], rec["station"]) if (a.ismn_dir and rec["network"] != "user") else None
    if f is not None:
        ser = reference.ismn_series(f); rec["ismn_file"] = str(f)
        for lab, sr in ser.items():
            if lab.endswith("_allflags") or not lab.startswith(("soil_moisture", "soil_temperature", "air_temperature", "precipitation")): continue
            rec[f"ismn_{lab}"] = reference.at_overpass(sr, times).tolist()
        sm_keys = [k for k in rec if k.startswith("ismn_soil_moisture")]
        rec["has_ismn"] = bool(any(np.isfinite(np.array(rec[k], float)).sum() >= 3 for k in sm_keys))
        pk = [k for k in ser if k.startswith("precipitation") and not k.endswith("_allflags")]
        if pk: rec["ismn_rain_between_mm"] = reference.rain_between(ser[pk[0]], times).tolist()
        log(f"[3] {rec['station']}: ISMN file {pathlib.Path(f).name}, depths {[k.replace('ismn_soil_moisture_', '') for k in sm_keys]}, usable {rec['has_ismn']}")
    elif rec["network"] != "user":
        log(f"[3] {rec['station']}: no ISMN file {rec['network']}+{rec['station']}.csv below {a.ismn_dir} -> models only")
    for m in rec["sm_sources"]:
        log(f"[3] {rec['station']} {m:10s} SM 0-7 cm at overpasses {np.round(rec.get(f'{m}_soil_moisture_0_to_7cm', []), 3).tolist()}")
    log(f"[3] {rec['station']}: slope {rec.get('aux_slope_mean_deg', np.nan):.1f} deg, land cover {rec.get('aux_landcover_pct')}")
    return rec


def reference_series(rec):
    """Ordered dict label -> SM sequence at the overpasses: station depths first, then models (0-7, 7-28 cm)."""
    refs = {}
    for k in sorted([k for k in rec if k.startswith("ismn_soil_moisture") and isinstance(rec[k], list)], key=plots._depth):
        refs["ISMN " + k.replace("ismn_soil_moisture_", "")] = np.array(rec[k], float)
    for m in rec.get("sm_sources", []):
        for lay, lab in (("0_to_7cm", "0-7 cm"), ("7_to_28cm", "7-28 cm")):
            if f"{m}_soil_moisture_{lay}" in rec: refs[f"{m} {lab}"] = np.array(rec[f"{m}_soil_moisture_{lay}"], float)
    return refs


def driving_sequence(rec):
    """Moisture sequence that drives the closure-phase model: shallowest station depth with >= 5 values, else the models in
    the order of the figures (ERA5-Land, ERA5, IFS)."""
    refs = reference_series(rec)
    order = [k for k in refs if k.startswith("ISMN")] + [f"{m} 0-7 cm" for m in plots.REF_MODELS]
    for k in order:
        if k in refs and np.isfinite(refs[k]).sum() >= 5:
            return k, pd.Series(refs[k]).interpolate(limit_direction="both").to_numpy()
    return None, None


# ---- stage 4-5: SAR ------------------------------------------------------------------------------------------------
def sar_for_point(p, rec, tab, times, a, out):
    T = len(tab); i_prim = int(np.argmax(tab["primary"].astype(bool).values)) if tab["primary"].astype(bool).any() else 0
    info = io_l1c.image_info(tab["abs_tiff"].iloc[0], catalog.token(), p["lat"], p["lon"]); H, W = info["height"], info["width"]
    luts = [io_l1c.download(u, out / "lut" / f"{str(d)[:10]}_lut.nc", catalog.token()) for u, d in zip(tab["lut_nc"], tab["datetime"])]
    loc = io_l1c.lut_locate(luts[i_prim], p["lat"], p["lon"], H, W)
    if not np.isfinite(loc.get("offset_m", np.nan)) or loc["offset_m"] > 1000:
        return dict(skipped=f"point not covered by the LUT grid (nearest node {loc.get('offset_m', np.nan):.0f} m)")
    th_loc = loc["incidence"] - (loc["rg_slope"] if np.isfinite(loc["rg_slope"]) else 0.0)
    log(f"[4] {rec['station']}: image {W} x {H}; LUT geolocation row {loc['row']}, col {loc['col']} ({loc['offset_m']:.0f} m from the node; "
        f"GCP fit differs by {loc['row'] - info['row_gcp']} rows, {loc['col'] - info['col_gcp']} cols)")
    log(f"[4] {rec['station']}: height {loc['height']:.0f} m, incidence {loc['incidence']:.1f} deg, terrain slope az/rg {loc['az_slope']:.1f}/{loc['rg_slope']:.1f} deg, "
        f"sigmaNought {loc['sigma0_factor']:.3f} vs sin(inc - rg slope) {np.sin(np.radians(th_loc)):.3f}")
    win = io_l1c.window_from_center(loc["row"], loc["col"], a.half_m, info["m_row"], info["m_col"])
    if a.lut_check:
        for gp, vs in io_l1c.lut_groups(luts[i_prim]): log(f"[4] LUT group {gp}: {vs}")
    try: log(f"[4] annotation flags (first secondary): {io_l1c.annotation_flags(tab['annot_xml_1'].iloc[(i_prim + 1) % T], catalog.token())}")
    except Exception as ex: log(f"[4] annotation flags not read ({str(ex)[:60]})")
    slcs, names = [], None
    for k, prod in tab.iterrows():
        s, names = io_l1c.read_slc_window(prod["abs_tiff"], prod["phase_tiff"], win, catalog.token()); slcs.append(s)
        p2 = np.abs(s) ** 2
        log(f"[4] read {str(prod['datetime'])[:10]}{' (primary)' if k == i_prim else ''}: {s.shape}, valid {100 * np.isfinite(p2).mean():.0f} %, "
            f"mean beta0 {[round(float(10 * np.log10(np.nanmean(p2[b]))), 1) if np.isfinite(p2[b]).any() else None for b in range(s.shape[0])]} dB")
    slc = np.stack(slcs).astype(np.complex128)
    vfrac = float(np.isfinite(slc).mean())
    if vfrac < a.min_valid:
        return dict(skipped=f"only {100 * vfrac:.0f} % valid pixels in the window (swath edge / nodata); try a neighbouring sub-swath stack")
    nr, nc = max(int(round(a.look_m / info["m_row"])), 1), max(int(round(a.look_m / info["m_col"])), 1)

    def mean_coh(x):
        return float(np.nanmean([np.nanmean(np.abs(observables.coherence(x[i], x[i_prim], nr, nc))) for i in range(T) if i != i_prim]))
    flat_note = "off"
    if a.flatten != "off":
        try:
            fl = np.nan_to_num(np.stack([io_l1c.lut_field_window(f, "coregistration", "flatteningPhaseScreen", win, H, W, phase=True) for f in luts]))
            slc_f = slc * np.exp(1j * fl[:, None, :, :])                      # '+' convention, PFD v1.6.1
            c_raw, c_flat = mean_coh(slc), mean_coh(slc_f)
            spread = [float(np.nanmax(fl[i]) - np.nanmin(fl[i])) for i in range(T) if i != i_prim]
            log(f"[4] flattening screen (DEM + flat-earth phase from the LUT): spread inside the window {min(spread):.0f}-{max(spread):.0f} rad per secondary; "
                f"mean |coh| to primary without {c_raw:.3f}, with {c_flat:.3f}")
            if a.flatten == "on" or c_flat > c_raw + 0.005: slc, flat_note = slc_f, f"applied (|coh| {c_raw:.2f} -> {c_flat:.2f})"
            else: flat_note = f"not applied (no coherence gain {c_raw:.2f} -> {c_flat:.2f}; phases probably already flattened)"
        except Exception as ex:
            flat_note = f"failed ({str(ex)[:80]})"
    log(f"[4] flattening: {flat_note}")

    lutw = io_l1c.lut_window_arrays(luts[i_prim], win, H, W)
    factor = lutw.get("sigmaNought"); factor = np.where(np.isfinite(factor), factor, np.nan) if factor is not None else np.ones(slc.shape[-2:])
    noise, noise_note, nsr_tab = None, "off", {}
    if not a.no_denoise:
        try:
            nz = np.stack([np.stack([io_l1c.lut_field_window(f, "denoising", f"denoising{n}", win, H, W) for n in names]) for f in luts])
            s_lin = observables.ml(np.abs(slc) ** 2 * factor, nr, nc); n_lin = observables.ml(nz * factor, nr, nc)
            nsr = 10 * np.log10(np.nanmean(n_lin, axis=(-2, -1)) / np.nanmean(s_lin, axis=(-2, -1)))       # [t, pol]
            nsr_tab = {str(d)[:10]: dict(zip(names, np.round(nsr[i], 1).tolist())) for i, d in enumerate(tab["datetime"])}
            log(f"[4] noise-to-signal ratio per acquisition [dB] (LUT noise map / measured power): {nsr_tab}")
            if np.nanmax(nsr) >= 0: noise_note = f"NOT applied: noise map >= signal (NSR max {np.nanmax(nsr):.1f} dB) - check the LUT"
            else: noise, noise_note = nz, "applied; noise-to-signal per pol [dB] " + str(dict(zip(names, np.round(np.nanmean(nsr, axis=0), 1).tolist())))
        except Exception as ex:
            noise_note = f"failed ({str(ex)[:80]})"
    log(f"[4] denoising: {noise_note}")

    der = derive_observables(slc, names, factor, noise, nr, nc, i_prim, rec, a, th_loc, log_values=True, m_row=info["m_row"], m_col=info["m_col"])
    lc, li, ph = [np.nan] * T, [np.nan] * T, [np.nan] * T
    try:
        lc = [io_l1c.lut_value_at(f, "interferometry", "coherence", loc["lut_i"], loc["lut_j"]) for f in luts]
        li = [io_l1c.lut_value_at(f, "interferometry", "interferogram", loc["lut_i"], loc["lut_j"]) for f in luts]
        ph = [float(np.angle(np.nanmean(der["G"][i, i_prim, 0]))) for i in range(T)]
        log(f"[4] cross-check at the station node: LUT coherence {np.round(lc, 2).tolist()} | LUT interferogram {np.round(li, 2).tolist()} rad | pipeline arg<s_i s_prim*> HH {np.round(ph, 2).tolist()} rad")
    except Exception as ex:
        log(f"[4] LUT cross-check not possible ({str(ex)[:60]})")
    inc_t = [np.nan] * T
    try: inc_t = [io_l1c.lut_value_at(f, "geometry", "incidenceAngle", loc["lut_i"], loc["lut_j"]) for f in luts]
    except Exception as ex: log(f"[4] incidence per acquisition not read ({str(ex)[:60]})")
    log(f"[4] incidence angle at the point per acquisition [deg]: {np.round(np.asarray(inc_t, float), 3).tolist()}")
    return dict(names=names, slc=slc, win=win, m_row=info["m_row"], m_col=info["m_col"], nr=nr, nc=nc, i_prim=i_prim, loc=loc, th_loc=th_loc, valid_frac=vfrac,
                flat_note=flat_note, noise_note=noise_note, nsr=nsr_tab, lut_coh=lc, lut_itf=li, pipe_itf_hh=ph, inc_t=inc_t, lutw=lutw, H_img=H, W_img=W, **der)


def soil_texture(rec, a):
    """Clay and sand fraction [0-1] for the dielectric model: --clay/--sand if given, else SoilGrids 0-5 cm at the point,
    else the defaults 0.2 / 0.4."""
    soil = rec.get("aux_soil") or {}
    clay = a.clay if getattr(a, "clay", None) is not None else (soil["clay_pct_0_5cm"] / 100 if "clay_pct_0_5cm" in soil else None)
    sand = a.sand if getattr(a, "sand", None) is not None else (soil["sand_pct_0_5cm"] / 100 if "sand_pct_0_5cm" in soil else None)
    src = "command line" if getattr(a, "clay", None) is not None else ("SoilGrids 0-5 cm" if clay is not None else "default")
    return (0.2 if clay is None else clay), (0.4 if sand is None else sand), src


def derive_observables(slc, names, factor, noise, nr, nc, i_prim, rec, a, th_loc, log_values=False, m_row=None, m_col=None):
    """All observables and estimator outputs from the (flattened) SLC window: sigma0/beta0 window means, multilooked images,
    coherence matrix, closure phases, PyPolSARpro decompositions (own H/alpha without PyPolSARpro), De Zan model and experimental inversion. Also used by
    tests/replot_from_npz.py to rebuild the stage-6 outputs without data access."""
    T = slc.shape[0]
    lin_sig = observables.ml(np.abs(slc) ** 2 * factor, nr, nc) - (observables.ml(noise * factor, nr, nc) if noise is not None else 0.0)
    lin_beta = observables.ml(np.abs(slc) ** 2, nr, nc)
    win_db = lambda lin: observables.to_db(np.nanmean(np.where(lin > 0, lin, np.nan), axis=(-2, -1)))
    sig_t, beta_t = win_db(lin_sig), win_db(lin_beta)                                           # [t, pol]
    img_db = observables.to_db(np.where(lin_sig > 0, lin_sig, np.nan))                           # same looks as the analysis
    img_disp, disp_looks, disp_off = img_db, (nr, nc), (0, 0)
    if m_row and m_col:                                    # window_images: finer display cells (--image-look-m, default 50 m), centred crop
        dl = float(getattr(a, "image_look_m", 50.0) or 50.0)
        nrd, ncd = max(int(round(dl / m_row)), 1), max(int(round(dl / m_col)), 1)
        R_, C_ = slc.shape[-2:]; r0, c0 = (R_ % nrd) // 2, (C_ % ncd) // 2
        rs, cs = slice(r0, r0 + (R_ // nrd) * nrd), slice(c0, c0 + (C_ // ncd) * ncd)
        p_d = observables.ml((np.abs(slc) ** 2 * factor)[..., rs, cs], nrd, ncd)
        if noise is not None: p_d = p_d - observables.ml((noise * factor)[..., rs, cs], nrd, ncd)
        img_disp, disp_looks, disp_off = observables.to_db(np.where(p_d > 0, p_d, np.nan)), (nrd, ncd), (r0, c0)
    G = observables.coherence_matrix(slc, nr, nc); Gabs = np.nanmean(np.abs(G), axis=(-2, -1))  # [t, t, pol]
    trip = observables.all_triplets(T); clos = np.array([observables.closure_phase_mean(G, *tr) for tr in trip])  # [ntrip, pol]
    cons = [trip.index(tr) for tr in observables.consecutive_triplets(T)]
    try:                                                                                     # polarimetric decompositions: PyPolSARpro
        pp = [observables.pypolsarpro_decompositions(slc[t], names, nr, nc) for t in range(T)]
        pp = None if any(x is None for x in pp) else {k: np.array([x[k] for x in pp]) for k in pp[0]}
    except Exception as ex:
        pp = None; log(f"[4] PyPolSARpro decompositions failed ({type(ex).__name__}: {str(ex)[:80]}) -> own H/alpha only")
    if pp is not None and "pp_h_a_alpha" in pp:                                              # H, A, alpha from PyPolSARpro
        hA = np.asarray(pp["pp_h_a_alpha"], float); Ht, At, ha_src = hA[:, 0], hA[:, 2], "PyPolSARpro"
    else:                                                                                    # without PyPolSARpro: own H/alpha only
        C3 = [observables.pauli_and_covariance(slc[t], nr, nc, tuple(names)) for t in range(T)]
        ha = [observables.h_alpha(c) for c in C3]; Ht = np.array([np.nanmean(h) for h, _ in ha]); At = np.array([np.nanmean(x) for _, x in ha]); ha_src = "own"
    if log_values: log("[4] PyPolSARpro decompositions (Freeman-Durden, Yamaguchi y4r, van Zyl, H/A/alpha): " + ("computed" if pp is not None else "not installed -> own H/alpha only, no power partitions (see README)"))
    if log_values:
        log(f"[4] sigma0 window means [dB] {dict(zip(names, np.round(sig_t.T, 2).tolist()))}")
        log(f"[4] |coherence| 3-day pairs {dict(zip(names, np.round(np.array([Gabs[k, k + 1, :] for k in range(len(Gabs) - 1)]).T, 2).tolist()))}")
    ref_lab, seq = driving_sequence(rec)
    clay, sand, tex_src = soil_texture(rec, a)
    if log_values: log(f"[4] dielectric model {a.dielectric}: clay {clay:.2f}, sand {sand:.2f} ({tex_src})")
    eps_fn = estimators.make_eps(a.dielectric, clay, sand)
    mod = estimators.closure_model(seq, trip, th_loc, eps_fn) if seq is not None else None
    inv = [estimators.invert_closure(clos[:, b], trip, T, theta_deg=th_loc, eps_fn=eps_fn) for b in range(len(names))]
    dsm = np.array([v["m"] - v["m"][0] for v in inv]).T                                         # [t, pol]
    coh_pairs = np.array([Gabs[i, i + 1, :] for i in range(T - 1)])                             # [T-1, pol] consecutive (3-day) pairs
    return dict(sig_t=sig_t, beta_t=beta_t, img_db=img_db, img_disp=img_disp, disp_looks=disp_looks, disp_off=disp_off, G=G, Gabs=Gabs, coh_prim=Gabs[:, i_prim, :], coh_pairs=coh_pairs, trip=trip, clos=clos, cons=cons, pp=pp, H=Ht, alpha=At, ha_src=ha_src,
                ref_lab=ref_lab, ref_seq=seq, mod=mod, dsm=dsm)


# ---- stage 6: outputs ----------------------------------------------------------------------------------------------
def hourly_context(rec, run_dir, times, before_d=5, after_d=2):
    """Full-resolution context for the main figure (hourly, as fine as the sources go): in-situ soil moisture of every depth
    (ISMN file, flag G), the model reference 0-7 cm (ERA5-Land hourly, saved in the run folder), daily rain (station gauge if
    present, else the model), 2 m air temperature and soil temperature 0-7 cm (model), station air temperature and snow depth.
    Window: [first acquisition - before_d, last + after_d]. Missing sources are simply left out."""
    t = pd.to_datetime(pd.Series(times), utc=True); lo, hi = t.min() - pd.Timedelta(days=before_d), t.max() + pd.Timedelta(days=after_d)
    cut = lambda sr: sr[(sr.index >= lo) & (sr.index <= hi)].dropna()
    out = dict(window=(lo, hi), sm=[], temp=[], rain=None, rain_label="", snow=None)
    f = rec.get("ismn_file")
    if f and pathlib.Path(f).exists():
        try:
            ser = reference.ismn_series(f)
            for k in sorted([k for k in ser if k.startswith("soil_moisture") and not k.endswith("_allflags")], key=lambda k: float(k.split("_")[-1].split("-")[0].rstrip("m"))):
                s = cut(pd.to_numeric(ser[k], errors="coerce"))
                if len(s): out["sm"].append(("in situ " + k.replace("soil_moisture_", ""), s))
            pk = [k for k in ser if k.startswith("precipitation") and not k.endswith("_allflags")]
            if pk:
                pr = cut(pd.to_numeric(ser[pk[0]], errors="coerce"))
                if len(pr): out["rain"], out["rain_label"] = pr.resample("1D").sum(min_count=1), "station gauge, daily"
            ak = [k for k in ser if k.startswith("air_temperature") and not k.endswith("_allflags")]
            if ak:
                at = cut(pd.to_numeric(ser[ak[0]], errors="coerce"))
                if len(at): out["temp"].append(("station air T", at))
        except Exception as ex:
            log(f"[6] {rec['station']}: ISMN hourly series not read ({str(ex)[:60]})")
    m = plots.ref_model(rec)
    fm = pathlib.Path(run_dir) / f"{m}_{rec['station']}.csv" if m else None
    if fm is not None and fm.exists():
        d = pd.read_csv(fm, index_col=0, parse_dates=True)
        if d.index.tz is None: d.index = d.index.tz_localize("UTC")
        lab = plots.MODEL_LABEL[m]
        if "soil_moisture_0_to_7cm" in d: out["sm"].append((f"{lab} 0-7 cm (model, hourly)", cut(pd.to_numeric(d["soil_moisture_0_to_7cm"], errors="coerce"))))
        if "temperature_2m" in d: out["temp"].insert(0, (f"{lab} T 2 m", cut(pd.to_numeric(d["temperature_2m"], errors="coerce"))))
        if "soil_temperature_0_to_7cm" in d: out["temp"].insert(1, (f"{lab} soil T 0-7 cm", cut(pd.to_numeric(d["soil_temperature_0_to_7cm"], errors="coerce"))))
        if "snow_depth" in d:
            sn = cut(pd.to_numeric(d["snow_depth"], errors="coerce"))
            if len(sn) and sn.max() > 0.005: out["snow"] = sn
    if out["rain"] is None:                                      # daily rain from the first model that has it
        for mm in ([m] if m else []) + [x for x in plots.REF_MODELS if x != m]:
            f2 = pathlib.Path(run_dir) / f"{mm}_{rec['station']}.csv"
            if not f2.exists(): continue
            d2 = pd.read_csv(f2, index_col=0, parse_dates=True)
            if d2.index.tz is None: d2.index = d2.index.tz_localize("UTC")
            pr = cut(pd.to_numeric(d2["precipitation"], errors="coerce")) if "precipitation" in d2 else pd.Series(dtype=float)
            if len(pr): out["rain"], out["rain_label"] = pr.resample("1D").sum(min_count=1), f"{plots.MODEL_LABEL[mm]}, daily"; break
    return out


def point_outputs(rec, sar, tab, a, pdir):
    dts = tab["datetime"].tolist(); names = sar["names"]; st = rec["station"]
    w = sar["win"]
    np.savez_compressed(pdir / "slc_window.npz", slc=sar["slc"].astype(np.complex64), names=np.array(names), datetimes=np.array(dts),
                        window=np.array([w.col_off, w.row_off, w.width, w.height]), m_row=sar["m_row"], m_col=sar["m_col"], flattening=sar["flat_note"],
                        image_shape=np.array([sar["H_img"], sar["W_img"]]), point_rc=np.array(sar["rc"]))
    try: hourly = hourly_context(rec, pdir.parent, dts)
    except Exception as ex: hourly = None; log(f"[6] {st}: hourly context failed ({str(ex)[:60]}) -> overpass values only")
    t0 = pd.to_datetime(pd.Series(dts), utc=True)
    xlim = hourly["window"] if hourly else (t0.min() - pd.Timedelta(days=5), t0.max() + pd.Timedelta(days=2))   # time axis of the main figure
    plots.combined_panel(dts, names, sar["sig_t"], sar["coh_pairs"], rec, pdir / "biomass_vs_reference.png", a.stack, st, hourly=hourly)
    s2f = pdir.parent / str(rec.get("s2_file", ""))
    if rec.get("s2_file") and s2f.exists():
        try: plots.s2_figure(pd.read_csv(s2f), dts, pdir / "s2_ndvi_ndmi.png", a.stack, st, rec.get("s2_box_m"))
        except Exception as ex: log(f"[6] {st}: Sentinel-2 figure failed ({str(ex)[:60]})")
    plots.sar_panels(dts, names, sar["clos"][sar["cons"]], pdir / "sar_observables.png", a.stack, st, pp=sar.get("pp"), H=sar["H"], alpha=sar["alpha"],
                     ha_src=sar.get("ha_src", ""), xlim=xlim)
    pair_fits = {n: evaluate.pair_regression(sar["Gabs"][:, :, b], pd.to_datetime(dts, utc=True), sar["ref_seq"]) for b, n in enumerate(names)} if sar["ref_seq"] is not None else None
    plots.coherence_figure(sar["Gabs"], dts, names, sar["ref_seq"], pdir / "coherence_pairs.png", a.stack, st, sar["ref_lab"] or "SM", pair_fits=pair_fits)
    nrd, ncd = sar.get("disp_looks", (sar["nr"], sar["nc"])); r0, c0 = sar.get("disp_off", (0, 0))
    rc_d = ((sar["rc"][0] - r0) / nrd, (sar["rc"][1] - c0) / ncd) if "rc" in sar else None
    plots.window_images(dts, names, sar.get("img_disp", sar["img_db"]), pdir / "window_images.png", sar["m_row"] * nrd, sar["m_col"] * ncd, station_rc=rc_d,
                        title=f"{a.stack} @ {st}: sigma0 in the {2 * a.half_m:.0f} m window, display cells {sar['m_row'] * nrd:.0f} m x {sar['m_col'] * ncd:.0f} m "
                              f"({nrd} x {ncd} pixels; the analysis uses {a.look_m:.0f} m looks); + = point")
    try: plots.window_geometry(sar["lutw"], pdir / "window_geometry.png", sar["m_row"], sar["m_col"], station_rc=sar.get("rc"), title=f"{a.stack} @ {st}: LUT geometry in the window")
    except Exception as ex: log(f"[6] window geometry figure failed ({str(ex)[:60]})")
    plots.scatter_key_refs(dts, names, sar["sig_t"], rec, pdir / "sigma0_vs_reference.png", a.stack, st)
    if sar["mod"] is not None: plots.closure_check_plot(sar["clos"], sar["mod"], names, pdir / "closure_check.png", a.stack, st, sar["ref_lab"])
    if not a.no_quicklooks:
        try: plots.quicklook_sheet(tab, catalog.token(), pdir / "quicklooks_pauli.png", w, (sar["H_img"], sar["W_img"]), io_l1c.download, st, m_row=sar["m_row"], m_col=sar["m_col"], half_m=a.half_m)
        except Exception as ex: log(f"[6] quicklook sheet failed ({str(ex)[:80]})")
    info = dict(stack=a.stack, station=st, lat=float(rec.get("lat", np.nan)), lon=float(rec.get("lon", np.nan)), window_m=2 * a.half_m, look_m=a.look_m,
                image_row=int(sar["loc"]["row"]), image_col=int(sar["loc"]["col"]), lut_offset_m=float(sar["loc"]["offset_m"]), valid_pct=100 * sar["valid_frac"],
                height_m=sar["loc"].get("height"), incidence_deg=sar["loc"].get("incidence"), lut_slope_az_deg=sar["loc"].get("az_slope"), lut_slope_rg_deg=sar["loc"].get("rg_slope"),
                local_incidence_deg=sar["th_loc"], incidence_per_acquisition_deg=[None if not np.isfinite(v) else float(v) for v in np.asarray(sar.get("inc_t", []), float)],
                h_alpha_source=sar.get("ha_src"), flattening=sar["flat_note"], denoising=sar["noise_note"], nsr_db=sar["nsr"], acquisitions=[str(d) for d in dts],
                lut_coherence_at_node=[None if not np.isfinite(v) else float(v) for v in sar["lut_coh"]], pipeline_coh_to_primary_HH=[float(v) for v in sar["coh_prim"][:, 0]],
                coh_3day_pairs_HH=[float(v) for v in sar["coh_pairs"][:, 0]], pair_dt_days=[round(float((pd.Timestamp(dts[i + 1]) - pd.Timestamp(dts[i])).total_seconds() / 86400), 1) for i in range(len(dts) - 1)], mean_coh_3day_HH=float(np.nanmean(sar["coh_pairs"][:, 0])),
                lut_interferogram_at_node_rad=[None if not np.isfinite(v) else float(v) for v in sar["lut_itf"]], pipeline_interferogram_HH_rad=[None if not np.isfinite(v) else float(v) for v in sar["pipe_itf_hh"]],
                dem_slope_mean_deg=rec.get("aux_slope_mean_deg"), landcover_pct=rec.get("aux_landcover_pct"), tier=rec.get("tier"))
    json.dump(info, open(pdir / "sar_info.json", "w"), indent=1, default=_jsonable)
    raw = pd.DataFrame({"acq_datetime_utc": dts, **{f"beta0_db_{n}": sar["beta_t"][:, b] for b, n in enumerate(names)},
                        **{f"sigma0_db_{n}": sar["sig_t"][:, b] for b, n in enumerate(names)}, **{f"coh_to_primary_{n}": sar["coh_prim"][:, b] for b, n in enumerate(names)},
                        **{f"coh_pair_with_previous_{n}": np.r_[np.nan, sar["coh_pairs"][:, b]] for b, n in enumerate(names)}})
    for lab, v in reference_series(rec).items(): raw[f"ref {lab}"] = v
    for k in ("s2_ndvi_interp", "s2_ndmi_interp", "s2_ndvi_nearest", "s2_ndmi_nearest", "s2_dt_days"):
        if k in rec and len(rec[k]) == len(raw): raw[k] = pd.to_numeric(pd.Series(rec[k]), errors="coerce").values
    raw.round(4).to_csv(pdir / "backscatter_raw.csv", index=False)
    log(f"[6] {st}: raw backscatter table\n" + raw[[c for c in raw.columns if c.startswith(("acq", "beta0", "sigma0"))]].round(2).to_string(index=False))
    obs = []
    series = {"sigma0_db": sar["sig_t"], "beta0_db": sar["beta_t"], "dSM_closure_rel_t0 (experimental)": sar["dsm"]}
    for kind, arr in series.items():
        for b, n in enumerate(names):
            obs += [dict(station=st, pol=n, kind=kind, acq_datetime_utc=d, value=float(arr[i, b])) for i, d in enumerate(dts)]
    for b, n in enumerate(names):   # coherence of consecutive pairs, stamped with the LATER acquisition of the pair
        obs += [dict(station=st, pol=n, kind="coh_3day_pair (with previous acquisition)", acq_datetime_utc=dts[i + 1], value=float(sar["coh_pairs"][i, b])) for i in range(len(dts) - 1)]
    for key, _, _, comps in (observables.PP_METHODS if sar.get("pp") is not None else ()):
        P = sar["pp"][key]; pt = np.nansum(P, axis=1); ps = 100 * P / np.where(pt > 0, pt, np.nan)[:, None]
        for k, c in enumerate(comps):
            obs += [dict(station=st, pol="quad", kind=f"{key}_{c}_pct", acq_datetime_utc=d, value=float(ps[i, k])) for i, d in enumerate(dts)]
    if sar.get("pp") is not None and "pp_h_a_alpha" in sar["pp"]:
        for k, c in enumerate(("entropy_H", "anisotropy_A", "alpha_deg")):
            obs += [dict(station=st, pol="quad", kind=f"pp_{c}", acq_datetime_utc=d, value=float(np.asarray(sar["pp"]["pp_h_a_alpha"])[i, k])) for i, d in enumerate(dts)]
    if sar.get("ha_src") != "PyPolSARpro":                    # own H/alpha only without PyPolSARpro (otherwise the pp_ rows above)
        obs += [dict(station=st, pol="quad", kind="entropy_H (own)", acq_datetime_utc=d, value=float(sar["H"][i])) for i, d in enumerate(dts)]
        obs += [dict(station=st, pol="quad", kind="alpha_deg (own)", acq_datetime_utc=d, value=float(sar["alpha"][i])) for i, d in enumerate(dts)]
    clos = [dict(station=st, i=i, j=j, k=k, date_i=dts[i][:10], date_j=dts[j][:10], date_k=dts[k][:10], pol=n, closure_obs_rad=float(sar["clos"][q, b]),
                 closure_model_rad=float(sar["mod"][q]) if sar["mod"] is not None else np.nan, model_driver=sar["ref_lab"])
            for q, (i, j, k) in enumerate(sar["trip"]) for b, n in enumerate(names)]
    return obs, clos


def metrics_for_point(rec, sar, times, a):
    rows = []; names = sar["names"]; st = rec["station"]; nan = np.nan
    for lab, sm in reference_series(rec).items():
        if np.isfinite(sm).sum() < 3: continue
        rng = float(np.nanmax(sm) - np.nanmin(sm)); quant = rng >= a.min_sm_range
        note = "" if quant else f"reference range {rng:.3f} < {a.min_sm_range} m3/m3: slope not quantifiable"
        for b, n in enumerate(names):
            y = sar["sig_t"][:, b]; f = evaluate.linfit(sm, y)
            rows.append(dict(station=st, reference=lab, observable=f"sigma0 {n}", metric="slope [dB per m3/m3]", value=f["slope"] if quant else nan,
                             ci_low=f["ci_low"] if quant else nan, ci_high=f["ci_high"] if quant else nan, r=f["r"], rho=f["rho"], r_diff=evaluate.diff_corr(sm, y), r_diff_p=evaluate.diff_corr_p(sm, y),
                             n=f["n"], ref_range=rng, obs_range=float(np.nanmax(y) - np.nanmin(y)), note=note))
            pr = evaluate.pair_regression(sar["Gabs"][:, :, b], times, sm)
            rows.append(dict(station=st, reference=lab, observable=f"|coh| {n}", metric="d|coh| per m3/m3 of |dSM|, controlled for time", value=pr["b_dsm"],
                             ci_low=pr["b_dsm"] - 1.96 * pr["se_dsm"], ci_high=pr["b_dsm"] + 1.96 * pr["se_dsm"], r=pr["r_dsm"], rho=nan, r_diff=nan, n=pr["n"],
                             ref_range=rng, obs_range=nan, note=f"b_dt {pr['b_dt']:.4f} per day, R2 {pr['r2']:.2f}"))
            d = sar["dsm"][:, b]; dref = sm - sm[0]
            rows.append(dict(station=st, reference=lab, observable=f"closure-inversion dSM {n}", metric="RMSE [m3/m3] (experimental)", value=evaluate.rmse(dref, d),
                             ci_low=nan, ci_high=nan, r=evaluate.pearson(dref, d), rho=nan, r_diff=nan, n=int(np.isfinite(d).sum()), ref_range=rng, obs_range=nan, note=""))
        for key, _, _, comps in (observables.PP_METHODS if sar.get("pp") is not None else ()):   # same for the PyPolSARpro decompositions
            P = np.asarray(sar["pp"][key], float); pt = np.nansum(P, axis=1); ps = 100 * P / np.where(pt > 0, pt, np.nan)[:, None]
            for k, c in enumerate(comps):
                y = ps[:, k]; f = evaluate.linfit(sm, y)
                rows.append(dict(station=st, reference=lab, observable=f"{key} {c} share [%]", metric="slope [% per m3/m3]", value=f["slope"] if quant else nan,
                                 ci_low=f["ci_low"] if quant else nan, ci_high=f["ci_high"] if quant else nan, r=f["r"], rho=f["rho"], r_diff=evaluate.diff_corr(sm, y),
                                 n=f["n"], ref_range=rng, obs_range=float(np.nanmax(y) - np.nanmin(y)), note=note + " (PyPolSARpro)"))
    if sar["mod"] is not None:
        for b, n in enumerate(names):
            c = evaluate.closure_check(sar["clos"][:, b], sar["mod"])
            rows.append(dict(station=st, reference=sar["ref_lab"], observable=f"closure phase {n}", metric="observed vs De Zan model, all triplets: slope", value=c["slope"],
                             ci_low=nan, ci_high=nan, r=c["r"], rho=nan, r_diff=nan, n=c["n"], ref_range=nan, obs_range=c["obs_std_deg"],
                             note=f"sign agreement {c['sign_agree']:.2f}; std obs/model {c['obs_std_deg']:.1f}/{c['mod_std_deg']:.1f} deg; circ. RMSE {c['rmse_deg']:.1f} deg; r < 0 -> opposite phase convention"))
    return rows


def main(a):
    out = pathlib.Path(os.path.expanduser(a.out)) / a.stack; out.mkdir(parents=True, exist_ok=True)
    LOGFILE[0] = out / "run.log"; open(LOGFILE[0], "w").write(f"# run_stack.py {' '.join(sys.argv[1:])}\n")
    tab, times = products(a); tab.to_csv(out / "products.csv", index=False)
    t0, t1 = times.min(), times.max()
    pts = points(a, tab, t0)
    s, e = (t0 - pd.Timedelta(days=10)).date().isoformat(), (t1 + pd.Timedelta(days=3)).date().isoformat()
    clat, clon = float(pts["lat"].mean()), float(pts["lon"].mean())
    try: smos = auxiliary.smos_items(clat, clon, s, e)
    except Exception as ex: smos = pd.DataFrame(); log(f"[3] SMOS listing failed ({str(ex)[:60]})")
    smos.to_csv(out / "smos_items.csv", index=False); log(f"[3] SMOS L2 soil-moisture products in the window: {len(smos)} (listed only; values not read in this version)")
    refs = [reference_for_point(p, times, a, out) for _, p in pts.iterrows()]
    usable = [r["station"] for r in refs if r.get("has_ismn")]
    tier = 1 if usable else 3                     # tier 2 needs SMOS/SMAP VALUES at the point; this version only lists SMOS files
    json.dump(dict(stack=a.stack, tier=tier, tier_text=TIERS[tier], stations=usable, points=[r["station"] for r in refs]), open(out / "tier.json", "w"), indent=1)
    log(f"[3] reference tier {tier}: {TIERS[tier]}")
    json.dump(refs, open(out / "reference_at_overpass.json", "w"), indent=1, default=_jsonable)
    plots.reference_panel(refs, tab["datetime"].tolist(), out / "reference_sources.png", title=f"{a.stack}: reference data at the BIOMASS overpasses")
    log(f"[6] wrote {out / 'reference_sources.png'}")
    if a.no_sar:
        log(f"[done] reference only: {out}"); return
    if tab["abs_tiff"].isna().any() or tab["phase_tiff"].isna().any():
        sys.exit(f"[4] stop: {int(tab['abs_tiff'].isna().sum())} of {len(tab)} products have no amplitude/phase COG. This is a {tab['pid'].iloc[0][4:12]} "
                 "product set (L1B DGM = detected, no phase). The SAR stage needs L1C STA stacks -> choose the BiomassLevel1c entry of this stack key.")
    code, note = io_l1c.check_token(catalog.token(), tab["abs_tiff"].iloc[0])
    log(f"[4] access token from MAAP IAM; check on the first amplitude COG: HTTP {code} -> {note}")
    if code in (401, 403): sys.exit("[4] stop: renew the offline token in the MAAP portal -> ~/my-private-bucket/token.txt (one line)")
    obs_rows, clos_rows, met_rows = [], [], []
    for (_, p), rec in zip(pts.iterrows(), refs):
        pdir = out / rec["station"]; pdir.mkdir(exist_ok=True)
        try:
            sar = sar_for_point(p, rec, tab, times, a, out)
            if not sar.get("skipped"):
                sar["rc"] = (float(sar["loc"]["row"] - sar["win"].row_off), float(sar["loc"]["col"] - sar["win"].col_off))   # point in window pixels (row, col)
                o, c = point_outputs(rec, sar, tab, a, pdir); obs_rows += o; clos_rows += c; met_rows += metrics_for_point(rec, sar, times, a)
                log(f"[6] figures and tables in {pdir}")
        except Exception as ex:
            traceback.print_exc(); sar = dict(skipped=f"error {type(ex).__name__}: {str(ex)[:150]}")
        if sar.get("skipped"):
            log(f"[4] {rec['station']}: SAR skipped -> {sar['skipped']}"); met_rows.append(dict(station=rec["station"], observable="all", metric="skipped", note=sar["skipped"]))
    pd.DataFrame(obs_rows).to_csv(out / "observables_long.csv", index=False)
    pd.DataFrame(clos_rows).to_csv(out / "closure_phases.csv", index=False)
    M = pd.DataFrame(met_rows); M.to_csv(out / "metrics.csv", index=False)
    if len(M) and "reference" in M:
        key = M[M["observable"].str.startswith(("sigma0 HH", "sigma0 VV", "sigma0 HV", "closure phase"), na=False)]
        pd.set_option("display.width", 220)
        log("[6] key metrics:\n" + key[["station", "reference", "observable", "value", "ci_low", "ci_high", "r", "r_diff", "ref_range", "obs_range"]].round(3).to_string(index=False))
    log(f"[done] {out}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True); ap.add_argument("--collection", default="BiomassLevel1c")
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float); ap.add_argument("--name")
    ap.add_argument("--only-point", action="store_true", help="process only the given point (or the station at it), not all stations in the footprint")
    ap.add_argument("--ismn", default=reference.ISMN_GEOJSON_DEFAULT, help="ismn_locations.geojson (default: shared bucket)")
    ap.add_argument("--ismn-dir", default=reference.ISMN_DIR_DEFAULT, help="folder with ISMN export files <network>+<station>.csv (searched recursively)")
    ap.add_argument("--assets-csv", help="product list instead of a catalogue search")
    ap.add_argument("--half-m", type=float, default=300, help="half width of the analysis window [m]")
    ap.add_argument("--look-m", type=float, default=100, help="multilook cell size [m]")
    ap.add_argument("--out", default="~/data/runs", help="output root (local home is much faster than the S3-backed my-private-bucket)")
    ap.add_argument("--no-sar", action="store_true", help="reference part only (no token)")
    ap.add_argument("--no-s2", action="store_true", help="skip the Sentinel-2 NDVI / NDMI series (Earth Search on AWS)")
    ap.add_argument("--image-look-m", type=float, default=50, help="display cell size of window_images.png [m] (the analysis uses --look-m)")
    ap.add_argument("--no-quicklooks", action="store_true"); ap.add_argument("--lut-check", action="store_true", help="print the LUT groups and variables")
    ap.add_argument("--min-sm-range", type=float, default=0.05, help="minimum reference SM range [m3/m3] for reporting a sensitivity slope")
    ap.add_argument("--min-valid", type=float, default=0.8, help="minimum fraction of valid pixels in the window")
    ap.add_argument("--flatten", choices=["auto", "on", "off"], default="auto", help="LUT flattening screen: auto = apply if coherence increases")
    ap.add_argument("--no-denoise", action="store_true", help="do not subtract the LUT noise maps")
    ap.add_argument("--dielectric", choices=sorted(estimators.DIELECTRIC), default="mironov"); ap.add_argument("--clay", type=float, default=None, help="clay fraction 0-1 (default: SoilGrids at the point, else 0.2)"); ap.add_argument("--sand", type=float, default=None, help="sand fraction 0-1 (default: SoilGrids at the point, else 0.4)")
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
