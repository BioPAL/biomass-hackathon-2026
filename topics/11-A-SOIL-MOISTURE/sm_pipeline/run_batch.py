"""Run the pipeline for a list of stacks without interaction and aggregate the results.

    python run_batch.py my_stacks.csv                    # runs every row, then aggregates
    python run_batch.py --aggregate-only                 # only rebuild all_metrics.csv / all_tiles_summary.csv from existing runs
    nohup python run_batch.py my_stacks.csv > batch.log 2>&1 &        # keeps running when the browser tab closes

Stack list (CSV) columns: stack (required), lat, lon, name (optional), extra (optional extra CLI args, e.g. "--half-m 500 --only-point").
Rows whose run folder already contains metrics.csv are skipped unless --force.
Outputs in ~/data/runs/: batch_log.csv (one row per stack: status, duration, tier, points),
all_metrics.csv (concatenated metrics.csv with stack column), all_tiles_summary.csv (one row per stack x point with the key numbers).
"""
import argparse, glob, json, os, pathlib, subprocess, sys, time
import numpy as np, pandas as pd

RAIN_ORDER = ("ismn_rain_between_mm", "era5_land_rain_between_mm", "era5_rain_between_mm", "ecmwf_ifs_rain_between_mm")   # as plots.RAIN_KEYS

HERE = pathlib.Path(__file__).resolve().parent


def _mean(v):
    x = np.array([np.nan if e is None else e for e in (v or [])], float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def run_one(row, out_root, force):
    out = out_root / row["stack"]
    if (out / "metrics.csv").exists() and not force: return dict(stack=row["stack"], status="skipped (exists)", seconds=0)
    cmd = [sys.executable, str(HERE / "run_stack.py"), "--stack", row["stack"], "--out", str(out_root)]
    if pd.notna(row.get("lat")) and pd.notna(row.get("lon")): cmd += ["--lat", str(row["lat"]), "--lon", str(row["lon"])]
    if isinstance(row.get("name"), str) and row["name"]: cmd += ["--name", row["name"]]
    if isinstance(row.get("extra"), str) and row["extra"].strip(): cmd += row["extra"].split()
    t = time.time(); out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    tier = json.load(open(out / "tier.json")) if (out / "tier.json").exists() else {}
    return dict(stack=row["stack"], status="ok" if p.returncode == 0 else f"exit {p.returncode}", seconds=round(time.time() - t),
                tier=tier.get("tier"), points=";".join(tier.get("points", [])), stations=";".join(tier.get("stations", [])))


def _read_or_none(f):
    """CSV as DataFrame, None if missing, empty or broken."""
    try: return pd.read_csv(f) if pathlib.Path(f).exists() else None
    except (pd.errors.EmptyDataError, pd.errors.ParserError): return None


def aggregate(out_root):
    mets = []
    for f in glob.glob(str(out_root / "*" / "metrics.csv")):
        try: m = pd.read_csv(f)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as ex:
            print(f"[aggregate] skipping {f}: {type(ex).__name__} (empty or broken file; re-run or replot that stack)"); continue
        m.insert(0, "stack", pathlib.Path(f).parent.name); mets.append(m)
    M = pd.concat(mets, ignore_index=True) if mets else pd.DataFrame()
    M.to_csv(out_root / "all_metrics.csv", index=False)
    rows = []
    for tj in glob.glob(str(out_root / "*" / "tier.json")):
        d = pathlib.Path(tj).parent; tier = json.load(open(tj)); refs = json.load(open(d / "reference_at_overpass.json")) if (d / "reference_at_overpass.json").exists() else []
        log = open(d / "run.log").read() if (d / "run.log").exists() else ""
        for r in refs:
            st = r["station"]; pd_ = d / st
            info = json.load(open(pd_ / "sar_info.json")) if (pd_ / "sar_info.json").exists() else {}
            raw = _read_or_none(pd_ / "backscatter_raw.csv")
            mm = M[(M["stack"] == d.name) & (M["station"] == st)] if len(M) and "reference" in M else pd.DataFrame()
            def pick(obs, ref_pref):
                if not len(mm) or "observable" not in mm: return None, None          # reference-only run (no SAR metrics)
                for ref in ref_pref:
                    q = mm[(mm.observable == obs) & (mm.reference == ref)]
                    if len(q): return q.iloc[0], ref
                return None, None
            ismn_refs = sorted([x for x in mm.reference.dropna().unique() if str(x).startswith("ISMN")], key=lambda x: float(x.split()[1].rstrip("m"))) if len(mm) else []
            pref = ismn_refs + ["era5_land 0-7 cm", "era5 0-7 cm", "ecmwf_ifs 0-7 cm"]   # same order as the figures (plots.REF_MODELS)
            s_hh, ref_used = pick("sigma0 HH", pref); s_vv, _ = pick("sigma0 VV", pref); cc, _ = pick("closure phase HH", pref)
            skipped = next((l.split("SAR skipped -> ")[1] for l in log.splitlines() if f"[4] {st}: SAR skipped" in l), "")
            if not skipped and len(mm) and "metric" in mm:
                sk = mm[mm["metric"] == "skipped"]
                if len(sk): skipped = str(sk["note"].iloc[0])
            lc = r.get("aux_landcover_pct") or {}
            rows.append(dict(stack=d.name, station=st, network=r.get("network"), lat=r.get("lat"), lon=r.get("lon"), tier=tier.get("tier"), has_ismn=r.get("has_ismn"),
                             first=raw["acq_datetime_utc"].iloc[0][:10] if raw is not None else "", last=raw["acq_datetime_utc"].iloc[-1][:10] if raw is not None else "",
                             n_acq=len(raw) if raw is not None else 0, height_m=info.get("height_m"), incidence_deg=info.get("incidence_deg"),
                             dem_slope_deg=r.get("aux_slope_mean_deg"), lut_slope_az_deg=info.get("lut_slope_az_deg"), lut_slope_rg_deg=info.get("lut_slope_rg_deg"),
                             grassland_pct=lc.get("grassland"), tree_pct=lc.get("tree"), bare_pct=lc.get("bare"), valid_pct=info.get("valid_pct"),
                             clay_pct=(r.get("aux_soil") or {}).get("clay_pct_0_5cm"), sand_pct=(r.get("aux_soil") or {}).get("sand_pct_0_5cm"),
                             soc_g_kg=(r.get("aux_soil") or {}).get("soc_g_kg_0_5cm"),
                             ndvi_mean=_mean(r.get("s2_ndvi_nearest")), ndmi_mean=_mean(r.get("s2_ndmi_nearest")),
                             tmin_C=r.get("ecmwf_ifs_tmin_window_C"), snow_max_m=r.get("ecmwf_ifs_snow_depth_max_window_m"),
                             rain_total_mm=float(np.nansum([v for v in next((r[k] for k in RAIN_ORDER if r.get(k) and np.isfinite(np.array(r[k], float)).any()), []) if v is not None])),
                             ref_used=ref_used, ref_range=s_hh.ref_range if s_hh is not None else np.nan,
                             dsigma0_HH_dB=s_hh.obs_range if s_hh is not None else np.nan, dsigma0_VV_dB=s_vv.obs_range if s_vv is not None else np.nan,
                             slope_HH=s_hh.value if s_hh is not None else np.nan, slope_VV=s_vv.value if s_vv is not None else np.nan,
                             r_HH=s_hh.r if s_hh is not None else np.nan, r_VV=s_vv.r if s_vv is not None else np.nan,
                             r_diff_HH=s_hh.r_diff if s_hh is not None else np.nan, r_diff_VV=s_vv.r_diff if s_vv is not None else np.nan,
                             closure_r_HH=cc.r if cc is not None else np.nan, closure_note=cc.note if cc is not None else "",
                             mean_coh_HH=info.get("mean_coh_3day_HH", np.nan), r_diff_p_HH=s_hh.r_diff_p if s_hh is not None and "r_diff_p" in s_hh else np.nan,
                             flattening=info.get("flattening", ""), denoising=info.get("denoising", ""), skipped=skipped))
    S = pd.DataFrame(rows).sort_values(["tier", "stack", "station"]) if rows else pd.DataFrame()
    S.to_csv(out_root / "all_tiles_summary.csv", index=False)
    return M, S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("batch_csv", nargs="?"); ap.add_argument("--out", default="~/data/runs")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--aggregate-only", action="store_true"); a = ap.parse_args()
    out_root = pathlib.Path(os.path.expanduser(a.out)); out_root.mkdir(parents=True, exist_ok=True)
    if not a.aggregate_only:
        b = pd.read_csv(a.batch_csv); logs = []
        for i, row in b.iterrows():
            print(f"[{i + 1}/{len(b)}] {row['stack']} ...", flush=True); res = run_one(row, out_root, a.force); logs.append(res); print("    ", res, flush=True)
            pd.DataFrame(logs).to_csv(out_root / "batch_log.csv", index=False)
    M, S = aggregate(out_root)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    if len(S): print(S[["stack", "station", "has_ismn", "first", "last", "dem_slope_deg", "grassland_pct", "ref_used", "ref_range", "dsigma0_HH_dB", "r_diff_HH", "mean_coh_HH", "closure_r_HH", "skipped"]].round(2).to_string(index=False))
    print(f"[done] {len(S)} tile x point rows -> {out_root / 'all_tiles_summary.csv'}")
