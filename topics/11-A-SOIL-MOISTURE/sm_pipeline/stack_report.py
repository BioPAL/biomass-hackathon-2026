#!/usr/bin/env python
"""Full report of ONE computed stack as a self-contained HTML file (images embedded): <run>/report.html.

    python stack_report.py ~/data/runs/S3_T018_F138_G01_M03

Sections: stack facts (acquisitions, primary, reference tier, points), key numbers per point (from all_tiles_summary.csv, updated
on the fly), reference data at the overpasses, and per point every figure of run_stack.py with a one-line explanation, the
metrics rows and the processing decisions from run.log. Called by the dashboard app after each run.
"""
import base64, html, importlib.util, io, json, pathlib, sys
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CSS = """body{font-family:Helvetica,Arial,sans-serif;margin:0;background:#f4f4f4;color:#222}header{background:#0072B2;color:#fff;padding:12px 24px}
header h1{margin:0;font-size:19px}main{max-width:1500px;margin:0 auto;padding:12px 24px}
section{background:#fff;border-radius:6px;padding:12px 18px;margin:12px 0;box-shadow:0 1px 2px #0002}h2{font-size:16px;margin:4px 0 8px}
table{border-collapse:collapse;font-size:12.5px}th,td{border:1px solid #ddd;padding:3px 7px;text-align:right}th{background:#eee}
td:first-child,th:first-child,td.l{text-align:left}img{max-width:100%;height:auto;border:1px solid #ddd}.fig{margin:10px 0}.fig p{margin:2px 0 4px;font-size:12.5px;color:#444}
details summary{cursor:pointer;font-weight:bold}pre{background:#f7f7f7;font-size:11.5px;padding:8px;overflow:auto;max-height:360px}
.bad{color:#D55E00}.note{font-size:12.5px;color:#555}.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
.tabs a{padding:3px 9px;background:#e8eef5;border-radius:4px;text-decoration:none;color:#0072B2;font-size:12.5px}.good{background:#e6f4ea}.weak{background:#fdecea}
table.site{font-size:12px;margin:2px 0 4px}table.site th{background:#f2f2f2;text-align:left;font-weight:normal;color:#444}"""
FIGS = [("biomass_vs_reference.png", "Main figure, one time axis: in-situ soil moisture of every station depth (hourly; panel only if the point has a station) and ERA5-Land 0-7 cm (hourly), markers at the overpasses; BIOMASS sigma0 HH and VV; |coherence| of consecutive 3-day pairs (HH, VV); daily rain; hourly temperature. Coherence and temperature axes are scaled to the data. Site facts in the table above each figure."),
        ("s2_ndvi_ndmi.png", "Sentinel-2 L2A NDVI (B08, B04) and NDMI (B8A, B11) over the analysis window, 30 days before the first to 30 days after the last acquisition; filled = clear scenes (>= 80 % of the box clear), hollow = partly cloudy; method from the team notebook 'Biomass L1C Station Ancillary'."),
        ("sigma0_vs_reference.png", "sigma0 [dB] against the same soil-moisture references; slopes are only meaningful when the reference range is >= 0.05 m3/m3."),
        ("coherence_pairs.png", "All 21 pair coherences against temporal baseline, the coherence matrices and |coh| vs |dSM| controlled for time (y axes scaled to the data)."),
        ("sar_observables.png", "Closure phase of consecutive triplets on the time axis of the main figure (bar over the three dates, dot at the middle date, as the coherence pairs there); power partitions of Freeman-Durden, Yamaguchi-4 y4r and van Zyl 1992 and H / A / alpha, all from PyPolSARpro."),
        ("closure_check.png", "Observed closure phases against the De Zan soil-moisture model driven by the reference series."),
        ("quicklooks_pauli.png", "Pauli quicklooks: full primary scene with the window (dotted) and every acquisition zoomed to the window + 15 %."),
        ("window_images.png", "The analysis window in 50 m display cells (the analysis itself uses 100 m looks) for every date and all four polarisations (shared scale) and the temporal standard deviation per cell."),
        ("window_geometry.png", "LUT geometry in the window: height, incidence, terrain slopes, beta0 -> sigma0 factor.")]
KEY = ["station", "has_ismn", "ref_used", "ref_range", "dsigma0_HH_dB", "dsigma0_VV_dB", "r_diff_HH", "r_diff_p_HH", "r_diff_VV", "mean_coh_HH",
       "closure_r_HH", "dem_slope_deg", "grassland_pct", "tree_pct", "valid_pct", "flattening", "denoising", "skipped"]


def img(path, max_w=1600):
    """PNG -> embedded JPEG (q85, at most max_w px wide): keeps the report a few MB."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > max_w: im = im.resize((max_w, int(im.height * max_w / im.width)))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or str(v) in ("nan", "None", ""): return ""
    if isinstance(v, (float, np.floating)): return f"{v:.3g}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def table(df, cls_fn=None):
    h = ["<table><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"]
    for _, r in df.iterrows():
        h.append("<tr>" + "".join(f'<td class="{(cls_fn(c, r) if cls_fn else "")}">{fmt(r[c])}</td>' for c in df.columns) + "</tr>")
    return "\n".join(h) + "</table>"


def cls(c, r):
    if c == "r_diff_HH" and np.isfinite(r.get("r_diff_HH", np.nan)) and r.get("ref_range", 0) >= 0.05:
        p = r.get("r_diff_p_HH", np.nan)                 # two-sided p of r_diff; the |r| needed depends on the number of dates
        if p is not None and np.isfinite(p): return "good" if p < 0.05 else "weak"
        return "good" if abs(r["r_diff_HH"]) >= 0.81 else "weak"
    return ""


def _f(v, nd=1, unit=""):
    try: v = float(v)
    except (TypeError, ValueError): return ""
    return f"{v:.{nd}f}{unit}" if np.isfinite(v) else ""


def _info(pdir):
    f = pathlib.Path(pdir) / "sar_info.json"
    try: return json.load(open(f)) if f.exists() else {}
    except Exception: return {}


def _s2_text(r):
    v = [np.array([np.nan if e is None else e for e in (r.get(k) or [])], float) for k in ("s2_ndvi_nearest", "s2_ndmi_nearest")]
    if not len(v[0]) or not np.isfinite(v[0]).any(): return ""
    return f"{np.nanmean(v[0]):.2f} / {np.nanmean(v[1]):.2f} (clear scene within 15 days for {int(np.isfinite(v[0]).sum())} of {len(v[0])} dates)"


def site_table(r, info=None):
    """Small table with the site facts of one point (LUT geometry, DEM slope, land cover, SoilGrids soil, window, processing)."""
    info = info or {}; so = r.get("aux_soil") or {}
    lc = sorted((r.get("aux_landcover_pct") or {}).items(), key=lambda kv: -kv[1])[:3]
    sl, gt5 = _f(r.get("aux_slope_mean_deg"), 1, "°"), _f(r.get("aux_slope_frac_gt5_pct"), 0, " %")
    tex = lambda d: (" / ".join(_f(so.get(f"{k}_pct_{d}"), 0) for k in ("clay", "sand", "silt")) + " %") if f"clay_pct_{d}" in so else ""
    items = [("elevation (LUT)", _f(info.get("height_m"), 0, " m")),
             ("incidence, LUT / local", " / ".join(x for x in (_f(info.get("incidence_deg"), 1, "°"), _f(info.get("local_incidence_deg"), 1, "°")) if x)),
             ("terrain slope az / rg (LUT)", " / ".join(x for x in (_f(info.get("lut_slope_az_deg"), 1, "°"), _f(info.get("lut_slope_rg_deg"), 1, "°")) if x)),
             ("DEM slope (share > 5°)", sl + (f" ({gt5})" if gt5 else "")),
             ("land cover (WorldCover)", ", ".join(f"{k} {v:.0f} %" for k, v in lc)),
             ("clay / sand / silt 0-5 cm", tex("0_5cm")), ("clay / sand / silt 5-15 cm", tex("5_15cm")),
             ("organic carbon, bulk density 0-5 cm", f"{_f(so.get('soc_g_kg_0_5cm'), 0)} g/kg, {_f(so.get('bulk_density_kg_dm3_0_5cm'), 2)} kg/dm³" if "soc_g_kg_0_5cm" in so else ""),
             ("analysis window", f"{_f(info.get('window_m'), 0)} m, {_f(info.get('look_m'), 0)} m cells, valid {_f(info.get('valid_pct'), 0)} %" if info.get("window_m") else ""),
             ("NDVI / NDMI (Sentinel-2, at the overpasses)", _s2_text(r)),
             ("flattening", str(info.get("flattening") or "")), ("denoising", str(info.get("denoising") or ""))]
    cells = [f"<th>{html.escape(k)}</th><td class='l'>{html.escape(v)}</td>" for k, v in items if v]
    if not cells: return ""
    return ("<table class='site'>" + "".join("<tr>" + "".join(cells[i:i + 3]) + "</tr>" for i in range(0, len(cells), 3)) + "</table>"
            "<p class='note'>Soil: ISRIC SoilGrids 2.0 (250 m, modelled). Land cover (ESA WorldCover) and DEM slope: around the point.</p>")


def _summary_rows(run):
    """Key numbers for this stack: aggregate all runs next to it (run_batch.aggregate) and keep this stack's rows."""
    spec = importlib.util.spec_from_file_location("run_batch", HERE / "run_batch.py"); rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
    M, S = rb.aggregate(run.parent)
    return (S[S["stack"] == run.name] if len(S) else pd.DataFrame()), (M[M["stack"] == run.name] if len(M) and "stack" in M else pd.DataFrame())


def build(run):
    run = pathlib.Path(run).expanduser().resolve(); stack = run.name
    rows, M = _summary_rows(run)
    tier = json.load(open(run / "tier.json")) if (run / "tier.json").exists() else {}
    refs = json.load(open(run / "reference_at_overpass.json")) if (run / "reference_at_overpass.json").exists() else []
    try: prod = pd.read_csv(run / "products.csv")
    except Exception: prod = pd.DataFrame()
    log = (run / "run.log").read_text() if (run / "run.log").exists() else ""
    b = []
    main = [(r, run / r["station"] / "biomass_vs_reference.png") for r in refs if (run / r["station"] / "biomass_vs_reference.png").exists()]
    if main:                                  # main topic first: soil moisture and BIOMASS on one time axis, one figure per point
        b.append("<section><h2>Soil moisture and BIOMASS at each point</h2><p class='note'>" + FIGS[0][1] + "</p>" +
                 "".join(f"<div class='fig' id='{r['station']}-main'><p><b>{r['station']}</b> ({r.get('network', '')}; {r['lat']:.4f}, {r['lon']:.4f})</p>"
                         + site_table(r, _info(run / r["station"])) + f"<img src='{img(f)}'></div>" for r, f in main) + "</section>")
    b.append("<section><h2>Stack facts</h2><table>")
    if len(prod): b.append(f"<tr><td class='l'>acquisitions</td><td class='l'>{len(prod)}: {', '.join(str(d)[:10] for d in prod['datetime'])}; primary {', '.join(str(d)[:10] for d in prod.loc[prod['primary'].astype(bool), 'datetime'])}</td></tr>")
    b.append(f"<tr><td class='l'>reference tier</td><td class='l'>{tier.get('tier', '')}: {tier.get('tier_text', '')}</td></tr>")
    b.append(f"<tr><td class='l'>points</td><td class='l'>{', '.join(r['station'] for r in refs)}</td></tr><tr><td class='l'>folder</td><td class='l'>{run}</td></tr></table></section>")
    if len(rows):
        b.append("<section><h2>Key numbers per point</h2>" + table(rows[[c for c in KEY if c in rows]], cls) +
                 "<p class='note'>r_diff = correlation of first differences of sigma0 HH and the soil-moisture reference (green if p &lt; 0.05; 7 acquisitions = 6 differences need |r_diff| &ge; 0.81, 6 acquisitions need &ge; 0.88); "
                 "green / red: reference range &ge; 0.05 m3/m3 with strong / weak response. Slope and land cover are information. Reading guide: INTERPRETATION_GUIDE.md.</p></section>")
    if (run / "reference_sources.png").exists() and not any((run / r["station"] / "biomass_vs_reference.png").exists() for r in refs):  # reference-only run
        b.append(f"<section><h2>Reference data at the overpasses</h2><div class='fig'><img src='{img(run / 'reference_sources.png')}'></div></section>")
    for r in refs:
        st = r["station"]; pdir = run / st
        b.append(f"<section id='{st}'><h2>{st} ({r.get('network', '')}; {r['lat']:.4f} N, {r['lon']:.4f} E)</h2>")
        skip = next((l.split("SAR skipped -> ")[1] for l in log.splitlines() if f"[4] {st}: SAR skipped" in l), "")
        if not skip and len(rows) and len(rows[rows["station"] == st]): skip = fmt(rows[rows["station"] == st]["skipped"].iloc[0])
        if not (pdir / "biomass_vs_reference.png").exists(): b.append(site_table(r, _info(pdir)))   # no main figure (and table) above
        if skip: b.append(f"<p class='note'><span class='bad'>SAR skipped: {skip}</span></p>")
        figs = [(f, d) for f, d in FIGS[1:] if (pdir / f).exists()]                       # the main figure is at the top
        if figs: b.append("<div class='tabs'>" + "".join(f"<a href='#{st}-{f[:-4]}'>{f[:-4].replace('_', ' ')}</a>" for f, _ in figs) + "</div>")
        for f, d in figs: b.append(f"<div class='fig' id='{st}-{f[:-4]}'><p><b>{f}</b> - {d}</p><img src='{img(pdir / f)}'></div>")
        if len(M):
            mm = M[(M["station"] == st) & (M["observable"].astype(str).str.startswith(("sigma0 HH", "sigma0 VV", "closure phase HH")))]
            mm = mm[[c for c in ["reference", "observable", "metric", "value", "ci_low", "ci_high", "r", "r_diff", "r_diff_p", "n", "ref_range", "obs_range", "note"] if c in mm]]
            if len(mm): b.append(f"<details><summary>metrics.csv rows for {st} ({len(mm)})</summary>" + table(mm) + "</details>")
        b.append("</section>")
    if log:
        keep = [l for l in log.splitlines() if l.startswith(("[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[done]")) and any(k in l for k in ("skipped", "flatten", "denois", "tier", "products", "points", "LUT"))]
        b.append("<section><details><summary>run.log excerpt (tier, points, flattening, denoising, skips)</summary><pre>" + "\n".join(keep[:200]) + "</pre></details></section>")
    out = run / "report.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{stack}</title><style>{CSS}</style></head><body><header><h1>{stack}: BIOMASS vs soil moisture - full report</h1></header><main>{''.join(b)}</main></body></html>")
    print(f"[report] {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from sm_pipeline import deps; deps.ensure(polsarpro=False)
    build(sys.argv[1])
