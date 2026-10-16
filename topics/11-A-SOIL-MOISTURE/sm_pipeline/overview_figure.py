"""Overview figure of one station across one or more BIOMASS L1C STA stacks.

Rows on one shared time axis:
  stack strip, Pauli thumbnails (one row per stack if stacks overlap in time),
  soil moisture (ISMN hourly + ERA5-Land hourly), sigma0 HH/VV,
  3-day coherence pairs, closure phase of consecutive triplets,
  environment (station rain, Sentinel-2 NDVI/NDMI, ERA5-Land air temperature).

Inputs are the run folders written by run_stack.py:
  <runs>/<stack>/<station>/slc_window.npz, backscatter_raw.csv
  <runs>/<stack>/closure_phases.csv, reference_at_overpass.json, products.csv,
  era5_land_<station>.csv, s2_<station>_*m.csv
Optional: hourly ISMN file (long format: sensor, date_time, soil_moisture /
precipitation columns with *_flag).

Usage:
  python overview_figure.py --runs ~/data/runs --station Kukuihaele \
      --stacks S2_T018_F137_G01_M03 S3_T018_F137_G01_M03 \
      --ismn ISMN/SCAN+Kukuihaele.csv --out overview_Kukuihaele.png
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter

# colour-vision-safe palette (no red/green pairs carrying meaning)
HHc, VVc = "#1E88E5", "#FF6F00"
SMc, ERAc = "#1E88E5", "#90CAF9"
NDVIc, NDMIc, TEMPc, RAINc = "#00C853", "#2979FF", "#212121", "#4FC3F7"
BANDc = "#E1F5FE"
STACKc = ["#546E7A", "#FFB300", "#8E24AA"]
GRID = dict(color="0.25", lw=0.9, ls=":", zorder=0.5)


def _naive(s):
    s = pd.to_datetime(s, utc=True, format="mixed")
    return s.dt.tz_localize(None) if hasattr(s, "dt") else s.tz_localize(None)


def load_stack(runs, stack, station):
    d = os.path.join(runs, stack)
    z = np.load(os.path.join(d, station, "slc_window.npz"), allow_pickle=True)
    b = pd.read_csv(os.path.join(d, station, "backscatter_raw.csv"))
    b["t"] = _naive(b["acq_datetime_utc"])
    ref = [r for r in json.load(open(os.path.join(d, "reference_at_overpass.json"))) if r["station"] == station][0]
    cl = pd.read_csv(os.path.join(d, "closure_phases.csv"))
    cl = cl[(cl.station == station) & (cl.j == cl.i + 1) & (cl.k == cl.i + 2)].copy()
    e = pd.read_csv(os.path.join(d, f"era5_land_{station}.csv"))
    e["t"] = _naive(e["time"])
    s2f = sorted(glob.glob(os.path.join(d, f"s2_{station}_*m.csv")))
    s2 = pd.read_csv(s2f[0]) if s2f else pd.DataFrame(columns=["datetime", "clear", "ndvi_mean", "ndmi_mean"])
    if len(s2):
        s2 = s2[s2["clear"].astype(bool)].copy(); s2["t"] = _naive(s2["datetime"])
    orbit = ""
    pf = os.path.join(d, "products.csv")
    if os.path.exists(pf):
        p = pd.read_csv(pf)
        if "orbit_state" in p and len(p):
            orbit = str(p["orbit_state"].iloc[0])
    return dict(stack=stack, z=z, b=b, ref=ref, cl=cl, era=e, s2=s2, orbit=orbit)


def ismn_hourly(path, depth_m):
    """Return (soil moisture series at depth_m, hourly precipitation series or None)."""
    if not path or not os.path.exists(path):
        return None, None
    d = pd.read_csv(path, low_memory=False)
    d["t"] = _naive(d["date_time"])
    sm = pr = None
    if "soil_moisture" in d:
        x = d[d["sensor"].str.contains("soil_moisture") & d["soil_moisture"].notna()].copy()
        x["depth"] = x["sensor"].str.extract(r"_(\d+\.\d+)_\d+\.\d+$")[0].astype(float)
        x = x[np.isclose(x["depth"], depth_m, atol=0.003)]
        if "soil_moisture_flag" in x:
            x = x[x["soil_moisture_flag"].astype(str).str.startswith("G")]
        sm = x.groupby("t")["soil_moisture"].mean().sort_index()
    if "precipitation" in d:
        y = d[d["precipitation"].notna()]
        if "precipitation_flag" in y:
            y = y[y["precipitation_flag"].astype(str).str.startswith("G")]
        pr = y.groupby("t")["precipitation"].sum().sort_index()
    return sm, pr


def shallowest_ref(b):
    cols = [c for c in b.columns if c.startswith("ref ISMN ") and b[c].notna().sum() >= 3]
    if not cols:
        return None, None
    c = min(cols, key=lambda c: float(re.findall(r"[\d.]+", c)[0]))
    return c, float(re.findall(r"[\d.]+", c)[0])


def pauli_rgb(z, look_m=100.0):
    slc, names = z["slc"], list(z["names"])
    iHH, iHV, iVV = names.index("HH"), names.index("HV"), names.index("VV")
    nr = max(1, int(round(look_m / float(z["m_row"])))); nc = max(1, int(round(look_m / float(z["m_col"]))))
    P = []
    for S in slc:
        ch = [np.abs(S[iHH] - S[iVV]) ** 2 / 2, 2 * np.abs(S[iHV]) ** 2, np.abs(S[iHH] + S[iVV]) ** 2 / 2]
        P.append(np.stack([10 * np.log10(np.maximum(uniform_filter(np.nan_to_num(c), size=(nr, nc), mode="nearest"), 1e-12)) for c in ch], -1))
    P = np.array(P)
    lims = [np.nanpercentile(P[..., c], [2, 98]) for c in range(3)]
    return np.clip(np.stack([(P[..., c] - lims[c][0]) / (lims[c][1] - lims[c][0]) for c in range(3)], -1), 0, 1)


def deco(ax, tn_all, bands):
    for a0, a1 in bands:
        ax.axvspan(a0, a1, color=BANDc, zorder=0)
    for t in tn_all:
        ax.axvline(t, **GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(a):
    plt.rcParams.update({"font.size": 11.5, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
                         "legend.fontsize": 10.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    S = [load_stack(a.runs, s, a.station) for s in a.stacks]
    S.sort(key=lambda s: s["b"]["t"].min())
    for s in S:
        s["tn"] = list(s["b"]["t"])
    tn_all = sorted(t for s in S for t in s["tn"])
    overlap = any(S[i + 1]["tn"][0] < S[i]["tn"][-1] for i in range(len(S) - 1))
    spacing = np.median(np.diff([mdates.date2num(t) for t in tn_all])) if len(tn_all) > 1 else 3.0
    half = pd.Timedelta(days=0.46 * (3.0 if overlap else min(spacing, 3.0)))
    X0, X1 = tn_all[0] - pd.Timedelta(days=2), tn_all[-1] + pd.Timedelta(days=2)

    refcol, depth = shallowest_ref(S[0]["b"])
    if depth is None and a.ismn and os.path.exists(a.ismn):        # user point: use the nearby station file as context
        dd = pd.read_csv(a.ismn, low_memory=False)
        dep = dd.loc[dd["sensor"].str.contains("soil_moisture"), "sensor"].str.extract(r"_(\d+\.\d+)_\d+\.\d+$")[0].astype(float)
        depth = float(dep.min()) if len(dep) else None
    sm_h, pr_h = ismn_hourly(a.ismn, depth) if depth else (None, None)
    st_name = os.path.basename(a.ismn).replace(".csv", "").split("+")[-1] if a.ismn else ""

    # rain intervals between consecutive overpasses (per stack): hourly station rain if available, else interval totals
    bands, rain_iv = [], []
    for s in S:
        rb = s["ref"].get("ismn_rain_between_mm") or [np.nan] * len(s["tn"])
        for k in range(1, len(s["tn"])):
            t0, t1 = s["tn"][k - 1], s["tn"][k]
            tot = pr_h[(pr_h.index > t0) & (pr_h.index <= t1)].sum() if pr_h is not None and len(pr_h) else rb[k]
            rain_iv.append((t0, t1, tot))
            if np.isfinite(tot) and tot >= a.rain_mm:
                bands.append((t0, t1))

    n_pauli = 1                                   # one image row; stacks that overlap in time get one lane each
    L = len(S) if overlap else 1
    hr = [0.2] + [0.62 if L == 1 else 0.5 * L] + [0.75, 0.75, 0.55, 0.55, 0.9]
    W = 15.5 if len(tn_all) <= 7 else 19.0
    fig, axs = plt.subplots(len(hr), 1, figsize=(W, 2.2 * sum(hr) / 0.9 * 1.25), sharex=True,
                            gridspec_kw=dict(height_ratios=hr, hspace=0.62))
    aS, aP, aSM, aSig, aCo, aCl, aEn = axs[0], axs[1:1 + n_pauli], axs[1 + n_pauli], axs[2 + n_pauli], axs[3 + n_pauli], axs[4 + n_pauli], axs[5 + n_pauli]

    # stack strip
    for i, s in enumerate(S):
        y = 0.5 if not overlap else 1 - (i + 0.5) / len(S)
        aS.plot([s["tn"][0] - half, s["tn"][-1] + half], [y, y], color=STACKc[i % 3], lw=15 if not overlap else 9, solid_capstyle="butt")
        sty = ("" if not overlap else ("; filled markers, solid bars" if i == 0 else "; open markers, dashed bars"))
        aS.text(s["tn"][0] - half, y, f"  {s['stack']} ({s['orbit']}, {s['tn'][0]:%d %b}–{s['tn'][-1]:%d %b}{sty})  ",
                va="center", ha="left", fontsize=10.5, color="white" if i % 3 != 1 else "k", fontweight="bold")
    aS.set_ylim(0, 1); aS.axis("off")
    win_m = S[0]["z"]["slc"].shape[-2] * float(S[0]["z"]["m_row"])
    rain_src = "hourly station gauge" if pr_h is not None and len(pr_h) else "station gauge"
    aS.set_title(f"{a.station}: {len(tn_all)} BIOMASS overpasses in {len(S)} stack(s); dotted lines = overpasses, "
                 f"blue bands = rain ≥ {a.rain_mm:g} mm between overpasses ({rain_src})", loc="left", pad=10)

    # Pauli thumbnails
    ax = aP[0]
    for i, s in enumerate(S):
        z = s["z"]; look = 100.0 if z["slc"].shape[-2] * float(z["m_row"]) >= 500 else 50.0
        rgb = pauli_rgb(z, look)
        ext = [0, rgb.shape[2] * float(z["m_col"]), rgb.shape[1] * float(z["m_row"]), 0]
        prc = (float(z["point_rc"][0]) * float(z["m_row"]), float(z["point_rc"][1]) * float(z["m_col"]))
        for k, t in enumerate(s["tn"]):
            x0, x1 = mdates.date2num(t - half), mdates.date2num(t + half)
            ly0, lh = (1 - (i + 1) / L) + 0.03 / L, 0.94 / L
            ins = ax.inset_axes([x0, ly0, x1 - x0, lh], transform=ax.get_xaxis_transform())
            ins.imshow(rgb[k], extent=ext, interpolation="bilinear", aspect="auto")
            ins.plot(prc[1], prc[0], "+", color="#FFEB3B", ms=8, mew=1.8)
            ins.set_xticks([]); ins.set_yticks([])
            after = any(t1 == t for t0, t1 in bands)
            for sp in ins.spines.values():
                sp.set_color(HHc if after else (STACKc[i % 3] if L > 1 else "0.6")); sp.set_linewidth(2.5 if (after or L > 1) else 1)
        if L > 1:
            ax.text(-0.004, 1 - (i + 0.5) / L, f"{s['orbit']}\n{s['tn'][0]:%H:%M} UTC", transform=ax.transAxes, ha="right", va="center",
                    fontsize=10.5, color="k", fontweight="bold")
    ax.set_yticks([]); [ax.spines[x].set_visible(False) for x in ("top", "right", "left")]
    lab = "; frame colour = stack (blue = after rain)" if L > 1 else "; blue frame = after rain"
    ax.set_title(f"Pauli colour composite of the {win_m:.0f} m window ({look:.0f} m moving average): red = double-bounce, "
                 f"green = volume, blue = surface{lab}", loc="left")

    # soil moisture
    deco(aSM, tn_all, bands)
    e = pd.concat([s["era"] for s in S]).drop_duplicates("t").sort_values("t")
    e = e[(e.t >= X0) & (e.t <= X1)]
    aSM.plot(e.t, e.soil_moisture_0_to_7cm, color=ERAc, lw=1.6, label="ERA5-Land 0–7 cm (model, hourly)")
    if refcol:
        if sm_h is not None and len(sm_h):
            h = sm_h[(sm_h.index >= X0) & (sm_h.index <= X1)]
            aSM.plot(h.index, h.values, color=SMc, lw=2.2, zorder=4)
        for i, s in enumerate(S):
            aSM.plot(s["b"]["t"], s["b"][refcol], "o", ms=9, color=SMc, mec="k", mew=0.6, zorder=5,
                     label=(f"station {depth * 100:.0f} cm (" + ("hourly; " if sm_h is not None and len(sm_h) else "") + "dots = overpasses)") if i == 0 else None)
    if not refcol and sm_h is not None and len(sm_h):
        h = sm_h[(sm_h.index >= X0) & (sm_h.index <= X1)]
        aSM.plot(h.index, h.values, color=SMc, lw=2.2, zorder=4)
        at = h.reindex(pd.DatetimeIndex(tn_all).round("h"))
        aSM.plot(tn_all, at.values, "o", ms=9, color=SMc, mec="k", mew=0.6, zorder=5,
                 label=f"station {st_name} {depth * 100:.0f} cm (hourly{', ' + a.ismn_note if a.ismn_note else ''})")
    aSM.set_ylabel("soil moisture\n[m³/m³]"); aSM.legend(loc="lower right", bbox_to_anchor=(1, 0.98), ncols=2, frameon=False)
    aSM.set_title("Soil moisture at the station and in ERA5-Land", loc="left")

    # sigma0
    deco(aSig, tn_all, bands)
    for i, s in enumerate(S):
        mk = dict(mec="k", mew=0.6) if i == 0 or not overlap else dict(mfc="white", mew=1.6)
        aSig.plot(s["b"]["t"], s["b"]["sigma0_db_HH"], "o-", color=HHc, lw=2.5, ms=8, label="HH" if i == 0 else None, **({"mec": "k", "mew": 0.6} if not (overlap and i) else {"mfc": "white", "mew": 1.6}))
        aSig.plot(s["b"]["t"], s["b"]["sigma0_db_VV"], "s--", color=VVc, lw=2.5, ms=7, label="VV" if i == 0 else None, **({"mec": "k", "mew": 0.6} if not (overlap and i) else {"mfc": "white", "mew": 1.6}))
    aSig.set_ylabel("σ⁰ [dB]"); aSig.legend(loc="lower right", bbox_to_anchor=(1, 0.98), ncols=2, frameon=False)
    aSig.set_title("Radar backscatter" + (" (lines break between stacks: different geometry)" if len(S) > 1 else ""), loc="left")

    # coherence pairs
    deco(aCo, tn_all, bands)
    for i, s in enumerate(S):
        b = s["b"]
        for k in range(1, len(b)):
            for pol, c, dy in (("HH", HHc, -0.003), ("VV", VVc, 0.003)):
                v = b[f"coh_pair_with_previous_{pol}"].iloc[k]
                if np.isfinite(v):
                    second = overlap and i > 0
                    aCo.plot([b.t.iloc[k - 1], b.t.iloc[k]], [v + dy] * 2, color=c, lw=3.5 if second else 4, ls=(0, (2, 1)) if second else "-",
                             solid_capstyle="butt", label=pol if (i == 0 and k == 1) else None)
                    aCo.plot(b.t.iloc[k - 1] + (b.t.iloc[k] - b.t.iloc[k - 1]) / 2, v + dy, "o", color=c, ms=6,
                             **({"mfc": "white", "mew": 1.4} if second else {"mec": "k", "mew": 0.5}))
    aCo.set_ylabel("|coherence|\n3-day pairs"); aCo.legend(loc="lower right", bbox_to_anchor=(1, 0.98), ncols=2, frameon=False)
    aCo.set_title("Interferometric coherence of consecutive pairs", loc="left")

    # closure phase
    deco(aCl, tn_all, bands); aCl.axhline(0, color="k", lw=1)
    for i, s in enumerate(S):
        cl, tn = s["cl"], s["tn"]; second = overlap and i > 0
        for _, r in cl[cl.pol.isin(["HH", "VV"])].iterrows():
            c = HHc if r.pol == "HH" else VVc
            v = np.degrees(r.closure_obs_rad) + (-0.08 if r.pol == "HH" else 0.08)
            aCl.plot([tn[int(r.i)], tn[int(r.k)]], [v, v], color=c, lw=3, alpha=0.8, solid_capstyle="butt", ls=(0, (2, 1)) if second else "-")
            aCl.plot(tn[int(r.j)], v, "o", color=c, ms=7, **({"mfc": "white", "mew": 1.4} if second else {"mec": "k", "mew": 0.5}))
    lo, hi = aCl.get_ylim(); aCl.set_ylim(min(lo, 0) - 0.4, max(hi, 0) + 0.6)
    aCl.set_ylabel("closure phase\n[deg]")
    aCl.set_title("Closure phase of consecutive triplets (bar = the three dates, dot = middle date)", loc="left")

    # environment
    deco(aEn, tn_all, bands)
    if pr_h is not None and len(pr_h):
        p = pr_h[(pr_h.index >= X0) & (pr_h.index <= X1)]
        aEn.bar(p.index + pd.Timedelta(minutes=30), p.values, width=1 / 24, color=RAINc, edgecolor="#0288D1", lw=0.4, label="rain at the station (hourly)")
        aEn.set_ylabel("rain [mm h⁻¹]", color="#0288D1")
    else:
        has_gauge = any(np.isfinite(t[2]) for t in rain_iv)
        for t0, t1, tot in rain_iv:
            if np.isfinite(tot) and tot > 0:
                aEn.bar(t0 + (t1 - t0) / 2, tot, width=0.6, color=RAINc, edgecolor="k", lw=0.5)
        from matplotlib.patches import Patch
        rain_patch = Patch(facecolor=RAINc, edgecolor="k", lw=0.5, label="rain at the station (total between overpasses)" if has_gauge else "no rain gauge at the station")
        aEn.set_ylabel("rain [mm]\n(total between overpasses)" if has_gauge else "", color="#0288D1")
        if not has_gauge:
            aEn.set_yticks([]); aEn.spines["left"].set_visible(False)
    aEn.set_ylim(0, max(aEn.get_ylim()[1] * 1.15, 5))
    aV = aEn.twinx()
    s2 = pd.concat([s["s2"] for s in S if len(s["s2"])]) if any(len(s["s2"]) for s in S) else None
    if s2 is not None:
        s2 = s2.drop_duplicates("t").sort_values("t"); s2 = s2[(s2.t >= X0) & (s2.t <= X1)]
        aV.plot(s2.t, s2.ndvi_mean, "o-", color=NDVIc, lw=2.5, ms=8, mec="k", mew=0.5, label="NDVI (Sentinel-2)")
        aV.plot(s2.t, s2.ndmi_mean, "s-", color=NDMIc, lw=2.5, ms=7, mec="k", mew=0.5, label="NDMI (Sentinel-2)")
    aV.set_ylim(-1, 1); aV.set_ylabel("vegetation index [-]")
    aT = aEn.twinx(); aT.spines["right"].set_position(("axes", 1.06))
    aT.plot(e.t, e.temperature_2m, color=TEMPc, lw=1.4, label="air temperature (ERA5-Land, hourly)")
    aT.set_ylabel("air temperature [°C]")
    tlo, thi = np.nanmin(e.temperature_2m), np.nanmax(e.temperature_2m)
    aT.set_ylim(min(0, tlo - 2), max(30, thi + 2))
    h1, l1 = aEn.get_legend_handles_labels()
    if "rain_patch" in locals():
        h1, l1 = h1 + [rain_patch], l1 + [rain_patch.get_label()]
    _h, _l = None, None; h2, l2 = aV.get_legend_handles_labels(); h3, l3 = aT.get_legend_handles_labels()
    aEn.legend(h1 + h2 + h3, l1 + l2 + l3, loc="lower right", bbox_to_anchor=(1, 0.98), ncols=2, frameon=False)
    aEn.set_title(f"Environment: air temperature {tlo:.0f}–{thi:.0f} °C (ERA5-Land)", loc="left")

    aEn.set_xlim(X0, X1)
    aEn.xaxis.set_major_locator(mdates.DayLocator(interval=3 if len(tn_all) <= 7 else 6))
    aEn.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True); p.add_argument("--station", required=True)
    p.add_argument("--stacks", nargs="+", required=True); p.add_argument("--ismn", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--ismn-note", dest="ismn_note", default="", help="e.g. '2.1 km from the point' when the point is not the station")
    p.add_argument("--rain-mm", dest="rain_mm", type=float, default=5.0, help="rain total between overpasses that marks an interval (default 5 mm)")
    main(p.parse_args())
