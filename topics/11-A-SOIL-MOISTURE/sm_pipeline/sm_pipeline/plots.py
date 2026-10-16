"""Figures of a run. Layout rules: one quantity per panel, shared time axis, legends outside the axes (right),
fixed colours per polarisation and per reference source (Okabe-Ito palette, safe for red-green colour-vision deficiency),
acquisition dates as dark dotted lines, grid and zero lines darker and thicker than the matplotlib defaults (visible on
projectors and in print), every figure carries the stack, the point and the processing notes in its title."""
import json, pathlib, re, textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Okabe & Ito (2008) palette: blue, orange, reddish purple, vermillion, bluish green, sky blue, black
POLCOL = {"HH": "#0072B2", "HV": "#E69F00", "VH": "#CC79A7", "VV": "#D55E00", "XX": "#E69F00"}
MODCOL = {"era5_land": "#009E73", "era5": "#56B4E9", "ecmwf_ifs": "#000000"}
FDCOL = {"surface": "#0072B2", "double-bounce": "#D55E00", "volume": "#009E73"}
DEPTH_COLS = ("#000000", "#E69F00", "#56B4E9", "#CC79A7", "#009E73", "#8C510A", "#999999", "#F0E442")   # station depths, shallow -> deep
DEPTH_MARKERS = ("^", "s", "D", "v", "o", "P", "X", "*")                 # distinct hue AND marker per depth (Okabe-Ito based)
RAIN_KEYS = ["ismn_rain_between_mm", "era5_land_rain_between_mm", "era5_rain_between_mm", "ecmwf_ifs_rain_between_mm"]   # station, then REF_MODELS order


# Background / reference lines: darker and thicker than the matplotlib defaults (grid default: light grey, alpha 0.25-0.3 here before)
GRID_KW = dict(color="0.72", lw=0.8, ls="-")                  # grid, below all data
ACQ_KW = dict(color="0.3", ls=(0, (1.5, 2.0)), lw=1.0)        # BIOMASS acquisitions (dotted), above the grid, below the data
ZERO_KW = dict(color="0.15", lw=1.0)                          # zero lines (0 deg C, closure phase 0)


def _acq_lines(ax, t):
    for tt in t: ax.axvline(tt, zorder=0.6, **ACQ_KW)


def _grid(ax):
    """Grid below everything (bars included), darker and thicker than the default."""
    ax.set_axisbelow(True); ax.grid(True, **GRID_KW)


def _legend_right(ax, **kw):
    """Legend right of the axes; further right if the axes has a twin y axis (its tick labels sit there)."""
    h, l = ax.get_legend_handles_labels()
    twin = any(o is not ax and o.get_position().bounds == ax.get_position().bounds for o in ax.figure.axes)
    if h: ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.09 if twin else 1.01, 1.0), fontsize=7, frameon=False, **kw)


def _vals(v):
    return np.array([np.nan if x is None else x for x in v], float)


def _depth(k):
    return float(k.replace("ismn_soil_moisture_", "").replace("m", "").split("-")[0])


def _rain(r):
    k = next((k for k in RAIN_KEYS if k in r and np.isfinite(_vals(r[k])).any()), None)
    return (k, _vals(r[k])) if k else (None, None)


def _rain_label(k):
    return "station gauge" if k.startswith("ismn") else MODEL_LABEL.get(k.split("_rain")[0], k.split("_rain")[0])


def depth_style(i):
    """Colour and marker of the i-th station depth (sorted shallow -> deep)."""
    return DEPTH_COLS[i % len(DEPTH_COLS)], DEPTH_MARKERS[i % len(DEPTH_MARKERS)]


def _depth_label(k):
    return re.sub(r"m$", " m", k.replace("ismn_soil_moisture_", "").replace("in situ ", ""))


def _station_sm(ax, t, r):
    depths = sorted([k for k in r if k.startswith("ismn_soil_moisture") and isinstance(r[k], list)], key=_depth)
    for i, k in enumerate(depths):
        c, mk = depth_style(i); deep = _depth(k) > 0.4
        ax.plot(t, _vals(r[k]), "--" if deep else "-", marker=mk, color=c, lw=0.9 if deep else 1.5, ms=5, label="station " + _depth_label(k))
    if not depths: ax.text(0.5, 0.5, "no in-situ station data for this point", ha="center", va="center", transform=ax.transAxes, color="0.4")
    ax.set_ylabel("in-situ soil moisture\n[m³/m³]")


def _rel_ylim(ax, vals, min_span, lo_cap=None, hi_cap=None):
    """y axis scaled to the data (not to a fixed absolute range): data range + 10 % on each side, at least min_span wide."""
    v = np.asarray(vals, float).ravel(); v = v[np.isfinite(v)]
    if not len(v): return
    lo, hi = float(v.min()), float(v.max()); span = max(hi - lo, min_span); mid = 0.5 * (lo + hi)
    a, b = mid - 0.6 * span, mid + 0.6 * span
    ax.set_ylim(a if lo_cap is None else max(a, lo_cap), b if hi_cap is None else min(b, hi_cap))


def _zero_line(ax):
    """0 deg C line only if the data cross it: the axis is scaled to the data and not forced to include 0."""
    lo, hi = ax.get_ylim()
    if lo < 0 < hi: ax.axhline(0, **ZERO_KW); ax.set_ylim(lo, hi)


PLOT_POLS = ("HH", "VV")                    # polarisations shown in the figures (window_images: all four)
REF_MODELS = ("era5_land", "era5", "ecmwf_ifs")      # model reference of the figures: the first one present
MODEL_LABEL = {"era5_land": "ERA5-Land", "era5": "ERA5", "ecmwf_ifs": "ECMWF IFS"}


def ref_model(r):
    return next((m for m in REF_MODELS if f"{m}_soil_moisture_0_to_7cm" in r and np.isfinite(_vals(r[f"{m}_soil_moisture_0_to_7cm"])).sum() >= 3), None)


def _model_sm(ax, t, r):
    for m, c in MODCOL.items():
        if f"{m}_soil_moisture_0_to_7cm" in r: ax.plot(t, _vals(r[f"{m}_soil_moisture_0_to_7cm"]), "o-", color=c, lw=1.2, ms=4, label=f"{m} 0-7 cm")
        if f"{m}_soil_moisture_7_to_28cm" in r: ax.plot(t, _vals(r[f"{m}_soil_moisture_7_to_28cm"]), ":", color=c, lw=1.0, label=f"{m} 7-28 cm")
    ax.set_ylabel("model soil moisture\n[m³/m³]")


def _rain_panel(ax, t, r):
    k, v = _rain(r)
    if k: ax.bar(t, np.nan_to_num(v), width=1.2, color="#56B4E9", alpha=.9, label=f"rain since previous acquisition ({_rain_label(k)})")
    else: ax.text(0.5, 0.5, "no precipitation source", ha="center", va="center", transform=ax.transAxes, color="0.4")
    ax.set_ylabel("rain [mm]"); ax.set_ylim(bottom=0)


def _temperature_panel(ax, t, r):
    for m in ("era5_land", "ecmwf_ifs"):
        c = MODCOL[m]
        if f"{m}_temperature_2m" in r: ax.plot(t, _vals(r[f"{m}_temperature_2m"]), "-", marker=".", color=c, label=f"{m} T 2 m")
        if f"{m}_dew_point_2m" in r: ax.plot(t, _vals(r[f"{m}_dew_point_2m"]), ":", color=c, alpha=.8, label=f"{m} dew point 2 m")
        if f"{m}_soil_temperature_0_to_7cm" in r: ax.plot(t, _vals(r[f"{m}_soil_temperature_0_to_7cm"]), "--", color=c, alpha=.8, label=f"{m} soil T 0-7 cm")
    for k in [k for k in r if k.startswith("ismn_air_temperature") and isinstance(r[k], list)]: ax.plot(t, _vals(r[k]), "^-", color="0.45", label="station air T")
    ax.set_ylabel("temperature [°C]")
    sk = next((k for k in ["era5_land_snow_depth", "ecmwf_ifs_snow_depth"] if k in r and np.nanmax(np.r_[_vals(r[k]), 0]) > 0.005), None)
    if sk:
        ax2 = ax.twinx(); ax2.bar(t, 100 * np.nan_to_num(_vals(r[sk])), width=1.2, alpha=.25, color="grey"); ax2.set_ylabel("snow depth [cm]"); ax2.set_ylim(bottom=0)
        ax.plot([], [], color="grey", lw=6, alpha=.25, label=f"snow depth ({sk.split('_snow')[0]}, right axis)")
    _zero_line(ax)


def reference_panel(ref_rows, times, out_png, title=""):
    """Reference data only (no token): per point in-situ SM, model SM, rain, temperature/snow."""
    t = pd.to_datetime(times, utc=True); n = len(ref_rows)
    fig, axes = plt.subplots(4 * n, 1, figsize=(12, 2.6 * 4 * n), squeeze=False, sharex=True)
    for i, r in enumerate(ref_rows):
        a = axes[4 * i:4 * i + 4, 0]
        _station_sm(a[0], t, r); _model_sm(a[1], t, r); _rain_panel(a[2], t, r); _temperature_panel(a[3], t, r)
        a[0].set_title(f"{r['station']} ({r['network']})", loc="left", fontsize=9)
        for ax in a: _acq_lines(ax, t); _grid(ax); _legend_right(ax)
    axes[-1, 0].set_xlabel("acquisition date (UTC)"); fig.autofmt_xdate()
    fig.suptitle(title or "Reference data at the BIOMASS overpasses", fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.09, right=0.73, top=0.95, bottom=0.05, hspace=0.18); fig.savefig(out_png, dpi=130); plt.close(fig)


def _pair_coh_panel(ax, t, names, coh_pairs, pols=None):
    """|coherence| of consecutive acquisitions (3-day pairs), drawn as a step between the two dates of each pair; y axis
    scaled to the data (not 0-1) so that the changes between the pairs are visible."""
    vals = []
    for b, n in enumerate(names):
        if pols and n not in pols: continue
        for i in range(len(t) - 1):
            ax.plot([t[i], t[i + 1]], [coh_pairs[i, b]] * 2, "-", color=POLCOL.get(n), lw=2.0, alpha=.85, label=n if i == 0 else None)
        ax.plot([t[i] + (t[i + 1] - t[i]) / 2 for i in range(len(t) - 1)], coh_pairs[:, b], "o", color=POLCOL.get(n), ms=4); vals += list(coh_pairs[:, b])
    _rel_ylim(ax, vals, 0.1, 0.0, 1.0); ax.set_ylabel("|coherence|\n3-day pairs")


def combined_panel(times, names, sig_db_t, coh_pairs, ref_row, out_png, stack, station, hourly=None, **_old):
    """Main figure of a point, one time axis (5 days before the first to 2 days after the last acquisition): in-situ soil
    moisture of every station depth (hourly; this panel only if the point has a station), the model reference ERA5-Land
    0-7 cm (hourly; markers = values at the BIOMASS overpasses), sigma0 HH and VV (window mean), |coherence| of consecutive
    3-day pairs, daily rain and hourly temperature (+ snow). Coherence and temperature axes are scaled to the data (the
    incidence angle per acquisition is in sar_info.json, not in the figure). Site facts are not in the title: the report shows
    them in a table above the figure."""
    t = pd.to_datetime(times, utc=True); r = ref_row; H = hourly or {}; m = ref_model(r)
    hsm = H.get("sm") or []
    ins_h = [(lab, sr) for lab, sr in hsm if lab.startswith("in situ")]; mod_h = [(lab, sr) for lab, sr in hsm if not lab.startswith("in situ")]
    depths = sorted([k for k in r if k.startswith("ismn_soil_moisture") and isinstance(r[k], list) and np.isfinite(_vals(r[k])).any()], key=_depth)
    rows = (["insitu"] if (ins_h or depths) else []) + ["model", "sigma", "coh", "rain", "temp"]
    hr = dict(insitu=1.4, model=1.0, sigma=1.2, coh=0.9, rain=0.7, temp=0.9)
    fig_h = 1.9 * sum(hr[k] for k in rows) + 1.4
    fig, axs = plt.subplots(len(rows), 1, figsize=(12, fig_h), sharex=True, gridspec_kw={"height_ratios": [hr[k] for k in rows]})
    ax = dict(zip(rows, axs))
    if "insitu" in ax:
        a = ax["insitu"]
        if ins_h:                                                    # hourly series of every depth + overpass markers
            for i, (lab, sr) in enumerate(ins_h):
                key = "ismn_soil_moisture_" + lab.replace("in situ ", ""); c, mk = depth_style(i); deep = _depth(key) > 0.4
                a.plot(sr.index, sr.values, "--" if deep else "-", color=c, lw=1.0 if deep else 1.5, label=_depth_label(key))
                if key in r: a.plot(t, _vals(r[key]), mk, color=c, ms=6, mec="k", mew=.5, ls="none")
        else:                                                        # only the values at the overpasses (ISMN file not readable)
            for i, k in enumerate(depths):
                c, mk = depth_style(i); deep = _depth(k) > 0.4
                a.plot(t, _vals(r[k]), "--" if deep else "-", marker=mk, color=c, lw=1.0 if deep else 1.5, ms=6, mec="k", mew=.5, label=_depth_label(k))
        a.set_ylabel("in-situ soil moisture\n[m³/m³]")
    a = ax["model"]
    for lab, sr in mod_h: a.plot(sr.index, sr.values, "-", color=MODCOL.get(m or "era5_land", "k"), lw=1.3, label=lab)
    if m: a.plot(t, _vals(r[f"{m}_soil_moisture_0_to_7cm"]), "o", color=MODCOL[m], ms=6, mec="k", mew=.5, ls="none" if mod_h else "-",
                 label=None if mod_h else f"{MODEL_LABEL[m]} 0-7 cm (model, at the overpasses)")
    if not mod_h and not m: a.text(0.5, 0.5, "no model soil moisture", ha="center", va="center", transform=a.transAxes, color="0.4")
    a.set_ylabel(f"{MODEL_LABEL.get(m, 'model')} 0-7 cm\n[m³/m³]")
    axs[0].plot([], [], "o", color="w", mec="k", label="markers: BIOMASS overpasses")
    a = ax["sigma"]
    for b, n in enumerate(names):
        if n in PLOT_POLS: a.plot(t, sig_db_t[:, b], "o-", color=POLCOL.get(n), lw=1.4, ms=5, label=n)
    a.set_ylabel("sigma0 [dB]\n(window mean)")
    _pair_coh_panel(ax["coh"], t, names, coh_pairs, pols=PLOT_POLS)
    a = ax["rain"]
    if H.get("rain") is not None and len(H["rain"]):
        rr = H["rain"]; a.bar(rr.index + pd.Timedelta(hours=12), rr.values, width=0.8, color="#56B4E9", alpha=.9, label=f"rain ({H['rain_label']})")
        a.set_ylabel("rain [mm/day]"); a.set_ylim(bottom=0)
    else:
        _rain_panel(a, t, r)
    a = ax["temp"]
    if H.get("temp"):
        sty = {0: ("-", 1.2), 1: ("--", 1.0)}
        for q, (lab, sr) in enumerate(H["temp"]):
            ls, lw = sty.get(q, ("-", 1.0)) if not lab.startswith("station") else ("-", 1.0)
            a.plot(sr.index, sr.values, ls, color="0.45" if lab.startswith("station") else MODCOL.get(m or "era5_land", "k"), lw=lw, label=lab)
        a.set_ylabel("temperature [°C]\n(hourly)")
        if H.get("snow") is not None:
            a2 = a.twinx(); a2.fill_between(H["snow"].index, 0, 100 * H["snow"].values, color="grey", alpha=.25); a2.set_ylabel("snow depth [cm]"); a2.set_ylim(bottom=0)
            a.plot([], [], color="grey", lw=6, alpha=.25, label="snow depth (model, right axis)")
        _zero_line(a)
    else:
        _temperature_panel(a, t, r)
    axs[0].set_title(f"{stack} @ {station}", loc="left", fontsize=10)
    if H.get("window"): axs[0].set_xlim(*[x.to_pydatetime() for x in H["window"]])
    for a in axs: _acq_lines(a, t); _grid(a); _legend_right(a)
    axs[-1].set_xlabel("date (UTC); dotted lines = BIOMASS acquisitions"); fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.76, top=1 - 0.45 / fig_h, bottom=1.05 / fig_h, hspace=0.12); fig.savefig(out_png, dpi=130); plt.close(fig)


PP_METHODS_PLOT = (("pp_freeman", None, None, ("odd", "double", "volume")), ("pp_yamaguchi4_y4r", None, None, ("odd", "double", "volume", "helix")),
                   ("pp_vanzyl", None, None, ("odd", "double", "volume")))
PP_LABEL = {"pp_freeman": "Freeman-Durden (PSP)", "pp_yamaguchi4_y4r": "Yamaguchi-4 y4r (PSP)", "pp_vanzyl": "van Zyl 1992 (PSP)"}


PP_COMPCOL = {"odd": None, "double": None, "volume": None, "helix": "#CC79A7"}
PP_COMPLAB = {"odd": "surface (odd)", "double": "double-bounce", "volume": "volume", "helix": "helix"}


def _pp_stack_panel(ax, t, key, P, comps):
    """Power shares of one PyPolSARpro decomposition as a stacked area (surface blue, double-bounce vermillion, volume green,
    helix purple)."""
    P = np.asarray(P, float); tot = np.nansum(P, axis=1, keepdims=True); sh = 100 * P / np.where(tot > 0, tot, np.nan)
    col = {"odd": FDCOL["surface"], "double": FDCOL["double-bounce"], "volume": FDCOL["volume"], "helix": PP_COMPCOL["helix"]}
    ax.stackplot(t, sh.T, colors=[col[c] for c in comps], labels=[PP_COMPLAB[c] for c in comps], alpha=.85,
                 zorder=0.4)                                 # below grid (0.5) and acquisition lines (0.6): both stay visible
    ax.set_ylim(0, 100); ax.set_ylabel(PP_LABEL.get(key, key).replace(" (PSP)", "") + "\n(PyPolSARpro) [%]")


def sar_panels(times, names, closures_consec, out_png, stack, station, pp=None, H=None, alpha=None, ha_src="", xlim=None, **_old):
    """Phase and polarimetric observables that are NOT in the main figure, on the same time axis as the main figure: the
    closure phase of consecutive triplets drawn like the 3-day coherence pairs there (a bar over the three dates of each
    triplet, dot at the middle date), the power partitions of Freeman-Durden, Yamaguchi-4 y4r and van Zyl 1992 and entropy
    H / anisotropy A / mean alpha, all from PyPolSARpro (without PyPolSARpro: own H/alpha only, no power partitions)."""
    t = pd.to_datetime(times, utc=True)
    ppm = [(k, c) for k, _, _, c in PP_METHODS_PLOT if pp is not None and k in pp]
    nP = 1 + len(ppm) + (H is not None)
    fig_h = 2.3 * nP + 1.2
    fig, ax = plt.subplots(nP, 1, figsize=(12, fig_h), sharex=True, squeeze=False); ax = ax[:, 0]
    vals = [0.0]
    for b, n in enumerate(names):
        if n not in PLOT_POLS: continue
        y = np.degrees(np.asarray(closures_consec[:, b], float)); vals += list(y)
        for i in range(len(y)):                                      # triplet (i, i+1, i+2): bar from t[i] to t[i+2]
            ax[0].plot([t[i], t[i + 2]], [y[i]] * 2, "-", color=POLCOL.get(n), lw=2.0, alpha=.55, label=f"{n}: triplet (i-1, i, i+1)" if i == 0 else None)
        ax[0].plot(t[1:-1], y, "o", color=POLCOL.get(n), ms=5)
    ax[0].axhline(0, **ZERO_KW); _rel_ylim(ax[0], vals, 2.0); ax[0].set_ylabel("closure phase [deg]\nconsecutive triplets")
    for q, (k, comps) in enumerate(ppm): _pp_stack_panel(ax[1 + q], t, k, pp[k], comps)
    if H is not None:
        k = nP - 1; src = ha_src or "own"
        ax[k].plot(t, H, "o-", color="#0072B2", label="entropy H"); ax[k].set_ylabel(f"H, A (0-1)\n({src})"); ax[k].set_ylim(0, 1)
        if pp is not None and "pp_h_a_alpha" in pp:
            ax[k].plot(t, np.asarray(pp["pp_h_a_alpha"], float)[:, 1], "^-", color="#009E73", label="anisotropy A")
        a2 = ax[k].twinx(); a2.plot(t, alpha, "s--", color="#D55E00"); a2.set_ylabel("alpha [deg]"); a2.set_ylim(0, 90)
        ax[k].plot([], [], "s--", color="#D55E00", label="mean alpha (right axis)")
    note = "" if ppm else "  (PyPolSARpro not installed: no power partitions)"
    ax[0].set_title(f"{stack} @ {station}: closure phase and polarimetry (window means, {len(t)} acquisitions){note}", loc="left", fontsize=10)
    if xlim is not None: ax[0].set_xlim(*[pd.Timestamp(x).to_pydatetime() for x in xlim])
    for a in ax: _acq_lines(a, t); _grid(a); _legend_right(a)
    ax[-1].set_xlabel("date (UTC); dotted lines = BIOMASS acquisitions"); fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.76, top=1 - 0.55 / fig_h, bottom=1.05 / fig_h, hspace=0.12); fig.savefig(out_png, dpi=130); plt.close(fig)


def coherence_figure(G_abs, times, names, ref_sm, out_png, stack, station, sm_label="reference SM", pair_fits=None):
    """Top row: |coherence| of all pairs vs temporal baseline (HH and VV, medians as lines) and the coherence matrices
    of the co-polar channels. Bottom row: one panel per polarisation, |coherence| vs |dSM| between the two dates, points coloured
    by temporal baseline; annotation = regression of |coh| on dt and |dSM| (metrics.csv). y axes scaled to the data."""
    t = pd.to_datetime(times, utc=True); T = len(t); days = [str(x)[5:10] for x in times]
    fig = plt.figure(figsize=(14, 9.5)); gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.38, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0:2])
    dts = np.array([[abs((t[j] - t[i]).days) for j in range(T)] for i in range(T)], float)
    iu = np.triu_indices(T, 1)
    sel = [(b, n) for b, n in enumerate(names) if n in PLOT_POLS]
    yall = np.concatenate([np.asarray(G_abs[:, :, b], float)[iu] for b, _ in sel]) if sel else np.array([])   # common, data-scaled y axis
    for b, n in sel:
        y = G_abs[:, :, b][iu]; x = dts[iu]
        ax0.plot(x, y, "o", color=POLCOL.get(n), alpha=.6, ms=5, label=n)
        ux = np.unique(x); ax0.plot(ux, [np.nanmedian(y[x == u]) for u in ux], "-", color=POLCOL.get(n), lw=1.2)
    ax0.set_xlabel("temporal baseline [days]"); ax0.set_ylabel("|coherence| (all pairs)"); _rel_ylim(ax0, yall, 0.1, 0.0, 1.0); _grid(ax0)
    ax0.set_title(f"all {T * (T - 1) // 2} pairs, lines = median per baseline", fontsize=9, loc="left"); ax0.legend(fontsize=8, frameon=False, ncol=4)
    for k, pol in enumerate([p for p in ("HH", "VV") if p in names][:2]):
        axm = fig.add_subplot(gs[0, 2 + k]); b = list(names).index(pol)
        Gm = np.array(G_abs[:, :, b], float); np.fill_diagonal(Gm, np.nan)                  # diagonal = self-coherence (1), not data
        im = axm.imshow(np.ma.masked_invalid(Gm), vmin=0, vmax=1, cmap="viridis"); axm.set_facecolor("#dddddd"); axm.set_title(f"coherence matrix {pol} (diagonal = self, grey)", fontsize=9, loc="left")
        axm.set_xticks(range(T)); axm.set_yticks(range(T)); axm.set_xticklabels(days, rotation=90, fontsize=7); axm.set_yticklabels(days, fontsize=7)
        for i in range(T):
            for j in range(T):
                if i != j: axm.text(j, i, f"{G_abs[i, j, b]:.2f}", ha="center", va="center", fontsize=6, color="w" if G_abs[i, j, b] < 0.6 else "k")
        if k == 1: fig.colorbar(im, ax=axm, fraction=0.046, pad=0.03, label="|coherence|")
    has_sm = ref_sm is not None and np.isfinite(np.asarray(ref_sm, float)).sum() >= 3
    for q, (b, n) in enumerate(sel[:2]):
        ax = fig.add_subplot(gs[1, 2 * q:2 * q + 2])
        if has_sm:
            sm = np.asarray(ref_sm, float); d = np.abs(sm[:, None] - sm[None, :])[iu]; y = G_abs[:, :, b][iu]
            sc = ax.scatter(d, y, c=dts[iu], cmap="cividis", vmin=0, vmax=np.nanmax(dts), s=28, edgecolor="none")
            ax.set_xlabel(f"|d {sm_label}| [m³/m³]")
            if pair_fits and n in pair_fits:
                f = pair_fits[n]; ax.text(0.03, 0.04, f"|coh| = a + {f['b_dt']:+.3f}·dt + {f['b_dsm']:+.1f}·|dSM|\nR² {f['r2']:.2f}, partial r(|dSM|) {f['r_dsm']:+.2f}", transform=ax.transAxes, fontsize=7, va="bottom")
            if q == len(sel[:2]) - 1: fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="temporal baseline [days]")
        else:
            ax.text(0.5, 0.5, "no reference SM sequence", ha="center", va="center", transform=ax.transAxes, color="0.4")
        _rel_ylim(ax, yall, 0.1, 0.0, 1.0); _grid(ax); ax.set_title(n, fontsize=9, loc="left", color=POLCOL.get(n))
        if q == 0: ax.set_ylabel("|coherence| (all pairs)")
    fig.suptitle(f"{stack} @ {station}: interferometric coherence (100 m looks, window mean)", fontsize=10, y=0.98)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def s2_figure(s2, times, out_png, stack, station, box_m=None):
    """Sentinel-2 L2A NDVI and NDMI over the analysis window (method of the team notebook 'Biomass L1C Station Ancillary'):
    filled = clear scenes (>= 80 % of the box clear in the scene classification), error bar = std over the clear pixels;
    hollow = partly cloudy scenes (mean of the clear pixels only); dotted lines = BIOMASS acquisitions; fixed y ranges as in
    the notebook (NDVI 0-1, NDMI -1-1) so that stations can be compared."""
    if s2 is None or not len(s2) or "ndvi_mean" not in s2: return False
    t = pd.to_datetime(times, utc=True); d = s2.copy(); d["t"] = pd.to_datetime(d["datetime"], utc=True); cl = d["clear"].astype(bool)
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True)
    for a, v, lim, col in zip(ax, ("ndvi", "ndmi"), ((0, 1), (-1, 1)), ("#009E73", "#56B4E9")):
        c, p = d[cl], d[~cl]
        a.errorbar(c["t"], c[f"{v}_mean"], yerr=c[f"{v}_std"] if f"{v}_std" in c else None, fmt="o-", ms=5, capsize=2, color=col, label="clear (>= 80 % of the box)")
        a.plot(p["t"], p[f"{v}_mean"], "o", mfc="none", color=col, label="partly cloudy (clear pixels only)")
        a.set_ylim(*lim); a.set_ylabel(v.upper()); _acq_lines(a, t); _grid(a); _legend_right(a)
    ax[0].set_title(f"{stack} @ {station}: Sentinel-2 L2A over the {box_m or 0:.0f} m window ({len(d)} dates, {int(cl.sum())} clear)", loc="left", fontsize=10)
    ax[-1].set_xlabel("date (UTC); dotted lines = BIOMASS acquisitions"); fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.76, top=0.92, bottom=0.17, hspace=0.12); fig.savefig(out_png, dpi=130); plt.close(fig)
    return True


def window_images(times, names, imgs_db, out_png, m_row, m_col, station_rc=None, title="", vrange=None):
    """Multilooked sigma0 [dB] of the analysis window (display cells, default 50 m): rows = polarisations, columns = dates + temporal std; ONE colour scale for
    all panels (2-98 % of all values) so that polarisations and dates are comparable; axes in metres (vertical = azimuth,
    horizontal = slant range), + = point."""
    T, P, R, C = imgs_db.shape
    fin = imgs_db[np.isfinite(imgs_db)]; vmin, vmax = vrange or (np.percentile(fin, [2, 98]) if fin.size else (-25, 0))
    std = np.nanstd(imgs_db, axis=0)                                           # [P, R, C]
    fig, axes = plt.subplots(P, T + 1, figsize=(1.9 * (T + 1) + 1.6, 2.4 * P + 0.8), squeeze=False, constrained_layout=True)
    ext = [0, C * m_col, R * m_row, 0]
    for b in range(P):
        for i in range(T + 1):
            ax = axes[b, i]
            if i < T: im = ax.imshow(imgs_db[i, b], cmap="gray", vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
            else: ims = ax.imshow(std[b], cmap="magma", vmin=0, vmax=max(1.0, float(np.nanpercentile(std, 98))), extent=ext, interpolation="nearest")
            if station_rc is not None: ax.plot(station_rc[1] * m_col, station_rc[0] * m_row, "+", color="#D55E00", ms=10, mew=1.6)
            ax.set_aspect("equal"); ax.tick_params(labelsize=6)
            if b == 0: ax.set_title(str(times[i])[:10] if i < T else "temporal std [dB]", fontsize=8)
            if i == 0: ax.set_ylabel(f"{names[b]}\nazimuth [m]", fontsize=9)
            else: ax.set_yticklabels([])
            if b == P - 1 and i < T: ax.set_xlabel("slant range [m]", fontsize=7)
            else: ax.set_xticklabels([])
    fig.colorbar(im, ax=axes[:, :T].ravel().tolist(), fraction=0.02, pad=0.01, label="sigma0 [dB] (shared scale)")
    fig.colorbar(ims, ax=axes[:, T].ravel().tolist(), fraction=0.08, pad=0.02, label="std over dates [dB]")
    fig.suptitle(title or "multilooked sigma0 in the analysis window", fontsize=10); fig.savefig(out_png, dpi=120); plt.close(fig)


def window_geometry(fields, out_png, m_row, m_col, station_rc=None, title=""):
    """LUT geometry in the window: height, incidence angle, azimuth and range terrain slope (bilinear from the LUT grid)."""
    keys = [k for k in ("height", "incidenceAngle", "azimuthTerrainSlope", "rangeTerrainSlope", "sigmaNought") if k in fields]
    if not keys: return
    labels = {"height": "height [m]", "incidenceAngle": "incidence [deg]", "azimuthTerrainSlope": "azimuth slope [deg]", "rangeTerrainSlope": "range slope [deg]", "sigmaNought": "beta0->sigma0 factor"}
    R, C = fields[keys[0]].shape; ext = [0, C * m_col, R * m_row, 0]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys) + 1, 3.6), squeeze=False)
    for ax, k in zip(axes[0], keys):
        im = ax.imshow(fields[k], cmap="cividis", extent=ext, interpolation="nearest"); ax.set_aspect("equal"); ax.set_title(labels[k], fontsize=9)
        if station_rc is not None: ax.plot(station_rc[1] * m_col, station_rc[0] * m_row, "+", color="#D55E00", ms=10, mew=1.6)
        ax.tick_params(labelsize=6); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0, 0].set_ylabel("azimuth [m]"); axes[0, 0].set_xlabel("slant range [m]")
    fig.suptitle(title or "LUT geometry in the analysis window", fontsize=10); fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def scatter_key_refs(times, names, sig_t, ref_row, out_png, stack, station, max_refs=3):
    """sigma0 [dB] (absolute, window mean) vs the soil-moisture references of the main figure: the two shallowest in-situ
    depths (if present) and the model reference 0-7 cm (ERA5-Land)."""
    r = ref_row; refs = []
    depths = sorted([k for k in r if k.startswith("ismn_soil_moisture") and isinstance(r[k], list)], key=_depth)
    refs += [("in situ " + k.replace("ismn_soil_moisture_", ""), _vals(r[k])) for k in depths[:2]]
    m = ref_model(r)
    if m: refs.append((f"{MODEL_LABEL[m]} 0-7 cm (model)", _vals(r[f"{m}_soil_moisture_0_to_7cm"])))
    refs = [x for x in refs if np.isfinite(x[1]).sum() >= 3][:max_refs]
    if not refs: return
    fig, axes = plt.subplots(1, len(refs), figsize=(3.6 * len(refs) + 1.4, 3.9), squeeze=False)
    for ax, (lab, sm) in zip(axes[0], refs):
        txt = []
        for b, n in enumerate(names):
            if n not in PLOT_POLS: continue
            y = sig_t[:, b]; ok = np.isfinite(sm) & np.isfinite(y)
            ax.plot(sm, y, "o", color=POLCOL.get(n), label=n)
            if ok.sum() >= 3 and np.ptp(sm[ok]) > 0:
                k, c = np.polyfit(sm[ok], y[ok], 1); xx = np.linspace(np.nanmin(sm), np.nanmax(sm), 10); ax.plot(xx, k * xx + c, "-", color=POLCOL.get(n), lw=0.8)
                txt.append(f"{n}: {k:+.0f} dB/(m³/m³), r {np.corrcoef(sm[ok], y[ok])[0, 1]:+.2f}")
        ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, fontsize=6.5, va="top")
        ax.set_xlabel(f"{lab} [m³/m³]  (range {np.nanmax(sm) - np.nanmin(sm):.3f})"); _grid(ax)
    axes[0, 0].set_ylabel("sigma0 [dB] (window mean)"); _legend_right(axes[0, -1])
    fig.suptitle(f"{stack} @ {station}: sigma0 vs reference soil moisture ({len(times)} overpasses; lines = least squares; slopes only meaningful for range >= 0.05)", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.17, top=0.85, wspace=0.3); fig.savefig(out_png, dpi=130); plt.close(fig)


def closure_check_plot(obs, mod, names, out_png, stack, station, ref_label):
    """Observed closure phases of all triplets vs the De Zan model driven by the reference moisture sequence, per polarisation."""
    md = np.degrees(mod); od = np.degrees(obs)
    lim = 1.1 * max(5.0, np.nanmax(np.abs(od)) if np.isfinite(od).any() else 5.0, np.nanmax(np.abs(md)) if np.isfinite(md).any() else 5.0)
    sel = [(b, n) for b, n in enumerate(names) if n in PLOT_POLS]
    fig, axes = plt.subplots(1, len(sel), figsize=(max(9.0, 3.6 * len(sel) + 0.8), 4.4), squeeze=False)
    for ax, (b, n) in zip(axes[0], sel):
        o = od[:, b]; ok = np.isfinite(o) & np.isfinite(md)
        r = np.corrcoef(md[ok], o[ok])[0, 1] if ok.sum() > 3 and np.ptp(md[ok]) > 0 and np.ptp(o[ok]) > 0 else np.nan
        ax.plot(md, o, "o", color=POLCOL.get(n), ms=4, alpha=.8); ax.plot([-lim, lim], [-lim, lim], color="k", ls="--", lw=1.1)
        ax.axhline(0, **ZERO_KW); ax.axvline(0, **ZERO_KW); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(f"{n}: r = {r:.2f}", fontsize=10); ax.set_xlabel("model closure phase [deg]"); _grid(ax)
    axes[0, 0].set_ylabel("observed closure phase [deg]")
    ms, os_ = np.nanstd(md), np.nanstd(od[:, [b for b, _ in sel]])
    if ms < 0.5 * os_: verdict = "model spread smaller than observed scatter -> the moisture change is too small to test the closure model here"
    elif ms > 2 * os_: verdict = "model predicts much larger closure phases than observed -> weaker phase response than the De Zan model expects"
    else: verdict = "model spread comparable to the observations -> test is meaningful"
    fig.suptitle(f"{stack} @ {station}: observed closure phases (all {len(md)} triplets) vs De Zan model driven by {ref_label}\n"
                 + "\n".join(textwrap.wrap(f"std model {ms:.1f}°, std observed {os_:.1f}°: {verdict}", 110)), fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.74, wspace=0.3); fig.savefig(out_png, dpi=130); plt.close(fig)


def quicklook_sheet(tab, tok, out_png, window=None, img_shape=None, download=None, station="", m_row=None, m_col=None, zoom_m=None, half_m=300):
    """BPS Pauli quicklooks (R = HH-VV, G = HV, B = HH+VV). Left: full scene of the primary with the analysis window (dotted)
    and the point (small dot). Right: every acquisition zoomed to the window half width + 15 % (zoom_m), window dotted, point
    as a small dot, nothing else on top. The quicklook pixel size is given in the title (the zoom cannot be finer than that)."""
    from PIL import Image
    out_png = pathlib.Path(out_png); files = []; zoom_m = zoom_m or 1.15 * half_m
    for _, r in tab.iterrows():
        url = r.get("ql_pauli_png")
        if not isinstance(url, str): files.append(None); continue
        try: files.append(download(url, out_png.parent / "quicklooks" / f"{str(r['datetime'])[:10]}_pauli.png", tok))
        except Exception: files.append(None)
    n = len(tab); prim = int(np.argmax(tab["primary"].astype(bool).values)) if tab["primary"].astype(bool).any() else 0
    ncol = 4 if n > 4 else n; nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.2 + 3.1 * ncol, 3.3 * nrow + 0.8)); gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1.3] + [1] * ncol, wspace=0.08, hspace=0.25)
    axf = fig.add_subplot(gs[:, 0]); axf.axis("off"); axf.set_title(f"{str(tab['datetime'].iloc[prim])[:10]} (primary), full scene", fontsize=8)
    geo = window is not None and img_shape is not None; qpx = ""
    if files[prim] is not None:
        im = Image.open(files[prim]); axf.imshow(im)
        if geo:
            sy, sx = im.size[1] / img_shape[0], im.size[0] / img_shape[1]
            if m_row and m_col: qpx = f"; quicklook pixel {m_row / sy:.0f} m (azimuth) x {m_col / sx:.0f} m (slant range)"
            axf.add_patch(Rectangle((window.col_off * sx, window.row_off * sy), max(window.width * sx, 3), max(window.height * sy, 3), fill=False, ec="#D55E00", lw=1.0, ls=":"))
            axf.plot((window.col_off + window.width / 2) * sx, (window.row_off + window.height / 2) * sy, "o", color="#D55E00", ms=2.5)
    for k, ((_, r), f) in enumerate(zip(tab.iterrows(), files)):
        ax = fig.add_subplot(gs[k // ncol, 1 + k % ncol]); ax.set_title(f"{str(r['datetime'])[:10]}{' (primary)' if k == prim else ''}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        if f is None: ax.text(0.5, 0.5, "no quicklook", ha="center", va="center", transform=ax.transAxes); continue
        im = Image.open(f); ax.imshow(im, interpolation="nearest")
        if geo and m_row and m_col:
            sy, sx = im.size[1] / img_shape[0], im.size[0] / img_shape[1]
            cx, cy = (window.col_off + window.width / 2) * sx, (window.row_off + window.height / 2) * sy
            ax.add_patch(Rectangle((window.col_off * sx, window.row_off * sy), window.width * sx, window.height * sy, fill=False, ec="#D55E00", lw=1.4, ls=":"))
            ax.plot(cx, cy, "o", color="#D55E00", ms=3)
            ax.set_xlim(cx - zoom_m / m_col * sx, cx + zoom_m / m_col * sx); ax.set_ylim(cy + zoom_m / m_row * sy, cy - zoom_m / m_row * sy)
            ax.set_aspect((m_row / sy) / (m_col / sx))
    fig.suptitle(f"{tab['stack'].iloc[0]} @ {station}: Pauli quicklooks (R = HH-VV, G = HV, B = HH+VV); zoom +-{zoom_m:.0f} m (window half width + 15 %),"
                 f" slant-range geometry (vertical = azimuth)\ndotted = {2 * half_m:.0f} m analysis window, dot = point{qpx}", fontsize=9)
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)


def global_tile_map(stack_index_csv, ismn_geojson, out_html, out_png=None, min_acq=6):
    """World map of all L1C stacks (folium HTML, optional PNG) with ISMN stations."""
    import folium
    st = pd.read_csv(stack_index_csv); st = st[st.n_acq >= min_acq]
    m = folium.Map(location=[10, 20], zoom_start=2, tiles="OpenStreetMap")
    for _, r in st.iterrows():
        folium.Rectangle([[r.s, r.w], [r.n, r.e]], color="#0072B2", weight=0.5, fill=True, fill_opacity=0.06, tooltip=f"{r.stack} {str(r.first_date)[:10]}..{str(r.last_date)[:10]} n={r.n_acq}").add_to(m)
    if ismn_geojson:
        for f in json.load(open(ismn_geojson))["features"]:
            lo, la = f["geometry"]["coordinates"][:2]; folium.CircleMarker([la, lo], radius=2.5, color="#D55E00", fill=True, tooltip=f["properties"].get("station", "")).add_to(m)
    m.save(out_html)
    if out_png:
        fig, ax = plt.subplots(figsize=(14, 7))
        for _, r in st.iterrows(): ax.add_patch(Rectangle((r.w, r.s), r.e - r.w, r.n - r.s, fc="#0072B2", ec="none", alpha=0.15))
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal"); _grid(ax); ax.set_title(f"BIOMASS L1C stacks with >= {min_acq} acquisitions: {len(st)}")
        fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


