"""Figures of a run. Layout rules: one quantity per panel, shared time axis, legends outside the axes (right),
fixed colours per polarisation and per reference source (Okabe-Ito palette, safe for red-green colour-vision deficiency),
acquisition dates as thin dotted lines, every figure carries the stack, the point and the processing notes in its title."""
import json, pathlib, textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

# Okabe & Ito (2008) palette: blue, orange, reddish purple, vermillion, bluish green, sky blue, black
POLCOL = {"HH": "#0072B2", "HV": "#E69F00", "VH": "#CC79A7", "VV": "#D55E00", "XX": "#E69F00"}
MODCOL = {"era5_land": "#009E73", "era5": "#56B4E9", "ecmwf_ifs": "#000000"}
FDCOL = {"surface": "#0072B2", "double-bounce": "#D55E00", "volume": "#009E73"}
STATION_CMAP = plt.get_cmap("viridis")          # station depths: sequential, colour-vision safe
RAIN_KEYS = ["ismn_rain_between_mm", "ecmwf_ifs_rain_between_mm", "era5_land_rain_between_mm", "era5_rain_between_mm"]


def _acq_lines(ax, t):
    for tt in t: ax.axvline(tt, color="0.5", ls=":", lw=0.6, zorder=0)


def _legend_right(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False, **kw)


def _vals(v):
    return np.array([np.nan if x is None else x for x in v], float)


def _depth(k):
    return float(k.replace("ismn_soil_moisture_", "").replace("m", "").split("-")[0])


def _rain(r):
    k = next((k for k in RAIN_KEYS if k in r and np.isfinite(_vals(r[k])).any()), None)
    return (k, _vals(r[k])) if k else (None, None)


def _rain_label(k):
    return "station gauge" if k.startswith("ismn") else k.split("_rain")[0]


def site_line(r, loc=None, win_m=None, valid_pct=None):
    """One line with the site facts that matter for interpretation (used as second title line of the point figures)."""
    bits = []
    if loc:
        if np.isfinite(loc.get("height", np.nan)): bits.append(f"height {loc['height']:.0f} m")
        if np.isfinite(loc.get("incidence", np.nan)): bits.append(f"incidence {loc['incidence']:.1f}°")
        if np.isfinite(loc.get("az_slope", np.nan)): bits.append(f"LUT slope az/rg {loc['az_slope']:.1f}/{loc['rg_slope']:.1f}°")
    if r.get("aux_slope_mean_deg") is not None and np.isfinite(r.get("aux_slope_mean_deg", np.nan)):
        bits.append(f"DEM slope {r['aux_slope_mean_deg']:.1f}° (>5°: {r.get('aux_slope_frac_gt5_pct', float('nan')):.0f} %)")
    lc = r.get("aux_landcover_pct") or {}
    if lc: bits.append("land cover " + ", ".join(f"{k} {v:.0f} %" for k, v in sorted(lc.items(), key=lambda x: -x[1])[:3]))
    if win_m: bits.append(f"window {win_m:.0f} m" + (f", valid {valid_pct:.0f} %" if valid_pct is not None else ""))
    return "\n".join(textwrap.wrap("; ".join(bits), 120))


def _station_sm(ax, t, r):
    depths = sorted([k for k in r if k.startswith("ismn_soil_moisture") and isinstance(r[k], list)], key=_depth)
    for i, k in enumerate(depths):
        lab = "station " + k.replace("ismn_soil_moisture_", ""); c = STATION_CMAP(0.1 + 0.75 * i / max(len(depths) - 1, 1))
        if _depth(k) <= 0.4: ax.plot(t, _vals(r[k]), "^-", color=c, lw=1.5, ms=5, label=lab)
        else: ax.plot(t, _vals(r[k]), "--", color=c, lw=0.9, alpha=.8, label=lab)
    if not depths: ax.text(0.5, 0.5, "no in-situ station data for this point", ha="center", va="center", transform=ax.transAxes, color="0.4")
    ax.set_ylabel("in-situ soil moisture\n[m³/m³]")


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
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("temperature [°C]")
    sk = next((k for k in ["era5_land_snow_depth", "ecmwf_ifs_snow_depth"] if k in r and np.nanmax(np.r_[_vals(r[k]), 0]) > 0.005), None)
    if sk:
        ax2 = ax.twinx(); ax2.bar(t, 100 * np.nan_to_num(_vals(r[sk])), width=1.2, alpha=.25, color="grey"); ax2.set_ylabel("snow depth [cm]"); ax2.set_ylim(bottom=0)
        ax.plot([], [], color="grey", lw=6, alpha=.25, label=f"snow depth ({sk.split('_snow')[0]}, right axis)")


def reference_panel(ref_rows, times, out_png, title=""):
    """Reference data only (no token): per point in-situ SM, model SM, rain, temperature/snow."""
    t = pd.to_datetime(times, utc=True); n = len(ref_rows)
    fig, axes = plt.subplots(4 * n, 1, figsize=(12, 2.6 * 4 * n), squeeze=False, sharex=True)
    for i, r in enumerate(ref_rows):
        a = axes[4 * i:4 * i + 4, 0]
        _station_sm(a[0], t, r); _model_sm(a[1], t, r); _rain_panel(a[2], t, r); _temperature_panel(a[3], t, r)
        a[0].set_title(f"{r['station']} ({r['network']}): {site_line(r)}", loc="left", fontsize=9)
        for ax in a: _acq_lines(ax, t); ax.grid(alpha=.25); _legend_right(ax)
    axes[-1, 0].set_xlabel("acquisition date (UTC)"); fig.autofmt_xdate()
    fig.suptitle(title or "Reference data at the BIOMASS overpasses", fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.09, right=0.73, top=0.95, bottom=0.05, hspace=0.18); fig.savefig(out_png, dpi=130); plt.close(fig)


def combined_panel(times, names, sig_db_t, coh_abs_primary, ref_row, out_png, stack, station, beta0_db_t=None, notes="", site=""):
    """Main figure of a point: absolute backscatter, sigma0 anomaly, |coherence| to primary, in-situ SM, model SM, rain, temperature/snow."""
    t = pd.to_datetime(times, utc=True); r = ref_row
    fig, ax = plt.subplots(7, 1, figsize=(12, 16.5), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1.0, 1.3, 1.1, 0.7, 1.0]})
    for b, n in enumerate(names):
        if beta0_db_t is not None: ax[0].plot(t, beta0_db_t[:, b], "o-", color=POLCOL.get(n), lw=1.3, label=f"{n} beta0 (raw)")
        ax[0].plot(t, sig_db_t[:, b], "s--", color=POLCOL.get(n), lw=1.0, ms=4, alpha=.8, label=f"{n} sigma0")
        ax[1].plot(t, sig_db_t[:, b] - np.nanmean(sig_db_t[:, b]), "o-", color=POLCOL.get(n), lw=1.4, label=f"{n} (mean {np.nanmean(sig_db_t[:, b]):.1f} dB)")
        ax[2].plot(t, coh_abs_primary[:, b], "s-", color=POLCOL.get(n), lw=1.2, label=n)
    ax[0].set_ylabel("backscatter [dB]\n(window mean)")
    ax[0].set_title(f"{stack} @ {station}" + (f"   [{notes}]" if notes else "") + (f"\n{site}" if site else ""), loc="left", fontsize=9)
    ax[1].axhline(0, color="k", lw=.5); ax[1].set_ylabel("sigma0 anomaly [dB]\n(minus stack mean)")
    ax[2].set_ylabel("|coherence|\nto primary"); ax[2].set_ylim(0, 1)
    _station_sm(ax[3], t, r); _model_sm(ax[4], t, r); _rain_panel(ax[5], t, r); _temperature_panel(ax[6], t, r)
    for a in ax: _acq_lines(a, t); a.grid(alpha=.25); _legend_right(a)
    ax[-1].set_xlabel("acquisition date (UTC)"); fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.74, top=0.955, bottom=0.05, hspace=0.12); fig.savefig(out_png, dpi=130); plt.close(fig)


def sar_panels(times, names, sig_db_t, coh_abs_primary, closures_consec, fd, out_png, stack, station, H=None, alpha=None, lut_coh=None, site=""):
    """SAR observables: sigma0 anomaly, |coherence| to primary (pipeline + BPS LUT value at the node), closure phase of
    consecutive triplets, Freeman-Durden power shares, H/alpha."""
    t = pd.to_datetime(times, utc=True); nP = 5 if H is not None else 4
    fig, ax = plt.subplots(nP, 1, figsize=(11, 2.6 * nP + 1), sharex=True)
    for b, n in enumerate(names):
        ax[0].plot(t, sig_db_t[:, b] - np.nanmean(sig_db_t[:, b]), "o-", color=POLCOL.get(n), label=n)
        ax[1].plot(t, coh_abs_primary[:, b], "s-", color=POLCOL.get(n), label=n)
        ax[2].plot(t[1:-1], np.degrees(closures_consec[:, b]), "o-", color=POLCOL.get(n), label=f"{n} (i-1, i, i+1)")
    if lut_coh is not None and np.isfinite(lut_coh).any(): ax[1].plot(t, lut_coh, "x--", color="0.3", lw=0.9, label="BPS LUT coherence at the node")
    tot = np.nansum(fd, axis=1, keepdims=True); share = 100 * fd / np.where(tot > 0, tot, np.nan)
    ax[3].stackplot(t, share.T, colors=[FDCOL[k] for k in FDCOL], labels=list(FDCOL), alpha=.85)
    ax[0].axhline(0, color="k", lw=.5); ax[0].set_ylabel("sigma0 anomaly [dB]"); ax[1].set_ylabel("|coherence| to primary"); ax[1].set_ylim(0, 1)
    ax[2].axhline(0, color="k", lw=.5); ax[2].set_ylabel("closure phase [deg]"); ax[3].set_ylabel("Freeman-Durden\nshare of total power [%]"); ax[3].set_ylim(0, 100)
    if H is not None:
        ax[4].plot(t, H, "o-", color="#0072B2", label="entropy H (0-1)"); ax[4].set_ylabel("entropy H"); ax[4].set_ylim(0, 1)
        a2 = ax[4].twinx(); a2.plot(t, alpha, "s--", color="#D55E00", label="mean alpha [deg]"); a2.set_ylabel("alpha [deg]"); a2.set_ylim(0, 90)
        ax[4].plot([], [], "s--", color="#D55E00", label="mean alpha [deg] (right axis)")
    ax[0].set_title(f"{stack} @ {station}: SAR observables (window means, {len(t)} acquisitions)" + (f"\n{site}" if site else ""), loc="left", fontsize=9)
    for a in ax: _acq_lines(a, t); a.grid(alpha=.25); _legend_right(a)
    fig.autofmt_xdate(); fig.subplots_adjust(left=0.09, right=0.78, top=0.94, bottom=0.07, hspace=0.12); fig.savefig(out_png, dpi=130); plt.close(fig)


def coherence_figure(G_abs, times, names, ref_sm, out_png, stack, station, sm_label="reference SM", pair_fits=None):
    """Top row: |coherence| of all pairs vs temporal baseline (all polarisations, medians as lines) and the coherence matrices
    of the co-polar channels. Bottom row: one panel per polarisation, |coherence| vs |dSM| between the two dates, points coloured
    by temporal baseline; annotation = regression of |coh| on dt and |dSM| (metrics.csv)."""
    t = pd.to_datetime(times, utc=True); T = len(t); P = len(names); days = [str(x)[5:10] for x in times]
    fig = plt.figure(figsize=(14, 9.5)); gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.38, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0:2])
    dts = np.array([[abs((t[j] - t[i]).days) for j in range(T)] for i in range(T)], float)
    iu = np.triu_indices(T, 1)
    for b, n in enumerate(names):
        y = G_abs[:, :, b][iu]; x = dts[iu]
        ax0.plot(x, y, "o", color=POLCOL.get(n), alpha=.6, ms=5, label=n)
        ux = np.unique(x); ax0.plot(ux, [np.nanmedian(y[x == u]) for u in ux], "-", color=POLCOL.get(n), lw=1.2)
    ax0.set_xlabel("temporal baseline [days]"); ax0.set_ylabel("|coherence| (all pairs)"); ax0.set_ylim(0, 1); ax0.grid(alpha=.3)
    ax0.set_title(f"all {T * (T - 1) // 2} pairs, lines = median per baseline", fontsize=9, loc="left"); ax0.legend(fontsize=8, frameon=False, ncol=4)
    for k, pol in enumerate([p for p in ("HH", "VV") if p in names][:2]):
        axm = fig.add_subplot(gs[0, 2 + k]); b = list(names).index(pol)
        im = axm.imshow(G_abs[:, :, b], vmin=0, vmax=1, cmap="viridis"); axm.set_title(f"coherence matrix {pol}", fontsize=9, loc="left")
        axm.set_xticks(range(T)); axm.set_yticks(range(T)); axm.set_xticklabels(days, rotation=90, fontsize=7); axm.set_yticklabels(days, fontsize=7)
        for i in range(T):
            for j in range(T):
                if i != j: axm.text(j, i, f"{G_abs[i, j, b]:.2f}", ha="center", va="center", fontsize=6, color="w" if G_abs[i, j, b] < 0.6 else "k")
        if k == 1: fig.colorbar(im, ax=axm, fraction=0.046, pad=0.03, label="|coherence|")
    has_sm = ref_sm is not None and np.isfinite(np.asarray(ref_sm, float)).sum() >= 3
    for b, n in enumerate(names[:4]):
        ax = fig.add_subplot(gs[1, b])
        if has_sm:
            sm = np.asarray(ref_sm, float); d = np.abs(sm[:, None] - sm[None, :])[iu]; y = G_abs[:, :, b][iu]
            sc = ax.scatter(d, y, c=dts[iu], cmap="cividis", vmin=0, vmax=np.nanmax(dts), s=28, edgecolor="none")
            ax.set_xlabel(f"|d {sm_label}| [m³/m³]")
            if pair_fits and n in pair_fits:
                f = pair_fits[n]; ax.text(0.03, 0.04, f"|coh| = a + {f['b_dt']:+.3f}·dt + {f['b_dsm']:+.1f}·|dSM|\nR² {f['r2']:.2f}, partial r(|dSM|) {f['r_dsm']:+.2f}", transform=ax.transAxes, fontsize=7, va="bottom")
            if b == 3 or b == P - 1: fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="temporal baseline [days]")
        else:
            ax.text(0.5, 0.5, "no reference SM sequence", ha="center", va="center", transform=ax.transAxes, color="0.4")
        ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.set_title(n, fontsize=9, loc="left", color=POLCOL.get(n))
        if b == 0: ax.set_ylabel("|coherence| (all pairs)")
    fig.suptitle(f"{stack} @ {station}: interferometric coherence (100 m looks, window mean)", fontsize=10, y=0.98)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def window_images(times, names, imgs_db, out_png, m_row, m_col, station_rc=None, title="", vrange=None):
    """Multilooked sigma0 [dB] of the analysis window: rows = polarisations, columns = dates + temporal std; ONE colour scale for
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


def scatter_key_refs(times, names, sig_t, ref_row, out_png, stack, station, max_refs=4):
    """sigma0 anomaly vs the key soil-moisture references (two shallowest station depths, ERA5-Land and IFS 0-7 cm)."""
    r = ref_row; refs = []
    depths = sorted([k for k in r if k.startswith("ismn_soil_moisture") and isinstance(r[k], list)], key=_depth)
    refs += [("station " + k.replace("ismn_soil_moisture_", ""), _vals(r[k])) for k in depths[:2]]
    refs += [(f"{m} 0-7 cm", _vals(r[f"{m}_soil_moisture_0_to_7cm"])) for m in ("era5_land", "ecmwf_ifs", "era5") if f"{m}_soil_moisture_0_to_7cm" in r]
    refs = [x for x in refs if np.isfinite(x[1]).sum() >= 3][:max_refs]
    if not refs: return
    fig, axes = plt.subplots(1, len(refs), figsize=(3.4 * len(refs) + 1.4, 3.8), squeeze=False)
    for ax, (lab, sm) in zip(axes[0], refs):
        txt = []
        for b, n in enumerate(names):
            y = sig_t[:, b] - np.nanmean(sig_t[:, b]); ok = np.isfinite(sm) & np.isfinite(y)
            ax.plot(sm, y, "o", color=POLCOL.get(n), label=n)
            if ok.sum() >= 3 and np.ptp(sm[ok]) > 0:
                k, c = np.polyfit(sm[ok], y[ok], 1); xx = np.linspace(np.nanmin(sm), np.nanmax(sm), 10); ax.plot(xx, k * xx + c, "-", color=POLCOL.get(n), lw=0.8)
                txt.append(f"{n}: {k:+.0f} dB/(m³/m³), r {np.corrcoef(sm[ok], y[ok])[0, 1]:+.2f}")
        ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, fontsize=6.5, va="top")
        ax.set_xlabel(f"{lab} [m³/m³]  (range {np.nanmax(sm) - np.nanmin(sm):.3f})"); ax.grid(alpha=.3); ax.axhline(0, color="0.6", lw=.5)
    axes[0, 0].set_ylabel("sigma0 anomaly [dB]"); _legend_right(axes[0, -1])
    fig.suptitle(f"{stack} @ {station}: sigma0 anomaly vs reference soil moisture ({len(times)} overpasses; lines = least squares; slopes only meaningful for range >= 0.05)", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.17, top=0.85, wspace=0.3); fig.savefig(out_png, dpi=130); plt.close(fig)


def closure_check_plot(obs, mod, names, out_png, stack, station, ref_label):
    """Observed closure phases of all triplets vs the De Zan model driven by the reference moisture sequence, per polarisation."""
    md = np.degrees(mod); od = np.degrees(obs)
    lim = 1.1 * max(5.0, np.nanmax(np.abs(od)) if np.isfinite(od).any() else 5.0, np.nanmax(np.abs(md)) if np.isfinite(md).any() else 5.0)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.9), squeeze=False)
    for ax, (b, n) in zip(axes[0], enumerate(names)):
        o = od[:, b]; ok = np.isfinite(o) & np.isfinite(md)
        r = np.corrcoef(md[ok], o[ok])[0, 1] if ok.sum() > 3 and np.ptp(md[ok]) > 0 and np.ptp(o[ok]) > 0 else np.nan
        ax.plot(md, o, "o", color=POLCOL.get(n), ms=4, alpha=.8); ax.plot([-lim, lim], [-lim, lim], "k--", lw=.7)
        ax.axhline(0, color="0.6", lw=.5); ax.axvline(0, color="0.6", lw=.5); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(f"{n}: r = {r:.2f}", fontsize=10); ax.set_xlabel("model closure phase [deg]"); ax.grid(alpha=.3)
    axes[0, 0].set_ylabel("observed closure phase [deg]")
    ms, os_ = np.nanstd(md), np.nanstd(od)
    verdict = "model spread smaller than observed scatter -> the moisture change is too small to test the closure model here" if ms < 0.5 * os_ else "model spread comparable to the observations -> test is meaningful"
    fig.suptitle(f"{stack} @ {station}: observed closure phases (all {len(md)} triplets) vs De Zan model driven by {ref_label}\nstd model {ms:.1f}°, std observed {os_:.1f}°: {verdict}", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.78, wspace=0.35); fig.savefig(out_png, dpi=130); plt.close(fig)


def quicklook_sheet(tab, tok, out_png, window=None, img_shape=None, download=None, station="", m_row=None, m_col=None, zoom_m=6000, half_m=300):
    """BPS Pauli quicklooks (R = HH-VV, G = HV, B = HH+VV). Left: full scene of the primary with the analysis window marked.
    Right: zoom (+- zoom_m around the point) of every acquisition with the analysis window (rectangle), the 500 m and 1 km
    rings (ellipses in slant-range geometry) and the point."""
    from PIL import Image
    out_png = pathlib.Path(out_png); files = []
    for _, r in tab.iterrows():
        url = r.get("ql_pauli_png")
        if not isinstance(url, str): files.append(None); continue
        try: files.append(download(url, out_png.parent / "quicklooks" / f"{str(r['datetime'])[:10]}_pauli.png", tok))
        except Exception: files.append(None)
    n = len(tab); prim = int(np.argmax(tab["primary"].astype(bool).values)) if tab["primary"].astype(bool).any() else 0
    ncol = 4 if n > 4 else n; nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.2 + 3.1 * ncol, 3.3 * nrow + 0.8)); gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1.3] + [1] * ncol, wspace=0.08, hspace=0.25)
    axf = fig.add_subplot(gs[:, 0]); axf.axis("off"); axf.set_title(f"{str(tab['datetime'].iloc[prim])[:10]} (primary), full scene", fontsize=8)
    geo = window is not None and img_shape is not None
    if files[prim] is not None:
        im = Image.open(files[prim]); axf.imshow(im)
        if geo:
            sy, sx = im.size[1] / img_shape[0], im.size[0] / img_shape[1]
            axf.add_patch(Rectangle((window.col_off * sx, window.row_off * sy), max(window.width * sx, 4), max(window.height * sy, 4), fill=False, ec="#D55E00", lw=1.5))
            axf.add_patch(Ellipse(((window.col_off + window.width / 2) * sx, (window.row_off + window.height / 2) * sy), 2 * zoom_m / m_col * sx, 2 * zoom_m / m_row * sy, fill=False, ec="w", lw=0.8, ls="--"))
    for k, ((_, r), f) in enumerate(zip(tab.iterrows(), files)):
        ax = fig.add_subplot(gs[k // ncol, 1 + k % ncol]); ax.set_title(f"{str(r['datetime'])[:10]}{' (primary)' if k == prim else ''}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        if f is None: ax.text(0.5, 0.5, "no quicklook", ha="center", va="center", transform=ax.transAxes); continue
        im = Image.open(f); ax.imshow(im)
        if geo and m_row and m_col:
            sy, sx = im.size[1] / img_shape[0], im.size[0] / img_shape[1]
            cx, cy = (window.col_off + window.width / 2) * sx, (window.row_off + window.height / 2) * sy
            ax.add_patch(Rectangle((window.col_off * sx, window.row_off * sy), window.width * sx, window.height * sy, fill=False, ec="#D55E00", lw=1.4))
            for rad, col in ((500, "w"), (1000, "#F0E442")): ax.add_patch(Ellipse((cx, cy), 2 * rad / m_col * sx, 2 * rad / m_row * sy, fill=False, ec=col, lw=0.9, ls="--"))
            ax.plot(cx, cy, "+", color="#D55E00", ms=9, mew=1.5)
            ax.set_xlim(cx - zoom_m / m_col * sx, cx + zoom_m / m_col * sx); ax.set_ylim(cy + zoom_m / m_row * sy, cy - zoom_m / m_row * sy)
            ax.set_aspect((m_row / sy) / (m_col / sx))
    fig.suptitle(f"{tab['stack'].iloc[0]} @ {station}: Pauli quicklooks (R = HH-VV, G = HV, B = HH+VV); zoom +-{zoom_m / 1000:.0f} km, slant-range geometry (vertical = azimuth)\n"
                 f"orange = {2 * half_m:.0f} m analysis window, white = 500 m, yellow = 1 km radius, + = point", fontsize=9)
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
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title(f"BIOMASS L1C stacks with >= {min_acq} acquisitions: {len(st)}")
        fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


WC_COLORS = {"tree": "#006400", "shrub": "#ffbb22", "grassland": "#ffff4c", "cropland": "#f096ff", "built": "#fa0000", "bare": "#b4b4b4",
             "snow_ice": "#f0f0f0", "water": "#0064c8", "wetland": "#0096a0", "mangrove": "#00cf75", "moss_lichen": "#fae6a0"}   # ESA WorldCover legend


def candidate_map(cands, chosen, out_png, stack, stations=None, criteria=None):
    """Footprint blocks coloured by their dominant WorldCover class (official legend), suitable blocks hatched, chosen windows as
    numbered stars, ISMN stations as triangles; right: slope map of the same blocks."""
    names = [k for k in WC_COLORS if f"{k}_pct" in cands]
    dom = cands[[f"{k}_pct" for k in names]].values.argmax(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    dlat = np.median(np.diff(np.sort(cands.lat.unique()))) if cands.lat.nunique() > 1 else 0.005; dlon = np.median(np.diff(np.sort(cands.lon.unique()))) if cands.lon.nunique() > 1 else 0.005
    for (la, lo, d, su) in zip(cands.lat, cands.lon, dom, cands.suitable):
        ax[0].add_patch(Rectangle((lo - dlon / 2, la - dlat / 2), dlon, dlat, fc=WC_COLORS[names[d]], ec="k" if su else "none", lw=0.4 if su else 0, hatch="///" if su else None))
    sc = ax[1].scatter(cands.lon, cands.lat, c=cands.slope_mean_deg, s=6, marker="s", cmap="cividis", vmin=0, vmax=20, edgecolor="none")
    fig.colorbar(sc, ax=ax[1], fraction=0.03, pad=0.02, label="mean slope [deg] (DEM 30 m, 90 m samples)")
    for a in ax:
        if stations is not None and len(stations):
            a.plot(stations.lon, stations.lat, "^", color="k", ms=9, mfc="w", mew=1.5, label="ISMN station")
        for i, (_, r) in enumerate(chosen.iterrows()):
            a.plot(r.lon, r.lat, "*", color="#D55E00", ms=16, mec="k", mew=0.8); a.annotate(str(i + 1), (r.lon, r.lat), xytext=(6, 6), textcoords="offset points", fontsize=9, fontweight="bold")
        a.set_xlim(cands.lon.min() - dlon, cands.lon.max() + dlon); a.set_ylim(cands.lat.min() - dlat, cands.lat.max() + dlat); a.set_aspect(1 / np.cos(np.radians(cands.lat.mean()))); a.grid(alpha=.3)
        a.set_xlabel("longitude"); 
    ax[0].set_ylabel("latitude")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=WC_COLORS[k]) for k in names] + [plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch="///"), plt.Line2D([], [], marker="*", color="#D55E00", ls="", ms=12, mec="k")]
    ax[0].legend(handles, names + ["suitable block", "chosen window"], fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=5, frameon=False)
    n_s = int(cands.suitable.sum())
    ax[0].set_title(f"dominant land cover per 600 m block; {n_s} of {len(cands)} blocks suitable ({100 * n_s / max(len(cands), 1):.0f} %)", fontsize=9, loc="left")
    ax[1].set_title("mean slope per block", fontsize=9, loc="left")
    fig.suptitle(f"{stack}: candidate analysis windows\ncriteria: {criteria}", fontsize=9)
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
