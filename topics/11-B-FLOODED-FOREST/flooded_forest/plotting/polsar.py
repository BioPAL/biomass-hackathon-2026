"""PolSAR figures (port of polsar_stack.ipynb) from processing.polsar outputs, kept scenes only.

Per scene, into <roi>/polsar/<stack>/figures/, on the coherence grid in radar geometry:
    <date>_pauli.png      Pauli RGB of the averaged T3 (R = T22 double bounce, G = T33 volume, B = T11 surface)
    <date>_hsi.png        polarimetric HSI: hue = 4 * alpha_averaged, saturation = 1 - entropy,
                          intensity = total power (span) scaled to 0-1
    <date>_alphas.png     alpha_1, alpha_2, alpha_3
    <date>_h_a_alpha.png  entropy, anisotropy, alpha_averaged
    <date>_h_alpha.png    entropy / alpha_averaged plane (polsarpro plot_h_alpha_plane)
Per stack:
    stack_pauli_hsi.png        every kept scene in chronological order (columns): Pauli RGB (top), HSI (bottom)
    mechanism_timeseries.png   cells per dominant scattering mechanism (of lambda_1) vs temporal baseline
    ../mechanism_counts.json   the counts of that plot

Pauli and HSI use stack-wide scalings (same for every scene of a stack) so the scenes can be compared.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch
from polsarpro.util import plot_h_alpha_plane

from ..core.scenes import removed_ids
from ..processing.polsar import (ALPHA_SURFACE_MAX, ALPHA_VOLUME_MAX, MECHANISMS, MIN_VALID_FRACTION,
                                 dominant_mechanism, polsar_paths)
from .common import CMAP, cell_geometry, hsi_to_rgb, save, show_map

PAULI_Q = 0.98                  # stack-wide clip quantile of sqrt(T22), sqrt(T33), sqrt(T11) (as polsarpro pauli_rgb)
SPAN_PERCENTILES = (2, 98)      # stack-wide stretch of the span in dB -> HSI intensity 0-1
MECHANISM_COLORS = {"surface": "#2a64d6", "volume": "#2fa84f", "double bounce": "#e0302a"}
PAULI_KEY = [Patch(color=c, label=label) for c, label in (
    ("#e0302a", "R  T22 = <|HH−VV|²>/2  double bounce"),
    ("#2fa84f", "G  T33 = <|HV+VH|²>/2  volume"),
    ("#2a64d6", "B  T11 = <|HH+VV|²>/2  surface"),
)]
HSI_TEXT = "hue = 4 · alpha_averaged, saturation = 1 − entropy H,\nintensity = span T11+T22+T33 [dB] scaled 0−1"
PER_SCENE = ("t3", "pauli_rgb", "lambdas", "alphas", "entropy", "anisotropy", "alpha_averaged", "span")


def _panel_height(st, width_in):
    return float(np.clip(width_in * st["rows"] * st["geo"]["aspect"] / st["cols"], 2.5, 11.0))


def _load_stack(path, stack_dir, min_valid=MIN_VALID_FRACTION):
    """npz + json of one stack restricted to the kept scenes, with the stack-wide Pauli and HSI images."""
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path) as data:
        st = {k: data[k] for k in data.files}
    removed = removed_ids(stack_dir)
    keep = [k for k, pid in enumerate(meta["products"]) if pid not in removed]
    for key in PER_SCENE:
        st[key] = st[key][keep]
    primary = meta.get("primary_index")
    meta["primary_index"] = keep.index(primary) if primary in keep else None
    for key in ("products", "dates", "phase_calibration"):
        meta[key] = [meta[key][k] for k in keep]
    st["meta"] = meta
    st["good"] = st["valid_fraction"] >= min_valid
    st["geo"] = cell_geometry(meta["window"], st["az_index"], st["rg_index"])
    st["rows"], st["cols"] = st["good"].shape
    st["days"] = np.array([(np.datetime64(d) - np.datetime64(meta["dates"][0])) / np.timedelta64(1, "D")
                           for d in meta["dates"]]).astype(int) if keep else np.array([], int)
    if keep:
        st["pauli"] = _stack_pauli(st)
        st["hsi"] = _stack_hsi(st)
    return st


def _rgba(st, rgb):
    ok = st["good"] & np.isfinite(rgb).all(axis=-1)
    return np.concatenate([np.nan_to_num(rgb), ok[..., None].astype(float)], axis=-1)


def _stack_pauli(st):
    """(Nb, Naz, Nrg, 4) Pauli RGBA: sqrt(T22), sqrt(T33), sqrt(T11) clipped at their stack-wide 98 % quantile."""
    diag = np.sqrt(np.clip(st["t3"][:, [1, 2, 0]].real, 0, None))          # (Nb, 3, Naz, Nrg) R G B
    clip = np.array([np.nanquantile(diag[:, b][:, st["good"]], PAULI_Q) for b in range(3)])
    rgb = np.clip(diag / clip[None, :, None, None], 0, 1)
    return np.stack([_rgba(st, np.moveaxis(rgb[k], 0, -1)) for k in range(len(rgb))])


def _stack_hsi(st):
    """(Nb, Naz, Nrg, 4) HSI RGBA with the span stretched between its stack-wide 2nd and 98th percentile."""
    span_db = 10 * np.log10(np.clip(st["span"], 1e-12, None))
    lo, hi = np.nanpercentile(span_db[:, st["good"]], SPAN_PERCENTILES)
    intensity = np.clip((span_db - lo) / (hi - lo + 1e-12), 0, 1)
    st["span_db_limits"] = (float(lo), float(hi))
    return np.stack([
        _rgba(st, hsi_to_rgb(4 * st["alpha_averaged"][k], 1 - st["entropy"][k], intensity[k]))
        for k in range(len(span_db))
    ])


def _title(st, k):
    m = st["meta"]
    primary = ", primary" if k == m["primary_index"] else ""
    return f"{m['stack']} · {m['dates'][k]}{primary} · θ {m['theta_deg']}°"


def _draw_hsi_key(ax, intensity=0.5):
    """Hue / saturation key: alpha_averaged (x) vs entropy (y) at a fixed intensity."""
    alpha, entropy = np.meshgrid(np.linspace(0, 90, 181), np.linspace(0, 1, 101))
    ax.imshow(hsi_to_rgb(4 * alpha, 1 - entropy, np.full_like(alpha, intensity)), origin="lower",
              extent=[0, 90, 0, 1], aspect="auto")
    ax.set_xticks([0, 30, 60, 90])
    ax.set_xlabel("alpha_averaged [deg]", fontsize=7)
    ax.set_ylabel("entropy H", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(f"HSI key (intensity {intensity:g})", fontsize=7)


# -------------------- per scene --------------------


def _plot_pauli(st, k, out_dir, show):
    fig, ax = plt.subplots(figsize=(7, _panel_height(st, 5.0)))
    show_map(ax, st["pauli"][k], st["geo"])
    ax.set_title(f"{_title(st, k)}\nPauli RGB of the averaged T3 (amplitudes sqrt(Tii), "
                 f"stack-wide {PAULI_Q:.0%} clip)", fontsize=9, loc="left")
    ax.legend(handles=PAULI_KEY, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    return save(fig, out_dir, f"{st['meta']['dates'][k]}_pauli", show)


def _plot_hsi(st, k, out_dir, show):
    fig, (ax, key) = plt.subplots(1, 2, figsize=(9, _panel_height(st, 5.0)), width_ratios=[4, 1.4],
                                  layout="constrained")
    show_map(ax, st["hsi"][k], st["geo"])
    lo, hi = st["span_db_limits"]
    ax.set_title(f"{_title(st, k)}\nPolarimetric HSI: {HSI_TEXT.replace(chr(10), ' ')} "
                 f"({lo:.1f} to {hi:.1f} dB, stack-wide)", fontsize=8, loc="left")
    _draw_hsi_key(key)
    return save(fig, out_dir, f"{st['meta']['dates'][k]}_hsi", show)


def _plot_triplet(st, layers, tag, title, out_dir, show, shared_cbar=None):
    """Three maps side by side; layers = [(image, panel title, vmin, vmax, colorbar label)]."""
    fig, axes = plt.subplots(1, 3, figsize=(12, _panel_height(st, 3.2)), sharex=True, sharey=True,
                             layout="constrained")
    im = None
    for ax, (img, name, vmin, vmax, unit) in zip(axes, layers):
        im = show_map(ax, img, st["geo"], st["good"], labels=(ax is axes[0]), cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=8)
        if shared_cbar is None:
            fig.colorbar(im, ax=ax, shrink=0.7, label=unit)
    if shared_cbar is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label=shared_cbar)
    fig.suptitle(title, fontsize=9)
    return save(fig, out_dir, tag, show)


def _plot_h_alpha_plane(st, k, out_dir, show, bins=200, min_pts=5):
    ds = xr.Dataset(
        {"entropy": (("y", "x"), np.where(st["good"], st["entropy"][k], np.nan)),
         "alpha": (("y", "x"), np.where(st["good"], st["alpha_averaged"][k], np.nan))},
        attrs={"poltype": "h_a_alpha"},
    )
    ax, fig = plot_h_alpha_plane(ds, bins=bins, min_pts=min_pts)
    ax.set_xlabel("entropy H = −Σ p_i log3 p_i")
    ax.set_ylabel("alpha_averaged = Σ p_i alpha_i [deg]")
    ax.set_title(f"{_title(st, k)}\nentropy / alpha_averaged plane ({int(st['good'].sum())} cells)", fontsize=9)
    return save(fig, out_dir, f"{st['meta']['dates'][k]}_h_alpha", show)


def _scene_figures(st, k, out_dir, show):
    date = st["meta"]["dates"][k]
    yield _plot_pauli(st, k, out_dir, show)
    yield _plot_hsi(st, k, out_dir, show)
    yield _plot_triplet(
        st, [(st["alphas"][k, i], f"alpha_{i + 1}: eigenvector of λ{i + 1}" + (" (dominant)" if i == 0 else ""),
              0, 90, "deg") for i in range(3)],
        f"{date}_alphas",
        f"{_title(st, k)}: alpha_i = scattering angle of the eigenvector of λ_i of T3 (λ1 ≥ λ2 ≥ λ3)",
        out_dir, show, shared_cbar="alpha_i [deg]")
    yield _plot_triplet(
        st, [(st["entropy"][k], "entropy H = −Σ p_i log3 p_i", 0, 1, "entropy H"),
             (st["anisotropy"][k], "anisotropy A = (p2 − p3) / (p2 + p3)", 0, 1, "anisotropy A"),
             (st["alpha_averaged"][k], "alpha_averaged = Σ p_i alpha_i", 0, 90, "alpha_averaged [deg]")],
        f"{date}_h_a_alpha",
        f"{_title(st, k)}: Cloude–Pottier H / A / alpha_averaged, p_i = λ_i / (λ1 + λ2 + λ3).\n"
        f"alpha_averaged is the average of alpha_1, alpha_2 and alpha_3 weighted by the eigenvalue "
        f"probabilities p_i", out_dir, show)
    yield _plot_h_alpha_plane(st, k, out_dir, show)


# -------------------- per stack --------------------


def _plot_stack_comparison(st, out_dir, show):
    """Kept scenes in chronological order (columns): Pauli RGB (top row) and HSI (bottom row)."""
    m = st["meta"]
    n = len(m["dates"])
    panel_h = _panel_height(st, 2.8)
    fig, axes = plt.subplots(2, n + 1, squeeze=False, layout="constrained",
                             figsize=(2.8 * n + 3.2, 2 * panel_h + 1.6), width_ratios=[1] * n + [0.9])
    for k in range(n):
        for row, (images, name) in enumerate(((st["pauli"], "Pauli RGB"), (st["hsi"], "HSI"))):
            ax = axes[row, k]
            show_map(ax, images[k], st["geo"], labels=(k == 0))
            primary = " (primary)" if k == m["primary_index"] else ""
            ax.set_title(f"{name}  {m['dates'][k]}{primary}\n+{st['days'][k]} d", fontsize=8)
    axes[0, -1].axis("off")
    axes[0, -1].legend(handles=PAULI_KEY, loc="center left", frameon=False, fontsize=7)
    _draw_hsi_key(axes[1, -1])
    lo, hi = st["span_db_limits"]
    fig.suptitle(f"{m['stack']}: {n} scenes in chronological order — Pauli RGB of the averaged T3 "
                 f"(stack-wide {PAULI_Q:.0%} clip) and polarimetric HSI ({HSI_TEXT.replace(chr(10), ' ')}, "
                 f"{lo:.1f} to {hi:.1f} dB)", fontsize=10)
    return save(fig, out_dir, "stack_pauli_hsi", show)


def mechanism_counts(st):
    """Cells per dominant mechanism (of lambda_1, from alpha_1) per scene, on the cells valid in every scene."""
    mech = dominant_mechanism(st["alphas"][:, 0])                      # (Nb, Naz, Nrg)
    common = st["good"] & (mech >= 0).all(axis=0)
    counts = {name: [int((mech[k][common] == i).sum()) for k in range(len(mech))]
              for i, name in enumerate(MECHANISMS)}
    return counts, int(common.sum())


def _plot_mechanism_timeseries(st, out_dir, show):
    m = st["meta"]
    counts, n_cells = mechanism_counts(st)
    (out_dir.parent / "mechanism_counts.json").write_text(json.dumps({
        "stack": m["stack"], "dates": m["dates"], "days_since_first": st["days"].tolist(),
        "cells_valid_in_every_scene": n_cells,
        "rule": f"dominant mechanism = the one of lambda_1, from alpha_1: surface < {ALPHA_SURFACE_MAX:g} <= "
                f"volume <= {ALPHA_VOLUME_MAX:g} < double bounce [deg]",
        "counts": counts,
    }, indent=1))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for name in MECHANISMS:
        ax.plot(st["days"], counts[name], "-o", color=MECHANISM_COLORS[name], lw=2, ms=6, label=name)
    ax.set_xticks(st["days"], [f"{d}\n{date[5:]}" for d, date in zip(st["days"], m["dates"])], fontsize=8)
    ax.set_xlabel("temporal baseline [days since the first kept scene] / date (MM-DD)")
    ax.set_ylabel(f"number of pixels\n(cells of the {m['window']['window_m']:g} m averaging grid)")
    ax.grid(color="#d9d8d2", lw=0.5)
    ax.spines[["top"]].set_visible(False)
    if n_cells:
        right = ax.secondary_yaxis("right", functions=(lambda c: 100 * c / n_cells, lambda p: p * n_cells / 100))
        right.set_ylabel("share of the cells [%]")
    ax.legend(title=f"dominant mechanism of λ1 (alpha_1 < {ALPHA_SURFACE_MAX:g}° surface, "
                    f"{ALPHA_SURFACE_MAX:g}–{ALPHA_VOLUME_MAX:g}° volume, > {ALPHA_VOLUME_MAX:g}° double bounce)",
              loc="upper left", bbox_to_anchor=(0, -0.28), ncols=3, frameon=False, fontsize=8, title_fontsize=8)
    ax.set_title(f"{m['stack']}: dominant scattering mechanism per scene, {n_cells} cells valid in every "
                 f"kept scene", fontsize=10, loc="left")
    return save(fig, out_dir, "mechanism_timeseries", show)


def plot_polsar_outputs(roi_paths, window, show=False):
    """PolSAR figures of the kept scenes of every stack into <roi>/polsar/<stack>/figures/."""
    stacks = [(d, polsar_paths(roi_paths.polsar / d.name, window)[0]) for d in roi_paths.stack_dirs()]
    stacks = [(d, f) for d, f in stacks if f.exists()]
    if not stacks:
        print(f"No PolSAR results for a {window:g} m window in {roi_paths.polsar}")
        return []
    print(f"PolSAR figures ({window:g} m window):")
    figures = []
    for stack_dir, path in stacks:
        st = _load_stack(path, stack_dir)
        out_dir = path.parent / "figures"
        print(f"  {path.parent.name}: {len(st['meta']['dates'])} kept scenes")
        if not st["meta"]["dates"] or not st["good"].any():
            print("    no kept scene or no valid cell inside the common area, skipped")
            continue
        for k in range(len(st["meta"]["dates"])):
            figures.extend(_scene_figures(st, k, out_dir, show))
        figures.append(_plot_stack_comparison(st, out_dir, show))
        figures.append(_plot_mechanism_timeseries(st, out_dir, show))
        del st
    return figures
