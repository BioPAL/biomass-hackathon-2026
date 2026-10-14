"""MBMP coherence: run the marcos_dev coherence computer and plot its output.

Two parts:
- run_coherence: shells out to marcos_dev/1_data_extraction/compute_coherence.py
  (kept untouched). For every stack folder of the ROI it downloads the
  flattening LUT of each product (cached in flattening_lut.nc) and writes
  coherence_w<L>m.npz / coherence_w<L>m.json next to the products.
- plot_coherence_outputs: reads those npz files and saves the figures of
  marcos_dev/1_data_extraction/visualize_coherence.ipynb into <stack>/figures/:
  the mean MBMP correlation matrix, the |gamma| pair grids of the kept
  acquisitions for every polarisation and the RGB of the temporal
  decorrelation. The stack summary and the acquisition-filter table go to
  <stack>/coherence_filter.md.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

COMPUTE_SCRIPT = (
    Path(__file__).resolve().parent.parent / "marcos_dev" / "1_data_extraction" / "compute_coherence.py"
)

MIN_VALID = 0.5        # cells with a smaller share of valid pixels in the window are masked
N_STD = 1.0            # acquisitions below mean - N_STD * std of the 3-day |gamma| are dropped
RGB_LAGS_DAYS = (3, 6, 9)      # temporal baselines shown as R, G, B
RGB_POLS = ("HH", "XP", "VV")  # XP = (HV + VH) / 2, block order as in BPS


# -------------------- coherence computer (marcos_dev subprocess) --------------------


def _valid_products(stack_dir):
    """Products compute_coherence.py uses: extracted and still on disk, sorted by date."""
    info_path = stack_dir / "stack_info.json"
    if not info_path.exists():
        return []
    info = json.loads(info_path.read_text(encoding="utf-8"))
    products = [
        p for p in info["products"]
        if p.get("status") in ("ok", "skipped (exists)") and (stack_dir / p["file"]).exists()
    ]
    products.sort(key=lambda p: p["datetime"])
    return products


def coherence_paths(stack_dir, window):
    """npz / json written by compute_coherence.py for this averaging window."""
    stem = f"coherence_w{window:g}m"
    return stack_dir / f"{stem}.npz", stack_dir / f"{stem}.json"


def _up_to_date(stack_dir, window):
    """True when the coherence of this stack exists and matches its current products."""
    npz_path, json_path = coherence_paths(stack_dir, window)
    if not (npz_path.exists() and json_path.exists()):
        return False
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return [p["id"] for p in _valid_products(stack_dir)] == list(meta.get("products", []))


def pending_stacks(roi_dir, window):
    """Stack folders of an ROI folder whose coherence is missing or stale."""
    roi_dir = Path(roi_dir)
    todo = []
    for stack_dir in sorted(p.parent for p in roi_dir.glob("*/stack_info.json")):
        if _up_to_date(stack_dir, window):
            continue
        if len(_valid_products(stack_dir)) < 2:
            print(f"  {stack_dir.name}: fewer than 2 extracted products, no coherence")
            continue
        todo.append(stack_dir)
    return todo


def build_command(config, roi_slug, window, token_file, stacks=None, workers=None, yes=True):
    """Command line for marcos_dev/1_data_extraction/compute_coherence.py."""
    cmd = [
        sys.executable,
        str(COMPUTE_SCRIPT),
        "--roi-dir", str(config.roi_dir(roi_slug)),
        "--window", f"{window:g}",
        "--token-file", str(token_file),
    ]
    if stacks:
        names = list(stacks) if isinstance(stacks, (list, tuple)) else [stacks]
        cmd += ["--stacks", ",".join(names)]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if yes:
        cmd.append("--yes")
    return cmd


def run_coherence(config, roi, window, token_file, workers=None, yes=True):
    """Compute the coherence of every stack of the ROI whose result is missing or stale.

    Stacks whose coherence_w<L>m.npz already matches their products are skipped,
    so re-running the pipeline (or a plots-only run) does not redo the work.
    Delete <stack>/coherence_w<L>m.npz to force a recompute.
    """
    roi_dir = config.roi_dir(roi["slug"])
    todo = pending_stacks(roi_dir, window)
    if not todo:
        print(f"Coherence ({window:g} m window) already up to date in {roi_dir}")
        return roi_dir
    cmd = build_command(config, roi["slug"], window, token_file,
                        stacks=[d.name for d in todo], workers=workers, yes=yes)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return roi_dir


# -------------------- figures (port of visualize_coherence.ipynb) --------------------


def _save(fig, out_dir, tag, show=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"    saved {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


def _load_stack(path, min_valid=MIN_VALID):
    """npz + json of one stack, plus the maps/axes the plots need."""
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path) as data:
        st = {k: data[k] for k in data.files}
    st["meta"] = meta
    st["name"] = path.parent.name
    st["good"] = st["valid_fraction"] >= min_valid
    win = meta["window"]
    # maps in radar geometry with crop pixel axes: one output cell = d_az x d_rg crop pixels
    if st["az_index"].size >= 2 and st["rg_index"].size >= 2:
        d_az, d_rg = np.diff(st["az_index"][:2])[0], np.diff(st["rg_index"][:2])[0]
    else:  # single-cell grid
        d_az, d_rg = win["output_spacing_az_m"], win["output_spacing_rg_m"]
    st["extent"] = [
        st["rg_index"][0] - d_rg / 2, st["rg_index"][-1] + d_rg / 2,
        st["az_index"][-1] + d_az / 2, st["az_index"][0] - d_az / 2,
    ]
    # imshow aspect = ground height / width of one crop pixel (azimuth vs ground range)
    st["px_aspect"] = (win["output_spacing_az_m"] / d_az) / (win["output_spacing_rg_m"] / d_rg)
    st["aspect"] = win["output_spacing_az_m"] / win["output_spacing_rg_m"]  # of one output cell
    st["days"] = np.array([np.datetime64(d) for d in meta["dates"]])
    nb = len(meta["dates"])
    # (Naz, Nrg, Nb, Nb) complex coherence per pol; XP = diagonal block of the MBMP matrix
    st["coh"] = {pol: st["coh_pol"][:, :, p] for p, pol in enumerate(meta["coh_pols"])}
    xp = meta["mbmp_pols"].index("XP")
    st["coh"]["XP"] = st["mbmp_corr"][:, :, xp * nb:(xp + 1) * nb, xp * nb:(xp + 1) * nb]
    return st


def _lag_days(st, i, j):
    return int((st["days"][j] - st["days"][i]) / np.timedelta64(1, "D"))


def _show_map(ax, st, img, labels=True, **kw):
    """(Naz, Nrg) image or RGB(A) in radar geometry, crop pixel axes, invalid cells masked."""
    if img.ndim == 2:
        img = np.where(st["good"], img, np.nan)
    im = ax.imshow(img, extent=st["extent"], aspect=st["px_aspect"], interpolation="nearest", **kw)
    ax.tick_params(labelsize=6)
    if labels:
        ax.set_xlabel("range pixel", fontsize=7)
        ax.set_ylabel("azimuth pixel", fontsize=7)
    return im


def _three_day_coherence(st, pol):
    """Per acquisition: mean of the spatially averaged |gamma| of its pairs at +-3 days."""
    mean_abs = np.abs(st["coh"][pol][st["good"]]).mean(axis=0)
    nb = mean_abs.shape[0]
    return np.array([
        np.mean([mean_abs[i, j] for j in range(nb) if abs(_lag_days(st, i, j)) == 3] or [np.nan])
        for i in range(nb)
    ])


def _stack_summary(st):
    """Markdown header of coherence_filter.md: window, grid, dates and mean kz."""
    m, w = st["meta"], st["meta"]["window"]
    rows = "\n".join(
        f"| {k} | {d} | {kz:+.3f} |{' primary' if k == m['primary_index'] else ''}"
        for k, (d, kz) in enumerate(zip(m["dates"], m["kz_mean_rad_per_m"]))
    )
    return (
        f"## {st['name']}\n\n"
        f"{m['orbit_state']}, {len(m['dates'])} acquisitions. Window {w['NW_a']} x {w['NW_r']} px = "
        f"**{w['window_ground_az_m']:.0f} x {w['window_ground_rg_m']:.0f} m** "
        f"(az x rg, rg/az = {w['aspect_rg_over_az']:.2f}), "
        f"{w['looks']} looks, output grid {w['output_spacing_az_m']:.0f} x {w['output_spacing_rg_m']:.0f} m, "
        f"{int(st['good'].sum())} valid cells.\n\n"
        f"| idx | date | mean kz [rad/m] |\n|---|---|---|\n{rows}"
    )


def _filter_acquisitions(st, n_std=N_STD):
    """st['kept'][pol] = acquisitions reaching mean - n_std * std of the 3-day |gamma|."""
    m = st["meta"]
    st["kept"] = {}
    dates = " | ".join(f"{k}: {d[5:]}" for k, d in enumerate(m["dates"]))
    lines = [f"| pol | {dates} | mean | std | threshold | kept |", "|---" * (len(m["dates"]) + 5) + "|"]
    for pol in list(m["coh_pols"]) + ["XP"]:
        c3 = _three_day_coherence(st, pol)
        threshold = np.nanmean(c3) - n_std * np.nanstd(c3)
        kept = [k for k, v in enumerate(c3) if v >= threshold]
        st["kept"][pol] = kept
        cells = [f"{v:.3f}" if k in kept else f"~~{v:.3f}~~" for k, v in enumerate(c3)]
        lines.append(
            f"| {pol} | " + " | ".join(cells)
            + f" | {np.nanmean(c3):.3f} | {np.nanstd(c3):.3f} | **{threshold:.3f}** | {len(kept)}/{len(c3)} |"
        )
    print("    acquisition filter: " + ", ".join(
        f"{pol} {len(st['kept'][pol])}/{len(m['dates'])}" for pol in st["kept"]))
    return "\n".join(lines)


def _plot_mean_matrix(st, out_dir, show=False):
    """Mean over the valid cells of the 3Nb x 3Nb MBMP correlation: |gamma| and phase."""
    m = st["meta"]
    nb, pols = len(m["dates"]), m["mbmp_pols"]
    corr = st["mbmp_corr"][st["good"]]                 # (Ncells, 3Nb, 3Nb)
    mean_abs = np.abs(corr).mean(axis=0)
    mean_phase = np.rad2deg(np.angle(corr.mean(axis=0)))
    labels = [f"{p} {d[5:]}" for p in pols for d in m["dates"]]
    n = len(labels)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), layout="constrained")
    panels = [(mean_abs, "Blues", (0, 1), "mean |γ|"),
              (mean_phase, "twilight", (-180, 180), "phase of mean γ [deg]")]
    for ax, (mat, cmap, (vmin, vmax), title) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        for b in range(1, len(pols)):                  # polarisation block borders
            ax.axhline(b * nb - 0.5, color="black", lw=1.5)
            ax.axvline(b * nb - 0.5, color="black", lw=1.5)
        ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(n), labels, fontsize=8)
        ax.set_title(title, fontsize=13)
        fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(n):                                 # values of mean |gamma|
        for j in range(n):
            v = mean_abs[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                         color="white" if v > 0.55 else "black")
    fig.suptitle(f"{st['name']} — MBMP correlation {n} x {n} (HH, XP, VV x {nb} acquisitions), "
                 f"mean over {corr.shape[0]} cells", fontsize=14)
    return _save(fig, out_dir, "mbmp_matrix", show)


def _plot_pair_grid(st, pol, out_dir, show=False):
    """Grid of the kept acquisitions of one polarisation: |gamma| above the diagonal, intensity on it."""
    m = st["meta"]
    kept = st["kept"][pol]
    if len(kept) < 2:
        print(f"    {st['name']} {pol}: {len(kept)} acquisition(s) above the 3-day threshold, no pairs to show")
        return None
    p = m["coh_pols"].index(pol)
    coh = st["coh"][pol]
    power_db = 10 * np.log10(st["power_pol"][:, :, p] + 1e-12)
    db_lim = np.nanpercentile(power_db[st["good"]], [2, 98])
    n = len(kept)

    fig, axes = plt.subplots(n, n, figsize=(2.6 * n + 1.5, 2.6 * n * min(st["aspect"], 1.6) + 1),
                             layout="constrained", squeeze=False)
    for a, i in enumerate(kept):
        for b, j in enumerate(kept):
            ax = axes[a, b]
            if a > b:
                ax.axis("off")
                continue
            if a == b:
                img, kw = power_db[:, :, i], dict(cmap="gray", vmin=db_lim[0], vmax=db_lim[1])
                txt = f'{m["dates"][i]}  [dB]'
            else:
                img, kw = np.abs(coh[:, :, i, j]), dict(cmap="Blues", vmin=0, vmax=1)
                txt = f'|γ| {i}-{j} ({_lag_days(st, i, j)} d)  μ={np.nanmean(img[st["good"]]):.2f}'
            im = _show_map(ax, st, img, labels=(a == b), **kw)
            ax.set_title(txt, fontsize=8)
            if a == b == 0:
                im_db = im
            if (a, b) == (0, 1):
                im_abs = im
    # colorbars in the empty lower triangle
    half = max(n // 2, 1)
    fig.colorbar(im_abs, ax=list(axes[-1, :half]), location="bottom", shrink=0.9, label="|γ|")
    fig.colorbar(im_db, ax=list(axes[-1, half:]) if n > 1 else [axes[-1, 0]],
                 location="bottom", shrink=0.9, label="intensity [dB]")
    fig.suptitle(f"{st['name']} — {pol}: |γ| (upper), intensity (diagonal), {n}/{len(m['dates'])} acquisitions "
                 f"kept  [crop pixels: rows = azimuth, columns = slant range]", fontsize=13)
    return _save(fig, out_dir, f"coherence_{pol.lower()}", show)


def _rgb_anchors(st, pol, rgb_lags):
    """Kept acquisitions i whose partners at the given lags are kept too -> [(i, [j_lag1, ...]), ...]"""
    kept = st["kept"][pol]
    out = []
    for i in kept:
        partners = [next((j for j in kept if _lag_days(st, i, j) == lag), None) for lag in rgb_lags]
        if None not in partners:
            out.append((i, partners))
    return out


def _plot_rgb(st, out_dir, rgb_lags=RGB_LAGS_DAYS, rgb_pols=RGB_POLS, show=False):
    """One RGB image per valid anchor and per pol: R, G, B = |gamma| at the three lags."""
    m = st["meta"]
    anchors = {pol: _rgb_anchors(st, pol, rgb_lags) for pol in rgb_pols}
    n_col = max(len(a) for a in anchors.values())
    if n_col == 0:
        print(f"    {st['name']}: no kept acquisition has kept partners at {rgb_lags} days, no RGB to show")
        return None
    lag_txt = "/".join(map(str, rgb_lags))
    fig, axes = plt.subplots(len(rgb_pols), n_col, squeeze=False, layout="constrained",
                             figsize=(3.2 * n_col + 1, 3.2 * len(rgb_pols) * min(st["aspect"], 1.6) + 0.6))
    for r, pol in enumerate(rgb_pols):
        for c in range(n_col):
            ax = axes[r, c]
            if c >= len(anchors[pol]):
                ax.axis("off")
                continue
            i, partners = anchors[pol][c]
            # absolute 0-1 scale: each channel is the raw |gamma| (no stretch to the max of the image)
            rgb = np.stack([np.abs(st["coh"][pol][:, :, i, j]) for j in partners], axis=-1)
            rgba = np.concatenate([np.clip(rgb, 0, 1), st["good"][..., None].astype(float)], axis=-1)
            _show_map(ax, st, rgba, labels=(c == 0))
            means = rgb[st["good"]].mean(axis=0)
            ax.set_title(f"{pol}  {m['dates'][i]} ({i} -> {', '.join(map(str, partners))})\n"
                         f"mean R {means[0]:.2f}  G {means[1]:.2f}  B {means[2]:.2f}", fontsize=8)
        if len(anchors[pol]) == 0:
            axes[r, 0].text(0.5, 0.5, f"{pol}: no valid anchor", ha="center", va="center",
                            transform=axes[r, 0].transAxes)
    fig.suptitle(f"{st['name']} — RGB of |γ|: R = {rgb_lags[0]} d, G = {rgb_lags[1]} d, "
                 f"B = {rgb_lags[2]} d  (white = stable over {lag_txt} d, red = fast decorrelation)",
                 fontsize=12)
    return _save(fig, out_dir, "temporal_rgb", show)


def plot_coherence_outputs(roi_dir, window, min_valid=MIN_VALID, n_std=N_STD,
                           rgb_lags=RGB_LAGS_DAYS, rgb_pols=RGB_POLS, show=False):
    """Coherence figures of every stack of an ROI folder into <stack>/figures/.

    Stacks without a coherence_w<L>m.npz for this window are skipped, so this is
    safe to call whether or not the coherence was computed in this run.
    """
    roi_dir = Path(roi_dir)
    files = sorted(roi_dir.glob(f"*/coherence_w{window:g}m.npz"))
    if not files:
        print(f"No coherence results for a {window:g} m window in {roi_dir}")
        return []
    print(f"Coherence figures ({window:g} m window):")
    figures = []
    for path in files:
        stack_dir = path.parent
        print(f"  {stack_dir.name}:")
        st = _load_stack(path, min_valid)
        if not st["good"].any():
            print("    no valid cell inside the common area, skipped")
            del st
            continue
        report = [_stack_summary(st), _filter_acquisitions(st, n_std)]
        (stack_dir / "coherence_filter.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
        figures.append(_plot_mean_matrix(st, stack_dir / "figures", show))
        for pol in st["meta"]["coh_pols"]:
            figures.append(_plot_pair_grid(st, pol, stack_dir / "figures", show))
        figures.append(_plot_rgb(st, stack_dir / "figures", rgb_lags, rgb_pols, show))
        del st
    return [f for f in figures if f is not None]
