"""Coherence figures and the manual scene review (port of visualize_coherence.ipynb).

Per stack, into <roi>/coherence/<stack>/figures/ (all scenes, scene index [k] in every label):
    mbmp_matrix.png        mean |γ| of the 3Nb x 3Nb MBMP correlation matrix (HH, XP, VV blocks)
    coherence_<pol>.png    single-pol pair grid: |γ| maps above the diagonal, intensity on it
    temporal_rgb.png       RGB of |γ| at 3 / 6 / 9 days, KEPT scenes only

Manual scene review (review=True): for every stack the MBMP matrix and the HH, HV
and VV pair grids open in desktop windows and the scenes to keep are typed in the
terminal. The choice is saved in <roi>/extraction/<stack>/scene_selection.json
(core.scenes) and every later step (stack figures, PolSAR, comparisons, temporal
mechanism plot) uses only the kept scenes. The coherence itself is not recomputed:
the |γ| of a pair does not depend on the other scenes.

<roi>/coherence/<stack>/coherence_scenes.md: window, scenes (kept / removed), and the
3-day |γ| of every scene as a hint for the review.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from ..core.scenes import format_selection, parse_selection, removed_ids, save_selection
from ..processing.coherence import coherence_paths
from .common import CMAP, ask_with_figures, cell_geometry, save, show_map

MIN_VALID = 0.5        # cells with a smaller share of valid pixels in the window are masked
N_STD = 1.0            # 3-day |γ| below mean - N_STD * std is flagged in the review hint
RGB_LAGS_DAYS = (3, 6, 9)      # temporal baselines shown as R, G, B
RGB_POLS = ("HH", "XP", "VV")  # XP = (HV + VH) / 2, block order as in BPS
REVIEW_POLS = ("HH", "HV", "VV")


def _load_stack(path, min_valid=MIN_VALID):
    """npz + json of one stack, plus the maps/axes the plots need."""
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path) as data:
        st = {k: data[k] for k in data.files}
    st["meta"] = meta
    st["name"] = meta["stack"]
    st["good"] = st["valid_fraction"] >= min_valid
    # maps in radar geometry with crop pixel axes, same cells as the PolSAR maps
    st["geo"] = cell_geometry(meta["window"], st["az_index"], st["rg_index"])
    st["aspect"] = st["geo"]["aspect"]
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
    return show_map(ax, img, st["geo"], st["good"], labels, **kw)


def _three_day_coherence(st, pol):
    """Per scene: mean of the spatially averaged |γ| of its pairs at +-3 days."""
    mean_abs = np.abs(st["coh"][pol][st["good"]]).mean(axis=0)
    nb = mean_abs.shape[0]
    return np.array([
        np.mean([mean_abs[i, j] for j in range(nb) if abs(_lag_days(st, i, j)) == 3] or [np.nan])
        for i in range(nb)
    ])


def _three_day_table(st, n_std=N_STD):
    """Markdown table of the 3-day |γ| per scene and pol; values below mean - n_std * std are flagged (!)."""
    m = st["meta"]
    head = " | ".join(f"[{k}] {d[5:]}" for k, d in enumerate(m["dates"]))
    lines = [f"| pol | {head} | mean | std | flag below |", "|---" * (len(m["dates"]) + 4) + "|"]
    for pol in list(m["coh_pols"]) + ["XP"]:
        c3 = _three_day_coherence(st, pol)
        threshold = np.nanmean(c3) - n_std * np.nanstd(c3)
        cells = [f"{v:.3f}" + (" (!)" if v < threshold else "") for v in c3]
        lines.append(f"| {pol} | " + " | ".join(cells)
                     + f" | {np.nanmean(c3):.3f} | {np.nanstd(c3):.3f} | {threshold:.3f} |")
    return "\n".join(lines)


def _scenes_report(st, kept):
    """Markdown of coherence_scenes.md: window, grid, scenes (kept / removed) and the 3-day hint table."""
    m, w = st["meta"], st["meta"]["window"]
    rows = "\n".join(
        f"| {k} | {d} | {kz:+.4f} | {'kept' if k in kept else '**removed**'} |"
        f"{' primary' if k == m['primary_index'] else ''}"
        for k, (d, kz) in enumerate(zip(m["dates"], m["kz_mean_rad_per_m"]))
    )
    return (
        f"## {st['name']}\n\n"
        f"{m['orbit_state']}, {len(m['dates'])} scenes, {len(kept)} kept. Window {w['NW_a']} x {w['NW_r']} px = "
        f"**{w['window_ground_az_m']:.0f} x {w['window_ground_rg_m']:.0f} m** "
        f"(az x rg, rg/az = {w['aspect_rg_over_az']:.2f}), "
        f"{w['looks']} looks, output grid {w['output_spacing_az_m']:.0f} x {w['output_spacing_rg_m']:.0f} m, "
        f"{int(st['good'].sum())} valid cells.\n\n"
        f"| scene | date | mean kz [rad/m] | review |\n|---|---|---|---|\n{rows}\n\n"
        f"### Mean |γ| of each scene with its ±3-day partners (hint for the review)\n\n"
        f"(!) = below mean − {N_STD:g} std of the row.\n\n{_three_day_table(st)}"
    )


# -------------------- figures --------------------


def fig_mean_matrix(st):
    """Mean |γ| over the valid cells of the 3Nb x 3Nb MBMP correlation matrix."""
    m = st["meta"]
    nb, pols = len(m["dates"]), m["mbmp_pols"]
    mean_abs = np.abs(st["mbmp_corr"][st["good"]]).mean(axis=0)
    labels = [f"{p} [{k}] {d[5:]}" for p in pols for k, d in enumerate(m["dates"])]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(12, 10.5), layout="constrained")
    im = ax.imshow(mean_abs, cmap=CMAP, vmin=0, vmax=1)
    for b in range(1, len(pols)):                      # polarisation block borders
        ax.axhline(b * nb - 0.5, color="white", lw=2)
        ax.axvline(b * nb - 0.5, color="white", lw=2)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = mean_abs[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if 0.3 < v < 0.75 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean |γ| over the valid cells")
    ax.set_title(f"{st['name']}\nMBMP coherence: mean |γ| of the {n} x {n} correlation matrix "
                 f"(blocks HH, XP = (HV+VH)/2, VV x {nb} scenes [index] MM-DD), "
                 f"{int(st['good'].sum())} valid cells", fontsize=11)
    return fig


def fig_pair_grid(st, pol):
    """All scenes of one pol: |γ| maps above the diagonal, intensity <|pol|²> on the diagonal."""
    m = st["meta"]
    n = len(m["dates"])
    p = m["coh_pols"].index(pol)
    coh = st["coh"][pol]
    power_db = 10 * np.log10(st["power_pol"][:, :, p] + 1e-12)
    db_lim = np.nanpercentile(power_db[st["good"]], [2, 98])

    fig, axes = plt.subplots(n, n, figsize=(2.6 * n + 1.5, 2.6 * n * min(st["aspect"], 1.6) + 1),
                             layout="constrained", squeeze=False)
    im_db = im_abs = None
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i > j:
                ax.axis("off")
                continue
            if i == j:
                img, kw = power_db[:, :, i], dict(cmap=CMAP, vmin=db_lim[0], vmax=db_lim[1])
                txt = f"[{i}] {m['dates'][i]}\n<|{pol}|²> [dB]"
            else:
                img, kw = np.abs(coh[:, :, i, j]), dict(cmap=CMAP, vmin=0, vmax=1)
                txt = (f"|γ {pol}| [{i}]-[{j}] ({_lag_days(st, i, j)} d)\n"
                       f"mean {np.nanmean(img[st['good']]):.2f}")
            im = _show_map(ax, st, img, labels=(i == j), **kw)
            ax.set_title(txt, fontsize=7)
            if i == j and im_db is None:
                im_db = im
            if i != j and im_abs is None:
                im_abs = im
    # colorbars in the empty lower triangle
    half = max(n // 2, 1)
    if im_abs is not None:
        fig.colorbar(im_abs, ax=list(axes[-1, :half]), location="bottom", shrink=0.9,
                     label=f"single-pol coherence |γ {pol}|")
    fig.colorbar(im_db, ax=list(axes[-1, half:]) if n > 1 else [axes[-1, 0]],
                 location="bottom", shrink=0.9, label=f"intensity <|{pol}|²> [dB, uncalibrated]")
    fig.suptitle(f"{st['name']} — {pol}: single-pol coherence |γ| (upper triangle) and intensity (diagonal), "
                 f"{n} scenes [index]  [rows = azimuth, columns = slant range]", fontsize=12)
    return fig


def _rgb_anchors(st, kept, rgb_lags):
    """Kept scenes i whose partners at the given lags are kept too -> [(i, [j_lag1, ...]), ...]"""
    out = []
    for i in kept:
        partners = [next((j for j in kept if _lag_days(st, i, j) == lag), None) for lag in rgb_lags]
        if None not in partners:
            out.append((i, partners))
    return out


def _plot_rgb(st, kept, out_dir, rgb_lags=RGB_LAGS_DAYS, rgb_pols=RGB_POLS, show=False):
    """One RGB image per anchor and per pol: R, G, B = |γ| at the three lags (kept scenes only)."""
    m = st["meta"]
    anchors = _rgb_anchors(st, kept, rgb_lags)
    if not anchors:
        print(f"    {st['name']}: no kept scene has kept partners at {rgb_lags} days, no temporal RGB")
        return None
    lag_txt = "/".join(map(str, rgb_lags))
    n_col = len(anchors)
    fig, axes = plt.subplots(len(rgb_pols), n_col, squeeze=False, layout="constrained",
                             figsize=(3.2 * n_col + 1, 3.2 * len(rgb_pols) * min(st["aspect"], 1.6) + 0.6))
    for r, pol in enumerate(rgb_pols):
        for c, (i, partners) in enumerate(anchors):
            ax = axes[r, c]
            # absolute 0-1 scale: each channel is the raw |γ| (no stretch to the max of the image)
            rgb = np.stack([np.abs(st["coh"][pol][:, :, i, j]) for j in partners], axis=-1)
            rgba = np.concatenate([np.clip(rgb, 0, 1), st["good"][..., None].astype(float)], axis=-1)
            _show_map(ax, st, rgba, labels=(c == 0))
            means = rgb[st["good"]].mean(axis=0)
            ax.set_title(f"{pol}  [{i}] {m['dates'][i]} -> [{', '.join(map(str, partners))}]\n"
                         f"mean |γ| R {means[0]:.2f}  G {means[1]:.2f}  B {means[2]:.2f}", fontsize=8)
    fig.suptitle(f"{st['name']} — temporal RGB of |γ|: R = {rgb_lags[0]} d, G = {rgb_lags[1]} d, "
                 f"B = {rgb_lags[2]} d baseline (white = coherent over {lag_txt} d, red = fast decorrelation), "
                 f"{len(kept)}/{len(m['dates'])} scenes kept", fontsize=11)
    return save(fig, out_dir, "temporal_rgb", show)


# -------------------- review --------------------


def _ask_scenes(st, figs, default):
    """Interactive choice of the scenes to keep; the figures stay open (and zoomable) meanwhile."""
    n = len(st["meta"]["dates"])
    default_txt = format_selection(default)
    while True:
        answer = ask_with_figures(
            figs, f"  {st['name']}: scenes to keep (e.g. 0,2,4-6 | all | Enter = {default_txt}): ")
        try:
            chosen = parse_selection(answer or default_txt, n)
        except ValueError as exc:
            print(f"    {exc}")
            continue
        if not chosen:
            print("    keep at least one scene")
            continue
        removed = [k for k in range(n) if k not in chosen]
        note = " (fewer than 2 scenes: no coherence pairs left)" if len(chosen) < 2 else ""
        confirm = ask_with_figures(
            figs, f"    keep {format_selection(chosen)}, remove {format_selection(removed) or 'none'}{note}. "
                  f"OK? (y/n) [y]: ")
        if confirm.strip().lower() in ("", "y", "yes"):
            return chosen


def plot_coherence_outputs(roi_paths, window, review=False, interactive=True, min_valid=MIN_VALID,
                           rgb_lags=RGB_LAGS_DAYS, rgb_pols=RGB_POLS, show=False):
    """Coherence figures of every stack of the last extraction, with the manual scene review.

    review=True and interactive: the MBMP matrix and the HH, HV, VV pair grids of each stack
    open in windows and the scenes to keep are asked in the terminal (saved per stack).
    Otherwise the saved selection is used (all scenes when there is none).
    """
    stacks = [(d, coherence_paths(roi_paths.coherence / d.name, window)[0]) for d in roi_paths.stack_dirs()]
    stacks = [(d, f) for d, f in stacks if f.exists()]
    if not stacks:
        print(f"No coherence results for a {window:g} m window in {roi_paths.coherence}")
        return []
    print(f"Coherence figures ({window:g} m window){' + manual scene review' if review and interactive else ''}:")
    figures = []
    for stack_dir, path in stacks:
        out_dir = path.parent
        print(f"  {out_dir.name}:")
        st = _load_stack(path, min_valid)
        if not st["good"].any():
            print("    no valid cell inside the common area, skipped")
            continue
        products = [{"id": pid, "datetime": d} for pid, d in zip(st["meta"]["products"], st["meta"]["dates"])]
        removed = removed_ids(stack_dir)
        kept = [k for k, p in enumerate(products) if p["id"] not in removed]

        figs = {"mbmp_matrix": fig_mean_matrix(st)}
        figs.update({f"coherence_{pol.lower()}": fig_pair_grid(st, pol) for pol in st["meta"]["coh_pols"]})
        for tag, fig in figs.items():
            figures.append(save(fig, out_dir / "figures", tag, close=False))

        if review and interactive:
            print(f"    mean |γ| of each scene with its ±3-day partners ((!) = below mean − {N_STD:g} std):")
            print("    " + _three_day_table(st).replace("\n", "\n    "))
            shown = [figs["mbmp_matrix"]] + [figs[f"coherence_{pol.lower()}"] for pol in REVIEW_POLS]
            for tag, fig in figs.items():
                if fig not in shown:
                    plt.close(fig)
            kept = _ask_scenes(st, shown, kept)
            save_selection(stack_dir, products, kept)
        print(f"    scenes kept: {format_selection(kept)} of 0-{len(products) - 1}")
        plt.close("all")

        (out_dir / "coherence_scenes.md").write_text(_scenes_report(st, kept) + "\n", encoding="utf-8")
        figures.append(_plot_rgb(st, kept, out_dir / "figures", rgb_lags, rgb_pols, show))
        del st
    return [f for f in figures if f is not None]
