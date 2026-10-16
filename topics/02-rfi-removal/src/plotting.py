from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
import os

from processing import prescreen_pulses, range_power, range_spectrogram

POLS = ["HH", "HV", "VH", "VV"]  # assumed band order, check src.descriptions or the annotation XML


def fetch_quicklook(item, path=None, token=None):
    """Quicklook JPG as a PIL image: from a local file if `path` is given, else streamed with `token`."""
    if path is not None:
        return Image.open(path)
    r = requests.get(
        item.assets["quicklook_jpg"].href, headers={"Authorization": f"Bearer {token}"}, timeout=60
    )
    r.raise_for_status()
    return Image.open(BytesIO(r.content))


def plot_product(slc, title, quicklook=None, az_step=4, prescreen_thr=2.5, out_dir: str = None):
    """slc: complex array (pol, azimuth, range). Used for both streamed and downloaded products.
    az_step: spectrograms use every az_step-th azimuth line (1 = all lines)."""
    P = {k: range_power(v) for k, v in zip(POLS, slc)}
    f = np.fft.fftshift(np.fft.fftfreq(slc.shape[2]))  # normalized frequency, cycles/sample

    dB = {k: 10 * np.log10(p + 1e-20) for k, p in P.items()}
    vmin, vmax = np.percentile(np.concatenate([d.ravel() for d in dB.values()]), [2, 98])

    # Select the azimuth line with the highest RFI peak along range spectrum
    rfi_pulses, ratio_along_az = prescreen_pulses(
        P["VH"], threshold_factor=prescreen_thr, window_size=501
    )
    flagged_indices = np.where(rfi_pulses)[0]
    top_rfi_indices = flagged_indices[np.argsort(ratio_along_az[flagged_indices])[::-1][0:3]]
    az_line_peak_rfi = top_rfi_indices[0] if len(top_rfi_indices) else slc.shape[1] // 2

    lines = slice(az_line_peak_rfi, az_line_peak_rfi + 1)
    SG = {k: range_spectrogram(v[lines]) for k, v in zip(POLS, slc)}  # (f, t, P), mean over lines
    sg_dB = {k: 10 * np.log10(P_ + 1e-20) for k, (_, _, P_) in SG.items()}
    svmin, svmax = np.percentile(np.concatenate([d.ravel() for d in sg_dB.values()]), [2, 98])

    # top: average spectrum + quicklook, middle: 2D spectrum per pol, bottom: spectrogram per pol
    n = len(P)
    fig, axd = plt.subplot_mosaic(
        [["avg"] * (n - 1) + ["ql"], list(P), [f"{k}_sg" for k in P]],
        figsize=(4.5 * n, 13),
        height_ratios=[1, 2.2, 2.2],
        constrained_layout=True,
    )

    ax1 = axd["avg"]
    for k in P:
        ax1.plot(
            f, 10 * np.log10(P[k].mean(axis=0) + 1e-20), label=k, lw=0.8
        )  # mean in linear power, then dB
    ax1.set(
        xlabel="normalized range frequency", ylabel="power [dB]", title="Average range spectrum"
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    axq = axd["ql"]
    if quicklook is not None:
        axq.imshow(quicklook)
    axq.set_title("Quicklook")
    axq.axis("off")

    for k, d in dB.items():
        im = axd[k].imshow(
            d,
            aspect="auto",
            extent=[f[0], f[-1], d.shape[0], 0],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axd[k].set(xlabel="normalized range frequency", title=k)
    axd[POLS[0]].set_ylabel("azimuth line")
    fig.colorbar(im, ax=[axd[k] for k in P], label="dB", shrink=0.9)

    for k, (fs_, t_, _) in SG.items():
        ax = axd[f"{k}_sg"]
        im2 = ax.imshow(
            sg_dB[k],
            aspect="auto",
            origin="lower",
            extent=[t_[0], t_[-1], fs_[0], fs_[-1]],
            cmap="viridis",
            vmin=svmin,
            vmax=svmax,
        )
        ax.set(xlabel="range sample", title=f"{k} Range STFT (line {az_line_peak_rfi})")
    axd[f"{POLS[0]}_sg"].set_ylabel("normalized range frequency")
    fig.colorbar(im2, ax=[axd[f"{k}_sg"] for k in P], label="dB", shrink=0.9)

    fig.suptitle(title[:60], fontsize=9)
    if out_dir is not None:
        fig.savefig(out_dir, dpi=150, bbox_inches="tight")
    plt.show()


def _box(a, fy, fx):
    """Average a over fy x fx boxes (a is real, e.g. intensity)."""
    h, w = (a.shape[0] // fy) * fy, (a.shape[1] // fx) * fx
    return a[:h, :w].reshape(h // fy, fy, w // fx, fx).mean(axis=(1, 3))


def _power(a):
    """|a|^2, summed over the pol axis if a is (pol, rows, cols)."""
    p = np.abs(a) ** 2
    return p if a.ndim == 2 else p.sum(axis=0)


def _pauli_db(s, fy, fx):
    """Pauli powers [|HH-VV|^2/2, 2|(HV+VH)/2|^2, |HH+VV|^2/2] in dB, box-averaged: (3, rows, cols)."""
    hh, hv, vh, vv = s
    pw = [np.abs(hh - vv) ** 2 / 2, 2 * np.abs((hv + vh) / 2) ** 2, np.abs(hh + vv) ** 2 / 2]
    return np.stack([10 * np.log10(_box(p, fy, fx) + 1e-12) for p in pw])


def pauli_rgb(slc, max_px=4000, pct=(1, 99), ref=None):
    """Pauli RGB (R=HH-VV, G=HV+VH, B=HH+VV, in dB) of a full-pol (4, rows, cols) stack as a float image (rows, cols, 3)
    in [0, 1], the same image as the Pauli panels of plot_result. max_px: decimate as plot_result does, box-average by
    n // max_px per axis (so at most about max_px pixels per axis; None = full resolution). Each channel is stretched between the pct percentiles of ref (default slc itself)."""
    R, C = slc.shape[-2:]
    fy, fx = (1, 1) if max_px is None else (max(1, n // max_px) for n in (R, C))
    pa = _pauli_db(slc, fy, fx)
    lo, hi = (np.array([np.percentile(c, q) for c in (pa if ref is None else _pauli_db(ref, fy, fx))])[:, None, None]
              for q in pct)
    return np.clip((pa - lo) / (hi - lo), 0, 1).transpose(1, 2, 0)


def plot_result(slc, clean, mask=None, title="", roi=None, max_px=4000, pct=(1, 99), aspect="auto", out_dir: str=None, spectrum=False, only_pauli=False):
    """slc/clean: (rows, cols) or full-pol (4, rows, cols) in POLS order. mask is only used for the title (% flagged).
    Full pol: the power panels are summed over pols, a Pauli RGB (R=HH-VV, G=HV+VH, B=HH+VV) row is added and
    the spectrum is shown per pol. spectrum=True adds the 2D range spectrum (azimuth line vs range frequency).
    only_pauli (full pol only): a single row with the filtered Pauli and the removed RFI, no source, power panels
    or spectra, and the title as given (e.g. for a cropped window); the Pauli stretch still comes from the source.
    roi=(slice(r0, r1), slice(c0, c1)) zooms at full resolution; max_px = max displayed pixels per axis."""
    full_pol = slc.ndim == 3
    if only_pauli and not full_pol:
        raise ValueError("only_pauli needs a full-pol (4, rows, cols) input")
    spectrum = spectrum and not only_pauli
    if roi is not None:
        slc, clean = slc[..., roi[0], roi[1]], clean[..., roi[0], roi[1]]
        mask = None if mask is None else mask[roi]
    R, C = slc.shape[-2:]
    fy, fx = (max(1, n // max_px) for n in (R, C))              # decimation per axis, 1 = none
    db = lambda p: 10 * np.log10(_box(p, fy, fx) + 1e-12)
    src, flt, rem = db(_power(slc)), db(_power(clean)), db(_power(slc - clean))
    lo, hi = np.percentile(src, pct)
    rlo, rhi = np.percentile(rem, pct)
    chans = list(zip(POLS, slc, clean)) if full_pol else [("", slc, clean)]   # channels of the spectrum

    # each row is a list of panel names, one sub-figure per row
    rows = [["pa_flt", "rem"]] if only_pauli else [["src", "flt", "rem"]]
    if full_pol and not only_pauli:
        rows.append(["pa_src", "pa_flt"])
    if spectrum:
        rows += [[f"sp_{k}_{w}" for k, _, _ in chans] for w in ("src", "flt")] if full_pol else [["sp__src", "sp__flt"]]
    fig = plt.figure(figsize=(16, 4.5 * len(rows)), layout="constrained")
    axd = {}
    for sf, r in zip(fig.subfigures(len(rows), 1, squeeze=False)[:, 0], rows):   # one sub-figure per row
        axd.update(zip(r, sf.subplots(1, len(r), squeeze=False)[0]))
    imgs = [axd[n] for n in ("src", "flt", "rem", "pa_src", "pa_flt") if n in axd]
    for a in imgs[1:]:
        a.sharex(imgs[0]); a.sharey(imgs[0])

    for n, img, t, vmin, vmax, cmap in [("src", src, "source [dB]", lo, hi, "gray"),
                                        ("flt", flt, "filtered [dB]", lo, hi, "gray"),
                                        ("rem", rem, "removed RFI [dB]", rlo, rhi, "magma")]:
        if n not in axd:
            continue
        im = axd[n].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest",
                           extent=(0, C, R, 0))
        axd[n].set_title(t)
        axd[n].figure.colorbar(im, ax=axd[n], pad=0.02)
    if full_pol:
        pa = [_pauli_db(v, fy, fx) for v in (slc, clean)]
        lo3, hi3 = (np.array([np.percentile(c, q) for c in pa[0]])[:, None, None] for q in pct)   # stretch from the source
        for n, d, t in zip(("pa_src", "pa_flt"), pa, ("source", "filtered")):
            if n not in axd:
                continue
            axd[n].imshow(np.clip((d - lo3) / (hi3 - lo3), 0, 1).transpose(1, 2, 0), aspect=aspect,
                          interpolation="nearest", extent=(0, C, R, 0))
            axd[n].set_title(f"{t} Pauli (R=HH-VV, G=HV+VH, B=HH+VV)")
    if spectrum:
        f = np.fft.fftshift(np.fft.fftfreq(C))
        for k, s_, c_ in chans:
            sp = [10 * np.log10(_box(range_power(v), fy, 1) + 1e-20) for v in (s_, c_)]
            slo, shi = np.percentile(sp[0], [2, 98])                  # same scale for source and filtered
            for w, d in zip(("src", "flt"), sp):
                im = axd[f"sp_{k}_{w}"].imshow(d, aspect="auto", cmap="viridis", vmin=slo, vmax=shi,
                                               extent=[f[0], f[-1], R, 0])
                axd[f"sp_{k}_{w}"].set(xlabel="normalized range frequency", ylabel="azimuth line" if k in ("", "HH") else "",
                                       title=f"{k} {'source' if w == 'src' else 'filtered'} range spectrum [dB]".strip())
                axd[f"sp_{k}_{w}"].figure.colorbar(im, ax=axd[f"sp_{k}_{w}"], pad=0.02)
    fig.suptitle(title if only_pauli else
                 f"{title}  (decimation {fy}x{fx}" + ("" if mask is None else f", {mask.mean():.0%} of blocks flagged") + ")")
    if out_dir is not None:
        # check if the directory exists, if not create it
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        fig.savefig(out_dir, dpi=150, bbox_inches="tight")
    plt.show()
