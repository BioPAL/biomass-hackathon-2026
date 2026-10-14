from shapely.geometry import Polygon, shape, mapping
from rasterio.windows import Window
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pystac_client import Client
from folium import GeoJson
import geopandas as gpd
from io import BytesIO
import rasterio as rio
from PIL import Image
import pandas as pd
import numpy as np
import requests
import pathlib
import fsspec
import folium
import fiona
import os
import json
from datetime import datetime, timedelta
import time
import yaml
from pathlib import Path
import branca.colormap as cm
from processing import pol_mean, prescreen_pulses, range_power, range_spectrogram

POLS = ["HH", "HV", "VH", "VV"]   # assumed band order, check src.descriptions or the annotation XML


def fetch_quicklook(item, path=None):
    """Quicklook JPG as a PIL image: from a local file if `path` is given, else streamed with the token."""
    if path is not None:
        return Image.open(path)
    r = requests.get(item.assets["quicklook_jpg"].href,
                     headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def plot_product(slc, title, quicklook=None, az_step=4, prescreen_thr=2.5, out_dir: str = None):
    """slc: complex array (pol, azimuth, range). Used for both streamed and downloaded products.
    az_step: spectrograms use every az_step-th azimuth line (1 = all lines)."""
    P = {k: range_power(v) for k, v in zip(POLS, slc)}
    f = np.fft.fftshift(np.fft.fftfreq(slc.shape[2]))   # normalized frequency, cycles/sample

    dB = {k: 10 * np.log10(p + 1e-20) for k, p in P.items()}
    vmin, vmax = np.percentile(np.concatenate([d.ravel() for d in dB.values()]), [2, 98])

    # Select the azimuth line with the highest RFI peak along range spectrum
    rfi_pulses, ratio_along_az = prescreen_pulses(
        P["VH"], threshold_factor=prescreen_thr, window_size=501
    )
    flagged_indices = np.where(rfi_pulses)[0]
    top_rfi_indices = flagged_indices[
        np.argsort(ratio_along_az[flagged_indices])[::-1][0:3]
    ]
    az_line_peak_rfi = top_rfi_indices[0] if len(top_rfi_indices) else slc.shape[1] // 2
    
    lines = slice(az_line_peak_rfi, az_line_peak_rfi + 1)
    SG = {k: range_spectrogram(v[lines]) for k, v in zip(POLS, slc)}   # (f, t, P), mean over lines
    sg_dB = {k: 10 * np.log10(P_ + 1e-20) for k, (_, _, P_) in SG.items()}
    svmin, svmax = np.percentile(np.concatenate([d.ravel() for d in sg_dB.values()]), [2, 98])

    # top: average spectrum + quicklook, middle: 2D spectrum per pol, bottom: spectrogram per pol
    n = len(P)
    fig, axd = plt.subplot_mosaic(
        [["avg"] * (n - 1) + ["ql"], list(P), [f"{k}_sg" for k in P]],
        figsize=(4.5 * n, 13), height_ratios=[1, 2.2, 2.2], constrained_layout=True,
    )

    ax1 = axd["avg"]
    for k in P:
        ax1.plot(f, 10 * np.log10(P[k].mean(axis=0) + 1e-20), label=k, lw=0.8)   # mean in linear power, then dB
    ax1.set(xlabel="normalized range frequency", ylabel="power [dB]", title="Average range spectrum")
    ax1.legend(); ax1.grid(alpha=0.3)

    axq = axd["ql"]
    if quicklook is not None:
        axq.imshow(quicklook)
    axq.set_title("Quicklook"); axq.axis("off")

    for k, d in dB.items():
        im = axd[k].imshow(d, aspect="auto", extent=[f[0], f[-1], d.shape[0], 0],
                           cmap="viridis", vmin=vmin, vmax=vmax)
        axd[k].set(xlabel="normalized range frequency", title=k)
    axd[POLS[0]].set_ylabel("azimuth line")
    fig.colorbar(im, ax=[axd[k] for k in P], label="dB", shrink=0.9)

    for k, (fs_, t_, _) in SG.items():
        ax = axd[f"{k}_sg"]
        im2 = ax.imshow(sg_dB[k], aspect="auto", origin="lower", extent=[t_[0], t_[-1], fs_[0], fs_[-1]],
                        cmap="viridis", vmin=svmin, vmax=svmax)
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

def plot_result(slc, clean, mask, title="", roi=None, max_px=4000, pct=(1, 99), aspect="auto", out_dir: str=None):
    """roi=(slice(r0, r1), slice(c0, c1)) zooms at full resolution; max_px = max displayed pixels per axis."""
    if roi is not None:
        slc, clean, mask = slc[roi], clean[roi], mask[roi]
    fy, fx = (max(1, n // max_px) for n in slc.shape)           # decimation per axis, 1 = none
    db = lambda p: 10 * np.log10(_box(p, fy, fx) + 1e-12)
    src, flt, rem = db(np.abs(slc) ** 2), db(np.abs(clean) ** 2), db(np.abs(slc - clean) ** 2)
    lo, hi = np.percentile(src, pct)
    rlo, rhi = np.percentile(rem, pct)

    fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    panels = [(src, "source [dB]", lo, hi, "gray"),
              (flt, "filtered [dB]", lo, hi, "gray"),
              (rem, "removed RFI [dB]", rlo, rhi, "magma"),
              (_box(mask.astype(float), fy, fx), f"block mask ({mask.mean():.0%} flagged)", 0, 1, "Reds")]
    for a, (img, t, vmin, vmax, cmap) in zip(ax.ravel(), panels):
        im = a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest",  # 2: use it
                      extent=(0, slc.shape[1], slc.shape[0], 0))
        a.set_title(t)
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle(f"{title}  (decimation {fy}x{fx})")
    if out_dir is not None:
        fig.savefig(out_dir, dpi=150, bbox_inches="tight") 
        plt.show()
    