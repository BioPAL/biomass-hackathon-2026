from rasterio.windows import Window
from pystac_client import Client
from folium import GeoJson
import geopandas as gpd
from io import BytesIO
import rasterio as rio
from tqdm import tqdm
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
import time
from scipy import signal
from scipy.ndimage import median_filter
from numpy.typing import NDArray
import rasterio

def pol_mean(df: pd.DataFrame, kind: str, suffix: str):
    cols = [c for c in df.columns if c.startswith(f"{kind}_") and c.endswith(f"_{suffix}")]
    return df[cols].mean(axis=1)

def prescreen_pulses(
    range_f: NDArray[np.complexfloating],   # (n_az, n_range) range spectrum
    threshold_factor: float,
    window_size: int = 127,
) -> tuple[NDArray[np.bool_], NDArray[np.floating]]:
    """Return a boolean array indicating which pulses exceed the threshold factor
    relative to the median magnitude of the range spectrum, and the index of the pulse with the maximum ratio.

    Each bin's power is normalized by ITS OWN median across pulses before taking
    the per-pulse max: comparing a pulse's max against its own median (across
    ~tens of thousands of bins) instead conflates real RFI with the extreme-value
    statistics of maxing over that many noise-only bins -- a clean pulse's max is
    already several times its median from noise alone, so short/moderate RFI hits
    get lost in that ceiling's own pulse-to-pulse jitter. A bin's median-across-pulses
    stays a clean, stable reference (RFI only ever hits a minority of pulses at any
    one bin), so normalizing against it isolates genuine anomalies instead.

    The threshold then tracks a moving median of ``ratio`` (large ``window_size``)
    rather than one global scalar, so it follows slow baseline drift (e.g.
    antenna pattern shape) instead of being skewed by it.
    """
    magnitude = np.abs(range_f)  # (n_az, n_range)
    bin_baseline = np.median(
        magnitude, axis=0
    )  # (n_range,) -- each bin's typical level across pulses
    ratio = np.max(magnitude / bin_baseline[np.newaxis, :], axis=1)  # per pulse
    baseline = median_filter(ratio, size=window_size, mode="nearest")
    threshold = threshold_factor * baseline
    return ratio > threshold, ratio

def range_power(
    S: NDArray[np.complexfloating],          # (azimuth, range)
    axis: int = 1,
) -> NDArray[np.floating]:
    """Power spectrum along range for every azimuth line (range = `axis`)."""
    S = np.nan_to_num(S)
    F = np.fft.fftshift(np.fft.fft(S, axis=axis), axes=axis)
    return np.abs(F) ** 2

def range_spectrogram(
    S: NDArray[np.complexfloating],          # (azimuth, range)
    nperseg: int = 256,
    overlap: float = 0.75,
    chunk: int = 256,
    axis: int = 1,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """STFT along range for every azimuth line, power averaged over azimuth.
    S: complex (azimuth, range). Returns f [cycles/sample], t [range samples], P [freq, t]."""
    S = np.nan_to_num(S)
    noverlap = int(overlap * nperseg)
    acc, n = None, 0
    for i in range(0, S.shape[0], chunk):                      # chunks keep memory bounded
        f, t, Z = signal.stft(S[i:i + chunk], fs=1.0, window="hann", nperseg=nperseg,
                              noverlap=noverlap, return_onesided=False, padded=True,
                              scaling="spectrum", axis=axis)   # Z: (lines, freq, t)
        p = (np.abs(Z) ** 2).sum(axis=0)
        acc = p if acc is None else acc + p
        n += Z.shape[0]
    f = np.fft.fftshift(f)
    P = np.fft.fftshift(acc / n, axes=0)                       # (freq, t), mean over lines
    return f, t, P

def read_annot(
    item_id, 
    paths_by_id, 
):
    """Read the amplitude and phase of a product and return the SLC"""
    p = paths_by_id[item_id]
    with rasterio.open(p["enclosure_annot_xml"]) as annot:
        return annot

def read_item(
    item_id, 
    paths_by_id, 
    window=None
):
    """Read the amplitude and phase of a product and return the SLC"""
    p = paths_by_id[item_id]
    with rasterio.open(p["enclosure_i_abs_tiff"]) as a, rasterio.open(p["enclosure_i_phase_tiff"]) as ph:
        return a.read(window=window) * np.exp(1j * ph.read(window=window))   # (pol, azimuth, range)
