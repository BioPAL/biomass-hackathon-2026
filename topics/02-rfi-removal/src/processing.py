import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from scipy import fft as sfft
from scipy import signal
from scipy.ndimage import median_filter
from scipy.sparse.linalg import svds

from lambda1_bsf.yang_filter import get_eig_th


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

def read_item(
    item_id, 
    paths_by_id, 
    window=None
):
    """Read the amplitude and phase of a product and return the SLC"""
    p = paths_by_id[item_id]
    try:
        with rasterio.open(p["enclosure_i_abs_tiff"]) as a, rasterio.open(p["enclosure_i_phase_tiff"]) as ph:
            return a.read(window=window) * np.exp(1j * ph.read(window=window))   # (pol, azimuth, range)
    except rasterio.errors.RasterioIOError as e:
        print(f"{item_id}: file not found ({e})")
        return None


def _block_filter(slc, lblk, lpd, gamma, ns_max, ns_mul, axis, q_out, min_size, outlier, kw):
    """Block loop of lambda1_bsf on a (n_pol, rows, cols) stack. Returns (filtered stack, mask (rows, cols)).
    outlier: zero pixels above mean + outlier*std of |patch| before the SVD (None = off)."""
    n, R, C = slc.shape
    out, mask = slc.copy(), np.zeros((R, C), np.int8)
    for y in range(0, R, lblk[0]):
        for x in range(0, C, lblk[1]):
            y0, y1 = max(y - lpd[0], 0), min(y + lblk[0] + lpd[0], R)
            x0, x1 = max(x - lpd[1], 0), min(x + lblk[1] + lpd[1], C)
            P = slc[:, y0:y1, x0:x1].copy()
            if outlier:
                a = np.abs(P); P[a > a.mean() + outlier * a.std()] = 0     # same outlier rule as the repo
            X = np.concatenate(list(P), axis=axis)                         # spliced matrix
            if min(X.shape) <= min_size:                                   # too small for svds / edge sliver
                continue
            th = get_eig_th(np.var(X), *X.shape, gamma=gamma, **kw)
            u, s, vh = svds(X, k=ns_max)                                   # singular values ascending
            if s[-1] ** 2 <= th:                                           # lambda1 detector: no RFI in this block
                continue
            q = min(ns_max, int(np.sum(s ** 2 > th) * ns_mul))            # >= 1 since s[-1]^2 > th
            rem = np.stack(np.split((u[:, -q:] * s[-q:]) @ vh[-q:], n, axis=axis))   # (n, h, w)
            hy, hx = min(lblk[0], R - y), min(lblk[1], C - x)
            out[:, y:y + hy, x:x + hx] -= rem[:, y - y0:y - y0 + hy, x - x0:x - x0 + hx]
            mask[y:y + hy, x:x + hx] = 1
            if q_out is not None:
                q_out.append(q)
    return out, mask


def lambda1_bsf(slc, lblk=(500, 500), lpd=(50, 50), gamma=1.5, ns_max=15, ns_mul=2,
                axis=1, q_out=None, norm=False, domain="image", **kw):
    """Lambda-1 detector + block subspace filter on slc (n_pol, rows, cols); a 2D array is one channel.
    The n_pol patches of each block are spliced along `axis` into one matrix, so n_pol=1 is per-channel
    filtering and n_pol=4 the joint (polarimetric) one: same code, only the input stack changes.
    kw: ups1, ups2, bdr1, bdr2 (from ups_bdr). q_out: list that receives the eigenvalues removed per block.
    norm: scale each pol to unit rms before splicing (so strong co-pols do not dominate the joint SVD and
    threshold) and scale the result back; no effect for a single channel.
    domain: "image" = blocks of the SLC (lblk x lpd, as before). "freq" = FFT along range, keep the in-band bins
    (B/fs = 1/ups2), whiten by the Hamming window (coefficient inverted from bdr2), and cut the same lblk/lpd
    blocks on (azimuth line, frequency bin), e.g. lblk=(500, 100), lpd=(50, 0), with ups2 = bdr2 = 1; needs ups2
    and bdr2 in kw. No outlier
    zeroing in this mode (it would zero the narrowband RFI itself). mask is then "azimuth block flagged in
    at least one band", repeated along range, so it keeps the (rows, cols) shape.
    Returns (cleaned, mask (rows, cols)) with the shape of slc."""
    squeeze = slc.ndim == 2
    slc = slc[None] if squeeze else slc
    n, R, C = slc.shape
    if norm:
        scale = np.sqrt((np.abs(slc) ** 2).mean(axis=(1, 2), keepdims=True))  # rms per pol
        slc = slc / scale
    if domain == "image":
        out, mask = _block_filter(slc, lblk, lpd, gamma, ns_max, ns_mul, axis, q_out,
                                  max(ns_max, 100), 8, kw)
    elif domain == "freq":
        bw = 1 / kw["ups2"]                                               # B / fs, normalized range bandwidth
        alpha = (1 + np.sqrt(6 * kw["bdr2"] - 2)) / 3                     # Hamming coefficient, inverse of bdr2
        f = np.fft.fftshift(np.fft.fftfreq(C))
        inb = np.abs(f) < bw / 2
        w = (alpha + (1 - alpha) * np.cos(2 * np.pi * f[inb] / bw)).astype(np.float32)   # window over the band
        F = sfft.fftshift(sfft.fft(slc, axis=-1, norm="ortho"), axes=-1)  # (n, R, C) range spectrum
        G, mask_f = _block_filter(F[..., inb] / w, lblk, lpd, gamma, ns_max, ns_mul,
                                  axis, q_out, max(ns_max, 50), None, {**kw, "ups2": 1, "bdr2": 1})
        F[..., inb] = G * w                                               # out-of-band bins stay as they were
        out = sfft.ifft(sfft.ifftshift(F, axes=-1), axis=-1, norm="ortho")
        mask = np.repeat(mask_f.any(axis=1, keepdims=True), C, axis=1).astype(np.int8)
    else:
        raise ValueError(f"domain must be 'image' or 'freq', got {domain!r}")
    if norm:
        out = out * scale
    return (out[0] if squeeze else out), mask
