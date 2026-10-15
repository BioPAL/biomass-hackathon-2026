"""Self-contained Lambda-1 RFI detector + block subspace filter (BSF) for SAR single-look-complex data.

Dependencies: numpy, scipy only.

    from rfi_mitigation import lambda1_bsf, ups_bdr
    p = ups_bdr("annotation.xml")                      # ups1, ups2, bdr1, bdr2 from the product annotation
    clean, mask = lambda1_bsf(slc, gamma=1.5, **p)     # slc: (rows, cols) or (n_pol, rows, cols) complex

Algorithm: Huizhang Yang et al., "Lambda-1 Detector for Interference Detection in Synthetic Aperture Radar
Images" and "BSF: Block Subspace Filter for Removing Narrowband and Wideband Radio Interference Artifacts in
Single-Look Complex SAR Images", IEEE TGRS 2024 (reference code: github.com/huizhangyang/lambda1-bsf).
get_eig_th below is copied from that code; the block loop, the multi-pol splicing, the normalisation and the
range-frequency domain option are our own additions.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy import fft as sfft
from scipy.sparse.linalg import svds


def get_eig_th(var, n, p, gamma=2, ups1=1.5, ups2=1.5, bdr1=0.7, bdr2=0.7):
    """Lambda-1 threshold (Tracy-Widom, pfa = 1e-5) for an n x p block of noise with variance var."""
    ups1 = ups1 / bdr1
    ups2 = ups2 / bdr2
    n = np.trunc(n / ups1)
    p = np.trunc(p / ups2)
    var = var * ups1 * ups2 * gamma
    TW = 2.6672
    u = np.square(np.sqrt(n) + np.sqrt(p))
    sigma = np.sqrt(u) * np.power(1 / np.sqrt(n) + 1 / np.sqrt(p), 1 / 3)
    return (TW * sigma + u) * var


def ups_bdr(annot_xml):
    """ups/bdr for lambda1 from a Biomass annotation XML: axis 1 = azimuth, axis 2 = range."""
    root = ET.parse(annot_xml).getroot()

    def get(path):
        el = root.find(path)
        if el is None:
            raise KeyError(f"tag not found: {path}")
        return float(el.text)

    az, rg = "azimuthProcessingParameters", "rangeProcessingParameters"
    bdr = lambda blk: (lambda a: a**2 + (1 - a) ** 2 / 2)(get(f".//{blk}/windowCoefficient"))  # Hamming: B_eq/B
    return {
        "ups1": get(f".//{az}/totalBandwidth") / get(f".//{az}/processingBandwidth"),   # fs_az / B_az
        "ups2": get(f".//{rg}/totalBandwidth") / get(f".//{rg}/processingBandwidth"),   # fs_rg / B_rg
        "bdr1": bdr(az),
        "bdr2": bdr(rg),
    }


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


if __name__ == "__main__":      # self-check: injected broadband RFI must be flagged and reduced
    rng = np.random.default_rng(0)
    S = rng.standard_normal((4, 600, 700)) + 1j * rng.standard_normal((4, 600, 700))
    S[:, :, 100:160] += 1.5 * np.exp(1j * rng.uniform(0, 6, (4, 600, 1)))
    kw = dict(gamma=1, ups1=1.5, ups2=1.5, bdr1=0.7, bdr2=0.7)
    q = []
    out, mask = lambda1_bsf(S, q_out=q, **kw)
    assert out.shape == S.shape and mask.shape == S.shape[1:] and mask.any() and q
    assert (np.abs(out) ** 2).mean() < (np.abs(S) ** 2).mean()
    assert lambda1_bsf(S[0], **kw)[0].shape == S.shape[1:]
    print("ok: flagged", f"{mask.mean():.0%}", "blocks, mean q", np.mean(q))
