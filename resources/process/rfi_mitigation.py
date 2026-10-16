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
<<<<<<< HEAD
from scipy.ndimage import maximum_filter, median_filter
=======
>>>>>>> upstream/main
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


<<<<<<< HEAD
def block_matrix(slc, y, x, lblk, lpd, axis=1, outlier=8):
    """Matrix the filter works on for the block starting at (y, x) of a (n_pol, rows, cols) stack: the lpd-padded
    patches of all pols, outlier-zeroed (mean + outlier*std of |patch|, None = off) and spliced along `axis`.
    Returns (matrix, (y0, x0)), (y0, x0) = origin of the padded patch."""
    _, R, C = slc.shape
    y0, y1 = max(y - lpd[0], 0), min(y + lblk[0] + lpd[0], R)
    x0, x1 = max(x - lpd[1], 0), min(x + lblk[1] + lpd[1], C)
    P = slc[:, y0:y1, x0:x1].copy()
    if outlier:
        a = np.abs(P); P[a > a.mean() + outlier * a.std()] = 0         # same outlier rule as the repo
    return np.concatenate(list(P), axis=axis), (y0, x0)                # spliced matrix


def freq_whiten(slc, ups2, bdr2):
    """Range spectrum of a (n_pol, rows, cols) stack for the "freq" domain.
    Returns (F fft-shifted (n_pol, rows, cols), inb in-band bin mask, w Hamming window over the in-band bins);
    the matrices the filter works on are F[..., inb] / w."""
    C = slc.shape[-1]
    bw = 1 / ups2                                                     # B / fs, normalized range bandwidth
    alpha = (1 + np.sqrt(6 * bdr2 - 2)) / 3                           # Hamming coefficient, inverse of bdr2
    f = np.fft.fftshift(np.fft.fftfreq(C))
    inb = np.abs(f) < bw / 2
    w = (alpha + (1 - alpha) * np.cos(2 * np.pi * f[inb] / bw)).astype(np.float32)   # window over the band
    return sfft.fftshift(sfft.fft(slc, axis=-1, norm="ortho"), axes=-1), inb, w


def bright_mask(slc, thr=8, tile=(500, 500), dil=3):
    """Bright scatterers of a (n_pol, rows, cols) stack: pixels above mean + thr*std of |slc| (per pol and tile),
    dilated by dil pixels (the sidelobes of a bright target extend along both axes)."""
    n, R, C = slc.shape
    m = np.zeros(slc.shape, bool)
    for k in range(n):
        for y in range(0, R, tile[0]):
            for x in range(0, C, tile[1]):
                a = np.abs(slc[k, y:y + tile[0], x:x + tile[1]])
                m[k, y:y + tile[0], x:x + tile[1]] = a > a.mean() + thr * a.std()
    return maximum_filter(m, size=(1, 2 * dil + 1, 2 * dil + 1)) if dil else m


def flatten_spectrum(Fb, seg, fwin=101):
    """Separable scale a(line) * b(f) of a range spectrum Fb (n_pol, rows, bins): a = per-line level (median |F| over
    bins), b = per-azimuth-segment baseline (median over the segment's lines of |F|/a, then a running median of
    fwin bins along frequency so narrow persistent lines do not enter the baseline but the band edge does).
    Fb / scale is then ~unit-variance everywhere: bright lines, spectral shape, band edges and noise-only bins
    all drop out, leaving what is persistent over azimuth and narrow in frequency (RFI)."""
    A = np.abs(Fb)
    a = np.median(A, axis=-1, keepdims=True) + 1e-30                  # (n, rows, 1)
    A /= a
    b = np.empty_like(A[:, :1]).repeat(-(-A.shape[1] // seg), axis=1)  # (n, n_seg, bins)
    for i, y in enumerate(range(0, A.shape[1], seg)):
        b[:, i] = median_filter(np.median(A[:, y:y + seg], axis=1), size=(1, fwin), mode="nearest")
    return a * np.repeat(b, seg, axis=1)[:, :A.shape[1]]


def _block_filter(slc, lblk, lpd, gamma, ns_max, ns_mul, axis, q_out, min_size, outlier, kw, raw=None, margins=None, detect_only=False):
    """Block loop of lambda1_bsf on a (n_pol, rows, cols) stack. Returns (filtered stack, mask (rows, cols)).
    outlier: zero pixels above mean + outlier*std of |patch| before the SVD (None = off).
    raw: optional stack to filter instead of slc: slc only decides where the RFI is and gives the subspace, the
    removal is the projection of raw onto that subspace (raw - raw V V^H).
    margins: list that receives lambda1 / threshold at gamma = 1 of every evaluated block (a block is flagged at gamma
    if its margin > gamma, the threshold is proportional to gamma). detect_only: stop after the detection."""
    n, R, C = slc.shape
    out, mask = (slc if raw is None else raw).copy(), np.zeros((R, C), np.int8)
    for y in range(0, R, lblk[0]):
        for x in range(0, C, lblk[1]):
            X, (y0, x0) = block_matrix(slc, y, x, lblk, lpd, axis, outlier)
            if min(X.shape) <= min_size:                                   # too small for svds / edge sliver
                continue
            th = get_eig_th(np.var(X), *X.shape, gamma=gamma, **kw)
            u, s, vh = svds(X, k=1 if detect_only else ns_max)             # singular values ascending
            if margins is not None:
                margins.append(s[-1] ** 2 * gamma / th)
            if detect_only or s[-1] ** 2 <= th:                                           # lambda1 detector: no RFI in this block
                continue
            q = min(ns_max, int(np.sum(s ** 2 > th) * ns_mul))            # >= 1 since s[-1]^2 > th
            if raw is None:
                low = (u[:, -q:] * s[-q:]) @ vh[-q:]                       # rank-q part of X
            else:
                low = (block_matrix(raw, y, x, lblk, lpd, axis, None)[0] @ vh[-q:].conj().T) @ vh[-q:]
            rem = np.stack(np.split(low, n, axis=axis))                    # (n, h, w)
=======
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
            if svds(X, k=1, maxiter=20)[1][0] ** 2 <= th:
                continue
            u, s, vh = svds(X, k=ns_max)
            q = int(np.sum(s ** 2 > th) * ns_mul)
            rem = np.stack(np.split((u[:, -q:] * s[-q:]) @ vh[-q:], n, axis=axis))   # (n, h, w)
>>>>>>> upstream/main
            hy, hx = min(lblk[0], R - y), min(lblk[1], C - x)
            out[:, y:y + hy, x:x + hx] -= rem[:, y - y0:y - y0 + hy, x - x0:x - x0 + hx]
            mask[y:y + hy, x:x + hx] = 1
            if q_out is not None:
                q_out.append(q)
    return out, mask


def lambda1_bsf(slc, lblk=(500, 500), lpd=(50, 50), gamma=1.5, ns_max=15, ns_mul=2,
<<<<<<< HEAD
                axis=1, q_out=None, norm=False, domain="image", joint=True, flatten=False, fwin=101, bright=None, margins=None, detect_only=False, **kw):
=======
                axis=1, q_out=None, norm=False, domain="image", **kw):
>>>>>>> upstream/main
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
<<<<<<< HEAD
    flatten: freq domain only. Whiten the in-band spectrum with a measured separable scale (per-line level x
    per-azimuth-segment spectral baseline, see flatten_spectrum, fwin = running-median width in bins) instead of the
    Hamming window: bright azimuth lines/targets and band edges that differ from the assumed band no longer
    dominate the SVD, so the persistent narrowband RFI does. The removal is (Fw - G) * scale, still zero where
    nothing is flagged.
    bright: freq domain only, sigma threshold (e.g. 8) of the bright-scatterer mask (bright_mask, in the image
    domain, before the FFT). Masked pixels are zeroed in a copy that is used to detect the RFI and to estimate the
    subspace (whitening scale, lambda1, SVD); the removal is still the projection of the unmasked data onto it, so
    bright targets are not altered. None = off.
    margins / detect_only: list that receives lambda1 / threshold(gamma = 1) of every block, and stop after the
    detection (nothing is removed, the input is returned): the false-alarm rate at any gamma is mean(margins > gamma).
    joint: False = per-channel, the same filter on each pol separately (mask = flagged in any pol); no effect
    for a single channel.
    The result is the input minus the removed part, so data that is not filtered comes back bit-identical.
=======
>>>>>>> upstream/main
    Returns (cleaned, mask (rows, cols)) with the shape of slc."""
    squeeze = slc.ndim == 2
    slc = slc[None] if squeeze else slc
    n, R, C = slc.shape
<<<<<<< HEAD
    if not joint and n > 1:
        res = [lambda1_bsf(c, lblk, lpd, gamma, ns_max, ns_mul, axis, q_out, norm, domain, flatten=flatten, fwin=fwin,
                           bright=bright, margins=margins, detect_only=detect_only, **kw) for c in slc]
        return np.stack([r[0] for r in res]), np.maximum.reduce([r[1] for r in res])
    norm = norm and n > 1                                                 # a single channel is scale invariant
    orig = slc
=======
>>>>>>> upstream/main
    if norm:
        scale = np.sqrt((np.abs(slc) ** 2).mean(axis=(1, 2), keepdims=True))  # rms per pol
        slc = slc / scale
    if domain == "image":
        out, mask = _block_filter(slc, lblk, lpd, gamma, ns_max, ns_mul, axis, q_out,
<<<<<<< HEAD
                                  max(ns_max, 100), 8, kw, margins=margins, detect_only=detect_only)
    elif domain == "freq":
        F, inb, w = freq_whiten(slc, kw["ups2"], kw["bdr2"])
        Fe = freq_whiten(np.where(bright_mask(slc, bright), 0, slc), kw["ups2"], kw["bdr2"])[0] if bright else F
        fscale = flatten_spectrum(Fe[..., inb], lblk[0], fwin) if flatten else w
        Fw = F[..., inb] / fscale                                          # data that is filtered
        G, mask_f = _block_filter(Fe[..., inb] / fscale, lblk, lpd, gamma, ns_max, ns_mul, axis, q_out,
                                  max(ns_max, 50), None, {**kw, "ups2": 1, "bdr2": 1}, raw=Fw if bright else None,
                                  margins=margins, detect_only=detect_only)
        D = np.zeros_like(F)
        D[..., inb] = (Fw - G) * fscale                                       # removed part of the spectrum
        out = slc - sfft.ifft(sfft.ifftshift(D, axes=-1), axis=-1, norm="ortho")
=======
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
>>>>>>> upstream/main
        mask = np.repeat(mask_f.any(axis=1, keepdims=True), C, axis=1).astype(np.int8)
    else:
        raise ValueError(f"domain must be 'image' or 'freq', got {domain!r}")
    if norm:
<<<<<<< HEAD
        out = orig - (slc - out) * scale        # subtract only what was removed: no scale/unscale round-off
=======
        out = out * scale
>>>>>>> upstream/main
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
<<<<<<< HEAD
    pc, pc_mask = lambda1_bsf(S, joint=False, **kw)                  # per-channel = filtering each pol alone
    assert np.allclose(pc[1], lambda1_bsf(S[1], **kw)[0]) and pc_mask.shape == S.shape[1:]
    ff, fm = lambda1_bsf(S, domain="freq", flatten=True, lblk=(500, 100), lpd=(50, 0), **kw)   # measured-baseline whitening
    for opt in (dict(), dict(norm=True), dict(norm=True, bright=8), dict(joint=False, bright=8)):       # all freq-mode option mixes
        fo, fmk = lambda1_bsf(S, domain="freq", flatten=True, lblk=(500, 100), lpd=(50, 0), **opt, **kw)
        assert fo.shape == S.shape and fmk.shape == S.shape[1:], opt
    assert ff.shape == S.shape and fm.shape == S.shape[1:]
=======
>>>>>>> upstream/main
    print("ok: flagged", f"{mask.mean():.0%}", "blocks, mean q", np.mean(q))
