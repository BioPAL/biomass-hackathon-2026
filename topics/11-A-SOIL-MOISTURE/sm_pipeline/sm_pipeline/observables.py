"""Observables from a coregistered complex window stack slc[t, pol, rows, cols] (NaN = invalid pixel)."""
import warnings
import numpy as np


def ml(x, nr, nc):
    """NaN-aware box average (multilook) over nr x nc pixels for real or complex arrays; the remainder is cropped."""
    R, C = (x.shape[-2] // nr) * nr, (x.shape[-1] // nc) * nc
    y = x[..., :R, :C].reshape(*x.shape[:-2], R // nr, nr, C // nc, nc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(y, axis=(-3, -1))


def to_db(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.where(x > 0, x, np.nan))


def sigma0_series(slc, factor=1.0, nr=12, nc=2, noise=None):
    """Multilooked sigma0 [dB] = ML(|s|^2 x factor) - ML(noise x factor); noise = beta0 noise power (optional)."""
    lin = ml(np.abs(slc) ** 2 * factor, nr, nc)
    if noise is not None: lin = lin - ml(noise * factor, nr, nc)
    return to_db(lin)


def interferogram(si, sj, nr=12, nc=2):
    return ml(si * np.conj(sj), nr, nc)


def coherence(si, sj, nr=12, nc=2):
    """Complex coherence <si sj*> / sqrt(<|si|^2><|sj|^2>) per multilook cell."""
    num = interferogram(si, sj, nr, nc); den = np.sqrt(ml(np.abs(si) ** 2, nr, nc) * ml(np.abs(sj) ** 2, nr, nc))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.where(den > 0, den, np.nan)


def coherence_matrix(slc, nr=12, nc=2):
    """All pairs: complex coherence G[i, j, pol, R, C] (Hermitian, diagonal 1)."""
    T = slc.shape[0]; shape = coherence(slc[0], slc[0], nr, nc).shape
    G = np.full((T, T) + shape, np.nan + 0j, dtype=complex)
    for i in range(T):
        G[i, i] = 1.0
        for j in range(i + 1, T):
            G[i, j] = coherence(slc[i], slc[j], nr, nc); G[j, i] = np.conj(G[i, j])
    return G


def closure_phase(G, i, j, k):
    """Closure phase arg(g_ij g_jk g_ki) per cell [pol, R, C]."""
    return np.angle(G[i, j] * G[j, k] * G[k, i])


def closure_phase_mean(G, i, j, k):
    """Window closure phase per pol: argument of the cell-averaged triple product (circular mean, no wrapping bias)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.angle(np.nanmean(G[i, j] * G[j, k] * G[k, i], axis=(-2, -1)))


def all_triplets(T):
    return [(i, j, k) for i in range(T) for j in range(i + 1, T) for k in range(j + 1, T)]


def consecutive_triplets(T):
    return [(i, i + 1, i + 2) for i in range(T - 2)]


def pauli_and_covariance(slc_t, nr=12, nc=2, names=("HH", "HV", "VH", "VV")):
    """Multilooked C3 covariance (lexicographic HH, sqrt2*HV, VV) [3, 3, R, C]; HV/VH symmetrised (or XX for 3-band products)."""
    idx = {n: i for i, n in enumerate(names)}
    hh, vv = slc_t[idx["HH"]], slc_t[idx["VV"]]
    hv = 0.5 * (slc_t[idx["HV"]] + slc_t[idx["VH"]]) if ("HV" in idx and "VH" in idx) else slc_t[idx.get("HV", idx.get("XX"))]
    k = np.stack([hh, np.sqrt(2) * hv, vv]); C = None
    for a in range(3):
        for b in range(3):
            v = ml(k[a] * np.conj(k[b]), nr, nc)
            if C is None: C = np.empty((3, 3) + v.shape, dtype=complex)
            C[a, b] = v
    return C


PP_METHODS = (("pp_freeman", "freeman", {}, ("odd", "double", "volume")),
              ("pp_yamaguchi4_y4r", "yamaguchi4", {"mode": "y4r"}, ("odd", "double", "volume", "helix")),
              ("pp_vanzyl", "vanzyl", {}, ("odd", "double", "volume")))


def pypolsarpro_decompositions(slc_t, names, nr, nc):
    """Window-mean powers of three PyPolSARpro decompositions for one date (satim-co/PolSARpro, Apache-2.0; ESA-funded
    re-implementation of PolSARpro): Freeman-Durden, Yamaguchi 4-component with orientation compensation (y4r) and van Zyl
    1992, each on a boxcar of nr x nc pixels (= the analysis looks). Returns {key: [odd, double, volume(, helix)]} or None
    if PyPolSARpro is not installed (optional dependency) or the product is not quad-pol."""
    try:
        import xarray as xr
        from polsarpro import decompositions as D
    except ImportError:
        return None
    idx = {n: i for i, n in enumerate(names)}
    if not all(p in idx for p in ("HH", "HV", "VH", "VV")): return None
    S = xr.Dataset({p.lower(): (("y", "x"), np.asarray(slc_t[idx[p]], np.complex64)) for p in ("HH", "HV", "VH", "VV")},
                   attrs=dict(poltype="S", description="Scattering matrix")).chunk()      # PyPolSARpro expects dask arrays
    out = {}
    for key, fn, kw, comps in PP_METHODS:
        o = getattr(D, fn)(S, boxcar_size=[int(nr), int(nc)], **kw).compute()
        out[key] = [float(np.nanmean(o[c].values)) for c in comps]
    o = D.h_a_alpha(S, boxcar_size=[int(nr), int(nc)], flags=("entropy", "anisotropy", "alpha")).compute()
    out["pp_h_a_alpha"] = [float(np.nanmean(o[c].values)) for c in ("entropy", "anisotropy", "alpha")]
    return out


def h_alpha(C):
    """Cloude-Pottier entropy H and mean alpha [deg] from C3 [3, 3, R, C] (via T3 = U C3 U^H); NaN cells stay NaN."""
    s = 1 / np.sqrt(2); U = np.array([[s, 0, s], [s, 0, -s], [0, 1.0, 0]], dtype=complex)
    T = np.einsum("ab,bcRC,dc->adRC", U, C, np.conj(U))
    Tm = np.moveaxis(T, (0, 1), (-2, -1)); bad = ~np.isfinite(Tm).all(axis=(-2, -1))
    Tm = np.where(bad[..., None, None], np.eye(3), Tm)
    w, v = np.linalg.eigh(Tm); w = np.clip(w.real, 1e-12, None); p = w / w.sum(-1, keepdims=True)
    H = -(p * np.log(p) / np.log(3)).sum(-1); alpha = (p * np.degrees(np.arccos(np.clip(np.abs(v[..., 0, :]), 0, 1)))).sum(-1)
    H[bad] = np.nan; alpha[bad] = np.nan
    return H, alpha
