"""
Polarimetric tomography (Amacayacu_PolTomo_PolSARpro): coherent vertical focusing of the four
phase-corrected channels with one common filter per height, so that S(z) is a valid scattering
matrix, followed by PolSARpro decompositions of every height slice.

Arrays here use the PolSARpro layout (azimuth, range[, acquisition]) = (y, x).
"""
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

POLS = ("HH", "HV", "VH", "VV")
H_A_ALPHA_FLAGS = ("entropy", "anisotropy", "alpha")
FREEMAN_TERMS = {"Ps": "odd", "Pd": "double", "Pv": "volume"}


def capon_inverse_covariance(slc, kz, window_m, d_az, d_rg, theta, d_load=0.01):
    """Inverse of the polarimetric-trace multibaseline covariance, smoothed over window_m."""
    ny, nx, nb = kz.shape
    win = (max(int(round(window_m / d_az)) | 1, 1), max(int(round(window_m * np.sin(theta) / d_rg)) | 1, 1))
    R = np.zeros((ny, nx, nb, nb), dtype=np.complex128)
    for i in range(nb):
        for j in range(i, nb):
            acc = np.zeros((ny, nx), dtype=np.complex128)
            for pol in POLS:
                prod = slc[pol][:, :, i] * np.conj(slc[pol][:, :, j])
                acc += uniform_filter(prod.real, win) + 1j * uniform_filter(prod.imag, win)
            R[:, :, i, j], R[:, :, j, i] = acc, np.conj(acc)
    level = np.real(np.trace(R, axis1=2, axis2=3)) / nb
    empty = level <= 0
    R[empty] = np.eye(nb)
    R += (d_load * np.where(empty, 0.0, level))[..., None, None] * np.eye(nb)
    return np.linalg.inv(R)


def focus(slc, kz, z, R_inv=None):
    """Complex S(z): the same vertical filter (beamforming a/N, or Capon) for every channel."""
    a = np.exp(-1j * np.float32(z) * kz)
    if R_inv is None:
        w = a / kz.shape[-1]
    else:
        Ra = np.einsum("yxij,yxj->yxi", R_inv, a)
        w = Ra / np.einsum("yxi,yxi->yx", a.conj(), Ra)[..., None]
    return {pol: np.einsum("yxn,yxn->yx", w.conj(), slc[pol]).astype(np.complex64) for pol in POLS}


def to_polsarpro_S(channels, y, x, description=""):
    S = xr.Dataset({pol.lower(): (("y", "x"), channels[pol]) for pol in POLS}, coords={"y": y, "x": x},
                   attrs={"poltype": "S", "description": description})
    return S.chunk({"y": -1, "x": -1})


def pauli_components(channels, box):
    hh, vv = channels["HH"], channels["VV"]
    xx = 0.5 * (channels["HV"] + channels["VH"])
    terms = {"T11": 0.5 * np.abs(hh + vv) ** 2, "T22": 0.5 * np.abs(hh - vv) ** 2, "T33": 2.0 * np.abs(xx) ** 2}
    return {k: np.clip(uniform_filter(v.astype(np.float64), box), 0, None).astype(np.float32) for k, v in terms.items()}


def polarimetric_cube(slc, kz, heights, boxcar=(15, 5), step=(1, 1), focusing="beamforming",
                      capon_kw=None, decompositions=("pauli", "h_a_alpha", "freeman"), progress=None):
    """Focus every height and decompose it. Returns {parameter: (height, y, x)} subsampled by step."""
    from polsarpro.auxil import validate_dataset
    from polsarpro.decompositions import freeman, h_a_alpha

    ny, nx, _ = kz.shape
    y, x = np.arange(ny), np.arange(nx)
    R_inv = capon_inverse_covariance(slc, kz, **(capon_kw or {})) if focusing == "capon" else None
    sy, sx = slice(step[0] // 2, None, step[0]), slice(step[1] // 2, None, step[1])
    per_h = []
    for k, z in enumerate(heights):
        if progress:
            progress(k, z)
        ch = focus(slc, kz, z, R_inv)
        p = {}
        if "pauli" in decompositions:
            p.update(pauli_components(ch, boxcar))
        if {"h_a_alpha", "freeman"} & set(decompositions):
            S = to_polsarpro_S(ch, y, x, f"S at z = {z} m ({focusing})")
            if k == 0:
                validate_dataset(S, allowed_poltypes="S")
            if "h_a_alpha" in decompositions:
                haa = h_a_alpha(S, boxcar_size=list(boxcar), flags=H_A_ALPHA_FLAGS).compute()
                p.update({n: haa[n].values.astype(np.float32) for n in H_A_ALPHA_FLAGS})
            if "freeman" in decompositions:
                fd = freeman(S, boxcar_size=list(boxcar)).compute()
                p.update({n: fd[v].values.astype(np.float32) for n, v in FREEMAN_TERMS.items()})
        per_h.append({n: v[sy, sx] for n, v in p.items()})
    cube = {n: np.stack([p[n] for p in per_h]) for n in per_h[0]}
    if "T11" in cube:
        cube["span"] = cube["T11"] + cube["T22"] + cube["T33"]
        tot = np.where(cube["span"] > 0, cube["span"], np.nan)
        for n in ("T11", "T22", "T33"):
            cube[f"{n}_frac"] = cube[n] / tot
    if "Ps" in cube:
        tot = cube["Ps"] + cube["Pd"] + cube["Pv"]
        tot = np.where(tot > 0, tot, np.nan)
        for n in FREEMAN_TERMS:
            cube[f"{n}_frac"] = cube[n] / tot
    return cube, (sy, sx)


def rgb(cube, names, q=0.98):
    """RGB volume (height, y, x, 3) from three power parameters, one scaling per channel for all heights."""
    chans = []
    for n in names:
        a = np.sqrt(np.clip(cube[n], 0, None))
        chans.append(np.clip(a / np.nanquantile(a, q), 0, 1))
    return np.nan_to_num(np.stack(chans, axis=-1))
