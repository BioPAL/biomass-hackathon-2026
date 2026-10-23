"""
InSAR coherence, phase calibration, TomoSAR and forest height, as developed in the
Amacayacu course notebooks (Amacayacu_InSAR_TomoSAR_QuadPol_Forest_Height_Calculation,
Amacayacu_Capon_Canopy_Height_Calculation_VH). Arrays use the course layout
(range, azimuth[, acquisition]); coherence matrices are (range, azimuth, N, N).

Conventions
    corrected SLC   s_corr^i = s^i exp(j (SIGN_DEM phi_DEM^i + SIGN_CAL phi_CAL^i))
    steering        a_i(z) = exp(-j KZ_SIGN k_z^i z)
SIGN_DEM / SIGN_CAL / KZ_SIGN are not documented for the L1c LUTs and are determined from
the data (phase_signs, kz_sign_check).
"""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve


# multilook / coherence
def window_samples(window_m, dr, dx):
    """Odd window (range, azimuth samples) spanning window_m metres."""
    return int(2 * np.floor(window_m / dr / 2) + 1), int(2 * np.floor(window_m / dx / 2) + 1)


def filter_downsample(RS, window_r, window_x, downsample_flag=True):
    """Boxcar moving average, decimated by half the window (course function)."""
    filt = np.ones((window_r, window_x)) / (window_r * window_x)
    dsr, dsx = max(int(window_r / 2), 1), max(int(window_x / 2), 1)
    RS = fftconvolve(RS, filt, mode="same")
    return (RS[::dsr, ::dsx], dsr, dsx) if downsample_flag else (RS, 1, 1)


def compute_coherence(in1, in2, window_r, window_x, min_valid=0.9):
    """Coherence of one pair; NaN where less than min_valid of the window holds valid samples."""
    cov, dsr, dsx = filter_downsample(in1.astype(np.complex128) * np.conj(in2), window_r, window_x)
    en1, _, _ = filter_downsample(np.abs(in1).astype(np.float64) ** 2, window_r, window_x)
    en2, _, _ = filter_downsample(np.abs(in2).astype(np.float64) ** 2, window_r, window_x)
    frac, _, _ = filter_downsample(((in1 != 0) & (in2 != 0)).astype(np.float64), window_r, window_x)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = cov / np.sqrt(en1 * en2)
    coh[frac < min_valid] = np.nan
    return coh, dsr, dsx


def compute_coherence_mb(stack, window_r, window_x, min_valid=0.9, progress=None):
    """Full multibaseline coherence matrix per downsampled pixel (Hermitian symmetry used)."""
    nb = stack.shape[2]
    out = None
    pairs = [(i, j) for i in range(nb) for j in range(i, nb)]
    for k, (i, j) in enumerate(pairs, 1):
        c, dsr, dsx = compute_coherence(stack[:, :, i], stack[:, :, j], window_r, window_x, min_valid)
        if out is None:
            out = np.zeros(c.shape + (nb, nb), dtype=np.complex64)
        out[:, :, i, j], out[:, :, j, i] = c, np.conj(c)
        if progress:
            progress(k / len(pairs))
    return out, dsr, dsx


def coherence_matrix(stack, window_r, window_x, min_valid=0.9, progress=None):
    """compute_coherence_mb for full frames: boxcar by uniform_filter, each intensity computed once.
    Same decimation (half the window) and validity rule; returns (coherence, intensity, dsr, dsx)
    with intensity the mean multilooked intensity over acquisitions."""
    nb = stack.shape[2]
    dsr, dsx = max(window_r // 2, 1), max(window_x // 2, 1)
    win = (window_r, window_x)
    dec = lambda a: a[::dsr, ::dsx]
    smooth = lambda a: dec(uniform_filter(a, win, mode="constant"))
    power = [smooth(np.abs(stack[:, :, i]).astype(np.float32) ** 2) for i in range(nb)]
    valid = [stack[:, :, i] != 0 for i in range(nb)]
    out = np.zeros(power[0].shape + (nb, nb), dtype=np.complex64)
    pairs = [(i, j) for i in range(nb) for j in range(i, nb)]
    for k, (i, j) in enumerate(pairs, 1):
        if i == j:
            frac = smooth(valid[i].astype(np.float32))
            c = np.ones(power[i].shape, np.complex64)
        else:
            prod = stack[:, :, i] * np.conj(stack[:, :, j])
            cov = smooth(prod.real) + 1j * smooth(prod.imag)
            frac = smooth((valid[i] & valid[j]).astype(np.float32))
            with np.errstate(divide="ignore", invalid="ignore"):
                c = (cov / np.sqrt(power[i] * power[j])).astype(np.complex64)
        c[frac < min_valid] = np.nan
        out[:, :, i, j], out[:, :, j, i] = c, np.conj(c)
        if progress:
            progress(k / len(pairs))
    return out, np.mean(power, axis=0), dsr, dsx


def mean_intensity(stack, window_r, window_x):
    """Mean multilooked intensity over acquisitions, on the coherence grid (for power scaling)."""
    return np.mean([filter_downsample(np.abs(stack[:, :, i]).astype(np.float64) ** 2, window_r, window_x)[0]
                    for i in range(stack.shape[2])], axis=0)


# phase calibration
def phase_signs(slc, dem_phases, cal_phases, ref, sec, window_r, window_x):
    """Empirical signs of the flattening and SKP calibration screens on a short-baseline pair.
    Flattening sign: maximises mean |coherence|. Calibration sign: maximises the uniformity
    R = |mean(exp(j angle(gamma)))| of the interferogram."""
    scores = {}
    for sd in (+1, -1):
        for sc in (+1, -1):
            corr = {i: slc[:, :, i] * np.exp(1j * (sd * dem_phases[:, :, i] + sc * cal_phases[:, :, i]))
                    for i in (ref, sec)}
            coh, _, _ = compute_coherence(corr[ref], corr[sec], window_r, window_x)
            scores[(sd, sc)] = {"coherence": float(np.nanmean(np.abs(coh))),
                                "uniformity": float(np.abs(np.nanmean(np.exp(1j * np.angle(coh)))))}
    sign_dem = max((+1, -1), key=lambda sd: max(scores[(sd, sc)]["coherence"] for sc in (+1, -1)))
    sign_cal = max((+1, -1), key=lambda sc: scores[(sign_dem, sc)]["uniformity"])
    margins = {"dem": abs(scores[(sign_dem, sign_cal)]["coherence"] - scores[(-sign_dem, sign_cal)]["coherence"]),
               "cal": abs(scores[(sign_dem, sign_cal)]["uniformity"] - scores[(sign_dem, -sign_cal)]["uniformity"])}
    return sign_dem, sign_cal, scores, margins


def phase_correction(dem_phases, cal_phases, sign_dem, sign_cal):
    return np.exp(1j * (sign_dem * dem_phases + sign_cal * cal_phases)).astype(np.complex64)


# geometry
def kz_summary(kz):
    """kz (range, azimuth, N): scene-mean wavenumbers, Rayleigh resolution, height of ambiguity."""
    mean = np.nanmean(kz, axis=(0, 1))
    srt = np.sort(mean)
    span, step = srt[-1] - srt[0], np.min(np.diff(srt))
    return {"kz_mean": mean, "span": span, "resolution_m": 2 * np.pi / span, "ambiguity_m": 2 * np.pi / step}


# tomography
def tomograms(coherence_mb, kz, z_vector, d_load=0.0, methods=("Beamformer", "Capon")):
    """Beamformer a^H G a and Capon 1 / (a^H G^-1 a), shape (range, azimuth, height)."""
    nr, nx, nb, _ = coherence_mb.shape
    out = {m: np.full((nr, nx, z_vector.size), np.nan, dtype=np.float32) for m in methods}
    eye = np.eye(nb)
    for ir in range(nr):
        g = coherence_mb[ir].astype(np.complex128)
        valid = np.all(np.isfinite(g), axis=(1, 2)) & np.all(np.isfinite(kz[ir]), axis=1)
        if not valid.any():
            continue
        steer = np.exp(-1j * kz[ir][valid][:, :, None] * z_vector[None, None, :])
        if "Beamformer" in methods:
            out["Beamformer"][ir, valid] = np.real(np.einsum("nbz,nbc,ncz->nz", steer.conj(), g[valid], steer))
        if "Capon" in methods:
            g_inv = np.linalg.inv(g[valid] + d_load * eye)
            out["Capon"][ir, valid] = np.real(1 / np.einsum("nbz,nbc,ncz->nz", steer.conj(), g_inv, steer))
    return out


def kz_sign_check(profile_by_channel, z_vector, margin=5.0):
    """Ratio of mean beamformer power above (+margin) to below (-margin) the ground per channel.
    After calibration the ground is z = 0 and forest power lies above it: median ratio > 1
    means the kz sign is right; < 1 means the profiles are mirrored."""
    above, below = z_vector > margin, z_vector < -margin
    ratios = {ch: float(np.nansum(p[above]) / np.nansum(p[below])) for ch, p in profile_by_channel.items()}
    return ratios, float(np.median(list(ratios.values()))) > 1


# height estimators
def percentile_height(volume, z_vector, percentiles, z_range=(-30.0, 70.0), subtract_floor=True):
    """Height below which p % of the (floor-subtracted) profile energy lies; returns (P, ...)."""
    q = np.atleast_1d(percentiles).astype(float) / 100.0
    sel = (z_vector >= z_range[0]) & (z_vector <= z_range[1])
    z, dz = z_vector[sel], z_vector[1] - z_vector[0]
    prof = volume[..., sel].astype(np.float64)
    if subtract_floor:
        prof = prof - np.nanmin(prof, axis=-1, keepdims=True)
    prof = np.clip(np.nan_to_num(prof, nan=0.0), 0, None)
    total = prof.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cdf = np.cumsum(prof, axis=-1) / total[..., None]
    invalid = ~np.isfinite(volume[..., sel]).all(axis=-1) | (total <= 0)
    out = np.full((q.size,) + volume.shape[:-1], np.nan, dtype=np.float32)
    for k, qk in enumerate(q):
        idx = np.argmax(cdf >= qk, axis=-1)
        here = np.take_along_axis(cdf, idx[..., None], -1)[..., 0]
        prev = np.where(idx > 0, np.take_along_axis(cdf, np.maximum(idx - 1, 0)[..., None], -1)[..., 0], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.clip((qk - prev) / (here - prev), 0, 1)
        out[k] = np.where(invalid, np.nan, z[idx] - dz / 2 + np.nan_to_num(frac, nan=0.5) * dz)
    return out


def _take(a, idx):
    return np.take_along_axis(a, np.clip(idx, 0, a.shape[-1] - 1)[..., None], -1)[..., 0]


def minmax_edge(volume, z_vector, level=0.5, z_range=(-30.0, 70.0), z_top_max=60.0):
    """Peak height (parabolic refinement) and highest height above it where the min-max
    rescaled profile still reaches `level`."""
    sel = (z_vector >= z_range[0]) & (z_vector <= z_range[1])
    z, dz = z_vector[sel], z_vector[1] - z_vector[0]
    P = volume[..., sel].astype(np.float64)
    invalid = ~np.isfinite(P).all(axis=-1)
    P = np.nan_to_num(P, nan=0.0)
    pmin, pmax = P.min(axis=-1, keepdims=True), P.max(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        Pn = (P - pmin) / (pmax - pmin)
    i_peak = np.argmax(Pn, axis=-1)
    i = np.clip(i_peak, 1, z.size - 2)
    p0, p1, p2 = _take(Pn, i - 1), _take(Pn, i), _take(Pn, i + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.clip(0.5 * (p0 - p2) / (p0 - 2 * p1 + p2), -0.5, 0.5)
    z_peak = z[i] + np.nan_to_num(off) * dz
    index = np.arange(z.size)
    above = (index >= i_peak[..., None]) & (Pn >= level) & (z <= z_top_max)
    i_last = np.where(above.any(-1), z.size - 1 - np.argmax(above[..., ::-1], axis=-1), -1)
    p_in, p_out = _take(Pn, i_last), _take(Pn, i_last + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.clip((p_in - level) / (p_in - p_out), 0, 1)
    z_edge = np.where((i_last >= 0) & (i_last + 1 < z.size), z[np.clip(i_last, 0, None)] + frac * dz, np.nan)
    z_edge = np.where(z_edge <= z_top_max, z_edge, np.nan)
    return (np.where(invalid, np.nan, z_peak).astype(np.float32),
            np.where(invalid, np.nan, z_edge).astype(np.float32))


def canopy_top(volume, z_vector, threshold_db=-1.0):
    """Highest height, contiguous above the peak (z >= 0), where the profile stays above threshold_db."""
    prof = np.where(z_vector[None, None, :] >= 0, volume, np.nan)
    peak = np.nanmax(prof, axis=2, keepdims=True)
    above = prof / peak >= 10 ** (threshold_db / 10)
    peak_idx = np.nanargmax(np.nan_to_num(prof, nan=-np.inf), axis=2)
    z_idx = np.arange(z_vector.size)[None, None, :]
    below_after = (~above) & (z_idx > peak_idx[..., None])
    first_below = np.where(below_after.any(axis=2), np.argmax(below_after, axis=2), z_vector.size)
    top = z_vector[np.clip(first_below - 1, 0, z_vector.size - 1)].astype(np.float32)
    top[~np.isfinite(peak[..., 0])] = np.nan
    return top


def phase_centre(coherence_mb, kz, ref):
    """Median over baselines of z_pc = -angle(gamma_ref,m) / (kz_ref - kz_m)."""
    nb = kz.shape[2]
    zs = []
    for m in range(nb):
        if m == ref:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            zs.append(-np.angle(coherence_mb[:, :, ref, m]) / (kz[:, :, ref] - kz[:, :, m]))
    return np.nanmedian(np.stack(zs), axis=0).astype(np.float32)


def quality_mask(coherence_mb, capon, coh_min=0.1, contrast_min_db=1.0):
    """Pixels whose coherence carries height information: mean off-diagonal |gamma| >= coh_min
    and Capon profile max/mean >= contrast_min_db."""
    nb = coherence_mb.shape[2]
    off = ~np.eye(nb, dtype=bool)
    mean_coh = np.nanmean(np.abs(coherence_mb)[:, :, off], axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        contrast = 10 * np.log10(np.nanmax(capon, axis=2) / np.nanmean(capon, axis=2))
    return (mean_coh >= coh_min) & (contrast >= contrast_min_db)


# RVoG
def _rvog_volume(kappa, hv, p1):
    k = kappa[..., :, None, None]
    h = hv[None, :, None]
    p = np.maximum(p1[..., None, None, :], 1e-6)
    pk = p + 1j * k
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gv = (p / pk) * np.expm1(pk * h) / np.expm1(p * h)
    return np.where(h > 0, gv, 1.0 + 0j)


def rvog_inversion(coherence_mb, kz, incidence_deg, valid=None,
                   hv_grid=np.arange(0, 61, 1.0), sigma_db_grid=np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8]),
                   mu_grid=np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]), progress=None):
    """Random-Volume-over-Ground grid search over all pairs, ground at z = 0 (no temporal decorrelation)."""
    nr, nx, nb, _ = coherence_mb.shape
    sigma_np = sigma_db_grid * np.log(10) / 20
    pairs = np.array([(n, m) for n in range(nb) for m in range(n + 1, nb)])
    pn, pm = pairs.T
    cos_t = np.cos(np.deg2rad(incidence_deg))
    if valid is None:
        valid = np.all(np.isfinite(coherence_mb), axis=(2, 3))
    out = {k: np.full((nr, nx), np.nan, dtype=np.float32) for k in ("h_v", "sigma_db_per_m", "mu", "misfit")}
    obs_all = coherence_mb[:, :, pn, pm]
    kappa_all = kz[:, :, pm] - kz[:, :, pn]
    for ir in range(nr):
        ok = valid[ir] & np.all(np.isfinite(kappa_all[ir]), axis=1)
        if not ok.any():
            continue
        obs = obs_all[ir, ok].astype(np.complex64)
        p1 = 2 * sigma_np[None, :] / cos_t[ir, ok][:, None]
        gv = _rvog_volume(kappa_all[ir, ok], hv_grid, p1).astype(np.complex64)
        best_cost = np.full(obs.shape[0], np.inf)
        best = np.zeros((obs.shape[0], 3))
        for mu in mu_grid:
            cost = np.sum(np.abs(obs[:, :, None, None] - (gv + mu) / (1 + mu)) ** 2, axis=1)
            flat = cost.reshape(cost.shape[0], -1)
            idx = np.argmin(flat, axis=1)
            c = flat[np.arange(flat.shape[0]), idx]
            better = c < best_cost
            ih, isg = np.unravel_index(idx, cost.shape[1:])
            best_cost[better] = c[better]
            best[better] = np.column_stack([hv_grid[ih], sigma_db_grid[isg], np.full(idx.size, mu)])[better]
        out["h_v"][ir, ok], out["sigma_db_per_m"][ir, ok], out["mu"][ir, ok] = best.T
        out["misfit"][ir, ok] = np.sqrt(best_cost / len(pairs))
        if progress:
            progress((ir + 1) / nr)
    out["at_grid_limit"] = out["h_v"] >= hv_grid[-1]
    return out
