"""Soil-moisture estimators and forward models.

Dielectric models
  eps_mironov     Mironov, Kosolapova & Fomin (2009), "Physically and mineralogically based spectroscopic dielectric model for
                  moist soils", IEEE TGRS 47(7) 2059-2070, doi:10.1109/TGRS.2008.2011631 (CrossRef-verified 14 Oct 2026);
                  the generalized refractive mixing form is Mironov et al. (2004), IEEE TGRS 42(4) 773-785, doi:10.1109/TGRS.2003.823288.
                  Default here; coefficients are the clay-fraction regressions of the 2009 paper, valid 0.3-26 GHz incl. P-band.
  eps_hallikainen Hallikainen et al. (1985), 1.4 GHz polynomial coefficients (kept for comparison only)
Interferometric soil-volume model
  De Zan, Parizzi, Prats-Iraola & Lopez-Dekker (2014), IEEE TGRS 52(1) 418-425, doi:10.1109/TGRS.2013.2241069:
  coherence between moisture states m1, m2 of a homogeneous semi-infinite soil volume,
  gamma = 2j sqrt(Im k1 Im k2) / (k1 - conj(k2)),  k = k0 sqrt(eps - sin^2 theta) (vertical wavenumber in the soil).
  The absolute sign of the modelled phase depends on the SAR phase convention; run_stack.py tests it against the data
  (closure-phase forward check: r < 0 means the opposite convention).
Estimators
  stage A: relative change (backscatter anomaly / slope, closure-phase inversion relative to the first date)
  stage B: absolute (two-parameter calibration at a station) - only as an error estimate
"""
import numpy as np
from scipy.optimize import least_squares

C0 = 299792458.0; F_BIOMASS = 435e6; LAMBDA = C0 / F_BIOMASS; EPS0 = 8.8541878128e-12


def eps_hallikainen(m, f_hz=F_BIOMASS, clay=0.2, sand=0.4):
    S, Cl = 100 * sand, 100 * clay; m = np.asarray(m, float)
    er = (2.862 - 0.012 * S + 0.001 * Cl) + (3.803 + 0.462 * S - 0.341 * Cl) * m + (119.006 - 0.500 * S + 0.633 * Cl) * m ** 2
    ei = (0.356 - 0.003 * S - 0.008 * Cl) + (5.507 + 0.044 * S - 0.002 * Cl) * m + (17.753 - 0.313 * S + 0.206 * Cl) * m ** 2
    return er + 1j * ei


def eps_mironov(m, f_hz=F_BIOMASS, clay=0.2, sand=None):
    """Complex permittivity eps' + j eps'' of moist soil (volumetric moisture m [m3/m3], clay mass fraction 0-1)."""
    C = 100.0 * clay; m = np.asarray(m, float)
    nd = 1.634 - 0.539e-2 * C + 0.2748e-4 * C ** 2
    kd = 0.03952 - 0.04038e-2 * C
    mvt = 0.02863 + 0.30673e-2 * C
    e0b = 79.8 - 85.4e-2 * C + 32.7e-4 * C ** 2
    taub = 1.062e-11 + 3.450e-12 * 1e-2 * C
    sigb = 0.3112 + 0.467e-2 * C
    sigu = 0.3631 + 1.217e-2 * C
    e0u, tauu, einf = 100.0, 8.5e-12, 4.9
    w = 2 * np.pi * f_hz
    def debye(e0, tau, sig):
        return einf + (e0 - einf) / (1 + (w * tau) ** 2), (e0 - einf) * w * tau / (1 + (w * tau) ** 2) + sig / (w * EPS0)
    def nk(er, ei):
        a = np.sqrt(er ** 2 + ei ** 2); return np.sqrt((a + er) / 2), np.sqrt((a - er) / 2)
    nb, kb = nk(*debye(e0b, taub, sigb)); nu, ku = nk(*debye(e0u, tauu, sigu))
    low = m <= mvt
    n = np.where(low, nd + (nb - 1) * m, nd + (nb - 1) * mvt + (nu - 1) * (m - mvt))
    k = np.where(low, kd + kb * m, kd + kb * mvt + ku * (m - mvt))
    return (n ** 2 - k ** 2) + 1j * (2 * n * k)


DIELECTRIC = {"mironov": eps_mironov, "hallikainen": eps_hallikainen}


def make_eps(name="mironov", clay=0.2, sand=0.4, f_hz=F_BIOMASS):
    fn = DIELECTRIC[name]
    return lambda m: fn(m, f_hz=f_hz, clay=clay, sand=sand)


_DEFAULT_EPS = make_eps("mironov")


def k_soil(m, theta_deg=35.0, eps_fn=None):
    eps = (eps_fn or _DEFAULT_EPS)(np.asarray(m, float)); k0 = 2 * np.pi / LAMBDA
    return k0 * np.sqrt(eps - np.sin(np.radians(theta_deg)) ** 2)


def coherence_model(m1, m2, theta_deg=35.0, eps_fn=None):
    k1, k2 = k_soil(m1, theta_deg, eps_fn), k_soil(m2, theta_deg, eps_fn)
    return 2j * np.sqrt(k1.imag * k2.imag) / (k1 - np.conj(k2))


def closure_model(m, triplets, theta_deg=35.0, eps_fn=None):
    m = np.asarray(m, float)
    return np.array([np.angle(coherence_model(m[i], m[j], theta_deg, eps_fn) * coherence_model(m[j], m[k], theta_deg, eps_fn)
                              * coherence_model(m[k], m[i], theta_deg, eps_fn)) for i, j, k in triplets])


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def invert_closure(obs_closure, triplets, T, theta_deg=35.0, m0=None, bounds=(0.02, 0.50), coh_obs=None, pairs=None, w_coh=0.0, eps_fn=None, n_starts=6):
    """Least-squares moisture sequence m[t] from observed closure phases [rad] (EXPERIMENTAL). Closure phases constrain the
    relative sequence only weakly and the absolute level not at all (De Zan 2014; Zwieback et al. 2017, doi:10.1109/TGRS.2017.2702099):
    report m - m[0]. Several non-uniform starts because a constant sequence is a stationary point (all closure phases 0)."""
    obs = np.asarray(obs_closure, float); ok = np.isfinite(obs); trip_ok = [t for t, o in zip(triplets, ok) if o]
    if ok.sum() < 3: return dict(m=np.full(T, np.nan), residual_rms_rad=np.nan, success=False)
    def resid(m):
        r = [wrap(closure_model(m, trip_ok, theta_deg, eps_fn) - obs[ok])]
        if coh_obs is not None and w_coh > 0:
            r.append(w_coh * (np.array([np.abs(coherence_model(m[i], m[j], theta_deg, eps_fn)) for i, j in pairs]) - np.asarray(coh_obs)))
        return np.concatenate(r)
    rng = np.random.default_rng(0)
    starts = [np.asarray(m0, float)] if m0 is not None else [np.linspace(0.30, 0.10, T), np.linspace(0.10, 0.30, T)] + [np.clip(0.2 + rng.normal(0, 0.06, T), *bounds) for _ in range(n_starts)]
    best = None
    for x0 in starts:
        sol = least_squares(resid, np.clip(x0, bounds[0] + 1e-3, bounds[1] - 1e-3), bounds=(bounds[0], bounds[1]), x_scale=0.05)
        if best is None or sol.cost < best.cost: best = sol
    return dict(m=best.x, residual_rms_rad=float(np.sqrt(np.mean(best.fun[:ok.sum()] ** 2))), success=bool(best.success))


def fit_sensitivity(dsig_db, sm_ref):
    x, y = np.asarray(sm_ref, float), np.asarray(dsig_db, float); ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0: return dict(S=np.nan, a=np.nan, r=np.nan, n=int(ok.sum()))
    S, a = np.polyfit(x[ok], y[ok], 1); return dict(S=float(S), a=float(a), r=float(np.corrcoef(x[ok], y[ok])[0, 1]), n=int(ok.sum()))


def sm_from_backscatter(sig_db, S, a=None, ref_db=None):
    """Stage A (a=None): dSM = (sigma0 - ref) / S. Stage B: SM = (sigma0 - a) / S."""
    sig = np.asarray(sig_db, float)
    return (sig - a) / S if a is not None else (sig - (np.nanmean(sig) if ref_db is None else ref_db)) / S
