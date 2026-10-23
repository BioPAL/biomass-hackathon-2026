"""
Single-acquisition polarimetry on the L1c stack: radiometric calibration (beta0 -> sigma0 with the
/radiometry LUT), thermal-noise removal on the coherency matrix (/denoising LUT), and the H-alpha
plane. Arrays use the PolSARpro layout (azimuth, range) = (y, x); decompositions come from polsarpro.

Noise in the Pauli basis, for independent noise powers N_pq in each channel:
    T11, T22 += (N_HH + N_VV) / 2,   Re T12 += (N_HH - N_VV) / 2,   T33 += (N_HV + N_VH) / 2
"""
import numpy as np
import xarray as xr

POLS = ("HH", "HV", "VH", "VV")


def coherency_T3(slc, factor=None, noise=None, box=(9, 3), chunk=2048):
    """Boxcar-averaged Pauli coherency matrix T3 (polsarpro 'T3' Dataset) from four complex channels.
    factor: sigma0 (or gamma0) conversion factor per pixel applied to powers (None = beta0).
    noise: dict pol -> noise power per pixel in beta0 (None = no noise removal)."""
    from polsarpro.decompositions import S_to_T3, boxcar
    y, x = np.arange(slc["HH"].shape[0]), np.arange(slc["HH"].shape[1])
    amp = np.sqrt(factor).astype(np.float32) if factor is not None else np.float32(1)
    S = xr.Dataset({p.lower(): (("y", "x"), (slc[p] * amp).astype(np.complex64)) for p in POLS},
                   coords={"y": y, "x": x}, attrs={"poltype": "S"}).chunk({"y": chunk, "x": -1})
    T3 = boxcar(S_to_T3(S), int(box[0]), int(box[1])).compute()
    if noise is not None:
        f = factor if factor is not None else 1.0
        has_data = ((T3.m11 + T3.m22 + T3.m33) > 0).values          # no noise removal on no-data pixels
        n = {p: noise[p] * f * has_data for p in POLS}
        T3["m11"] = T3.m11 - 0.5 * (n["HH"] + n["VV"])
        T3["m22"] = T3.m22 - 0.5 * (n["HH"] + n["VV"])
        T3["m33"] = T3.m33 - 0.5 * (n["HV"] + n["VH"])
        T3["m12"] = T3.m12 - 0.5 * (n["HH"] - n["VV"])
        for d in ("m11", "m22", "m33"):
            T3[d] = T3[d].clip(min=1e-12).astype(np.float32)
        T3["m12"] = T3.m12.astype(np.complex64)
    T3.attrs = {"poltype": "T3", "description": "Coherency matrix (3x3)"}
    return T3.chunk({"y": chunk, "x": -1})                         # polsarpro decompositions expect dask arrays


def backscatter(T3):
    """sigma (or beta) nought per channel from T3: HH, VV, symmetrised cross-pol XX (power, linear)."""
    hh = 0.5 * (T3.m11 + T3.m22) + np.real(T3.m12)
    vv = 0.5 * (T3.m11 + T3.m22) - np.real(T3.m12)
    xx = 0.5 * T3.m33
    return {"HH": hh.values, "VV": vv.values, "XX": xx.values}


def h_alpha_plane(ax, labels=True):
    """Feasible-region boundaries (curves I and II) and the nine Cloude-Pottier zones."""
    def ent(p):
        p = np.asarray(p, float) / np.sum(p)
        p = p[p > 0]
        return -np.sum(p * np.log(p) / np.log(3))
    m = np.linspace(0, 1, 200)
    h1 = [ent([1, v, v]) for v in m]
    a1 = 90 * 2 * m / (1 + 2 * m)
    m2a = np.linspace(0, 0.5, 100)
    h2a, a2a = [ent([0, 1, 2 * v]) for v in m2a], np.full(m2a.size, 90.0)
    m2b = np.linspace(0.5, 1, 100)
    h2b, a2b = [ent([2 * v - 1, 1, 1]) for v in m2b], 180 / (1 + 2 * m2b)
    ax.plot(h1, a1, "k-", lw=1)
    ax.plot(np.r_[h2a, h2b], np.r_[a2a, a2b], "k-", lw=1)
    for h0, h1_, alphas in ((0, 0.5, (42.5, 47.5)), (0.5, 0.9, (40, 50)), (0.9, 1.0, (40, 55))):
        for a in alphas:
            ax.plot([h0, h1_], [a, a], "k--", lw=0.6)
    for h in (0.5, 0.9):
        ax.axvline(h, color="k", ls="--", lw=0.6)
    if labels:
        for (h, a, z) in ((0.25, 70, "Z7"), (0.25, 45, "Z8"), (0.25, 20, "Z9"), (0.7, 70, "Z4"), (0.7, 45, "Z5"),
                          (0.7, 20, "Z6"), (0.95, 75, "Z1"), (0.95, 47, "Z2"), (0.95, 20, "Z3")):
            ax.text(h, a, z, ha="center", va="center", fontsize=8, color="0.3")
    ax.set_xlim(0, 1); ax.set_ylim(0, 90)
    ax.set_xlabel("entropy H"); ax.set_ylabel(r"$\alpha$ [deg]")


def noise_check(noise_levels_db, tol_db=1.5):
    """noise_levels_db: dict pol -> per-acquisition median noise [dB]. An acquisition whose noise in any
    channel departs from the across-stack median by more than tol_db is flagged (its LUT is unreliable)."""
    flagged = set()
    for p, v in noise_levels_db.items():
        v = np.asarray(v)
        flagged |= set(np.flatnonzero(np.abs(v - np.median(v)) > tol_db).tolist())
    return sorted(flagged)
