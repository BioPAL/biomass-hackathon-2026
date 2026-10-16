"""Multi-baseline multi-polarimetric (MBMP) coherence of the extracted stacks (ported from compute_coherence.py).

Works stack by stack on the extraction output (no cross-stack combinations).
The products are saved with SKP + flattening already applied, so they are
used as they are (no phase correction here).

1. Window: averaging_window [m] on ground -> pixels with the BPS rule
   (processing.averaging), spacings from stack_info.json.
2. MBMP correlation (BPS port of MPMBCovarianceEstimation +
   Covariance4D2Correlation4D): channels HH, XP = (HV + VH) / 2, VV, each with
   all the acquisitions -> (3 Nb) x (3 Nb) matrix, block order pol-major:
   index = pol * Nb + acquisition.
3. Single-pol coherence of HH, HV, VH and VV separately, same window and grid:
   gamma_nm = <s_n s_m*> / sqrt(<|s_n|^2> <|s_m|^2>).

If the optional ``bps`` package is importable, the MBMP estimator is checked
against bps main_correlation_estimation_SR on a small block.

Output (<roi>/coherence/<stack>/)
---------------------------------
    coherence_w<L>m.npz   mbmp_corr   (Naz, Nrg, 3Nb, 3Nb) complex64
                          mbmp_power  (Naz, Nrg, 3Nb)      float32
                          coh_pol     (Naz, Nrg, 4, Nb, Nb) complex64 [HH HV VH VV]
                          power_pol   (Naz, Nrg, 4, Nb)     float32
                          valid_fraction (Naz, Nrg)  share of the window inside the common area with data
                          lat, lon (Naz, Nrg), az_index, rg_index (crop pixels of the cells)
    coherence_w<L>m.json  window, spacings, dates, primary, mean kz, ...
"""

import json
import time
from pathlib import Path

import numpy as np
import rasterio

from ..core.io import POLARISATIONS, load_stack_info
from ..extraction import lut
from .averaging import AveragingGrid, describe_window, print_window_table

MBMP_POLS = ["HH", "XP", "VV"]  # BPS FH/TOMO order, XP = (HV + VH) / 2
MIN_VALID_FRACTION = 0.5        # cells below this share of valid pixels are flagged in the json stats


def coherence_paths(out_dir, window):
    stem = f"coherence_w{window:g}m"
    return Path(out_dir) / f"{stem}.npz", Path(out_dir) / f"{stem}.json"


def correlation_from_covariance(cov):
    """Covariance4D2Correlation4D on a (Naz, Nrg, N, N) array."""
    diag = np.real(np.diagonal(cov, axis1=2, axis2=3))
    norm = 1.0 / np.sqrt(diag + np.spacing(1))
    return cov * norm[..., :, None] * norm[..., None, :], diag


def mbmp_covariance(channels, grid):
    """MPMBCovarianceEstimation: upper triangle of <c_i c_j*> + hermitian completion.

    channels: list of N (Naz, Nrg) complex arrays, NaN already set to 0.
    """
    n = len(channels)
    cov = np.zeros(grid.shape_out + (n, n), dtype=np.complex64)
    for i in range(n):
        ci = channels[i].astype(np.complex128)
        for j in range(i, n):
            prod = ci * np.conj(channels[j].astype(np.complex128))
            cov[:, :, i, j] = grid.average(prod).astype(np.complex64)
            if j != i:
                cov[:, :, j, i] = np.conj(cov[:, :, i, j])
    return cov


def check_calibrated(info):
    """Warn when products were saved without the phase corrections the extraction asks for."""
    wanted = info.get("phase_correction", {})
    expected = lut.correction_tag([k for k in ("skp", "flat") if wanted.get(k)])
    stale = [p["id"] for p in info["products"] if p.get("phase_calibration") != expected]
    if stale:
        print(f"  WARNING: {len(stale)} product(s) not saved with '{expected}' phase corrections "
              f"(re-run the extraction): {', '.join(s[:47] for s in stale)}")


def read_stack(info):
    """Calibrated complex data {pol: [Nb x (Naz, Nrg)]} (NaN -> 0) and the joint valid mask."""
    folder = info["_dir"]
    data = {pol: [] for pol in POLARISATIONS}
    valid = None
    for prod in info["products"]:
        with rasterio.open(folder / prod["file"]) as src:
            for band, pol in enumerate(POLARISATIONS, start=1):
                s = src.read(band)
                ok = np.isfinite(s)
                valid = ok if valid is None else valid & ok
                data[pol].append(s)
        print(f"    read {prod['file'][:47]}...")
    with rasterio.open(folder / "overlap_mask.tif") as src:
        valid &= src.read(1).astype(bool)
    for pol in POLARISATIONS:
        for s in data[pol]:
            s[~np.isfinite(s)] = 0  # BPS: no-data set to 0 before averaging
    return data, valid


def check_against_bps(channels_by_pol, win, sp):
    """Compare the port with bps main_correlation_estimation_SR on a small block (bps is optional)."""
    try:
        from bps.l2a_processor.tomo_fh.tomo_fh_commands import main_correlation_estimation_SR
    except ImportError:
        print("  (bps not importable: skipping the cross-check against the BPS implementation)")
        return None
    nb = len(channels_by_pol["HH"])
    n_az, n_rg = channels_by_pol["HH"][0].shape
    a0, r0 = n_az // 2, n_rg // 2
    sl = (slice(a0, min(a0 + 8 * win["NW_a"], n_az)), slice(r0, min(r0 + 12 * win["NW_r"], n_rg)))
    # BPS convention: {acq: {pol: (Rg, Az)}}, slant range spacing + incidence
    stack = {k: {pol: channels_by_pol[pol][k][sl].T.astype(np.complex128) for pol in MBMP_POLS}
             for k in range(nb)}
    inc = np.deg2rad(sp["incidence_mean_deg"])
    corr_bps, *_ = main_correlation_estimation_SR(
        stack, win["window_m"], sp["slant_range_m"], sp["azimuth_m"], inc, 435e6, 6e6)
    corr_bps = np.transpose(corr_bps, (3, 2, 0, 1))
    grid = AveragingGrid((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), win)
    chans = [channels_by_pol[pol][k][sl] for pol in MBMP_POLS for k in range(nb)]
    corr, _ = correlation_from_covariance(mbmp_covariance(chans, grid))
    err = float(np.max(np.abs(corr - corr_bps)))
    print(f"  cross-check vs BPS main_correlation_estimation_SR: max |diff| = {err:.2e}")
    return err


def compute_stack(info, out_dir, win):
    """MBMP + single-pol coherence of one stack -> out_dir/coherence_w<L>m.npz/.json."""
    t0 = time.time()
    folder = info["_dir"]
    sp = info["spacings"]
    check_calibrated(info)
    data, valid = read_stack(info)
    data["XP"] = [(hv + vh) / 2 for hv, vh in zip(data["HV"], data["VH"])]
    nb = len(info["products"])

    bps_err = check_against_bps(data, win, sp)

    grid = AveragingGrid(valid.shape, win)
    print(f"  crop {valid.shape[0]} x {valid.shape[1]} px -> grid {grid.shape_out[0]} x {grid.shape_out[1]}")

    print(f"  MBMP covariance ({len(MBMP_POLS) * nb} x {len(MBMP_POLS) * nb})...")
    channels = [data[pol][k] for pol in MBMP_POLS for k in range(nb)]
    mbmp_corr, mbmp_power = correlation_from_covariance(mbmp_covariance(channels, grid))
    del data["XP"], channels

    print("  single-pol coherence (HH, HV, VH, VV)...")
    coh_pol = np.zeros(grid.shape_out + (len(POLARISATIONS), nb, nb), dtype=np.complex64)
    power_pol = np.zeros(grid.shape_out + (len(POLARISATIONS), nb), dtype=np.float32)
    for p, pol in enumerate(POLARISATIONS):
        coh_pol[:, :, p], power_pol[:, :, p] = correlation_from_covariance(mbmp_covariance(data[pol], grid))
    del data

    valid_fraction = grid.average(valid.astype(np.float64)).astype(np.float32)
    with rasterio.open(folder / "latlon.tif") as src:
        lat, lon = grid.latlon(src.read())

    ids = [p["id"] for p in info["products"]]
    phase_lut = folder / "phase_lut.nc"
    kz_mean = lut.mean_wavenumbers(phase_lut, ids) if phase_lut.exists() else [float("nan")] * nb

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = coherence_paths(out_dir, win["window_m"])
    np.savez(
        npz_path,
        mbmp_corr=mbmp_corr.astype(np.complex64), mbmp_power=mbmp_power.astype(np.float32),
        coh_pol=coh_pol, power_pol=power_pol, valid_fraction=valid_fraction,
        lat=lat, lon=lon, az_index=grid.az_index, rg_index=grid.rg_index,
    )
    good = valid_fraction >= MIN_VALID_FRACTION
    meta = {
        "stack": folder.name,
        "stack_id": info["stack_id"],
        "orbit_state": info.get("orbit_state"),
        "products": ids,
        "dates": [p["datetime"][:10] for p in info["products"]],
        "repeat_cycles": [p["repeat_cycle"] for p in info["products"]],
        "primary_index": next((k for k, p in enumerate(info["products"]) if p["coregistration_primary"]), None),
        "kz_mean_rad_per_m": kz_mean,
        "mbmp_pols": MBMP_POLS,
        "mbmp_order": "index = pol * Nb + acquisition, pols HH, XP=(HV+VH)/2, VV",
        "coh_pols": POLARISATIONS,
        "phase_calibration": [p.get("phase_calibration") for p in info["products"]],
        "spacings": sp,
        "window": win,
        "grid_shape": list(grid.shape_out),
        "min_valid_fraction": MIN_VALID_FRACTION,
        "cells_valid": int(good.sum()),
        "mean_abs_mbmp_valid_cells": float(np.abs(mbmp_corr[good]).mean()) if good.any() else None,
        "bps_crosscheck_max_abs_diff": bps_err,
        "processing_seconds": round(time.time() - t0, 1),
    }
    json_path.write_text(json.dumps(meta, indent=1))
    print(f"  -> {npz_path}  ({time.time() - t0:.0f} s)")
    return npz_path


def _up_to_date(info, out_dir, window):
    npz_path, json_path = coherence_paths(out_dir, window)
    if not (npz_path.exists() and json_path.exists()):
        return False
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (meta.get("products") == [p["id"] for p in info["products"]]
            and meta.get("phase_calibration") == [p.get("phase_calibration") for p in info["products"]])


def run_coherence(roi_paths, window, force=False):
    """Coherence of every stack of the last extraction whose result is missing or stale.

    Delete <roi>/coherence/<stack>/coherence_w<L>m.npz (or pass force=True) to recompute.
    """
    todo = []
    for stack_dir in roi_paths.stack_dirs():
        info = load_stack_info(stack_dir)
        out_dir = roi_paths.coherence / stack_dir.name
        if len(info["products"]) < 2:
            print(f"  {stack_dir.name}: fewer than 2 extracted products, no coherence")
        elif force or not _up_to_date(info, out_dir, window):
            todo.append((info, out_dir, describe_window(window, info["spacings"])))
    if not todo:
        print(f"Coherence ({window:g} m window) up to date in {roi_paths.coherence}")
        return []
    print_window_table([(info["_dir"].name, win) for info, _, win in todo])
    done = []
    for info, out_dir, win in todo:
        print(f"[{info['_dir'].name}]  window {win['NW_a']} x {win['NW_r']} px")
        done.append(compute_stack(info, out_dir, win))
    return done
