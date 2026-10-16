"""PolSAR decompositions of the extracted stacks with PolSARpro (ported from polsar_stack.ipynb).

For every KEPT scene (manual review, core.scenes) of every stack of the last extraction,
on its common-area crop:

1. S = calibrated HH, HV, VH, VV scaled by sqrt(sin(theta)). theta is ONE
   incidence angle per swath (config.yaml swath_incidence_deg): the L1C
   amplitude is sqrt(beta0) and the scale brings it towards sigma0.
2. T3 = polsarpro S_to_T3(S), every element averaged with the SAME window and
   the SAME decimated grid as the MBMP coherence (processing.averaging), so
   T3, Pauli and H/A/alpha share the coherence cells and resolution. The
   average is normalised by the share of pixels with data in the window.
3. Pauli RGB = polsarpro pauli_rgb(T3)  (R = T22 double bounce, G = T33 volume, B = T11 surface).
4. polsarpro h_a_alpha(T3, boxcar_size=[1, 1]) (T3 is already averaged), with
   p_i = lambda_i / (lambda_1 + lambda_2 + lambda_3) and lambda_1 >= lambda_2 >= lambda_3:
       lambda_1..3      eigenvalues of T3
       alpha_1..3       scattering angle of the eigenvector of lambda_i [deg]
       entropy H        -sum p_i log3(p_i)
       anisotropy A     (p_2 - p_3) / (p_2 + p_3)
       alpha_averaged   sum p_i alpha_i: average of alpha_1, alpha_2, alpha_3 weighted by p_i
       span             total power T11 + T22 + T33
5. Dominant scattering mechanism of a cell (dominant_mechanism): the one of the
   largest eigenvalue lambda_1, from its alpha_1: surface < 40 deg <= volume <= 50 deg < double bounce.

Cells with less than MIN_VALID_FRACTION of their window inside the common area with data are NaN.

Output (<roi>/polsar/<stack>/)
------------------------------
    polsar_w<L>m.npz   t3          (Nb, 6, Naz, Nrg) complex64  [m11 m22 m33 m12 m13 m23]
                       pauli_rgb   (Nb, 3, Naz, Nrg) float32    0-1, R G B
                       lambdas, alphas (Nb, 3, Naz, Nrg) float32
                       entropy, anisotropy, alpha_averaged, span (Nb, Naz, Nrg) float32
                       valid_fraction (Naz, Nrg), lat, lon (Naz, Nrg), az_index, rg_index
    polsar_w<L>m.json  products, dates, swath, theta, window, spacings, ...
"""

import json
import time
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from polsarpro.decompositions import h_a_alpha
from polsarpro.util import S_to_T3, pauli_rgb

from ..core.io import POLARISATIONS, load_stack_info
from .averaging import AveragingGrid, describe_window, print_window_table
from .coherence import check_calibrated

T3_ELEMENTS = ("m11", "m22", "m33", "m12", "m13", "m23")
EIGEN_FLAGS = ("entropy", "anisotropy", "alpha", "lambdas", "alphas")
MIN_VALID_FRACTION = 0.5

# dominant mechanism = the one of lambda_1, classified by alpha_1 (Cloude-Pottier ranges)
MECHANISMS = ("surface", "volume", "double bounce")
ALPHA_SURFACE_MAX = 40.0   # alpha_1 < 40 deg: surface
ALPHA_VOLUME_MAX = 50.0    # 40 <= alpha_1 <= 50 deg: volume (dipole), > 50 deg: double bounce


def dominant_mechanism(alpha_1):
    """Index into MECHANISMS of every cell (-1 where alpha_1 is NaN)."""
    alpha_1 = np.asarray(alpha_1, dtype=np.float64)
    out = np.full(alpha_1.shape, -1, dtype=np.int8)
    out[alpha_1 < ALPHA_SURFACE_MAX] = 0
    out[(alpha_1 >= ALPHA_SURFACE_MAX) & (alpha_1 <= ALPHA_VOLUME_MAX)] = 1
    out[alpha_1 > ALPHA_VOLUME_MAX] = 2
    return out


def polsar_paths(out_dir, window):
    stem = f"polsar_w{window:g}m"
    return Path(out_dir) / f"{stem}.npz", Path(out_dir) / f"{stem}.json"


def incidence_scale(theta_deg):
    """sqrt(sin(|theta|)): L1C amplitude is sqrt(beta0), this converts it toward sigma0."""
    return np.float32(np.sqrt(np.sin(np.deg2rad(abs(theta_deg)))))


def sinclair(path, scale):
    """PolSARpro Sinclair dataset of a calibrated product tif (NaN -> 0) and its finite-pixel mask."""
    channels = {}
    finite = None
    with rasterio.open(path) as src:
        for band, pol in enumerate(POLARISATIONS, start=1):
            s = src.read(band)
            ok = np.isfinite(s)
            finite = ok if finite is None else finite & ok
            channels[pol] = s
    for pol in POLARISATIONS:
        channels[pol][~finite] = 0  # BPS: no-data set to 0 before averaging
    height, width = finite.shape
    dataset = xr.Dataset(
        {pol.lower(): (("y", "x"), (channels[pol] * scale).astype(np.complex64)) for pol in POLARISATIONS},
        coords={"y": np.arange(height), "x": np.arange(width)},
        attrs={"poltype": "S", "description": "Scattering matrix"},
    )
    return dataset, finite


def averaged_t3(scattering, finite, grid, invalid):
    """polsarpro S_to_T3 averaged on the coherence grid (NaN on invalid cells), as a T3 dataset."""
    t3_full = S_to_T3(scattering)
    finite_fraction = grid.average(finite.astype(np.float64))
    norm = np.where(finite_fraction > 0, 1.0 / np.maximum(finite_fraction, 1e-12), np.nan)
    norm[invalid] = np.nan
    data = {}
    for name in T3_ELEMENTS:
        element = np.asarray(t3_full[name].values)
        averaged = grid.average(element.astype(np.complex128 if np.iscomplexobj(element) else np.float64)) * norm
        dtype = np.complex64 if np.iscomplexobj(element) else np.float32
        data[name] = (("y", "x"), averaged.astype(dtype))
    del t3_full
    return xr.Dataset(
        data,
        coords={"y": np.arange(grid.shape_out[0]), "x": np.arange(grid.shape_out[1])},
        attrs={"poltype": "T3", "description": "Coherency matrix (3x3), coherence grid"},
    )


def decompose(t3):
    """Pauli RGB and H/A/alpha (eigenvalues, alphas) of an averaged T3 dataset."""
    rgb = pauli_rgb(t3).compute()
    eig = h_a_alpha(t3, boxcar_size=[1, 1], flags=EIGEN_FLAGS).compute()
    return {
        "pauli_rgb": np.asarray(rgb.transpose("band", "y", "x").values, dtype=np.float32),
        "lambdas": np.moveaxis(np.asarray(eig["lambdas"].values, dtype=np.float32), -1, 0),
        "alphas": np.moveaxis(np.asarray(eig["alphas"].values, dtype=np.float32), -1, 0),
        "entropy": np.asarray(eig["entropy"].values, dtype=np.float32),
        "anisotropy": np.asarray(eig["anisotropy"].values, dtype=np.float32),
        "alpha_averaged": np.asarray(eig["alpha"].values, dtype=np.float32),
        "span": np.asarray((t3["m11"] + t3["m22"] + t3["m33"]).values, dtype=np.float32),
    }


def compute_stack(info, out_dir, win, theta_deg):
    """T3, Pauli and H/A/alpha of every product of one stack -> out_dir/polsar_w<L>m.npz/.json."""
    t0 = time.time()
    folder = info["_dir"]
    check_calibrated(info)
    scale = incidence_scale(theta_deg)
    with rasterio.open(folder / "overlap_mask.tif") as src:
        inside = src.read(1).astype(bool)
    grid = AveragingGrid(inside.shape, win)
    print(f"  crop {inside.shape[0]} x {inside.shape[1]} px -> grid {grid.shape_out[0]} x {grid.shape_out[1]}, "
          f"theta {theta_deg} deg (scale {scale:.4f})")

    results = {key: [] for key in ("t3", "pauli_rgb", "lambdas", "alphas", "entropy", "anisotropy",
                                   "alpha_averaged", "span")}
    valid = inside
    for prod in info["products"]:
        scattering, finite = sinclair(folder / prod["file"], scale)
        valid = valid & finite
        invalid = grid.average((inside & finite).astype(np.float64)) < MIN_VALID_FRACTION
        t3 = averaged_t3(scattering, finite, grid, invalid)
        del scattering, finite
        out = decompose(t3)
        results["t3"].append(np.stack([t3[name].values.astype(np.complex64) for name in T3_ELEMENTS]))
        for key, value in out.items():
            results[key].append(value)
        print(f"    {prod['datetime'][:10]}  {prod['file'][:47]}...  "
              f"mean entropy {np.nanmean(out['entropy']):.2f}  alpha_averaged {np.nanmean(out['alpha_averaged']):.1f} deg")
        del t3, out

    valid_fraction = grid.average(valid.astype(np.float64)).astype(np.float32)
    with rasterio.open(folder / "latlon.tif") as src:
        lat, lon = grid.latlon(src.read())

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = polsar_paths(out_dir, win["window_m"])
    np.savez(npz_path, **{k: np.stack(v) for k, v in results.items()}, valid_fraction=valid_fraction,
             lat=lat, lon=lon, az_index=grid.az_index, rg_index=grid.rg_index)
    meta = {
        "stack": folder.name,
        "stack_id": info["stack_id"],
        "orbit_state": info.get("orbit_state"),
        "swath": info["swath"],
        "theta_deg": theta_deg,
        "scale": float(scale),
        "products": [p["id"] for p in info["products"]],
        "dates": [p["datetime"][:10] for p in info["products"]],
        "primary_index": next((k for k, p in enumerate(info["products"]) if p["coregistration_primary"]), None),
        "phase_calibration": [p.get("phase_calibration") for p in info["products"]],
        "t3_elements": list(T3_ELEMENTS),
        "alpha_averaged": "sum p_i alpha_i, p_i = lambda_i / sum(lambda): weighted average of alpha_1..3",
        "dominant_mechanism": f"of lambda_1 from alpha_1: surface < {ALPHA_SURFACE_MAX:g} <= volume <= "
                              f"{ALPHA_VOLUME_MAX:g} < double bounce [deg]",
        "pauli_rgb": "R = T22 |HH-VV|^2/2, G = T33 |HV+VH|^2/2, B = T11 |HH+VV|^2/2 (polsarpro pauli_rgb, q = 0.98)",
        "spacings": info["spacings"],
        "window": win,
        "grid_shape": list(grid.shape_out),
        "min_valid_fraction": MIN_VALID_FRACTION,
        "processing_seconds": round(time.time() - t0, 1),
    }
    json_path.write_text(json.dumps(meta, indent=1))
    print(f"  -> {npz_path}  ({time.time() - t0:.0f} s)")
    return npz_path


def _up_to_date(info, out_dir, window, theta_deg):
    npz_path, json_path = polsar_paths(out_dir, window)
    if not (npz_path.exists() and json_path.exists()):
        return False
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (meta.get("products") == [p["id"] for p in info["products"]]
            and meta.get("phase_calibration") == [p.get("phase_calibration") for p in info["products"]]
            and meta.get("theta_deg") == theta_deg
            and "alpha_averaged" in meta)  # results of older versions (key 'alpha', no span) are recomputed


def run_polsar(config, roi_paths, window, force=False):
    """PolSAR products of the kept scenes of every stack whose result is missing or stale."""
    todo = []
    for stack_dir in roi_paths.stack_dirs():
        info = load_stack_info(stack_dir, kept_only=True)
        if not info["products"]:
            print(f"  {stack_dir.name}: no extracted product, skipped")
            continue
        theta = config.incidence_deg(info["swath"])
        out_dir = roi_paths.polsar / stack_dir.name
        if force or not _up_to_date(info, out_dir, window, theta):
            todo.append((info, out_dir, describe_window(window, info["spacings"]), theta))
    if not todo:
        print(f"PolSAR ({window:g} m window) up to date in {roi_paths.polsar}")
        return []
    print_window_table([(info["_dir"].name, win) for info, _, win, _ in todo])
    done = []
    for info, out_dir, win, theta in todo:
        print(f"[{info['_dir'].name}]  {len(info['products'])} products, window {win['NW_a']} x {win['NW_r']} px")
        done.append(compute_stack(info, out_dir, win, theta))
    return done
