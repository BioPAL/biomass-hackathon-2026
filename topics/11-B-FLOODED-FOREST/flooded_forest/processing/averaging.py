"""Spatial averaging shared by the MBMP coherence and the PolSAR coherency matrix T3.

One window side L in metres on ground gives, per stack, the same pixel window
and the same decimated output grid for both products, so coherence, T3,
H/A/alpha and Pauli maps land on the same cells (same resolution).

BPS rule (covariance_options_struct_filler, bps.l2a_processor.tomo_fh)::

    NW = int(L / pixel_spacing_ground / 2) * 2 + 1,  subsampling = (NW - 1) / 2

ground range spacing = slant range spacing / sin(mean incidence) as in
main_covariance_estimation_SR; azimuth spacing measured on latlon.tif
(both written in stack_info.json "spacings" at extraction).
"""

import numpy as np
from scipy.sparse import csr_matrix


def bps_window(window_m, pixel_spacing_grd):
    """covariance_options_struct_filler: odd window size, half window and subsampling in pixels."""
    nw = int(window_m / pixel_spacing_grd / 2) * 2 + 1
    nw_mono = (nw - 1) // 2
    return nw, nw_mono, max(nw_mono, 1)


def describe_window(window_m, spacings):
    """Pixel window, subsampling and effective ground extent of a window_m window on one stack."""
    nw_r, mono_r, subs_r = bps_window(window_m, spacings["ground_range_m"])
    nw_a, mono_a, subs_a = bps_window(window_m, spacings["azimuth_m"])
    ext_az, ext_rg = nw_a * spacings["azimuth_m"], nw_r * spacings["ground_range_m"]
    return {
        "window_m": window_m,
        "NW_a": nw_a, "NW_r": nw_r, "NWmono_a": mono_a, "NWmono_r": mono_r,
        "subs_F_a": subs_a, "subs_F_r": subs_r,
        "window_ground_az_m": ext_az, "window_ground_rg_m": ext_rg,
        "aspect_rg_over_az": ext_rg / ext_az,
        "output_spacing_az_m": subs_a * spacings["azimuth_m"],
        "output_spacing_rg_m": subs_r * spacings["ground_range_m"],
        "looks": nw_a * nw_r,
    }


def print_window_table(rows):
    """rows: [(stack name, describe_window dict)]"""
    print("\n  stack                                          window (az x rg)       on ground (az x rg)   "
          "rg/az  looks  output grid")
    for name, win in rows:
        print(f"  {name:<46s} {win['NW_a']:3d} x {win['NW_r']:2d} px   "
              f"{win['window_ground_az_m']:7.0f} x {win['window_ground_rg_m']:5.0f} m   "
              f"{win['aspect_rg_over_az']:5.2f}  {win['looks']:5d}  "
              f"{win['output_spacing_az_m']:.0f} x {win['output_spacing_rg_m']:.0f} m")
    print()


def build_filtering_matrix(n_in, nw_mono, subs_f):
    """Sparse one-dimensional moving average + decimation, normalised by the samples inside the image."""
    x_out = np.arange(0, n_in, subs_f)
    taps = np.arange(-nw_mono, nw_mono + 1)
    col = (x_out[:, None] + taps[None, :]).ravel()
    row = np.repeat(np.arange(x_out.size), taps.size)
    ok = (col >= 0) & (col < n_in)
    f = csr_matrix((np.ones(ok.sum()), (row[ok], col[ok])), shape=(x_out.size, n_in))
    norm = 1.0 / np.asarray(f.sum(axis=1)).ravel()
    return csr_matrix(f.multiply(norm[:, None])), x_out


class AveragingGrid:
    """Boxcar average + decimation of (Naz, Nrg) images: Fa @ image @ Fr.T (BPS MPMBCovarianceEstimation)."""

    def __init__(self, shape, win):
        self.win = win
        self.fa, self.az_index = build_filtering_matrix(shape[0], win["NWmono_a"], win["subs_F_a"])
        fr, self.rg_index = build_filtering_matrix(shape[1], win["NWmono_r"], win["subs_F_r"])
        self.frt = fr.T.tocsc()
        self.shape_out = (self.az_index.size, self.rg_index.size)

    def average(self, image):
        """Average of a (Naz, Nrg) image (NaN must already be set to 0) on the output grid."""
        return self.fa @ image @ self.frt

    def latlon(self, latlon):
        """(2, Naz, Nrg) lat/lon of the window -> lat, lon at the output cell centres."""
        cells = latlon[:, self.az_index][:, :, self.rg_index]
        return cells[0], cells[1]
