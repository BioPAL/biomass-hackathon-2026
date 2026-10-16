"""Optional: 3-D curtain of the TOM profile along one GEDI beam track, with GEDI RH98 drawn on top.

Uses the geolocation of a GEDI L1B granule (only `BEAMxxxx/geolocation/{latitude,longitude}_bin0` is read) to
get the track, reads the 3x3-pixel mean TOM profile along it, and renders one figure per polarization.
The GEDI RH98 comes from the shot table made by steps 02-03 (data/interim/shots.parquet).

    python scripts/curtain3d.py --config configs/local.yaml --granule /path/to/GEDI01_B_...h5 \
        --beam BEAM1011 --date 2022-09-09 --out results/figures
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import cm, colors  # noqa: E402

from biomass_tom_gedi.config import Config  # noqa: E402
from biomass_tom_gedi.tom import TOMProduct, db, extract_profiles, read_slice  # noqa: E402

warnings.filterwarnings("ignore")
log = logging.getLogger("curtain3d")
E_ = 150.0          # vertical exaggeration (z is drawn in km x E_)
HMAX = 100.0        # top of the height ruler (m)


def track_profiles(prod: TOMProduct, granule: Path, beam: str, step: int = 2):
    g = prod.grid
    with h5py.File(granule, "r") as f:
        la = f[f"{beam}/geolocation/latitude_bin0"][:]
        lo = f[f"{beam}/geolocation/longitude_bin0"][:]
    m = (np.isfinite(la) & np.isfinite(lo) & (la > g.lat[0]) & (la < g.lat[-1]) & (lo > g.lon[0]) & (lo < g.lon[-1]))
    la, lo = la[m][::step], lo[m][::step]
    if len(la) == 0:
        raise SystemExit(f"{beam} does not cross the TOM scene in this granule")
    R, C = g.row_col(la, lo)
    with prod.open_cube() as cube:
        prof = extract_profiles(cube, R, C, half=1)
        iHV = g.pols.index("HV") if "HV" in g.pols else 0
        k20 = g.height_index(4.0 + 20)
        STEP = 10
        rr0, rr1 = max(R.min() - 200, 0), min(R.max() + 200, g.nrows)
        cc0, cc1 = max(C.min() - 300, 0), min(C.max() + 300, g.ncols)
        base = db(read_slice(cube, slice(rr0, rr1, STEP), slice(cc0, cc1, STEP), k20, iHV))
    blat = g.lat0 + g.dlat * np.arange(rr0, rr1, STEP)[: base.shape[0]]
    blon = g.lon0 + g.dlon * np.arange(cc0, cc1, STEP)[: base.shape[1]]
    return la, lo, prof, base, blat, blon


def render(pol, prof, valid, x, y, s, dvec, nvec, zh, hmin, base, BX, BY, band, kx, ky, gd, grid, out, beam, date):
    ip = grid.pols.index(pol)
    P = db(prof[:, :, ip])
    P[~valid] = np.nan
    norm = colors.Normalize(*np.nanpercentile(P, [3, 99.5]))
    X = np.repeat(x[:, None], len(zh), 1)
    Y = np.repeat(y[:, None], len(zh), 1)
    Zc = np.repeat(zh[None, :], len(x), 0) / 1000 * E_
    rgba = cm.turbo(norm(P))
    rgba[np.isnan(P)] = (1, 1, 1, 0)
    along = lambda xx, yy: (xx - x[0]) * dvec[0] + (yy - y[0]) * dvec[1]  # noqa: E731
    perp = lambda xx, yy: (xx - x[0]) * nvec[0] + (yy - y[0]) * nvec[1]    # noqa: E731

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    bn = colors.Normalize(*np.nanpercentile(base, [2, 98]))
    brgba = cm.gray(bn(base))
    brgba[np.isnan(base) | ~band] = (1, 1, 1, 0)
    brgba[..., :3] = 0.35 + 0.65 * brgba[..., :3]
    pp = perp(BX, BY)
    fb = brgba.copy(); fb[pp > 0, 3] = 0                                       # far half of the map: behind the curtain
    ff = brgba.copy(); ff[pp <= 0, 3] = 0; ff[..., 3] *= 0.45                  # near half: over the curtain, see-through
    kw = dict(rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)
    ax.plot_surface(BX, BY, np.zeros_like(BX), facecolors=fb, zorder=0, **kw)
    ax.plot_surface(X, Y, Zc, facecolors=rgba, zorder=1, **kw)
    ax.plot_surface(BX, BY, np.zeros_like(BX), facecolors=ff, zorder=2, **kw)

    gx_all, gy_all = kx(gd.longitude.values), ky(gd.latitude.values)
    keep = (along(gx_all, gy_all) >= 0) & (along(gx_all, gy_all) <= s[-1]) & (np.abs(perp(gx_all, gy_all)) < 2.5)
    gx, gy, gz = gx_all[keep], gy_all[keep], gd.rh_98.values[keep] / 1000 * E_
    o = np.argsort(along(gx, gy)); gx, gy, gz = gx[o], gy[o], gz[o]
    brk = np.flatnonzero(np.diff(along(gx, gy)) > 1.0) + 1                     # break the line at gaps > 1 km
    gx, gy, gz = (np.insert(a.astype(float), brk, np.nan) for a in (gx, gy, gz))
    ax.plot(gx, gy, gz, "k-", lw=0.9, zorder=3, label=f"GEDI RH98 ({beam}, {date})")
    ax.plot(x, y, np.zeros_like(x), color="k", lw=0.8, ls=(0, (4, 3)), zorder=2)   # ground line on the curtain

    xr, yr = x[0] - 5.0 * dvec[0], y[0] - 5.0 * dvec[1]                        # height ruler, 5 km before the curtain
    ax.plot([xr, xr], [yr, yr], [hmin / 1000 * E_, HMAX / 1000 * E_], "k-", lw=1.5, zorder=4)
    for hh in (-50, 0, 50, 100):
        ax.plot([xr - dvec[0], xr + dvec[0]], [yr - dvec[1], yr + dvec[1]], [hh / 1000 * E_] * 2, "k-", lw=1.2, zorder=4)
        ax.text(xr - 1.8 * dvec[0], yr - 1.8 * dvec[1], hh / 1000 * E_, f"{hh:+d}" if hh else "0", fontsize=10, ha="right", va="center", zorder=5)
    ax.text(xr, yr, (HMAX + 42) / 1000 * E_, "height above\nground (m)", fontsize=10, ha="center")

    ax.set_proj_type("ortho")
    ax.view_init(elev=34, azim=np.degrees(np.arctan2(nvec[1], nvec[0])) + 24)   # view perpendicular to the track
    xl, yl = (BX[band].min(), BX[band].max()), (BY[band].min(), BY[band].max())
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(1.03 * hmin / 1000 * E_, 1.12 * HMAX / 1000 * E_)
    ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], (1.12 * HMAX - 1.03 * hmin) / 1000 * E_), zoom=1.3)
    ax.set_axis_off()
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="turbo"), ax=ax, shrink=0.45, pad=0.01, label=f"TOM {pol} backscatter (dB)")
    ax.set_title(f"BIOMASS TomoSAR: {pol} vertical profile along GEDI {beam} ({s[-1]:.0f} km), base map = HV at 20 m", fontsize=13)
    ax.legend(loc="upper left")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--granule", required=True, help="GEDI L1B .h5 granule (only the geolocation is read)")
    ap.add_argument("--beam", default="BEAM1011")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD of the granule, to select the matching shots")
    ap.add_argument("--pols", nargs="+", default=None, help="default: all polarizations")
    ap.add_argument("--out", default=None, help="output directory (default: <results_dir>/figures)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = Config.load(a.config)
    prod = TOMProduct(cfg["paths"]["tom_product"])
    grid = prod.grid
    with np.load(cfg.profiles_path) as d:
        z_ground = float(d["z_ground"])
    zr = grid.z - z_ground
    out_dir = Path(a.out) if a.out else cfg.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    la, lo, prof, base, blat, blon = track_profiles(prod, Path(a.granule), a.beam)
    valid = np.isfinite(prof).all(axis=(1, 2))
    log.info("track samples %d, valid columns %d", len(la), valid.sum())
    lat_ref, lon_ref = la[valid][0], lo[valid][0]
    kx = lambda lon_: (lon_ - lon_ref) * 111.32 * np.cos(np.radians(lat_ref))  # noqa: E731
    ky = lambda lat_: (lat_ - lat_ref) * 110.57                                # noqa: E731
    x, y = kx(lo), ky(la)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    vi = np.flatnonzero(valid)
    i0, i1 = vi[0], vi[-1]
    x, y, prof, valid, s = x[i0:i1 + 1], y[i0:i1 + 1], prof[i0:i1 + 1], valid[i0:i1 + 1], s[i0:i1 + 1] - s[i0]
    dvec = np.array([x[-1] - x[0], y[-1] - y[0]]); dvec /= np.hypot(*dvec)
    nvec = np.array([dvec[1], -dvec[0]])
    BX, BY = np.meshgrid(kx(blon), ky(blat))
    along = (BX - x[0]) * dvec[0] + (BY - y[0]) * dvec[1]
    perp = (BX - x[0]) * nvec[0] + (BY - y[0]) * nvec[1]
    band = (np.abs(perp) < 14) & (along > -9) & (along < s[-1] + 6)

    gd = pd.read_parquet(cfg.shots_path, columns=["latitude", "longitude", "time", "beam_name", "rh_98"])
    gd = gd[(gd.beam_name == a.beam) & (gd.time.astype(str).str[:10] == a.date)]
    log.info("GEDI shots overlaid: %d", len(gd))
    for pol in (a.pols or grid.pols):
        out = out_dir / f"curtain3d_{pol}.png"
        render(pol, prof, valid, x, y, s, dvec, nvec, zr, float(zr.min()), base, BX, BY, band, kx, ky, gd, grid, out, a.beam, a.date)
        log.info("saved %s", out)


if __name__ == "__main__":
    main()
