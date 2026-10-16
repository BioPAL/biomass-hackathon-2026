"""Find a point near a station that lies well inside two stacks (e.g. two swaths of consecutive stacks).

Uses the LUT latitude/longitude grids already downloaded by run_stack.py (<runs>/<stack>/lut/*.nc).
For every LUT node of stack A within --max-km of the station it computes the distance to the grid edge
in A and, via the nearest node, in B. The margin of a node is the smaller of the two. Swath edges can hold
no-data pixels although the LUT grid covers them, so choose a margin of a few km.

Usage (on MAAP, from the package folder):
  python tools/find_overlap_point.py --runs ~/data/runs --stacks S1_T009_F118_G01_M04 S2_T009_F118_G01_M04 \
      --lat 41.99651 --lon 70.63916 --name Maydantal
"""
import argparse
import glob
import os

import numpy as np
import netCDF4
from scipy.spatial import cKDTree

R = 6371000.0


def grid(runs, stack):
    f = sorted(glob.glob(os.path.join(os.path.expanduser(runs), stack, "lut", "*.nc")))
    if not f:
        raise SystemExit(f"no LUT file in {runs}/{stack}/lut/ (run run_stack.py for this stack first)")
    with netCDF4.Dataset(f[0]) as d:
        g = d.groups["geometry"]
        return np.array(g.variables["latitude"][:], float), np.array(g.variables["longitude"][:], float)


def xy(la, lo, lat0, lon0):
    return np.radians(lo - lon0) * R * np.cos(np.radians(lat0)), np.radians(la - lat0) * R


def edge_m(la, lo, lat0, lon0):
    """Distance of every node to the grid edge in metres (index distance x local node spacing)."""
    x, y = xy(la, lo, lat0, lon0)
    dr = np.hypot(np.gradient(x, axis=0), np.gradient(y, axis=0))
    dc = np.hypot(np.gradient(x, axis=1), np.gradient(y, axis=1))
    nr, nc = la.shape
    i = np.arange(nr)[:, None]; j = np.arange(nc)[None, :]
    mr, mc = np.nanmedian(dr), np.nanmedian(dc)     # median spacing (robust against single distorted nodes)
    return np.minimum(np.minimum(i, nr - 1 - i) * mr, np.minimum(j, nc - 1 - j) * mc), x, y, max(mr, mc)


def main(a):
    laA, loA = grid(a.runs, a.stacks[0]); laB, loB = grid(a.runs, a.stacks[1])
    eA, xA, yA, spA = edge_m(laA, loA, a.lat, a.lon)
    eB, xB, yB, spB = edge_m(laB, loB, a.lat, a.lon)
    ok = np.isfinite(xA) & np.isfinite(yA) & (np.hypot(xA, yA) <= a.max_km * 1000)
    if not ok.any():
        raise SystemExit(f"stack {a.stacks[0]} has no LUT node within {a.max_km} km of the station")
    vb = np.isfinite(xB) & np.isfinite(yB)
    tree = cKDTree(np.c_[xB[vb], yB[vb]])
    dnn, k = tree.query(np.c_[xA[ok], yA[ok]])
    eBn = eB[vb][k]
    marg = np.where(dnn <= 1.5 * spB, np.minimum(eA[ok], eBn), 0.0)
    dist = np.hypot(xA[ok], yA[ok]); la, lo = laA[ok], loA[ok]
    print(f"station {a.name} {a.lat:.5f} {a.lon:.5f}; LUT node spacing ~{spA:.0f} m ({a.stacks[0]}), ~{spB:.0f} m ({a.stacks[1]})")
    print(f"largest common margin within {a.max_km:g} km: {marg.max() / 1000:.2f} km")
    best = None
    for m in (1000, 2000, 3000, 5000):
        sel = np.where(marg >= m)[0]
        if not len(sel):
            print(f"  margin >= {m / 1000:.0f} km: none"); continue
        b = sel[np.argmin(dist[sel])]
        print(f"  margin >= {m / 1000:.0f} km: nearest point {la[b]:.5f}, {lo[b]:.5f}  ({dist[b] / 1000:.2f} km from the station)")
        if m <= a.margin_m:
            best = b
    if best is None:
        best = int(np.argmax(marg))
    print(f"\nproposed point: --lat {la[best]:.5f} --lon {lo[best]:.5f}  (margin {marg[best] / 1000:.2f} km, {dist[best] / 1000:.2f} km from {a.name})")
    for s in a.stacks:
        print(f"python run_stack.py --stack {s} --lat {la[best]:.5f} --lon {lo[best]:.5f} --name {a.name}_overlap --only-point --half-m {a.half_m:g}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="~/data/runs"); p.add_argument("--stacks", nargs=2, required=True)
    p.add_argument("--lat", type=float, required=True); p.add_argument("--lon", type=float, required=True)
    p.add_argument("--name", default="point"); p.add_argument("--max-km", dest="max_km", type=float, default=25.0)
    p.add_argument("--margin-m", dest="margin_m", type=float, default=3000.0, help="wanted distance to both grid edges")
    p.add_argument("--half-m", dest="half_m", type=float, default=300.0)
    main(p.parse_args())
