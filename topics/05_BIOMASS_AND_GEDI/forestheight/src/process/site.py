"""Assemble everything for one site on one grid: BIOMASS passes (with viewing geometry), LVIS, GEDI, land
cover and terrain. Also builds the equivalent site dictionaries for the workshop FH products.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..load.biomass import read_raster, scene_files
from ..load.context import build_context, cover_class
from ..load.gedi import read_gedi
from ..load.lidar import find_lvis
from ..load.workshop import to_grid
from .geometry import heading_deg, incidence_field, local_incidence, look_azimuth_deg, slope_aspect
from .grid import Grid, aggregate_lvis_to_biomass, native_resolution


def load_site(
    key, cfg, root: Path, swath_incidence=(26.5, 31.4), min_coverage=0.5, min_pass_coverage=0.2, log=print
) -> dict:
    """Everything for one site on one grid: scenes (+ geometry), LVIS, GEDI, land cover, terrain."""
    site_dir = Path(root) / "data" / key
    bbox = cfg["bbox"]
    grid = Grid(bbox, native_resolution(site_dir))
    S = {"key": key, "name": cfg["name"], "role": cfg["role"], "bbox": bbox, "grid": grid, "dir": site_dir}

    ctx = build_context(bbox, grid.shape, grid.transform, grid.crs, cache=site_dir / "context.npz")
    S["ctx"] = ctx
    S["cover"] = cover_class(ctx["tree_frac"], ctx["grass_frac"])
    S["slope_grid"], S["aspect_grid"] = slope_aspect(ctx["dem"], grid.res, grid.res, np.mean(bbox[1::2]))

    scenes = []
    for mp in sorted((site_dir / "scenes").glob("*.json")):
        m = json.loads(mp.read_text())
        hp, qp, _ = scene_files(site_dir, m["id"])
        if not hp.exists():
            continue
        h, q = grid.resample(hp), grid.resample(qp)
        head = heading_deg(m["footprint"], m["orbit"])
        look = look_azimuth_deg(head, m.get("look", "left"))
        inc = incidence_field(grid.lon, grid.lat, m["footprint"], look, *swath_incidence)
        lia = local_incidence(inc, S["slope_grid"], S["aspect_grid"], look)
        scenes.append(
            dict(
                id=m["id"],
                date=pd.Timestamp(m["date"]),
                track=m["track"],
                frame=m["frame"],
                orbit=m["orbit"],
                heading=head,
                look_az=look,
                h=h,
                q=q,
                inc=inc,
                lia=lia,
                label=f"{pd.Timestamp(m['date']):%d %b} {m['orbit'][:4]} T{m['track']:03d}",
            )
        )
    frames = sorted(scenes, key=lambda d: (d["date"], d["frame"]))

    lvis_path = find_lvis(site_dir, cfg["lvis"])
    S["lvis"] = None
    if lvis_path is not None:
        lh, ltr, lcrs = read_raster(lvis_path)
        S["lvis"], _ = aggregate_lvis_to_biomass(
            lh, ltr, lcrs, grid.shape, grid.transform, grid.crs, min_coverage=min_coverage, lvis_valid_mask=None
        )
    gp = site_dir / f"gedi_{key}.parquet"
    S["gedi_shots"], S["gedi"], S["gedi_n"] = None, None, None
    if gp.exists():
        g = read_gedi(gp)
        g = g[g.longitude.between(bbox[0], bbox[2]) & g.latitude.between(bbox[1], bbox[3]) & g.rh_98.notna()]
        S["gedi_shots"] = g
        S["gedi"], S["gedi_n"] = grid.points_to_grid(g.longitude, g.latitude, g.rh_98.to_numpy())
    # One satellite pass is delivered as several adjacent frames: merge them (same date, track and
    # direction, so no artificial seams), then keep only passes that cover enough of the LiDAR survey.
    passes = {}
    for f in frames:
        k = (f["date"], f["track"], f["orbit"])
        if k not in passes:
            passes[k] = dict(f, frames=[f["frame"]])
            passes[k]["label"] = f"{f['date']:%d %b} {f['orbit'][:4]} T{f['track']:03d}"
            continue
        P = passes[k]
        P["frames"].append(f["frame"])
        take = ~np.isfinite(P["h"]) & np.isfinite(f["h"])
        for key in ("h", "q", "inc", "lia"):
            P[key] = np.where(take, f[key], P[key])
    ref = np.isfinite(S["lvis"]) if S["lvis"] is not None else np.isfinite(S["cover"])
    for P in passes.values():
        P["lidar_coverage"] = float((np.isfinite(P["h"]) & ref).sum() / max(ref.sum(), 1))
    allp = sorted(passes.values(), key=lambda d: d["date"])
    S["all_passes"] = allp
    S["scenes"] = [P for P in allp if P["lidar_coverage"] >= min_pass_coverage]
    S["best"] = max(S["scenes"], key=lambda d: d["lidar_coverage"]) if S["scenes"] else None
    # map window: where the LiDAR survey and the kept BIOMASS passes overlap
    any_pass = np.zeros(grid.shape, bool)
    for P in S["scenes"]:
        any_pass |= np.isfinite(P["h"])
    m = any_pass & ref
    if m.any():
        rr, cc = np.where(m)
        pad = 10
        r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, grid.shape[0])
        c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, grid.shape[1])
    else:
        r0, r1, c0, c1 = 0, grid.shape[0], 0, grid.shape[1]
    S["win"] = (slice(r0, r1), slice(c0, c1))
    S["win_extent"] = [
        grid.lon[0, c0] - grid.res / 2,
        grid.lon[0, c1 - 1] + grid.res / 2,
        grid.lat[r1 - 1, 0] - grid.res / 2,
        grid.lat[r0, 0] + grid.res / 2,
    ]
    log(
        f"{cfg['name']}: {len(frames)} frames -> {len(allp)} passes, {len(S['scenes'])} kept "
        f"(>= {min_pass_coverage:.0%} of the LiDAR area); best: {S['best']['label'] if S['best'] else '-'}"
    )
    log(
        f"{cfg['name']}: grid {grid.shape}, {len(S['scenes'])} BIOMASS passes, "
        f"LVIS {'yes' if S['lvis'] is not None else 'MISSING'}, GEDI shots {0 if S['gedi_shots'] is None else len(S['gedi_shots']):,}"
    )
    return S


def site_passes(S, inv: pd.DataFrame, swath_incidence=(26.5, 31.4), min_pass_coverage=0.2, look="left") -> dict:
    """A copy of the site dictionary 'S' whose 'scenes' are the workshop passes (frames of the same date and
    direction merged), with the same viewing geometry fields as the L2A passes."""
    grid = S["grid"]
    W, So, E, N = S["bbox"]
    passes = {}
    for _, r in inv.iterrows():
        fp = np.asarray(r.footprint)
        if fp[:, 0].max() < W or fp[:, 0].min() > E or fp[:, 1].max() < So or fp[:, 1].min() > N:
            continue
        h = to_grid(r.path, grid)
        if not np.isfinite(h).any():
            continue
        head = heading_deg(r.footprint, r.orbit)
        lk = look_azimuth_deg(head, look)
        inc = incidence_field(grid.lon, grid.lat, r.footprint, lk, *swath_incidence)
        lia = local_incidence(inc, S["slope_grid"], S["aspect_grid"], lk)
        k = (r.date, r.orbit)
        if k not in passes:
            passes[k] = dict(
                id=r.scene,
                date=r.date,
                orbit=r.orbit,
                track=int(r.date.toordinal()),
                frame=0,
                frames=[r.scene],
                heading=head,
                look_az=lk,
                h=h,
                q=np.full(grid.shape, np.nan, "float32"),
                inc=inc,
                lia=lia,
                label=f"{r.date:%d %b %Y} {r.orbit[:4]}",
            )
            continue
        P = passes[k]
        P["frames"].append(r.scene)
        take = ~np.isfinite(P["h"]) & np.isfinite(h)
        for key, v in (("h", h), ("inc", inc), ("lia", lia)):
            P[key] = np.where(take, v, P[key])
    ref = np.isfinite(S["lvis"]) if S["lvis"] is not None else np.isfinite(S["cover"])
    for P in passes.values():
        P["lidar_coverage"] = float((np.isfinite(P["h"]) & ref).sum() / max(ref.sum(), 1))
    allp = sorted(passes.values(), key=lambda d: d["date"])
    W2 = dict(S)
    W2["all_passes"] = allp
    W2["scenes"] = [P for P in allp if P["lidar_coverage"] >= min_pass_coverage]
    W2["best"] = max(W2["scenes"], key=lambda d: d["lidar_coverage"]) if W2["scenes"] else None
    return W2


def mosaic_site(S, mos):
    """The mosaic as a one-pass site, so the multisite statistics accept it."""
    nan = np.full(S["grid"].shape, np.nan, "float32")
    P = dict(
        id="mosaic",
        date=pd.Timestamp("2026-01-01"),
        orbit="mixed",
        track=0,
        frame=0,
        frames=["mosaic"],
        heading=np.nan,
        look_az=np.nan,
        h=mos,
        q=nan,
        inc=nan,
        lia=nan,
        label="mosaic",
        lidar_coverage=float((np.isfinite(mos) & np.isfinite(S["lvis"])).sum() / max(np.isfinite(S["lvis"]).sum(), 1))
        if S["lvis"] is not None
        else np.nan,
    )
    return dict(S, scenes=[P], all_passes=[P], best=P)
