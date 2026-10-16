"""Error as a function of spatial scale, GEDI as referee (block statistics) and a geolocation check."""

import numpy as np
import pandas as pd

from ..process.aggregate import at_scale, best_gedi, block_mean, gedi_blocks
from .accuracy import FOREST
from .metrics import agreement_metrics


def scale_curve(S, product, ks=(1, 3, 5, 9, 15)) -> pd.DataFrame:
    """Forest error against LVIS as a function of the averaging window (median over passes).
    RMSE^2 = bias^2 + SD^2: the SD part is what averaging can remove, the bias part is systematic."""
    if S["lvis"] is None or not S["scenes"]:
        return pd.DataFrame()
    m = S["cover"] == FOREST
    rows = []
    for k in ks:
        T = at_scale(S, k)
        mets = [agreement_metrics(p["h"][m], T["lvis"][m]) for p in T["scenes"]]
        mets = [x for x in mets if x["n"] >= 30]
        if not mets:
            continue
        rmse = float(np.median([x["rmse"] for x in mets]))
        bias = float(np.median([x["bias"] for x in mets]))
        rows.append(
            dict(
                site=S["name"],
                product=product,
                window=k,
                size_m=int(round(k * S["grid"].res * 111320, -1)),
                rmse=rmse,
                bias=bias,
                sd=float(np.sqrt(max(rmse**2 - bias**2, 0))),
                r=float(np.median([x["r"] for x in mets])),
            )
        )
    return pd.DataFrame(rows)


def best_shift(h, ref, mask, max_shift=5):
    """Integer pixel shift of 'h' that maximises the correlation with 'ref' (geolocation check)."""
    best = (-2.0, 0, 0)
    r0 = np.nan
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            sh = np.roll(np.roll(h, dy, 0), dx, 1)
            ok = mask & np.isfinite(sh) & np.isfinite(ref)
            if ok.sum() < 1000:
                continue
            r = float(np.corrcoef(sh[ok], ref[ok])[0, 1])
            if dy == 0 and dx == 0:
                r0 = r
            if r > best[0]:
                best = (r, dy, dx)
    return dict(r_at_zero=r0, r_best=best[0], shift_rows=best[1], shift_cols=best[2])


def block_comparison(sites: dict, products: dict, ks=(1, 3, 5, 10), min_forest=0.8) -> pd.DataFrame:
    """Forest blocks of k x k pixels: every product against LVIS and against GEDI, and GEDI against LVIS.
    'area' says whether the block lies inside the LiDAR survey or outside it (GEDI only).
    Product metrics are the median over the passes of the product."""
    rows = []
    for key, S in sites.items():
        for k in ks:
            size = int(round(k * S["grid"].res * 111320, -1))
            forest = block_mean((S["cover"] == FOREST).astype(float), k, 0) >= min_forest
            L = block_mean(S["lvis"], k) if S["lvis"] is not None else None
            G, _ = gedi_blocks(S, k)
            Gb, _ = gedi_blocks(S, k, best_gedi(S["gedi_shots"]))
            inside = np.isfinite(L) if L is not None else np.zeros_like(forest)
            base = dict(site=S["name"], k=k, size_m=size)
            if L is not None:
                for gname, GG in (("GEDI (all shots)", G), ("GEDI (power beams, night)", Gb)):
                    rows.append(
                        dict(
                            base,
                            estimate=gname,
                            reference="LVIS",
                            area="LiDAR area",
                            **agreement_metrics(GG[forest], L[forest]),
                        )
                    )
            for pname, P in products.items():
                for ref, R, area in (
                    ("LVIS", L, "LiDAR area"),
                    ("GEDI", np.where(inside, G, np.nan), "LiDAR area"),
                    ("GEDI", np.where(~inside, G, np.nan), "outside LiDAR"),
                ):
                    if R is None:
                        continue
                    mets = [agreement_metrics(block_mean(p["h"], k)[forest], R[forest]) for p in P[key]["scenes"]]
                    mets = [m for m in mets if m["n"] >= 30]
                    if mets:
                        rows.append(
                            dict(
                                base,
                                estimate=pname,
                                reference=ref,
                                area=area,
                                n=int(np.median([m["n"] for m in mets])),
                                **{
                                    c: float(np.median([m[c] for m in mets]))
                                    for c in ("bias", "mae", "rmse", "r", "rel_rmse")
                                },
                            )
                        )
    return pd.DataFrame(rows)
