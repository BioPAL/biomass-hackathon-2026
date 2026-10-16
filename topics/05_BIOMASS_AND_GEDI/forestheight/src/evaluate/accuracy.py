"""Accuracy of BIOMASS forest height against LVIS / GEDI by landscape stratum, viewing geometry and date.

Errors are always estimate - reference [m] (negative = BIOMASS too low).
"""

import itertools
import warnings

import numpy as np
import pandas as pd

from ..process.geometry import kz_height_ratio
from .metrics import agreement_metrics

FOREST, MOSAIC, SAVANNA = 0, 1, 2  # classes of load.context.cover_class


def strata(S):
    """Boolean masks of the landscape strata used everywhere in the story."""
    slope, cover = S["ctx"]["slope"], S["cover"]
    return {
        "all": np.isfinite(cover),
        "forest, flat (<5°)": (cover == FOREST) & (slope < 5),
        "forest, moderate (5–15°)": (cover == FOREST) & (slope >= 5) & (slope < 15),
        "forest, steep (≥15°)": (cover == FOREST) & (slope >= 15),
        "savanna": cover == SAVANNA,
    }


def accuracy_table(sites) -> pd.DataFrame:
    """One row per site x scene x reference x stratum."""
    rows = []
    for S in sites:
        st = strata(S)
        for sc in S["scenes"]:
            for ref_name in ("LVIS", "GEDI"):
                ref = S["lvis"] if ref_name == "LVIS" else S["gedi"]
                if ref is None:
                    continue
                for name, m in st.items():
                    rows.append(
                        dict(
                            site=S["name"],
                            scene=sc["label"],
                            date=sc["date"],
                            orbit=sc["orbit"],
                            reference=ref_name,
                            stratum=name,
                            **agreement_metrics(sc["h"][m], ref[m]),
                        )
                    )
    return pd.DataFrame(rows)


def site_summary(acc: pd.DataFrame, reference="LVIS") -> pd.DataFrame:
    """Median over scenes of each metric, per site and stratum (robust to one bad date)."""
    a = acc[(acc.reference == reference) & (acc.n >= 30)]
    return (
        a.groupby(["site", "stratum"], sort=False)
        .agg(
            scenes=("scene", "nunique"),
            pixels=("n", "median"),
            bias=("bias", "median"),
            rmse=("rmse", "median"),
            rel_rmse=("rel_rmse", "median"),
            r=("r", "median"),
        )
        .reset_index()
    )


def repeatability(S, mask=None) -> pd.DataFrame:
    """Pairwise agreement between all scenes of a site (BIOMASS vs BIOMASS)."""
    sc = S["scenes"]
    m = np.ones(S["grid"].shape, bool) if mask is None else mask
    rows = []
    for a, b in itertools.combinations(range(len(sc)), 2):
        met = agreement_metrics(sc[b]["h"][m], sc[a]["h"][m])
        same = sc[a]["track"] == sc[b]["track"] and sc[a]["orbit"] == sc[b]["orbit"]
        rows.append(
            dict(
                site=S["name"],
                a=sc[a]["label"],
                b=sc[b]["label"],
                days=abs((sc[b]["date"] - sc[a]["date"]).days),
                geometry="same track"
                if same
                else ("same direction" if sc[a]["orbit"] == sc[b]["orbit"] else "opposite directions"),
                **met,
            )
        )
    return pd.DataFrame(rows)


def by_orbit(S):
    """Median height and mean LIA per orbit direction (pixel-wise median across the dates of that direction)."""
    out = {}
    for orb in sorted({s["orbit"] for s in S["scenes"]}):
        ss = [s for s in S["scenes"] if s["orbit"] == orb]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[orb] = dict(
                h=np.nanmedian(np.stack([s["h"] for s in ss]), axis=0),
                lia=np.nanmean(np.stack([s["lia"] for s in ss]), axis=0),
                inc=np.nanmean(np.stack([s["inc"] for s in ss]), axis=0),
                n=len(ss),
            )
    return out


def error_vs_lia(S, ref="lvis", bins=(0, 15, 20, 25, 30, 35, 40, 50, 90)) -> pd.DataFrame:
    """Forest-only BIOMASS error binned by local incidence angle, separately per orbit direction,
    with the first-order kz prediction for comparison."""
    R = S[ref]
    if R is None:
        return pd.DataFrame()
    rows = []
    for orb, d in by_orbit(S).items():
        m = (S["cover"] == FOREST) & np.isfinite(d["h"]) & np.isfinite(R) & np.isfinite(d["lia"])
        df = pd.DataFrame(
            {
                "lia": d["lia"][m],
                "err": d["h"][m] - R[m],
                "ref": R[m],
                "pred": R[m] * (kz_height_ratio(d["inc"][m], d["lia"][m]) - 1),
            }
        )
        df["bin"] = pd.cut(df.lia, list(bins))
        g = (
            df.groupby("bin", observed=True)
            .agg(
                n=("err", "size"),
                bias=("err", "mean"),
                rmse=("err", lambda e: float(np.sqrt(np.mean(e**2)))),
                kz_prediction=("pred", "median"),
                lia_mid=("lia", "median"),
            )
            .reset_index()
        )
        g.insert(0, "orbit", orb)
        g.insert(0, "site", S["name"])
        rows.append(g[g.n >= 30])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def asc_desc_difference(S, bins=(-40, -20, -10, -5, 5, 10, 20, 40)) -> pd.DataFrame:
    """Same pixels, two viewing directions: height difference (asc - desc) versus LIA difference (asc - desc).
    If slope errors come from radar geometry, the difference must follow the LIA difference; the forest is the same."""
    d = by_orbit(S)
    if not {"ascending", "descending"} <= set(d):
        return pd.DataFrame(), np.nan, 0
    a, b = d["ascending"], d["descending"]
    m = (S["cover"] == FOREST) & np.isfinite(a["h"]) & np.isfinite(b["h"])
    df = pd.DataFrame({"dlia": a["lia"][m] - b["lia"][m], "dh": a["h"][m] - b["h"][m]})
    df["bin"] = pd.cut(df.dlia, list(bins))
    g = (
        df.groupby("bin", observed=True)
        .agg(n=("dh", "size"), dh_mean=("dh", "mean"), dh_median=("dh", "median"), dlia_mid=("dlia", "median"))
        .reset_index()
    )
    g.insert(0, "site", S["name"])
    r = float(np.corrcoef(df.dlia, df.dh)[0, 1]) if len(df) > 30 else np.nan
    return g[g.n >= 30], r, len(df)


def short_vegetation(S, low=10.0) -> pd.DataFrame:
    """Precision/recall of BIOMASS 'short' (< low m) against LVIS, per land cover, median over scenes."""
    if S["lvis"] is None:
        return pd.DataFrame()
    rows = []
    for sc in S["scenes"]:
        for k, name in [(FOREST, "forest"), (MOSAIC, "mosaic"), (SAVANNA, "savanna")]:
            m = (S["cover"] == k) & np.isfinite(sc["h"]) & np.isfinite(S["lvis"])
            if m.sum() < 30:
                continue
            b, l = sc["h"][m] < low, S["lvis"][m] < low
            rows.append(
                dict(
                    site=S["name"],
                    scene=sc["label"],
                    cover=name,
                    pixels=int(m.sum()),
                    lvis_median=float(np.median(S["lvis"][m])),
                    biomass_median=float(np.median(sc["h"][m])),
                    pct_short_lvis=100 * l.mean(),
                    pct_short_biomass=100 * b.mean(),
                    precision=100 * (b & l).sum() / max(b.sum(), 1),
                    recall=100 * (b & l).sum() / max(l.sum(), 1),
                )
            )
    df = pd.DataFrame(rows)
    return df.groupby(["site", "cover"], sort=False).median(numeric_only=True).reset_index() if len(df) else df
