"""Aggregate shots into ~500 m grid cells (and into spatial blocks for cross-validation).

Cells are blocks of `npx x npx` TOM pixels: id = (row // npx) * 100000 + (col // npx). With the 46 m pixel of the
Lope scene and 500 m cells, npx = 11.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

M_PER_DEG = 111_320.0


def pixels_per_cell(cell_m: float, dlat: float) -> int:
    return max(int(round(cell_m / (abs(dlat) * M_PER_DEG))), 1)


def cell_ids(row, col, npx: int) -> np.ndarray:
    return (np.asarray(row) // npx).astype(np.int64) * 100_000 + (np.asarray(col) // npx)


def block_ids(lat, lon, size_deg: float = 0.1) -> np.ndarray:
    """Spatial CV blocks (0.1 deg ~ 11 km)."""
    return np.floor(np.asarray(lat) / size_deg).astype(int) * 1000 + np.floor(np.asarray(lon) / size_deg).astype(int)


@dataclass
class CellIndex:
    """Which shots fall in which cell, for cells with at least `min_shots` shots."""

    idx: np.ndarray      # indices (into the shot table) of the shots that are kept, sorted by cell
    cells: np.ndarray    # unique cell ids, ascending
    inv: np.ndarray      # position of each kept shot's cell in `cells` (non-decreasing)
    n: np.ndarray        # shots per cell
    starts: np.ndarray   # offset of each cell's first shot in `idx`

    def __len__(self) -> int:
        return len(self.cells)


def group_cells(cell: np.ndarray, keep: np.ndarray, min_shots: int = 5) -> CellIndex:
    idx = np.flatnonzero(keep)
    cl = cell[idx]
    uniq, cnt = np.unique(cl, return_counts=True)
    good = uniq[cnt >= min_shots]
    sel = np.isin(cl, good)
    idx, cl = idx[sel], cl[sel]
    order = np.argsort(cl, kind="stable")
    idx, cl = idx[order], cl[order]
    cells, starts, n = np.unique(cl, return_index=True, return_counts=True)
    inv = np.repeat(np.arange(len(cells)), n)
    return CellIndex(idx=idx, cells=cells, inv=inv, n=n, starts=starts)


def mean_by_cell(arr: np.ndarray, ci: CellIndex) -> np.ndarray:
    """NaN-aware mean of `arr[ci.idx]` per cell; arr is (n_shots, ...) in any trailing shape."""
    a = np.asarray(arr)[ci.idx].astype("float64")
    ok = np.isfinite(a)
    s = np.add.reduceat(np.where(ok, a, 0.0), ci.starts, axis=0)
    c = np.add.reduceat(ok.astype("float64"), ci.starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out
