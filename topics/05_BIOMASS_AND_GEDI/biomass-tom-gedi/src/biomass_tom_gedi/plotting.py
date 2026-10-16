"""Figures. Matplotlib only, Agg-safe (no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def obs_vs_pred_hexbin(observed, predicted, xlabel: str, ylabel: str, out: str | Path, gridsize: int = 30) -> None:
    """Plain observed (x) vs predicted (y) density plot, viridis."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    hb = ax.hexbin(observed, predicted, gridsize=gridsize, cmap="viridis", mincnt=1, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(hb, ax=ax, label="samples per hexagon")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def correlation_heatmap(M: pd.DataFrame, title: str, out: str | Path, vlim: float = 0.5) -> None:
    fig, ax = plt.subplots(figsize=(0.55 * M.shape[1] + 3, 0.38 * M.shape[0] + 2), layout="constrained")
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.3 else "black")
    fig.colorbar(im, ax=ax, shrink=0.6, label="correlation")
    ax.set_title(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def delta_r2_bar(summary: pd.DataFrame, level: str, out: str | Path, top: int = 15) -> None:
    """Change in mean R2 vs baseline for the best configs at one level."""
    d = summary[summary.level == level].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.5), layout="constrained")
    ax.barh(d.config, d.dR2_vs_base, color="#2a6f97", label="filtered")
    ax.scatter(d.dR2_random_control, d.config, color="#e4572e", zorder=3, s=18, label="random removal (control)")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("change in mean R² vs baseline")
    ax.set_title(f"Bright-profile filters, {level} level")
    ax.legend(frameon=False)
    fig.savefig(out, dpi=150)
    plt.close(fig)
