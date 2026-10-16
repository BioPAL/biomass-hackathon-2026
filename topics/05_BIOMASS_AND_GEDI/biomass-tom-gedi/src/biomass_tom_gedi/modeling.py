"""Ridge regression with spatial-block cross-validation.

Held-out folds are whole 0.1 deg blocks (GroupKFold), so scores are out-of-sample in space, not picked-after-the-fact
correlations. Predictors are standardised inside each fold; alpha is chosen per target by RidgeCV (inside the training
fold only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

DEFAULT_ALPHAS = np.logspace(-1, 4, 30)


def spatial_cv_predict(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
                       alphas: np.ndarray = DEFAULT_ALPHAS) -> np.ndarray | None:
    """Out-of-fold predictions for every column of Y. Returns None if there are too few samples/blocks."""
    if len(np.unique(groups)) < n_splits or len(X) < 50:
        return None
    Y = np.asarray(Y, dtype="float64")
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits).split(X, Y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-9
        m = RidgeCV(alphas=alphas, alpha_per_target=True).fit((X[tr] - mu) / sd, (Y[tr] - ym) / ys)
        pred[te] = m.predict((X[te] - mu) / sd) * ys + ym
    return pred


def score(Y: np.ndarray, pred: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Pearson r, R^2, RMSE per target (pooled over the held-out predictions)."""
    Y = np.asarray(Y, dtype="float64")
    r = np.array([np.corrcoef(pred[:, j], Y[:, j])[0, 1] for j in range(Y.shape[1])])
    sse = ((Y - pred) ** 2).sum(0)
    r2 = 1 - sse / ((Y - Y.mean(0)) ** 2).sum(0)
    return pd.DataFrame({"metric": names, "r_cv": r, "R2_cv": r2, "RMSE": np.sqrt(sse / len(Y)),
                         "mean": Y.mean(0), "std": Y.std(0)})


def max_abs_layer_corr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per target: the largest |Pearson r| over single (pol, height) layers; no model."""
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    return np.abs(Xz.T @ Yz / len(X)).max(0)
