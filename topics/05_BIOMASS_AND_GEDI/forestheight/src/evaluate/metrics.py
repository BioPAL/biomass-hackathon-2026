"""Agreement metrics between an estimate and a reference [m]."""

import numpy as np


def agreement_metrics(est: np.ndarray, ref: np.ndarray) -> dict:
    """Agreement of an estimate with a reference, NaN ignored.

    Parameters
    ----------
    est, ref : numpy.ndarray
        Estimated and reference heights [m], same shape.

    Returns
    -------
    dict
        n (valid pairs), bias = mean(est - ref) [m], mae [m], rmse [m], r (Pearson),
        rel_rmse = 100 * rmse / mean(ref) [%]. Metrics are NaN when fewer than 30 pairs.
    """
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    e, r = est[ok], ref[ok]
    if len(e) < 30:
        return dict(n=int(len(e)), bias=np.nan, mae=np.nan, rmse=np.nan, r=np.nan, rel_rmse=np.nan)
    d = e - r
    return dict(
        n=int(len(e)),
        bias=float(d.mean()),
        mae=float(np.abs(d).mean()),
        rmse=float(np.sqrt((d**2).mean())),
        r=float(np.corrcoef(e, r)[0, 1]) if e.std() > 0 and r.std() > 0 else np.nan,
        rel_rmse=float(100 * np.sqrt((d**2).mean()) / r.mean()),
    )
