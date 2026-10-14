"""Metrics. With 7 acquisitions per stack every statistic is weak; the functions therefore report n, ranges and
difference-based correlations (which remove shared monotonic trends) next to slopes and their t-based confidence intervals."""
import numpy as np, pandas as pd
from scipy import stats


def _ok(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def pearson(x, y):
    x, y = _ok(x, y)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")


def linfit(x, y, alpha=0.05):
    """OLS slope of y on x with t-based (1-alpha) CI, Pearson r, p-value, Spearman rho, n."""
    x, y = _ok(x, y); n = len(x)
    out = dict(slope=np.nan, ci_low=np.nan, ci_high=np.nan, r=np.nan, p=np.nan, rho=np.nan, n=n)
    if n < 3 or np.ptp(x) == 0: return out
    lr = stats.linregress(x, y); t = stats.t.ppf(1 - alpha / 2, n - 2)
    out.update(slope=float(lr.slope), ci_low=float(lr.slope - t * lr.stderr), ci_high=float(lr.slope + t * lr.stderr), r=float(lr.rvalue), p=float(lr.pvalue))
    if np.ptp(y) > 0: out["rho"] = float(stats.spearmanr(x, y).correlation)
    return out


def diff_corr(x, y):
    """Correlation of first differences (consecutive acquisitions): insensitive to common trends such as a dry-down + phenology."""
    return pearson(np.diff(np.asarray(x, float)), np.diff(np.asarray(y, float)))


def pair_regression(G_abs, times, sm):
    """|coherence| of all pairs i<j regressed on temporal baseline [days] and |dSM| -> coefficients, standard errors, R2."""
    t = pd.to_datetime(list(times), utc=True); sm = np.asarray(sm, float); T = len(t)
    rows = [((t[j] - t[i]).total_seconds() / 86400.0, abs(sm[j] - sm[i]), float(G_abs[i, j])) for i in range(T) for j in range(i + 1, T)]
    A = np.array(rows, float); A = A[np.isfinite(A).all(axis=1)]; n = len(A)
    out = dict(b0=np.nan, b_dt=np.nan, b_dsm=np.nan, se_dt=np.nan, se_dsm=np.nan, r2=np.nan, r_dsm=np.nan, n=n)
    if n < 6 or np.ptp(A[:, 1]) == 0: return out
    X = np.c_[np.ones(n), A[:, 0], A[:, 1]]; y = A[:, 2]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ beta
    cov = (res @ res / (n - 3)) * np.linalg.pinv(X.T @ X); se = np.sqrt(np.clip(np.diag(cov), 0, None))
    r2 = float(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))) if np.ptp(y) > 0 else np.nan
    out.update(b0=float(beta[0]), b_dt=float(beta[1]), b_dsm=float(beta[2]), se_dt=float(se[1]), se_dsm=float(se[2]), r2=r2, r_dsm=pearson(A[:, 1], y))
    return out


def rmse(a, b):
    a, b = _ok(a, b); return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def circular_rmse(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float) + np.pi) % (2 * np.pi) - np.pi; d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")


def closure_check(obs, mod, thr_deg=1.0):
    """Observed vs modelled closure phases (all triplets): r, slope obs~mod, sign agreement where |mod| > thr, circular RMSE."""
    o, m = _ok(obs, mod); n = len(o)
    out = dict(r=np.nan, slope=np.nan, sign_agree=np.nan, rmse_deg=np.nan, obs_std_deg=np.nan, mod_std_deg=np.nan, n=n)
    if n < 5: return out
    out.update(obs_std_deg=float(np.degrees(np.std(o))), mod_std_deg=float(np.degrees(np.std(m))), rmse_deg=float(np.degrees(circular_rmse(o, m))))
    if np.ptp(m) > 0:
        out.update(r=pearson(m, o), slope=float(np.polyfit(m, o, 1)[0]))
    sel = np.abs(m) > np.radians(thr_deg)
    if sel.any(): out["sign_agree"] = float(np.mean(np.sign(o[sel]) == np.sign(m[sel])))
    return out
