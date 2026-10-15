"""'Straightened' view of the LVIS flight strip: a coordinate frame aligned with the strip and a sampler."""

import numpy as np


def strip_frame(S, bin_km=0.25, keep=0.5, step_km=None):
    """Coordinate frame aligned with the main LVIS flight strip.

    The LVIS survey was flown as long strips. We find the strip direction with a principal-component
    analysis of the LVIS pixel positions, keep the densest band across it, and return a sampler that
    'straightens' any grid array into a full rectangle (across-strip x along-strip).
    """
    g = S["grid"]
    m = np.isfinite(S["lvis"])
    lat0 = float(np.mean(S["bbox"][1::2]))
    kx, ky = 111.32 * np.cos(np.radians(lat0)), 111.32
    x, y = g.lon[m] * kx, g.lat[m] * ky
    c = np.array([x.mean(), y.mean()])
    w, v = np.linalg.eigh(np.cov(np.vstack([x - c[0], y - c[1]])))
    u = v[:, np.argmax(w)]  # along-strip unit vector
    if u[0] < 0:
        u = -u  # always run west -> east
    n = np.array([-u[1], u[0]])  # across-strip unit vector
    s = (x - c[0]) * u[0] + (y - c[1]) * u[1]
    t = (x - c[0]) * n[0] + (y - c[1]) * n[1]
    hist, edges = np.histogram(t, bins=np.arange(t.min(), t.max() + bin_km, bin_km))
    k = int(np.argmax(hist))
    lo = hi = k
    while lo > 0 and hist[lo - 1] >= keep * hist[k]:
        lo -= 1
    while hi < len(hist) - 1 and hist[hi + 1] >= keep * hist[k]:
        hi += 1
    t0, t1 = edges[lo], edges[hi + 1]
    inband = (t >= t0) & (t <= t1)
    s0, s1 = np.percentile(s[inband], [1, 99])
    step = step_km or g.res * 111.32 * 0.5
    ss = np.arange(s0, s1, step)
    tt = np.arange(t1, t0, -step)  # rows from one edge to the other
    SS, TT = np.meshgrid(ss, tt)
    X = c[0] + SS * u[0] + TT * n[0]
    Y = c[1] + SS * u[1] + TT * n[1]
    lon, lat = X / kx, Y / ky
    rows = np.floor((g.bbox[3] - lat) / g.res).astype(int)
    cols = np.floor((lon - g.bbox[0]) / g.res).astype(int)
    inside = (rows >= 0) & (rows < g.shape[0]) & (cols >= 0) & (cols < g.shape[1])

    def sample(arr):
        out = np.full(SS.shape, np.nan, dtype="float32")
        out[inside] = np.asarray(arr)[rows[inside], cols[inside]]
        return out

    def project(lon_pts, lat_pts):
        px, py = np.asarray(lon_pts) * kx - c[0], np.asarray(lat_pts) * ky - c[1]
        return px * u[0] + py * u[1], px * n[0] + py * n[1]

    azimuth = float(np.degrees(np.arctan2(u[0], u[1])) % 360)
    return dict(
        sample=sample,
        project=project,
        s=ss - s0,
        s_range=(s0, s1),
        t_range=(t0, t1),
        extent=[0, s1 - s0, 0, t1 - t0],
        length_km=float(s1 - s0),
        width_km=float(t1 - t0),
        azimuth=azimuth,
        lon=lon,
        lat=lat,
    )


def best_by_orbit(S):
    """Pass with the largest LiDAR coverage for each orbit direction."""
    out = {}
    for p in S["scenes"]:
        if p["orbit"] not in out or p["lidar_coverage"] > out[p["orbit"]]["lidar_coverage"]:
            out[p["orbit"]] = p
    return out
