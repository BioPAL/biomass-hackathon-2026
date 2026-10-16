"""Radar viewing geometry on a lat/lon grid: heading, incidence angle, slope/aspect and local incidence angle.

All functions are numpy-only. Angles in degrees; azimuths clockwise from north.

Conventions
-----------
- heading: direction of satellite motion over ground.
- look azimuth: horizontal direction from the radar towards the illuminated ground
  (heading - 90 deg for a left-looking SAR, heading + 90 deg for right-looking).
- aspect: azimuth of the direction a slope faces (direction of steepest descent).
- local incidence angle (LIA): angle between the line of sight and the local surface normal.
  On flat ground LIA equals the incidence angle. A slope facing the radar has a smaller LIA
  (foreshortening, layover when LIA < 0), a slope facing away a larger one (shadow when LIA > 90).
"""

import numpy as np

KM_PER_DEG = 111.32


def _local_xy(lon, lat, lon0, lat0):
    """Approximate local east/north coordinates [km] around (lon0, lat0)."""
    x = (np.asarray(lon) - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * KM_PER_DEG
    return x, y


def heading_deg(footprint_lonlat, orbit_state: str) -> float:
    """Satellite heading from a product footprint polygon.

    The footprint of a SAR frame is a long parallelogram: its longest edges run along-track.
    The orbit state (ascending = moving north, descending = moving south) fixes the sign.
    """
    pts = np.asarray(footprint_lonlat, dtype=float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    lon0, lat0 = pts[:, 0].mean(), pts[:, 1].mean()
    x, y = _local_xy(pts[:, 0], pts[:, 1], lon0, lat0)
    edges = [(x[(i + 1) % len(x)] - x[i], y[(i + 1) % len(y)] - y[i]) for i in range(len(x))]
    dx, dy = max(edges, key=lambda e: np.hypot(*e))
    if (orbit_state.lower().startswith("a") and dy < 0) or (orbit_state.lower().startswith("d") and dy > 0):
        dx, dy = -dx, -dy
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def look_azimuth_deg(heading: float, side: str = "left") -> float:
    """Horizontal look direction (radar -> ground)."""
    return float((heading - 90.0) % 360 if side.lower().startswith("l") else (heading + 90.0) % 360)


def incidence_field(lon, lat, footprint_lonlat, look_az: float, inc_near: float, inc_far: float):
    """Incidence angle [deg] at each grid point, linear in ground range across the product footprint.

    Ground range is measured along the look direction; the near edge of the footprint gets
    ``inc_near`` and the far edge ``inc_far``. Points outside are clipped to the edges.
    """
    pts = np.asarray(footprint_lonlat, dtype=float)
    lon0, lat0 = pts[:, 0].mean(), pts[:, 1].mean()
    ux, uy = np.sin(np.radians(look_az)), np.cos(np.radians(look_az))
    fx, fy = _local_xy(pts[:, 0], pts[:, 1], lon0, lat0)
    proj_fp = fx * ux + fy * uy
    gx, gy = _local_xy(lon, lat, lon0, lat0)
    frac = np.clip((gx * ux + gy * uy - proj_fp.min()) / (proj_fp.max() - proj_fp.min()), 0, 1)
    return inc_near + frac * (inc_far - inc_near)


def slope_aspect(dem, pixel_deg_x: float, pixel_deg_y: float, lat0: float):
    """Slope [deg] and aspect [deg, azimuth the slope faces] of a north-up lat/lon DEM [m]."""
    dx = abs(pixel_deg_x) * KM_PER_DEG * 1000 * np.cos(np.radians(lat0))
    dy = abs(pixel_deg_y) * KM_PER_DEG * 1000
    d_row, d_col = np.gradient(np.asarray(dem, dtype=float))
    dz_east = d_col / dx
    dz_north = -d_row / dy  # rows increase southwards
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_north)))
    aspect = np.degrees(np.arctan2(-dz_east, -dz_north)) % 360  # direction of steepest descent
    return slope, aspect


def local_incidence(inc, slope, aspect, look_az: float):
    """Local incidence angle [deg].

    cos(LIA) = cos(s) cos(theta) - sin(s) sin(theta) cos(aspect - look_az)
    A slope facing the radar (aspect = look_az + 180) gives LIA = theta - s.
    """
    th, s = np.radians(inc), np.radians(slope)
    rel = np.radians(np.asarray(aspect) - look_az)
    c = np.cos(s) * np.cos(th) - np.sin(s) * np.sin(th) * np.cos(rel)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def kz_height_ratio(inc, lia):
    """First-order prediction of height_estimated / height_true if the inversion uses the flat-terrain
    vertical wavenumber while the true one scales with 1/sin(local incidence)."""
    return np.sin(np.radians(inc)) / np.sin(np.radians(np.clip(lia, 1, 179)))
