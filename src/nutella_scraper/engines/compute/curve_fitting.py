"""B-spline fitting and reconstruction error metrics for InternalJarProfile."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splrep, splev


@dataclass(frozen=True)
class BSpline1D:
    """Radius-as-a-function-of-height B-spline (r = f(y))."""

    degree: int
    knots: tuple[float, ...]
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class BSpline2DLoop:
    """Closed planar curve in XZ (horizontal section, Y-up world)."""

    degree: int
    knots: tuple[float, ...]
    coefficients_x: tuple[float, ...]
    coefficients_z: tuple[float, ...]


@dataclass(frozen=True)
class CurveFitMetrics:
    max_error_mm: float
    rms_error_mm: float
    sample_count: int
    control_point_count: int


def fit_meridian_bspline(
    y_mm: np.ndarray,
    radius_mm: np.ndarray,
    *,
    max_error_mm: float = 0.12,
    degree: int = 3,
) -> tuple[BSpline1D, CurveFitMetrics]:
    y = np.asarray(y_mm, dtype=np.float64)
    r = np.asarray(radius_mm, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    r = r[order]
    if len(y) < degree + 1:
        degree = max(1, len(y) - 1)

    lo = 0.0
    hi = max(len(y) * 1e-4, 1e-9)
    best = splrep(y, r, k=degree, s=0.0)
    for _ in range(48):
        mid = (lo + hi) / 2.0
        candidate = splrep(y, r, k=degree, s=mid)
        fitted = np.asarray(splev(y, candidate), dtype=np.float64)
        max_err = float(np.max(np.abs(fitted - r)))
        if max_err <= max_error_mm:
            best = candidate
            lo = mid
        else:
            hi = mid

    fitted = np.asarray(splev(y, best), dtype=np.float64)
    residuals = fitted - r
    knots, coeffs, degree_out = best
    return (
        BSpline1D(
            degree=int(degree_out),
            knots=tuple(float(value) for value in knots),
            coefficients=tuple(float(value) for value in coeffs),
        ),
        CurveFitMetrics(
            max_error_mm=float(np.max(np.abs(residuals))),
            rms_error_mm=float(np.sqrt(np.mean(residuals**2))),
            sample_count=len(y),
            control_point_count=len(coeffs),
        ),
    )


def fit_top_contour_bspline(
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    max_error_mm: float = 0.12,
    degree: int = 3,
) -> tuple[BSpline2DLoop, CurveFitMetrics]:
    x = np.asarray(x_mm, dtype=np.float64)
    z = np.asarray(z_mm, dtype=np.float64)
    if len(x) < 4:
        raise ValueError("At least four section samples are required for top contour fitting")

    theta = np.arctan2(z, x)
    order = np.argsort(theta)
    x = x[order]
    z = z[order]
    points = np.column_stack([x, z])
    points = np.vstack([points, points[:1]])

    per = len(points) - 1
    smoothing = 0.0
    best_tck, _ = splprep(points.T, u=None, s=smoothing, per=True, k=min(degree, per - 1))
    lo = 0.0
    hi = max(per * 1e-4, 1e-9)
    for _ in range(48):
        mid = (lo + hi) / 2.0
        candidate_tck, _ = splprep(points.T, u=None, s=mid, per=True, k=min(degree, per - 1))
        u_fine = np.linspace(0.0, 1.0, len(points) - 1, endpoint=False)
        fitted = np.column_stack(splev(u_fine, candidate_tck))
        errors = np.linalg.norm(fitted - points[:-1], axis=1)
        max_err = float(np.max(errors))
        if max_err <= max_error_mm:
            best_tck = candidate_tck
            lo = mid
        else:
            hi = mid

    u_fine = np.linspace(0.0, 1.0, len(points) - 1, endpoint=False)
    fitted = np.column_stack(splev(u_fine, best_tck))
    errors = np.linalg.norm(fitted - points[:-1], axis=1)
    tck = best_tck
    knots, coeffs, degree_out = tck
    cx, cz = coeffs
    return (
        BSpline2DLoop(
            degree=int(degree_out),
            knots=tuple(float(value) for value in knots),
            coefficients_x=tuple(float(value) for value in cx),
            coefficients_z=tuple(float(value) for value in cz),
        ),
        CurveFitMetrics(
            max_error_mm=float(np.max(errors)),
            rms_error_mm=float(np.sqrt(np.mean(errors**2))),
            sample_count=len(points) - 1,
            control_point_count=len(cx),
        ),
    )


def evaluate_meridian_spline(
    spline: BSpline1D,
    y_mm: np.ndarray,
) -> np.ndarray:
    tck = (list(spline.knots), list(spline.coefficients), spline.degree)
    return np.asarray(splev(y_mm, tck), dtype=np.float64)


def evaluate_top_contour_spline(
    spline: BSpline2DLoop,
    *,
    sample_count: int = 128,
) -> np.ndarray:
    tck = (
        list(spline.knots),
        [list(spline.coefficients_x), list(spline.coefficients_z)],
        spline.degree,
    )
    u = np.linspace(0.0, 1.0, max(sample_count, 8), endpoint=False)
    return np.column_stack(splev(u, tck))


def circularity_score(x_mm: np.ndarray, z_mm: np.ndarray) -> float:
    radial = np.sqrt(np.asarray(x_mm, dtype=np.float64) ** 2 + np.asarray(z_mm, dtype=np.float64) ** 2)
    mean_radius = float(np.mean(radial))
    if mean_radius <= 1e-9:
        return 1.0
    variation = float(np.std(radial)) / mean_radius
    return max(0.0, 1.0 - variation)


def hausdorff_mm(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    forward = float(np.max(np.min(dists, axis=1)))
    backward = float(np.max(np.min(dists, axis=0)))
    return max(forward, backward)


def rms_mm(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    count = min(len(a), len(b))
    diffs = a[:count] - b[:count]
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
