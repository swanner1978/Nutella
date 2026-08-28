"""Geometric fitting of profile families. This error is never the coverage score."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from nutella_scraper.engines.compute.shape_constraints import polyline_length_mm
from nutella_scraper.engines.compute.shape_families import (
    SagittalFrame,
    SampledProfile,
    ShapeFamily,
    clip_params_to_bounds,
    sample_profile,
)

FIT_SAMPLES = 32


def geometric_residuals(
    family: ShapeFamily,
    params: NDArray[np.float64],
    frame: SagittalFrame,
    *,
    sample_count: int = FIT_SAMPLES,
) -> NDArray[np.float64]:
    profile = sample_profile(family, params, frame, sample_count=sample_count)
    wall = np.asarray(frame.r_wall_at_y(profile.y_mm), dtype=np.float64)
    return np.asarray(profile.r_mm - wall, dtype=np.float64)


def geometric_errors(
    profile: SampledProfile,
    frame: SagittalFrame,
) -> tuple[float, float]:
    wall = np.asarray(frame.r_wall_at_y(profile.y_mm), dtype=np.float64)
    abs_err = np.abs(np.asarray(profile.r_mm, dtype=np.float64) - wall)
    if len(abs_err) == 0:
        return 0.0, 0.0
    return float(np.mean(abs_err)), float(np.max(abs_err))


def orthogonal_errors(
    profile: SampledProfile,
    frame: SagittalFrame,
) -> tuple[float, float]:
    """Mean / max distance from profile samples to the A0 meridian polyline."""
    samples = np.asarray(profile.points_mm, dtype=np.float64)
    target = np.asarray(frame.meridian_xyz_mm, dtype=np.float64)
    if len(samples) == 0 or len(target) < 2:
        return 0.0, 0.0
    dists: list[float] = []
    segs = target[1:] - target[:-1]
    for point in samples:
        delta = point[None, :] - target[:-1]
        seg_len2 = np.sum(segs * segs, axis=1)
        t = np.sum(delta * segs, axis=1) / np.maximum(seg_len2, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        closest = target[:-1] + t[:, None] * segs
        dists.append(float(np.min(np.linalg.norm(point[None, :] - closest, axis=1))))
    arr = np.asarray(dists, dtype=np.float64)
    return float(np.mean(arr)), float(np.max(arr))


def fit_family_geometrically(
    family: ShapeFamily,
    frame: SagittalFrame,
    *,
    sample_count: int = FIT_SAMPLES,
) -> NDArray[np.float64]:
    start = clip_params_to_bounds(family, family.default_params(frame), frame)
    bounds = family.bounds(frame)

    def _fun(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        return geometric_residuals(family, vector, frame, sample_count=sample_count)

    result = least_squares(
        _fun,
        start,
        bounds=(bounds[:, 0], bounds[:, 1]),
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=80,
    )
    return clip_params_to_bounds(family, np.asarray(result.x, dtype=np.float64), frame)


def profile_length_mm(profile: SampledProfile) -> float:
    return polyline_length_mm(profile.points_mm)
