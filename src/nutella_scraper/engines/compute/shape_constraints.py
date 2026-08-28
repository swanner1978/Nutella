"""Geometric constructability gates for sagittal scraper profiles.

These checks reject curves the optimiser must not send to the loft. Physical
contact and collision stay in the existing engines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.engines.compute.shape_families import (
    BLADE_THICKNESS_MM,
    BLADE_WIDTH_MM,
    DEFAULT_SCRAPER_LENGTH_MM,
    SagittalFrame,
    SampledProfile,
)

CONTACT_TOLERANCE_MM = 0.5
# Absolute kink floor (manufacturing).
MIN_CURVATURE_RADIUS_MM = 2.5
# Mild-blade curvature cap, documented derivation (not fitted):
#   max sag = 8 % of the shortest search length (20 mm) → sag ≤ 1.6 mm
#   circle through chord L=20 mm: R = L² / (8 s) = 400 / 12.8 = 31.25 mm
#   MAX_CURVATURE_MM_INV = 1 / 31.25 = 0.032 mm^-1
MILD_SAG_FRACTION = 0.08
MAX_CURVATURE_MM_INV = 0.032
MIN_MILD_RADIUS_MM = 1.0 / MAX_CURVATURE_MM_INV
MAX_TURN_DEG = 55.0
LENGTH_REL_TOL = 0.25


@dataclass(frozen=True)
class GeometryValidity:
    valid: bool
    reasons: tuple[str, ...]
    length_mm: float
    min_curvature_radius_mm: float
    max_turn_deg: float
    self_intersects: bool
    monotonic_down: bool
    inside_envelope: bool
    thickness_mm: float = BLADE_THICKNESS_MM
    width_mm: float = BLADE_WIDTH_MM


def polyline_length_mm(points: NDArray[np.float64]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def min_menger_radius_mm(points: NDArray[np.float64]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return float("inf")
    radii: list[float] = []
    for i in range(1, len(pts) - 1):
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        c = pts[i + 1] - pts[i - 1]
        la = float(np.linalg.norm(a))
        lb = float(np.linalg.norm(b))
        lc = float(np.linalg.norm(c))
        if min(la, lb, lc) < 1e-9:
            continue
        area = 0.5 * float(np.linalg.norm(np.cross(a, b)))
        if area < 1e-12:
            radii.append(float("inf"))
            continue
        radii.append(la * lb * lc / (4.0 * area))
    if not radii:
        return float("inf")
    finite = [r for r in radii if np.isfinite(r)]
    return float(min(finite)) if finite else float("inf")


def max_turn_deg(points: NDArray[np.float64]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    tangents = np.diff(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-9)
    if len(tangents) < 2:
        return 0.0
    cos = np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0)
    return float(np.degrees(np.max(np.arccos(cos))))


def _segments_intersect_2d(
    a0: NDArray[np.float64],
    a1: NDArray[np.float64],
    b0: NDArray[np.float64],
    b1: NDArray[np.float64],
) -> bool:
    def _cross(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    da = a1 - a0
    db = b1 - b0
    den = _cross(da, db)
    if abs(den) < 1e-12:
        return False
    t = _cross(b0 - a0, db) / den
    u = _cross(b0 - a0, da) / den
    return 1e-6 < t < 1.0 - 1e-6 and 1e-6 < u < 1.0 - 1e-6


def sagittal_self_intersects(points: NDArray[np.float64]) -> bool:
    """True if the (y, r) polyline crosses itself. Graphs r(y) never do."""
    yr = np.column_stack(
        (
            np.asarray(points[:, 1], dtype=np.float64),
            np.asarray(points[:, 0], dtype=np.float64),
        )
    )
    n_seg = len(yr) - 1
    if n_seg < 3:
        return False
    for i in range(n_seg):
        for j in range(i + 2, n_seg):
            if _segments_intersect_2d(yr[i], yr[i + 1], yr[j], yr[j + 1]):
                return True
    return False


def y_is_monotonic_down(y_mm: NDArray[np.float64]) -> bool:
    y = np.asarray(y_mm, dtype=np.float64)
    if len(y) < 2:
        return False
    return bool(np.all(np.diff(y) <= 1e-9))


def inside_interior_envelope(
    profile: SampledProfile,
    frame: SagittalFrame,
    *,
    tolerance_mm: float = CONTACT_TOLERANCE_MM,
) -> bool:
    wall = np.asarray(frame.r_wall_at_y(profile.y_mm), dtype=np.float64)
    r = np.asarray(profile.r_mm, dtype=np.float64)
    if np.any(r < -1e-6):
        return False
    return bool(np.all(r <= wall + float(tolerance_mm)))


def validate_profile(
    profile: SampledProfile,
    frame: SagittalFrame,
    *,
    min_curvature_radius_mm: float = MIN_CURVATURE_RADIUS_MM,
    max_turn_deg_limit: float = MAX_TURN_DEG,
    requested_length_mm: float | None = None,
    max_curvature_mm_inv: float = MAX_CURVATURE_MM_INV,
) -> GeometryValidity:
    points = np.asarray(profile.points_mm, dtype=np.float64)
    length = polyline_length_mm(points)
    radius = min_menger_radius_mm(points)
    turn = max_turn_deg(points)
    crosses = sagittal_self_intersects(points)
    down = y_is_monotonic_down(profile.y_mm)
    inside = inside_interior_envelope(profile, frame)
    reasons: list[str] = []
    requested = float(
        requested_length_mm
        if requested_length_mm is not None
        else (profile.length_mm or DEFAULT_SCRAPER_LENGTH_MM)
    )
    requested = min(max(requested, 1.0), max(float(frame.useful_height_mm), 1.0))
    min_len = (1.0 - LENGTH_REL_TOL) * requested
    max_len = (1.0 + LENGTH_REL_TOL) * requested
    if length + 1e-6 < min_len:
        reasons.append("length_too_short")
    if length > max_len + 1e-6:
        reasons.append("length_exceeds_requested")
    if length > float(frame.useful_height_mm) * 1.05 + 1e-6:
        reasons.append("length_exceeds_useful_height")
    if not down:
        reasons.append("profile_climbs")
    if crosses:
        reasons.append("self_intersection")
    if not inside:
        reasons.append("outside_interior_envelope")
    mild_floor = max(float(min_curvature_radius_mm), float(1.0 / max_curvature_mm_inv))
    if np.isfinite(radius) and radius + 1e-9 < mild_floor:
        reasons.append("curvature_radius_too_small")
    if np.isfinite(radius) and radius > 1e-9:
        kappa = 1.0 / float(radius)
        if kappa > float(max_curvature_mm_inv) + 1e-12:
            reasons.append("curvature_too_high")
    if turn > float(max_turn_deg_limit) + 1e-9:
        reasons.append("turn_too_sharp")
    return GeometryValidity(
        valid=not reasons,
        reasons=tuple(reasons),
        length_mm=length,
        min_curvature_radius_mm=float(radius),
        max_turn_deg=turn,
        self_intersects=crosses,
        monotonic_down=down,
        inside_envelope=inside,
    )
