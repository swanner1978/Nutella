"""Pose sampling and motion limits in scraper space — not the 608-point cloud.

Targets (interior_matrix_a0_0_90) measure coverage only. They are never
waypoints. Motion constraints are millimetres / degrees of the scraper pose.

Label: TRAJECTORY_MODEL_V2_A0_ONLY / POSE_GRAPH.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    MATRIX_ANGLE_END_DEG,
    MATRIX_ANGLE_START_DEG,
    MATRIX_SPACING_MM,
    WALL_RADIUS_MIN_MM,
    surface_axis_xz,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span
from nutella_scraper.engines.compute.trajectory_search import MAX_LATERAL_STEP

TRAJECTORY_MODEL = "POSE_GRAPH"
TARGET_MATRIX = "interior_matrix_a0_0_90"
DIAGNOSTIC_LABEL = "TRAJECTORY_MODEL_V2_A0_ONLY"

# Sampling and limits are derived from already-named constants, not invented
# millimetre values. Callers may override every field.
DEFAULT_MAX_CANDIDATE_POSES = 2000
# Old MAX_LATERAL_STEP counted cloud columns on the 5 mm lattice.
DEFAULT_MAX_LATERAL_STEP_MM = float(MAX_LATERAL_STEP) * float(MATRIX_SPACING_MM)
Y_EQUAL_EPS_MM = 1e-6


@dataclass(frozen=True)
class PoseSampleSpec:
    """One candidate scraper pose in (height, azimuth). Independent of cloud index."""

    pose_id: int
    y_mm: float
    azimuth_deg: float


@dataclass(frozen=True)
class PoseSamplingConfig:
    """Discrete pose lattice. Counts are independent of the 608 target points."""

    height_step_mm: float = float(MATRIX_SPACING_MM)
    azimuth_step_deg: float | None = None
    azimuth_start_deg: float = float(MATRIX_ANGLE_START_DEG)
    azimuth_end_deg: float = float(MATRIX_ANGLE_END_DEG)
    max_candidate_poses: int = DEFAULT_MAX_CANDIDATE_POSES


@dataclass(frozen=True)
class PoseMotionLimits:
    """Physical transition limits in scraper motion space.

    ``max_vertical_step_mm`` is the maximum downward displacement of the
    sampled tip height between two poses. It is not a cloud-row index.
    """

    max_vertical_step_mm: float
    max_lateral_step_mm: float
    max_rotation_step_deg: float
    opening_y_mm: float
    min_useful_y_mm: float
    opening_band_mm: float
    finish_band_mm: float
    y_decreasing_is_down: bool = True

    def is_opening_pose(self, y_mm: float) -> bool:
        return float(y_mm) >= float(self.opening_y_mm) - float(self.opening_band_mm)

    def reached_useful_depth(self, y_mm: float) -> bool:
        return float(y_mm) <= float(self.min_useful_y_mm) + float(self.finish_band_mm)


def mean_wall_radius_mm(surface: InteriorSurfaceReference) -> float:
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError("Interior surface has no vertices")
    axis = surface_axis_xz(vertices)
    radii = np.hypot(vertices[:, 0] - float(axis[0]), vertices[:, 2] - float(axis[1]))
    wall = radii >= WALL_RADIUS_MIN_MM
    if not np.any(wall):
        wall = np.ones(len(radii), dtype=np.bool_)
    return float(np.median(radii[wall]))


def azimuth_step_from_surface(
    surface: InteriorSurfaceReference,
    *,
    height_step_mm: float = float(MATRIX_SPACING_MM),
) -> float:
    """Arc angle that matches MATRIX_SPACING_MM on the mean wall radius."""
    radius = max(mean_wall_radius_mm(surface), 1.0)
    return float(np.degrees(float(height_step_mm) / radius))


def motion_limits_from_surface(
    surface: InteriorSurfaceReference,
    *,
    scraper_length_mm: float,
    sampling: PoseSamplingConfig | None = None,
    max_vertical_step_mm: float | None = None,
    max_lateral_step_mm: float | None = None,
    max_rotation_step_deg: float | None = None,
    opening_band_mm: float | None = None,
    finish_band_mm: float | None = None,
    min_useful_y_mm: float | None = None,
) -> PoseMotionLimits:
    """Build limits from jar span + scraper length + named lattice spacing.

    Vertical default = manufactured scraper length (overlap budget between poses).
    Lateral default = MAX_LATERAL_STEP * MATRIX_SPACING_MM (same 10 mm as the
    old 2-column rule, now in millimetres).
    Rotation default = angle subtended by that lateral step at the wall.
    Opening / finish bands = one height sample from useful top / floor.
    """
    cfg = sampling or PoseSamplingConfig()
    opening_y, lower_y, max_length = scraper_length_span(surface)
    length = min(float(scraper_length_mm), float(max_length))
    height_step = float(cfg.height_step_mm)
    lateral = (
        float(max_lateral_step_mm)
        if max_lateral_step_mm is not None
        else DEFAULT_MAX_LATERAL_STEP_MM
    )
    radius = max(mean_wall_radius_mm(surface), 1.0)
    rotation = (
        float(max_rotation_step_deg)
        if max_rotation_step_deg is not None
        else float(np.degrees(lateral / radius))
    )
    band = float(opening_band_mm) if opening_band_mm is not None else height_step
    finish = float(finish_band_mm) if finish_band_mm is not None else height_step
    return PoseMotionLimits(
        max_vertical_step_mm=(
            float(max_vertical_step_mm) if max_vertical_step_mm is not None else length
        ),
        max_lateral_step_mm=lateral,
        max_rotation_step_deg=rotation,
        opening_y_mm=float(opening_y),
        min_useful_y_mm=(
            float(min_useful_y_mm) if min_useful_y_mm is not None else float(lower_y)
        ),
        opening_band_mm=band,
        finish_band_mm=finish,
    )


def sample_pose_specs(
    surface: InteriorSurfaceReference,
    config: PoseSamplingConfig | None = None,
) -> tuple[PoseSampleSpec, ...]:
    """Uniform (y, azimuth) lattice. Does not use cloud row/col indices."""
    cfg = config or PoseSamplingConfig()
    opening_y, lower_y, _span = scraper_length_span(surface)
    height_step = max(float(cfg.height_step_mm), 1e-3)
    az_step = (
        float(cfg.azimuth_step_deg)
        if cfg.azimuth_step_deg is not None
        else azimuth_step_from_surface(surface, height_step_mm=height_step)
    )
    az_step = max(az_step, 1e-3)
    heights = _inclusive_range(float(opening_y), float(lower_y), -height_step)
    azimuths = _inclusive_range(
        float(cfg.azimuth_start_deg),
        float(cfg.azimuth_end_deg),
        az_step,
    )
    n_total = len(heights) * len(azimuths)
    if n_total > int(cfg.max_candidate_poses):
        heights, azimuths = _coarsen_lattice(
            heights,
            azimuths,
            int(cfg.max_candidate_poses),
        )
    specs: list[PoseSampleSpec] = []
    pose_id = 0
    for y_mm in heights:
        for az in azimuths:
            specs.append(
                PoseSampleSpec(pose_id=pose_id, y_mm=float(y_mm), azimuth_deg=float(az))
            )
            pose_id += 1
    return tuple(specs)


def azimuth_delta_deg(src_deg: float, dst_deg: float) -> float:
    delta = abs(float(dst_deg) - float(src_deg)) % 360.0
    return float(min(delta, 360.0 - delta))


def lateral_mm(
    src_origin: tuple[float, float, float],
    dst_origin: tuple[float, float, float],
) -> float:
    dx = float(dst_origin[0]) - float(src_origin[0])
    dz = float(dst_origin[2]) - float(src_origin[2])
    return float((dx * dx + dz * dz) ** 0.5)


def transition_allowed(
    src_y_mm: float,
    src_azimuth_deg: float,
    src_origin_mm: tuple[float, float, float],
    dst_y_mm: float,
    dst_azimuth_deg: float,
    dst_origin_mm: tuple[float, float, float],
    limits: PoseMotionLimits,
) -> bool:
    """True iff the scraper step is physically admissible.

    Uses sampled tip height (not cloud row) and realised origin for lateral
    millimetres. Climbing the height lattice is forbidden.
    """
    d_y = float(dst_y_mm) - float(src_y_mm)
    if d_y > Y_EQUAL_EPS_MM:
        return False
    drop = -d_y
    if drop > float(limits.max_vertical_step_mm) + Y_EQUAL_EPS_MM:
        return False
    if azimuth_delta_deg(src_azimuth_deg, dst_azimuth_deg) > float(
        limits.max_rotation_step_deg
    ) + 1e-9:
        return False
    if lateral_mm(src_origin_mm, dst_origin_mm) > float(limits.max_lateral_step_mm) + 1e-9:
        return False
    if drop <= Y_EQUAL_EPS_MM:
        same_az = azimuth_delta_deg(src_azimuth_deg, dst_azimuth_deg) <= 1e-9
        same_xy = lateral_mm(src_origin_mm, dst_origin_mm) <= 1e-9
        return not (same_az and same_xy)
    return True


def _inclusive_range(start: float, stop: float, step: float) -> tuple[float, ...]:
    if abs(step) < 1e-12:
        return (float(start),)
    n = int(np.floor(abs(stop - start) / abs(step) + 1e-9)) + 1
    values = [float(start + i * step) for i in range(n)]
    last = float(stop)
    if abs(values[-1] - last) > 1e-6:
        values.append(last)
    return tuple(values)


def _coarsen_lattice(
    heights: tuple[float, ...],
    azimuths: tuple[float, ...],
    cap: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if cap < 2:
        return (heights[0],), (azimuths[0],)
    n_h = max(2, len(heights))
    n_a = max(2, len(azimuths))
    while n_h * n_a > cap and (n_h > 2 or n_a > 2):
        if n_h >= n_a and n_h > 2:
            n_h -= 1
        elif n_a > 2:
            n_a -= 1
        else:
            break
    return _take_evenly(heights, n_h), _take_evenly(azimuths, n_a)


def _take_evenly(values: tuple[float, ...], count: int) -> tuple[float, ...]:
    if count >= len(values):
        return values
    if count <= 1:
        return (values[0],)
    index = np.linspace(0, len(values) - 1, count)
    return tuple(float(values[int(round(i))]) for i in index)


def limits_payload(limits: PoseMotionLimits) -> dict[str, float | bool]:
    return {
        "max_vertical_step_mm": float(limits.max_vertical_step_mm),
        "max_lateral_step_mm": float(limits.max_lateral_step_mm),
        "max_rotation_step_deg": float(limits.max_rotation_step_deg),
        "opening_y_mm": float(limits.opening_y_mm),
        "min_useful_y_mm": float(limits.min_useful_y_mm),
        "opening_band_mm": float(limits.opening_band_mm),
        "finish_band_mm": float(limits.finish_band_mm),
        "y_decreasing_is_down": bool(limits.y_decreasing_is_down),
    }
