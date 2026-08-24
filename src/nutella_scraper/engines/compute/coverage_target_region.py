"""LEGACY 90° face-centroid region — kept for saved-campaign compatibility.

Do not use as the simulation target. New coverage uses
``coverage_reference_matrix`` (5 mm interior samples, A0 → +45°).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.mesh_utils import face_areas
from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span

COVERAGE_TARGET_SURFACE = "interior_product_surface"
COVERAGE_TARGET_REGION = "quadrant_90_from_a0"
A0_REFERENCE_AZIMUTH_DEG = 0.0
COVERAGE_TARGET_AZIMUTH_SPAN_DEG = 90.0
WALL_RADIUS_MIN_MM = 5.0
SYNTHETIC_EVALUATION_SOURCE = "synthetic_evaluation_interior"


@dataclass(frozen=True)
class CoverageTargetRegion:
    """Faces and centroids of the 90° interior-wall quadrant."""

    coverage_target_surface: str
    coverage_target_region: str
    coverage_target_azimuth_range: tuple[float, float]
    a0_azimuth_deg: float
    azimuth_span_deg: float
    face_ids: tuple[int, ...]
    face_centroids_mm: tuple[tuple[float, float, float], ...]
    area_mm2: float
    face_count: int
    point_count: int
    y_min_mm: float
    y_max_mm: float
    source: str
    vertex_count: int
    total_face_count: int
    wall_radius_min_mm: float
    simulator_invoked: bool = False
    coverage_recomputed: bool = False
    uses_visual_stl: bool = False
    uses_synthetic_evaluation_cylinder: bool = False
    symmetry_multiplier_applied: bool = False


def progress_azimuth_deg(x: float, z: float, *, axis_x: float = 0.0, axis_z: float = 0.0) -> float:
    """0° = +X, 90° = −Z. Same convention as CoverageSimulator."""
    return float(np.mod(np.degrees(np.arctan2(-(float(z) - axis_z), float(x) - axis_x)), 360.0))


def surface_axis_xz(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(vertices) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return 0.5 * (mins[[0, 2]] + maxs[[0, 2]])


def azimuths_deg(
    centroids: NDArray[np.float64],
    axis_xz: NDArray[np.float64],
) -> NDArray[np.float64]:
    dx = centroids[:, 0] - float(axis_xz[0])
    dz = centroids[:, 2] - float(axis_xz[1])
    raw = np.rad2deg(np.arctan2(-dz, dx))
    return np.mod(raw, 360.0)


def azimuth_in_quadrant_mask(
    azimuths: NDArray[np.float64],
    *,
    start_deg: float,
    span_deg: float = COVERAGE_TARGET_AZIMUTH_SPAN_DEG,
) -> NDArray[np.bool_]:
    """True when azimuth lies in [start, start+span] wrapping at 360°."""
    delta = np.mod(np.asarray(azimuths, dtype=np.float64) - float(start_deg), 360.0)
    return (delta >= -1e-9) & (delta <= float(span_deg) + 1e-9)


def wall_and_useful_mask(
    centroids: NDArray[np.float64],
    axis_xz: NDArray[np.float64],
    *,
    y_min_mm: float,
    y_max_mm: float,
    wall_radius_min_mm: float = WALL_RADIUS_MIN_MM,
) -> NDArray[np.bool_]:
    radii = np.hypot(
        centroids[:, 0] - float(axis_xz[0]),
        centroids[:, 2] - float(axis_xz[1]),
    )
    useful_y = (centroids[:, 1] >= float(y_min_mm) - 1e-3) & (
        centroids[:, 1] <= float(y_max_mm) + 1e-3
    )
    return useful_y & (radii >= float(wall_radius_min_mm) - 1e-9)


def build_coverage_target_region(
    surface: InteriorSurfaceReference,
    *,
    a0_azimuth_deg: float = A0_REFERENCE_AZIMUTH_DEG,
    span_deg: float = COVERAGE_TARGET_AZIMUTH_SPAN_DEG,
) -> CoverageTargetRegion:
    """Select interior-wall faces in the A0 → A0+90° quadrant. No simulation."""
    mesh = surface.to_trimesh()
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    areas = face_areas(mesh)
    centroids = vertices[faces].mean(axis=1)
    axis = surface_axis_xz(vertices)
    opening_y, lower_y, _span = scraper_length_span(surface)
    az = azimuths_deg(centroids, axis)
    wall = wall_and_useful_mask(
        centroids,
        axis,
        y_min_mm=lower_y,
        y_max_mm=opening_y,
    )
    in_quad = wall & azimuth_in_quadrant_mask(
        az, start_deg=float(a0_azimuth_deg), span_deg=float(span_deg)
    )
    ids = tuple(int(i) for i in np.flatnonzero(in_quad))
    cents = tuple(
        (float(centroids[i, 0]), float(centroids[i, 1]), float(centroids[i, 2]))
        for i in ids
    )
    end_deg = float(np.mod(float(a0_azimuth_deg) + float(span_deg), 360.0))
    source = str(surface.source)
    return CoverageTargetRegion(
        coverage_target_surface=COVERAGE_TARGET_SURFACE,
        coverage_target_region=COVERAGE_TARGET_REGION,
        coverage_target_azimuth_range=(float(a0_azimuth_deg), end_deg),
        a0_azimuth_deg=float(a0_azimuth_deg),
        azimuth_span_deg=float(span_deg),
        face_ids=ids,
        face_centroids_mm=cents,
        area_mm2=float(sum(areas[i] for i in ids)),
        face_count=len(ids),
        point_count=len(cents),
        y_min_mm=float(lower_y),
        y_max_mm=float(opening_y),
        source=source,
        vertex_count=int(len(vertices)),
        total_face_count=int(len(faces)),
        wall_radius_min_mm=float(WALL_RADIUS_MIN_MM),
        uses_synthetic_evaluation_cylinder=source == SYNTHETIC_EVALUATION_SOURCE,
    )


def region_to_payload(region: CoverageTargetRegion) -> dict[str, object]:
    return {
        "coverage_target_surface": region.coverage_target_surface,
        "coverage_target_region": region.coverage_target_region,
        "coverage_target_azimuth_range": list(region.coverage_target_azimuth_range),
        "a0_azimuth_deg": region.a0_azimuth_deg,
        "azimuth_span_deg": region.azimuth_span_deg,
        "face_ids": list(region.face_ids),
        "points_mm": [list(p) for p in region.face_centroids_mm],
        "area_mm2": region.area_mm2,
        "face_count": region.face_count,
        "point_count": region.point_count,
        "y_min_mm": region.y_min_mm,
        "y_max_mm": region.y_max_mm,
        "source": region.source,
        "interior_source": SOURCE_INTERIOR_PRODUCT_SURFACE,
        "vertex_count": region.vertex_count,
        "total_face_count": region.total_face_count,
        "wall_radius_min_mm": region.wall_radius_min_mm,
        "simulator_invoked": False,
        "coverage_recomputed": False,
        "uses_visual_stl": False,
        "uses_synthetic_evaluation_cylinder": region.uses_synthetic_evaluation_cylinder,
        "symmetry_multiplier_applied": False,
    }
