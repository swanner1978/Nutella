"""Interior-envelope reference point matrix for future coverage campaigns.

A0 (yellow scraper) is a visual meridian only. It is not this matrix.

Azimuth convention (same as CoverageSimulator progress):
  0°  = +X = A0 rest meridian
  90° = −Z = engine-positive / left when looking at A0 from inside (Y up)
  270° = +Z = engine-negative / right from inside

Reference zone (viewer white point cloud):
  0°  = A0 rest meridian (+X, yellow horizontal in top view)
  90° = engine-positive (−Z, white vertical axis in top view)
  span = 90° = one quadrant of the interior circumference
  height = full useful interior span (floor → opening)

Points lie on InteriorSurfaceReference, spaced ≈ 5 mm along the interior
meridian (wall + bottom fillet + floor) and along the wall arc (r·dθ).
Does not run evaluate_candidate, collision, or the simulation proximity
engine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray

from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.mesh_utils import face_areas
from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span

COVERAGE_TARGET_SURFACE = "interior_product_surface"
COVERAGE_TARGET_REGION = "interior_matrix_a0_0_90"
LEGACY_A0_QUADRANT_REGION = "quadrant_90_from_a0"

A0_MERIDIAN_AZIMUTH_DEG = 0.0
MATRIX_ANGLE_START_DEG = 0.0
MATRIX_ANGLE_END_DEG = 90.0
AZIMUTH_POSITIVE_SENSE = "toward_-Z"
AZIMUTH_NEGATIVE_SENSE = "toward_+Z"
REFERENCE_ZONE_SIDE = "a0_to_positive_90_quadrant"
REFERENCE_ZONE_SPAN_DEG = MATRIX_ANGLE_END_DEG - MATRIX_ANGLE_START_DEG
MATRIX_SPACING_MM = 5.0
WALL_RADIUS_MIN_MM = 5.0
FLOOR_RADIUS_MIN_MM = 0.5
ON_SURFACE_TOLERANCE_MM = 0.5
SECTION_SEARCH_STEP_MM = 0.25
Y_EDGE_MERGE_MM = 0.5
ROW_AZIMUTH_TOLERANCE_DEG = 1.0

# 0° = +X, 90° = −Z
def progress_azimuth_deg(x: float, z: float, *, axis_x: float = 0.0, axis_z: float = 0.0) -> float:
    return float(np.mod(np.degrees(np.arctan2(-(float(z) - axis_z), float(x) - axis_x)), 360.0))


def surface_axis_xz(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(vertices) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return 0.5 * (mins[[0, 2]] + maxs[[0, 2]])


def azimuths_deg(
    points: NDArray[np.float64],
    axis_xz: NDArray[np.float64],
) -> NDArray[np.float64]:
    dx = points[:, 0] - float(axis_xz[0])
    dz = points[:, 2] - float(axis_xz[1])
    return np.mod(np.rad2deg(np.arctan2(-dz, dx)), 360.0)


def azimuth_in_zone_mask(
    azimuths: NDArray[np.float64],
    *,
    start_deg: float = A0_MERIDIAN_AZIMUTH_DEG,
    span_deg: float = REFERENCE_ZONE_SPAN_DEG,
) -> NDArray[np.bool_]:
    delta = np.mod(np.asarray(azimuths, dtype=np.float64) - float(start_deg), 360.0)
    return (delta >= -1e-9) & (delta <= float(span_deg) + 1e-9)


def _direction_xz(az_deg: float) -> NDArray[np.float64]:
    rad = np.deg2rad(float(az_deg))
    return np.array([np.cos(rad), -np.sin(rad)], dtype=np.float64)


def _cross2(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _horizontal_contour(mesh: trimesh.Trimesh, y_mm: float) -> NDArray[np.float64] | None:
    section = mesh.section(
        plane_origin=[0.0, float(y_mm), 0.0],
        plane_normal=[0.0, 1.0, 0.0],
    )
    if section is None:
        for delta in (0.25, -0.25, 0.5, -0.5, 1.0, -1.0):
            section = mesh.section(
                plane_origin=[0.0, float(y_mm + delta), 0.0],
                plane_normal=[0.0, 1.0, 0.0],
            )
            if section is not None:
                break
    if section is None:
        return None
    discrete = getattr(section, "discrete", None)
    if discrete:
        poly = max(
            (np.asarray(part, dtype=np.float64) for part in discrete if len(part) >= 2),
            key=lambda arr: float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))),
            default=None,
        )
        return poly
    vertices = np.asarray(section.vertices, dtype=np.float64)
    if len(vertices) < 2:
        return None
    return vertices


def _ray_contour_hit(
    contour: NDArray[np.float64],
    *,
    axis_x: float,
    axis_z: float,
    az_deg: float,
) -> NDArray[np.float64] | None:
    origin = np.array([axis_x, axis_z], dtype=np.float64)
    direction = _direction_xz(az_deg)
    pts = np.asarray(contour, dtype=np.float64)
    best_t: float | None = None
    best: NDArray[np.float64] | None = None
    n = len(pts)
    if n < 2:
        return None
    closed = bool(np.linalg.norm(pts[0] - pts[-1]) < 1e-6)
    last = n - 1 if closed else n
    for i in range(last):
        a = pts[i]
        b = pts[(i + 1) % n]
        a2 = np.array([a[0], a[2]], dtype=np.float64)
        b2 = np.array([b[0], b[2]], dtype=np.float64)
        edge = b2 - a2
        det = _cross2(direction, edge)
        if abs(det) < 1e-12:
            continue
        w = a2 - origin
        t = _cross2(w, edge) / det
        s = _cross2(w, direction) / det
        if t <= 1e-6 or s < -1e-6 or s > 1.0 + 1e-6:
            continue
        if best_t is None or t < best_t:
            best_t = t
            y_hit = float(a[1] + s * (b[1] - a[1]))
            hit_xz = origin + t * direction
            best = np.array([hit_xz[0], y_hit, hit_xz[1]], dtype=np.float64)
    return best


def _y_samples(y_min: float, y_max: float, spacing: float) -> NDArray[np.float64]:
    """Include both useful edges; step ``spacing`` in between.

    If the remaining height is not a multiple of ``spacing``, the last row is
    still placed on ``y_max`` so a several-millimetre empty band cannot remain.
    """
    lo = float(y_min)
    hi = float(y_max)
    step = float(spacing)
    if hi - lo <= 1e-6:
        return np.array([lo], dtype=np.float64)
    samples: list[float] = [lo]
    nxt = lo + step
    while nxt < hi - 1e-9:
        samples.append(float(nxt))
        nxt += step
    if hi - samples[-1] > Y_EDGE_MERGE_MM:
        samples.append(hi)
    else:
        samples[-1] = hi
    return np.asarray(samples, dtype=np.float64)


def _row_at_y(
    mesh: trimesh.Trimesh,
    *,
    axis: NDArray[np.float64],
    y_mm: float,
    start_deg: float,
    span_deg: float,
    spacing_mm: float,
) -> list[NDArray[np.float64]]:
    contour = _horizontal_contour(mesh, float(y_mm))
    if contour is None:
        return []
    return _walk_row_azimuths(
        contour,
        axis=axis,
        start_deg=float(start_deg),
        span_deg=float(span_deg),
        spacing_mm=float(spacing_mm),
    )


def _row_covers_zone(
    row: list[NDArray[np.float64]],
    axis: NDArray[np.float64],
    *,
    start_deg: float,
    span_deg: float,
) -> bool:
    if not row:
        return False
    points = np.vstack(row)
    delta = np.mod(azimuths_deg(points, axis) - float(start_deg), 360.0)
    return bool(
        float(np.min(delta)) <= ROW_AZIMUTH_TOLERANCE_DEG
        and float(np.max(delta)) >= float(span_deg) - ROW_AZIMUTH_TOLERANCE_DEG
    )


def _first_sectionable_y(
    mesh: trimesh.Trimesh,
    *,
    axis: NDArray[np.float64],
    start_y: float,
    stop_y: float,
    start_deg: float,
    span_deg: float,
    spacing_mm: float,
) -> float | None:
    """Walk from ``start_y`` toward ``stop_y`` until a 0–span wall row exists."""
    lo = float(min(start_y, stop_y))
    hi = float(max(start_y, stop_y))
    direction = 1.0 if stop_y >= start_y else -1.0
    step = direction * SECTION_SEARCH_STEP_MM
    y = float(start_y)
    guard = 0
    while lo - 1e-9 <= y <= hi + 1e-9 and guard < 2000:
        guard += 1
        row = _row_at_y(
            mesh,
            axis=axis,
            y_mm=y,
            start_deg=start_deg,
            span_deg=span_deg,
            spacing_mm=spacing_mm,
        )
        if _row_covers_zone(
            row, axis, start_deg=start_deg, span_deg=span_deg
        ):
            return float(y)
        y += step
    return None


def _useful_height_bounds(
    mesh: trimesh.Trimesh,
    *,
    axis: NDArray[np.float64],
    envelope_y_min: float,
    envelope_y_max: float,
    start_deg: float,
    span_deg: float,
    spacing_mm: float,
) -> tuple[float, float]:
    """Lowest / highest sectionable interior Y (floor ring → A0 opening)."""
    y_lo = _first_sectionable_y(
        mesh,
        axis=axis,
        start_y=float(envelope_y_min),
        stop_y=float(envelope_y_max),
        start_deg=start_deg,
        span_deg=span_deg,
        spacing_mm=spacing_mm,
    )
    y_hi = _first_sectionable_y(
        mesh,
        axis=axis,
        start_y=float(envelope_y_max),
        stop_y=float(envelope_y_min),
        start_deg=start_deg,
        span_deg=span_deg,
        spacing_mm=spacing_mm,
    )
    if y_lo is None or y_hi is None or y_hi < y_lo:
        raise ValueError("Interior envelope has no sectionable 0–90° height")
    return float(y_lo), float(y_hi)


def _walk_row_azimuths(
    contour: NDArray[np.float64],
    *,
    axis: NDArray[np.float64],
    start_deg: float,
    span_deg: float,
    spacing_mm: float,
) -> list[NDArray[np.float64]]:
    """Place points every ``spacing_mm`` of wall arc from start to start+span."""
    row: list[NDArray[np.float64]] = []
    theta = 0.0
    guard = 0
    while theta <= float(span_deg) + 1e-9 and guard < 1000:
        guard += 1
        hit = _ray_contour_hit(
            contour,
            axis_x=float(axis[0]),
            axis_z=float(axis[1]),
            az_deg=float(start_deg) + float(theta),
        )
        if hit is None:
            break
        radius = float(np.hypot(hit[0] - axis[0], hit[2] - axis[1]))
        if radius < WALL_RADIUS_MIN_MM:
            break
        row.append(hit)
        if theta >= float(span_deg) - 1e-9:
            break
        step_deg = float(np.degrees(float(spacing_mm) / max(radius, 1e-3)))
        if step_deg < 1e-6:
            break
        nxt = theta + step_deg
        theta = float(span_deg) if nxt > float(span_deg) - 1e-6 else nxt
    return row


def _section_polylines(section: object) -> list[NDArray[np.float64]]:
    discrete = getattr(section, "discrete", None)
    if discrete:
        return [
            np.asarray(part, dtype=np.float64) for part in discrete if len(part) >= 2
        ]
    vertices = np.asarray(section.vertices, dtype=np.float64)
    polylines: list[NDArray[np.float64]] = []
    for entity in getattr(section, "entities", []):
        idx = np.asarray(getattr(entity, "points", []), dtype=np.int64)
        if len(idx) >= 2:
            polylines.append(vertices[idx])
    return polylines


def _approach_half_meridian(
    pts: NDArray[np.float64],
    *,
    axis: NDArray[np.float64],
    ux: float,
    uz: float,
) -> NDArray[np.float64]:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts
    proj = (pts[:, 0] - float(axis[0])) * ux + (pts[:, 2] - float(axis[1])) * uz
    radii = np.hypot(pts[:, 0] - float(axis[0]), pts[:, 2] - float(axis[1]))
    eligible = np.where(proj >= -1.0)[0]
    if len(eligible) < 2:
        eligible = np.arange(len(pts))
    i_open = int(eligible[int(np.argmax(pts[eligible, 1]))])

    def _walk(start: int, step: int) -> list[int]:
        chain = [start]
        index = start
        while True:
            nxt = index + step
            if nxt < 0 or nxt >= len(pts):
                break
            if float(proj[nxt]) < -1.0 and float(radii[nxt]) > 1.0:
                break
            chain.append(nxt)
            index = nxt
        return chain

    left = _walk(i_open, -1)
    right = _walk(i_open, 1)
    idxs = list(reversed(left[1:])) + [i_open] + right[1:]
    chain = pts[np.asarray(idxs, dtype=np.int64)]
    if float(chain[0, 1]) > float(chain[-1, 1]):
        chain = chain[::-1].copy()
    keep = np.concatenate(
        [[True], np.linalg.norm(np.diff(chain, axis=0), axis=1) > 1e-6]
    )
    return chain[keep]


def _resample_polyline(
    pts: NDArray[np.float64], spacing_mm: float
) -> NDArray[np.float64]:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        return pts[:1].copy()
    samples = _y_samples(0.0, total, spacing_mm)
    out = np.empty((len(samples), 3), dtype=np.float64)
    for i, s in enumerate(samples):
        k = int(np.searchsorted(cum, float(s), side="left"))
        if k <= 0:
            out[i] = pts[0]
            continue
        if k >= len(cum):
            out[i] = pts[-1]
            continue
        t = (float(s) - float(cum[k - 1])) / max(float(cum[k] - cum[k - 1]), 1e-12)
        out[i] = pts[k - 1] + t * (pts[k] - pts[k - 1])
    return out


def _interior_meridian(
    mesh: trimesh.Trimesh,
    axis: NDArray[np.float64],
    az_deg: float,
) -> NDArray[np.float64] | None:
    ux, uz = _direction_xz(az_deg)
    normal = np.array([-uz, 0.0, ux], dtype=np.float64)
    nrm = float(np.linalg.norm(normal))
    if nrm <= 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / nrm
    section = mesh.section(
        plane_origin=[float(axis[0]), 0.0, float(axis[1])],
        plane_normal=normal.tolist(),
    )
    if section is None:
        return _meridian_from_horizontal_slices(mesh, axis, az_deg)
    polylines = _section_polylines(section)
    if not polylines:
        return _meridian_from_horizontal_slices(mesh, axis, az_deg)
    raw = max(
        polylines,
        key=lambda arr: float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
        if len(arr) > 1
        else 0.0,
    )
    chain = _approach_half_meridian(raw, axis=axis, ux=float(ux), uz=float(uz))
    if len(chain) >= 2:
        return chain
    return _meridian_from_horizontal_slices(mesh, axis, az_deg)


def _meridian_from_horizontal_slices(
    mesh: trimesh.Trimesh,
    axis: NDArray[np.float64],
    az_deg: float,
) -> NDArray[np.float64] | None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    y_min = float(np.min(vertices[:, 1]))
    y_max = float(np.max(vertices[:, 1]))
    hits: list[NDArray[np.float64]] = []
    for y in _y_samples(y_min, y_max, min(MATRIX_SPACING_MM, 2.5)):
        contour = _horizontal_contour(mesh, float(y))
        if contour is None:
            continue
        hit = _ray_contour_hit(
            contour,
            axis_x=float(axis[0]),
            axis_z=float(axis[1]),
            az_deg=float(az_deg),
        )
        if hit is None:
            continue
        radius = float(np.hypot(hit[0] - axis[0], hit[2] - axis[1]))
        if radius < FLOOR_RADIUS_MIN_MM:
            continue
        hits.append(hit)
    if len(hits) < 2:
        return None
    return np.vstack(hits)


def _azimuth_steps_deg(radius_mm: float, span_deg: float, spacing_mm: float) -> list[float]:
    span = float(span_deg)
    if float(radius_mm) < FLOOR_RADIUS_MIN_MM:
        return [0.0]
    steps = [0.0]
    step_deg = float(np.degrees(float(spacing_mm) / max(float(radius_mm), 1e-3)))
    if step_deg < 1e-6:
        return [0.0, span]
    nxt = step_deg
    while nxt < span - 1e-9:
        steps.append(float(nxt))
        nxt += step_deg
    remainder_mm = float(radius_mm) * np.radians(span - steps[-1])
    if remainder_mm > Y_EDGE_MERGE_MM:
        steps.append(span)
    else:
        steps[-1] = span
    return steps


def _sample_meridian_zone(
    mesh: trimesh.Trimesh,
    *,
    axis: NDArray[np.float64],
    start_deg: float,
    span_deg: float,
    spacing_mm: float,
) -> NDArray[np.float64]:
    """5 mm stations along each interior meridian from A0 to +90°, including the floor."""
    reference = _interior_meridian(mesh, axis, float(start_deg))
    if reference is None or len(reference) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    mid = reference[len(reference) // 2]
    radius_ref = float(np.hypot(mid[0] - axis[0], mid[2] - axis[1]))
    radius_ref = max(radius_ref, 20.0)
    samples: list[NDArray[np.float64]] = []
    for theta in _azimuth_steps_deg(radius_ref, span_deg, spacing_mm):
        meridian = _interior_meridian(mesh, axis, float(start_deg) + float(theta))
        if meridian is None or len(meridian) < 2:
            continue
        samples.append(_resample_polyline(meridian, float(spacing_mm)))
    if not samples:
        return np.zeros((0, 3), dtype=np.float64)
    points = np.vstack(samples)
    az = azimuths_deg(points, axis)
    points = points[
        azimuth_in_zone_mask(az, start_deg=start_deg, span_deg=span_deg)
    ]
    if len(points) == 0:
        return points
    key = np.round(points, 3)
    _, idx = np.unique(key, axis=0, return_index=True)
    return points[np.sort(idx)]


@dataclass(frozen=True)
class CoverageReferenceMatrix:
    """5 mm interior-wall samples from A0 (0°) to the +90° meridian."""

    coverage_target_surface: str
    coverage_target_region: str
    coverage_target_azimuth_range: tuple[float, float]
    a0_azimuth_deg: float
    azimuth_span_deg: float
    azimuth_positive_sense: str
    azimuth_negative_sense: str
    reference_zone_side: str
    spacing_mm: float
    points_mm: tuple[tuple[float, float, float], ...]
    point_count: int
    y_min_mm: float
    y_max_mm: float
    azimuth_min_deg: float
    azimuth_max_deg: float
    target_face_ids: tuple[int, ...]
    target_area_mm2: float
    face_count: int
    fingerprint: str
    source: str
    mean_vertical_spacing_mm: float
    mean_tangential_spacing_mm: float
    neighbor_min_mm: float
    neighbor_max_mm: float
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]
    on_interior_envelope: bool
    max_distance_to_interior_mm: float
    any_point_outside_envelope: bool
    uses_visual_stl: bool = False
    uses_legacy_a0_point_matrix: bool = False
    simulator_invoked: bool = False
    coverage_recomputed: bool = False
    symmetry_multiplier_applied: bool = False


def _matrix_fingerprint(points: NDArray[np.float64]) -> str:
    payload = np.round(np.asarray(points, dtype=np.float64), 4).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _neighbor_spacings(
    points: NDArray[np.float64],
    axis: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    radii = np.hypot(points[:, 0] - float(axis[0]), points[:, 2] - float(axis[1]))
    wall = points[radii >= WALL_RADIUS_MIN_MM]
    az_wall = azimuths_deg(wall, axis) if len(wall) else np.array([])
    vertical: list[float] = []
    for a in np.unique(np.round(az_wall, 0)):
        mer = wall[np.abs(az_wall - float(a)) < 0.75]
        if len(mer) < 2:
            continue
        mer = mer[np.argsort(mer[:, 1])]
        deltas = np.linalg.norm(np.diff(mer, axis=0), axis=1)
        vertical.extend(float(v) for v in deltas if 2.5 < float(v) <= 7.5)
    if len(wall) < 2:
        wall = points
    ys = np.unique(np.round(wall[:, 1], 3))
    tangential: list[float] = []
    for y in ys:
        row = wall[np.abs(wall[:, 1] - y) < 0.6]
        if len(row) < 2:
            continue
        order = np.argsort(azimuths_deg(row, axis))
        ordered = row[order]
        deltas = np.linalg.norm(np.diff(ordered[:, [0, 2]], axis=0), axis=1)
        tangential.extend(float(v) for v in deltas if 1.0 < float(v) <= 12.0)
    all_n = vertical + tangential
    return (
        float(np.mean(vertical)) if vertical else 0.0,
        float(np.mean(tangential)) if tangential else 0.0,
        float(np.min(all_n)) if all_n else 0.0,
        float(np.max(all_n)) if all_n else 0.0,
    )


def _target_faces_for_zone(
    surface: InteriorSurfaceReference,
    *,
    axis: NDArray[np.float64],
    y_min: float,
    y_max: float,
) -> tuple[tuple[int, ...], float]:
    mesh = surface.to_trimesh()
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    centroids = vertices[faces].mean(axis=1)
    az = azimuths_deg(centroids, axis)
    radii = np.hypot(centroids[:, 0] - float(axis[0]), centroids[:, 2] - float(axis[1]))
    useful = (
        (centroids[:, 1] >= float(y_min) - 1e-3)
        & (centroids[:, 1] <= float(y_max) + 1e-3)
        & (radii >= WALL_RADIUS_MIN_MM - 1e-9)
        & azimuth_in_zone_mask(az)
    )
    ids = tuple(int(i) for i in np.flatnonzero(useful))
    areas = face_areas(mesh)
    area = float(sum(areas[i] for i in ids))
    return ids, area


def _point_envelope_check(
    mesh: trimesh.Trimesh,
    points: NDArray[np.float64],
) -> tuple[bool, float, bool]:
    if len(points) == 0:
        return False, 0.0, True
    _closest, distances, _tid = trimesh.proximity.closest_point(mesh, points)
    distances = np.asarray(distances, dtype=np.float64)
    max_d = float(np.max(distances)) if len(distances) else 0.0
    on_surface = bool(max_d <= ON_SURFACE_TOLERANCE_MM)
    outside = bool(np.any(distances > ON_SURFACE_TOLERANCE_MM))
    return on_surface, max_d, outside


def build_coverage_reference_matrix(
    surface: InteriorSurfaceReference,
    *,
    spacing_mm: float = MATRIX_SPACING_MM,
    a0_azimuth_deg: float = A0_MERIDIAN_AZIMUTH_DEG,
    span_deg: float = REFERENCE_ZONE_SPAN_DEG,
) -> CoverageReferenceMatrix:
    """Sample the interior wall on A0 → A0+90°. No coverage evaluation."""
    mesh = surface.to_trimesh()
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    axis = surface_axis_xz(vertices)
    opening_y, lower_y, _span = scraper_length_span(surface)
    points = _sample_meridian_zone(
        mesh,
        axis=axis,
        start_deg=float(a0_azimuth_deg),
        span_deg=float(span_deg),
        spacing_mm=float(spacing_mm),
    )
    if len(points) == 0:
        y_lo, y_hi = _useful_height_bounds(
            mesh,
            axis=axis,
            envelope_y_min=lower_y,
            envelope_y_max=opening_y,
            start_deg=float(a0_azimuth_deg),
            span_deg=float(span_deg),
            spacing_mm=float(spacing_mm),
        )
        samples: list[NDArray[np.float64]] = []
        for y in _y_samples(y_lo, y_hi, spacing_mm):
            row = _row_at_y(
                mesh,
                axis=axis,
                y_mm=float(y),
                start_deg=float(a0_azimuth_deg),
                span_deg=float(span_deg),
                spacing_mm=float(spacing_mm),
            )
            samples.extend(row)
        if not samples:
            raise ValueError("Interior reference matrix is empty")
        points = np.vstack(samples)
    order = np.lexsort((azimuths_deg(points, axis), np.round(points[:, 1], 6)))
    points = points[order]
    az = azimuths_deg(points, axis)
    face_ids, area = _target_faces_for_zone(
        surface, axis=axis, y_min=lower_y, y_max=opening_y
    )
    mean_v, mean_t, nmin, nmax = _neighbor_spacings(points, axis)
    on_surface, max_d, outside = _point_envelope_check(mesh, points)
    end_deg = float(np.mod(float(a0_azimuth_deg) + float(span_deg), 360.0))
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return CoverageReferenceMatrix(
        coverage_target_surface=COVERAGE_TARGET_SURFACE,
        coverage_target_region=COVERAGE_TARGET_REGION,
        coverage_target_azimuth_range=(float(a0_azimuth_deg), end_deg),
        a0_azimuth_deg=float(a0_azimuth_deg),
        azimuth_span_deg=float(span_deg),
        azimuth_positive_sense=AZIMUTH_POSITIVE_SENSE,
        azimuth_negative_sense=AZIMUTH_NEGATIVE_SENSE,
        reference_zone_side=REFERENCE_ZONE_SIDE,
        spacing_mm=float(spacing_mm),
        points_mm=tuple((float(p[0]), float(p[1]), float(p[2])) for p in points),
        point_count=int(len(points)),
        y_min_mm=float(np.min(points[:, 1])),
        y_max_mm=float(np.max(points[:, 1])),
        azimuth_min_deg=float(np.min(az)),
        azimuth_max_deg=float(np.max(az)),
        target_face_ids=face_ids,
        target_area_mm2=area,
        face_count=len(face_ids),
        fingerprint=_matrix_fingerprint(points),
        source=str(surface.source),
        mean_vertical_spacing_mm=mean_v,
        mean_tangential_spacing_mm=mean_t,
        neighbor_min_mm=nmin,
        neighbor_max_mm=nmax,
        bbox_min_mm=(float(mins[0]), float(mins[1]), float(mins[2])),
        bbox_max_mm=(float(maxs[0]), float(maxs[1]), float(maxs[2])),
        on_interior_envelope=on_surface,
        max_distance_to_interior_mm=max_d,
        any_point_outside_envelope=outside,
    )


def matrix_to_payload(matrix: CoverageReferenceMatrix) -> dict[str, object]:
    return {
        "coverage_target_surface": matrix.coverage_target_surface,
        "coverage_target_region": matrix.coverage_target_region,
        "coverage_target_azimuth_range": list(matrix.coverage_target_azimuth_range),
        "a0_azimuth_deg": matrix.a0_azimuth_deg,
        "azimuth_span_deg": matrix.azimuth_span_deg,
        "azimuth_positive_sense": matrix.azimuth_positive_sense,
        "azimuth_negative_sense": matrix.azimuth_negative_sense,
        "reference_zone_side": matrix.reference_zone_side,
        "spacing_mm": matrix.spacing_mm,
        "points_mm": [list(p) for p in matrix.points_mm],
        "point_count": matrix.point_count,
        "y_min_mm": matrix.y_min_mm,
        "y_max_mm": matrix.y_max_mm,
        "azimuth_min_deg": matrix.azimuth_min_deg,
        "azimuth_max_deg": matrix.azimuth_max_deg,
        "face_ids": list(matrix.target_face_ids),
        "face_count": matrix.face_count,
        "area_mm2": matrix.target_area_mm2,
        "fingerprint": matrix.fingerprint,
        "source": matrix.source,
        "interior_source": SOURCE_INTERIOR_PRODUCT_SURFACE,
        "mean_vertical_spacing_mm": matrix.mean_vertical_spacing_mm,
        "mean_tangential_spacing_mm": matrix.mean_tangential_spacing_mm,
        "neighbor_min_mm": matrix.neighbor_min_mm,
        "neighbor_max_mm": matrix.neighbor_max_mm,
        "bbox_min_mm": list(matrix.bbox_min_mm),
        "bbox_max_mm": list(matrix.bbox_max_mm),
        "on_interior_envelope": matrix.on_interior_envelope,
        "max_distance_to_interior_mm": matrix.max_distance_to_interior_mm,
        "any_point_outside_envelope": matrix.any_point_outside_envelope,
        "uses_visual_stl": False,
        "uses_legacy_a0_point_matrix": False,
        "uses_legacy_quadrant_90": False,
        "simulator_invoked": False,
        "coverage_recomputed": False,
        "symmetry_multiplier_applied": False,
        "point_color": "#ffffff",
    }


def diagnose_coverage_reference_matrix(
    matrix: CoverageReferenceMatrix,
) -> dict[str, object]:
    payload = matrix_to_payload(matrix)
    payload["diagnostic"] = True
    return payload
