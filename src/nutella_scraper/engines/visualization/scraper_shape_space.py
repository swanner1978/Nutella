"""Admissible scraper-shape space — visualization / search model only.

The control cage is a contact-constraint lattice on InteriorSurfaceReference,
not a volume and not a blade width. A candidate is a 1D contact curve C(s)
sampled at one lattice point per longitudinal station.

No collision imports. No mesh generation for the catalog. A future simulator
may apply SE(3) poses to a frozen ScraperShape; it must never reshape it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper import ScraperPose, ScraperShape
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_transform import pose_matrix
from nutella_scraper.engines.visualization.scraper_control_cage import (
    CAGE_CENTER_ROW_INDEX,
    CAGE_ROW_OFFSETS_MM,
)

# Match scraper A reference (thin FDM blade). Independent of cage span.
BLADE_THICKNESS_MM = 2.5
BLADE_WIDTH_MM = 2.5
CONTACT_TOLERANCE_MM = 0.5
DEFAULT_MAX_ROW_STEP = 1
MAX_SECOND_DIFFERENCE = 1
MAX_CANDIDATE_SHAPES = 1000
DEFAULT_CANDIDATE_COUNT = 100
REFERENCE_CANDIDATE_ID = "A0"
_WALK_SEED = 20260819
GAP_THRESHOLD_MM = 5.0
MAX_GARNISHED_TRAJECTORIES = 50
MIN_GARNISH_SEPARATION_MM = 1.5


@dataclass(frozen=True)
class ContactConstraintLattice:
    """Cage points as an envelope-validated contact lattice (not a solid)."""

    row_offsets_mm: tuple[float, ...]
    center_row_index: int
    points_mm: NDArray[np.float64]
    admissible: NDArray[np.bool_]
    signed_mm: NDArray[np.float64]
    unsigned_mm: NDArray[np.float64]
    source: str
    contact_tolerance_mm: float = CONTACT_TOLERANCE_MM
    fingerprint: str = ""

    @property
    def row_count(self) -> int:
        return int(self.points_mm.shape[0])

    @property
    def station_count(self) -> int:
        return int(self.points_mm.shape[1])

    @property
    def admissible_count(self) -> int:
        return int(np.count_nonzero(self.admissible))

    @property
    def nominal_count(self) -> int:
        return int(self.row_count * self.station_count)


@dataclass(frozen=True)
class GarnishReport:
    """Additive local fill of under-dense contact trajectories."""

    trajectories_before: int
    underdense_zones: int
    generated: int
    added: int
    duplicates_rejected: int
    off_envelope_rejected: int
    trajectories_after: int
    gap_threshold_mm: float = GAP_THRESHOLD_MM
    max_garnished: int = MAX_GARNISHED_TRAJECTORIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories_before": int(self.trajectories_before),
            "underdense_zones": int(self.underdense_zones),
            "generated": int(self.generated),
            "added": int(self.added),
            "duplicates_rejected": int(self.duplicates_rejected),
            "off_envelope_rejected": int(self.off_envelope_rejected),
            "trajectories_after": int(self.trajectories_after),
            "gap_threshold_mm": float(self.gap_threshold_mm),
            "max_garnished": int(self.max_garnished),
        }


@dataclass(frozen=True)
class CandidateShape:
    """Lightweight admissible curve. No FDM mesh until materialization."""

    candidate_id: str
    index: int
    row_indices: tuple[int, ...]
    shape: ScraperShape
    start_row: int
    end_row: int
    mean_abs_curvature: float
    family: str
    valid: bool
    reason_if_invalid: str | None

    @property
    def control_points_mm(self) -> tuple[tuple[float, float, float], ...]:
        return self.shape.control_points_mm

    @property
    def curve_length_mm(self) -> float:
        return float(self.shape.length_mm)

    @property
    def shape_fingerprint(self) -> str:
        return self.shape.fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "index": self.index,
            "row_indices": list(self.row_indices),
            "control_points_mm": [list(p) for p in self.control_points_mm],
            "curve_length_mm": round(self.curve_length_mm, 4),
            "curvature": round(self.mean_abs_curvature, 6),
            "start_row": self.start_row,
            "end_row": self.end_row,
            "family": self.family,
            "curvature_metrics": {
                "mean_abs_second_difference": round(self.mean_abs_curvature, 6),
                "max_row_step": DEFAULT_MAX_ROW_STEP,
            },
            "valid": self.valid,
            "reason_if_invalid": self.reason_if_invalid,
            "shape_fingerprint": self.shape_fingerprint,
            "thickness_mm": float(self.shape.thickness_mm),
            "width_mm": float(self.shape.width_mm),
            "has_mesh": self.shape.vertices is not None,
        }

    def as_rigid_shape(self) -> ScraperShape:
        """Promote a valid curve to a frozen ScraperShape. Invalid stays invalid."""
        if not self.valid:
            reason = self.reason_if_invalid or "invalid candidate"
            raise ValueError(f"invalid candidate cannot be promoted to ScraperShape: {reason}")
        return self.shape


def envelope_contact_metrics(
    surface: InteriorSurfaceReference,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interior-positive signed offset and unsigned nearest-surface distance.

    Positive signed = toward the jar cavity. Negative = through the glass /
    behind InteriorSurfaceReference. Does not import the collision engine.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(pts) == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    mesh = surface.to_trimesh()
    closest, distances, tri_ids = mesh.nearest.on_surface(pts)
    closest = np.asarray(closest, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    tri_ids = np.asarray(tri_ids, dtype=np.int64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    normals = face_normals[np.clip(tri_ids, 0, len(face_normals) - 1)].copy()
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-9)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    target = np.asarray(
        [0.0, 0.5 * (float(bounds[0, 1]) + float(bounds[1, 1])), 0.0],
        dtype=np.float64,
    )
    to_inside = target[None, :] - closest
    normals[np.sum(normals * to_inside, axis=1) < 0.0] *= -1.0
    signed = np.sum((pts - closest) * normals, axis=1)
    return signed, distances


def curve_behind_or_through_glass(
    surface: InteriorSurfaceReference,
    points: NDArray[np.float64],
    *,
    contact_tolerance_mm: float = CONTACT_TOLERANCE_MM,
) -> bool:
    """True if any vertex or chord midpoint sits behind the envelope."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return True
    samples = pts
    if len(pts) >= 2:
        mid = 0.5 * (pts[:-1] + pts[1:])
        samples = np.vstack([pts, mid])
    signed, _unsigned = envelope_contact_metrics(surface, samples)
    return bool(np.any(signed < -float(contact_tolerance_mm)))


def classify_curve_family(
    row_indices: tuple[int, ...],
    *,
    center: int,
    index: int,
) -> str:
    if index == 0:
        return "A0"
    rows = np.asarray(row_indices, dtype=np.int64)
    unique = {int(v) for v in rows}
    if len(unique) == 1:
        return "parallel"
    diffs = np.diff(rows)
    if len(rows) >= 3:
        second = rows[2:] - 2 * rows[1:-1] + rows[:-2]
        max_second = int(np.max(np.abs(second)))
    else:
        max_second = 0
    start, end = int(rows[0]), int(rows[-1])
    monotonic = bool(len(diffs) == 0 or np.all(diffs >= 0) or np.all(diffs <= 0))
    sign_changes = (
        int(np.sum((diffs[1:] * diffs[:-1]) < 0)) if len(diffs) >= 2 else 0
    )
    if sign_changes >= 1 and abs(start - end) <= 1:
        return "s_curve"
    if monotonic and start != end:
        changed = np.flatnonzero(rows != rows[0])
        held_prefix = int(changed[0]) if len(changed) else len(rows)
        changed_end = np.flatnonzero(rows[::-1] != rows[-1])
        held_suffix = int(changed_end[0]) if len(changed_end) else 0
        hold_limit = max(3, len(rows) // 4)
        if held_prefix >= hold_limit or held_suffix >= hold_limit:
            return "asymmetric"
        return "inclined"
    if abs(start - center) != abs(end - center):
        return "asymmetric"
    if max_second <= MAX_SECOND_DIFFERENCE:
        return "progressive"
    return "combined"


def _surface_axis_xz(surface: InteriorSurfaceReference) -> NDArray[np.float64]:
    verts = np.asarray(surface.vertices, dtype=np.float64)
    if len(verts) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    return 0.5 * (mins[[0, 2]] + maxs[[0, 2]])


def project_grid_onto_envelope(
    surface: InteriorSurfaceReference,
    points: NDArray[np.float64],
    *,
    center: int,
    spine: NDArray[np.float64],
    contact_tolerance_mm: float = CONTACT_TOLERANCE_MM,
) -> NDArray[np.float64]:
    """Snap every lateral row onto InteriorSurfaceReference as an offset of A.

    Theoretical samples sit on A's parallel (same radius, locked azimuth).
    Contact is the nearest envelope point to that seed — not the first hit of
    an axis ray, which on a flat floor skips the disk and sticks to the wall.
    The centreline is copied from ``spine`` and is not moved.
    """
    grid = np.asarray(points, dtype=np.float64).copy()
    spine = np.asarray(spine, dtype=np.float64)
    n_rows, n_stat, _ = grid.shape
    if spine.shape != (n_stat, 3):
        raise ValueError("Centreline length must match the lattice station count")
    lock_center = 0 <= int(center) < n_rows
    if lock_center:
        grid[center] = spine
    axis_xz = _surface_axis_xz(surface)
    mesh = surface.to_trimesh()
    opening = int(np.argmax(spine[:, 1]))
    ax = float(axis_xz[0])
    az = float(axis_xz[1])
    ys = np.asarray(spine[:, 1], dtype=np.float64)
    r_spine = np.maximum(np.hypot(spine[:, 0] - ax, spine[:, 2] - az), 0.5)
    row_azimuths = np.zeros(n_rows, dtype=np.float64)
    for row in range(n_rows):
        if lock_center and row == center:
            continue
        azimuth = _seed_row_azimuth(grid[row, opening], axis_xz, spine[opening])
        row_azimuths[row] = azimuth
        cosine = float(np.cos(azimuth))
        sine = float(np.sin(azimuth))
        seeds = np.column_stack((ax + r_spine * cosine, ys, az + r_spine * sine))
        closest, _dist, _tri = mesh.nearest.on_surface(seeds)
        closest = np.asarray(closest, dtype=np.float64)
        origins = np.column_stack((np.full(n_stat, ax), ys, np.full(n_stat, az)))
        directions = np.tile(
            np.array([cosine, 0.0, sine], dtype=np.float64),
            (n_stat, 1),
        )
        hits = _first_outward_hits(mesh, origins, directions)
        snapped = np.zeros((n_stat, 3), dtype=np.float64)
        for station in range(n_stat):
            seed = seeds[station]
            r_seed = float(r_spine[station])
            local = np.asarray(closest[station], dtype=np.float64).copy()
            if station in hits:
                hit = np.asarray(hits[station], dtype=np.float64)
                if float(np.linalg.norm(hit - seed)) < float(
                    np.linalg.norm(local - seed)
                ):
                    local = hit
            r_hit = _xz_radius(local, axis_xz)
            r_use = r_seed if r_hit > r_seed + 8.0 else max(r_hit, 0.25)
            snapped[station] = np.array(
                [ax + r_use * cosine, float(ys[station]), az + r_use * sine],
                dtype=np.float64,
            )
        projected, _dist, _tri = mesh.nearest.on_surface(snapped)
        projected = np.asarray(projected, dtype=np.float64)
        forced = projected.copy()
        forced[:, 1] = ys
        signed_row, unsigned_row = envelope_contact_metrics(surface, forced)
        keep_forced = (unsigned_row <= float(contact_tolerance_mm) + 1e-6) & (
            signed_row >= -float(contact_tolerance_mm)
        )
        grid[row] = np.where(keep_forced[:, None], forced, projected)
    if lock_center:
        grid[center] = spine
    signed, unsigned = envelope_contact_metrics(surface, grid.reshape(-1, 3))
    signed = signed.reshape(n_rows, n_stat)
    unsigned = unsigned.reshape(n_rows, n_stat)
    bad = (unsigned > float(contact_tolerance_mm) + 1e-6) | (
        signed < -float(contact_tolerance_mm)
    )
    if lock_center:
        bad[center, :] = False
    if np.any(bad):
        queries = []
        locations = []
        for row, station in zip(*np.nonzero(bad), strict=True):
            azimuth = float(row_azimuths[int(row)])
            r_seed = float(r_spine[int(station)])
            queries.append(
                [
                    ax + r_seed * float(np.cos(azimuth)),
                    float(ys[int(station)]),
                    az + r_seed * float(np.sin(azimuth)),
                ]
            )
            locations.append((int(row), int(station)))
        closest, _dist, _tri = mesh.nearest.on_surface(
            np.asarray(queries, dtype=np.float64)
        )
        for (row, station), point in zip(locations, closest, strict=True):
            projected = np.asarray(point, dtype=np.float64).copy()
            forced = projected.copy()
            forced[1] = float(ys[station])
            signed_pt, unsigned_pt = envelope_contact_metrics(
                surface, forced.reshape(1, 3)
            )
            if float(unsigned_pt[0]) <= float(contact_tolerance_mm) + 1e-6 and float(
                signed_pt[0]
            ) >= -float(contact_tolerance_mm):
                projected = forced
            grid[row, station] = projected
    if lock_center:
        grid[center] = spine
    return np.round(grid, 4)


def _xz_radius(point: NDArray[np.float64], axis_xz: NDArray[np.float64]) -> float:
    return float(
        np.hypot(float(point[0]) - float(axis_xz[0]), float(point[2]) - float(axis_xz[1]))
    )


def _azimuth_of(point: NDArray[np.float64], axis_xz: NDArray[np.float64]) -> float:
    return float(
        np.arctan2(float(point[2]) - float(axis_xz[1]), float(point[0]) - float(axis_xz[0]))
    )


def _seed_row_azimuth(
    seed: NDArray[np.float64],
    axis_xz: NDArray[np.float64],
    spine_point: NDArray[np.float64],
) -> float:
    if _xz_radius(seed, axis_xz) < 1e-9:
        seed = spine_point
    return _azimuth_of(seed, axis_xz)


def _first_outward_hits(
    mesh: object,
    origins: NDArray[np.float64],
    directions: NDArray[np.float64],
) -> dict[int, NDArray[np.float64]]:
    if len(origins) == 0:
        return {}
    locations, ray_ids, _tri = mesh.ray.intersects_location(  # type: ignore[attr-defined]
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
    )
    if locations is None or len(locations) == 0:
        return {}
    best: dict[int, tuple[float, NDArray[np.float64]]] = {}
    for loc, raw_id in zip(
        np.asarray(locations, dtype=np.float64),
        np.asarray(ray_ids, dtype=np.int64),
        strict=True,
    ):
        ray_id = int(raw_id)
        travel = float(np.dot(loc - origins[ray_id], directions[ray_id]))
        if travel < 0.0:
            continue
        previous = best.get(ray_id)
        if previous is None or travel < previous[0]:
            best[ray_id] = (travel, loc)
    return {ray_id: point for ray_id, (_travel, point) in best.items()}


def _station_gaps_mm(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.linalg.norm(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64),
        axis=1,
    )


def _trajectory_fingerprint(points: NDArray[np.float64]) -> str:
    rounded = np.round(np.asarray(points, dtype=np.float64), 3)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def _empty_garnish_report(
    lattice: ContactConstraintLattice,
    *,
    gap_threshold_mm: float,
    max_garnished: int,
) -> GarnishReport:
    return GarnishReport(
        trajectories_before=lattice.row_count,
        underdense_zones=0,
        generated=0,
        added=0,
        duplicates_rejected=0,
        off_envelope_rejected=0,
        trajectories_after=lattice.row_count,
        gap_threshold_mm=float(gap_threshold_mm),
        max_garnished=int(max_garnished),
    )


def _azimuth_mid_seed(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    axis_xz: NDArray[np.float64],
    y: float,
) -> NDArray[np.float64]:
    ax = float(axis_xz[0])
    az = float(axis_xz[1])
    az_a = float(np.arctan2(left[2] - az, left[0] - ax))
    az_b = float(np.arctan2(right[2] - az, right[0] - ax))
    delta = (az_b - az_a + np.pi) % (2.0 * np.pi) - np.pi
    azimuth = az_a + 0.5 * delta
    radius = 0.5 * (
        float(np.hypot(left[0] - ax, left[2] - az))
        + float(np.hypot(right[0] - ax, right[2] - az))
    )
    if radius < 1e-9:
        return np.array([ax, y, az], dtype=np.float64)
    return np.array(
        [ax + radius * float(np.cos(azimuth)), y, az + radius * float(np.sin(azimuth))],
        dtype=np.float64,
    )


def _surface_mid_trajectory(
    surface: InteriorSurfaceReference,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    spine: NDArray[np.float64],
    *,
    contact_tolerance_mm: float,
    mesh: object | None = None,
) -> NDArray[np.float64] | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    spine = np.asarray(spine, dtype=np.float64)
    n_stat = int(spine.shape[0])
    axis_xz = _surface_axis_xz(surface)
    seeds = np.zeros((n_stat, 3), dtype=np.float64)
    for station in range(n_stat):
        seeds[station] = _azimuth_mid_seed(
            left[station],
            right[station],
            axis_xz=axis_xz,
            y=float(spine[station, 1]),
        )
    mesh = mesh if mesh is not None else surface.to_trimesh()
    closest, _dist, _tri = mesh.nearest.on_surface(seeds)
    projected = np.asarray(closest, dtype=np.float64).copy()
    projected[:, 1] = spine[:, 1]
    signed, unsigned = envelope_contact_metrics(surface, projected)
    if float(np.min(signed)) < -float(contact_tolerance_mm) or float(
        np.max(unsigned)
    ) > float(contact_tolerance_mm) + 1e-6:
        closest, _dist, _tri = mesh.nearest.on_surface(projected)
        projected = np.asarray(closest, dtype=np.float64)
        signed, unsigned = envelope_contact_metrics(surface, projected)
        if float(np.min(signed)) < -float(contact_tolerance_mm):
            return None
        if float(np.max(unsigned)) > float(contact_tolerance_mm) + 1e-6:
            return None
    if not np.all(np.isfinite(projected)):
        return None
    if segments_self_intersect(projected):
        return None
    ys = projected[:, 1]
    diffs = np.diff(ys)
    if len(diffs) and not (
        bool(np.all(diffs >= -1e-6)) or bool(np.all(diffs <= 1e-6))
    ):
        return None
    return np.round(projected, 4)


def _mean_row_distance_mm(
    candidate: NDArray[np.float64],
    existing: NDArray[np.float64],
) -> float:
    return float(
        np.min(
            [
                float(np.mean(np.linalg.norm(candidate - existing[row], axis=1)))
                for row in range(existing.shape[0])
            ]
        )
    )


def garnish_contact_lattice(
    lattice: ContactConstraintLattice,
    surface: InteriorSurfaceReference,
    *,
    gap_threshold_mm: float = GAP_THRESHOLD_MM,
    max_garnished: int = MAX_GARNISHED_TRAJECTORIES,
    min_separation_mm: float = MIN_GARNISH_SEPARATION_MM,
) -> tuple[ContactConstraintLattice, GarnishReport]:
    """Insert intermediate envelope trajectories in under-dense gaps only.

    Existing rows are copied unchanged. A0 stays the centreline. New rows are
    complete opening→floor contact curves, never volume samples.
    """
    points = np.asarray(lattice.points_mm, dtype=np.float64).copy()
    offsets = [float(v) for v in lattice.row_offsets_mm]
    center = int(lattice.center_row_index)
    spine = points[center].copy()
    original = points.copy()
    original_offsets = list(offsets)
    threshold = float(gap_threshold_mm)
    underdense = 0
    for row in range(points.shape[0] - 1):
        if float(np.max(_station_gaps_mm(points[row], points[row + 1]))) > threshold:
            underdense += 1
    generated = 0
    added = 0
    duplicates = 0
    off_envelope = 0
    seen = {_trajectory_fingerprint(points[row]) for row in range(points.shape[0])}
    skipped: set[tuple[str, str]] = set()
    limit = max(0, int(max_garnished))
    mesh = surface.to_trimesh()
    while added < limit:
        best_gap = -1.0
        best_index = -1
        for row in range(points.shape[0] - 1):
            key = tuple(
                sorted(
                    (
                        _trajectory_fingerprint(points[row]),
                        _trajectory_fingerprint(points[row + 1]),
                    )
                )
            )
            if key in skipped:
                continue
            gap = float(np.max(_station_gaps_mm(points[row], points[row + 1])))
            if gap > threshold and gap > best_gap:
                best_gap = gap
                best_index = row
        if best_index < 0:
            break
        left = points[best_index]
        right = points[best_index + 1]
        pair_key = tuple(
            sorted((_trajectory_fingerprint(left), _trajectory_fingerprint(right)))
        )
        generated += 1
        mid = _surface_mid_trajectory(
            surface,
            left,
            right,
            spine,
            contact_tolerance_mm=lattice.contact_tolerance_mm,
            mesh=mesh,
        )
        if mid is None:
            off_envelope += 1
            skipped.add(pair_key)
            continue
        fingerprint = _trajectory_fingerprint(mid)
        if fingerprint in seen or _mean_row_distance_mm(mid, points) < float(
            min_separation_mm
        ):
            duplicates += 1
            skipped.add(pair_key)
            continue
        insert_at = best_index + 1
        points = np.insert(points, insert_at, mid, axis=0)
        offsets.insert(
            insert_at, 0.5 * (float(offsets[best_index]) + float(offsets[best_index + 1]))
        )
        if insert_at <= center:
            center += 1
        seen.add(fingerprint)
        added += 1
    for off, old_row in zip(original_offsets, original, strict=True):
        new_row = points[offsets.index(off)]
        if not np.allclose(old_row, new_row, atol=1e-9):
            raise RuntimeError("Garnish mutated an existing lattice trajectory")
    if not np.allclose(points[center], spine, atol=1e-9):
        raise RuntimeError("Garnish moved centreline A0")
    signed, unsigned = envelope_contact_metrics(surface, points.reshape(-1, 3))
    signed = signed.reshape(points.shape[0], points.shape[1])
    unsigned = unsigned.reshape(points.shape[0], points.shape[1])
    admissible = np.ones((points.shape[0], points.shape[1]), dtype=np.bool_)
    fingerprint = _lattice_fingerprint(
        admissible,
        points,
        center=center,
        tolerance=float(lattice.contact_tolerance_mm),
    )
    garnished = ContactConstraintLattice(
        row_offsets_mm=tuple(offsets),
        center_row_index=center,
        points_mm=points,
        admissible=admissible,
        signed_mm=np.asarray(signed, dtype=np.float64),
        unsigned_mm=np.asarray(unsigned, dtype=np.float64),
        source=lattice.source,
        contact_tolerance_mm=lattice.contact_tolerance_mm,
        fingerprint=fingerprint,
    )
    report = GarnishReport(
        trajectories_before=int(original.shape[0]),
        underdense_zones=int(underdense),
        generated=int(generated),
        added=int(added),
        duplicates_rejected=int(duplicates),
        off_envelope_rejected=int(off_envelope),
        trajectories_after=int(garnished.row_count),
        gap_threshold_mm=threshold,
        max_garnished=limit,
    )
    return garnished, report


def lattice_from_cage(
    cage: dict[str, Any],
    surface: InteriorSurfaceReference,
    *,
    contact_tolerance_mm: float = CONTACT_TOLERANCE_MM,
) -> ContactConstraintLattice:
    """Load the contact lattice. Existing locked trajectories are not moved.

    First construction radially projects the base rows onto the envelope.
    Already garnished overlays are reused as-is so added trajectories stay
    additive and A0 is not rewritten by a second projection.
    """
    rows = (
        cage.get("candidates")
        or cage.get("polylines_mm")
        or cage.get("nominal_candidates")
    )
    if not rows:
        raise ValueError("Control cage has no contact samples")
    points, present = _grid_from_rows(rows)
    raw_offsets = cage.get("row_offsets_mm") or list(CAGE_ROW_OFFSETS_MM)
    if len(raw_offsets) < points.shape[0]:
        raw_offsets = list(raw_offsets) + [
            float(raw_offsets[-1]) + 10.0 * (i + 1)
            for i in range(points.shape[0] - len(raw_offsets))
        ]
    offsets = tuple(float(v) for v in raw_offsets[: points.shape[0]])
    center = int(cage.get("center_row_index", CAGE_CENTER_ROW_INDEX))
    center = max(0, min(center, points.shape[0] - 1))
    centerline = cage.get("centerline_mm")
    if centerline is None:
        raise ValueError("Control cage is missing centreline A")
    spine = np.asarray(centerline, dtype=np.float64)
    if spine.shape != (points.shape[1], 3):
        raise ValueError("Centreline length must match the lattice station count")
    for row in range(points.shape[0]):
        missing = ~present[row]
        if np.any(missing):
            points[row, missing] = spine[missing]
            present[row, missing] = True
    if not bool(cage.get("envelope_locked")):
        points = project_grid_onto_envelope(
            surface,
            points,
            center=center,
            spine=spine,
            contact_tolerance_mm=contact_tolerance_mm,
        )
    else:
        points[center] = np.round(spine, 4)
    signed, unsigned = envelope_contact_metrics(surface, points.reshape(-1, 3))
    signed = signed.reshape(present.shape)
    unsigned = unsigned.reshape(present.shape)
    admissible = np.ones(present.shape, dtype=np.bool_)
    fingerprint = _lattice_fingerprint(
        admissible,
        points,
        center=center,
        tolerance=float(contact_tolerance_mm),
    )
    return ContactConstraintLattice(
        row_offsets_mm=offsets,
        center_row_index=center,
        points_mm=points,
        admissible=np.asarray(admissible, dtype=np.bool_),
        signed_mm=np.asarray(signed, dtype=np.float64),
        unsigned_mm=np.asarray(unsigned, dtype=np.float64),
        source=str(cage.get("source") or surface.source),
        contact_tolerance_mm=float(contact_tolerance_mm),
        fingerprint=fingerprint,
    )


def filter_control_cage(
    cage: dict[str, Any],
    surface: InteriorSurfaceReference,
    *,
    contact_tolerance_mm: float = CONTACT_TOLERANCE_MM,
    garnish: bool = True,
    gap_threshold_mm: float = GAP_THRESHOLD_MM,
    max_garnished: int = MAX_GARNISHED_TRAJECTORIES,
) -> dict[str, Any]:
    """Project base rows onto the envelope, then add intermediate trajectories."""
    lattice = lattice_from_cage(
        cage, surface, contact_tolerance_mm=contact_tolerance_mm
    )
    base_offsets = tuple(
        float(v) for v in (cage.get("row_offsets_mm") or list(CAGE_ROW_OFFSETS_MM))
    )
    base_row_count = int(cage.get("base_row_count") or min(11, lattice.row_count))
    if garnish and not bool(cage.get("lattice_garnished")):
        lattice, report = garnish_contact_lattice(
            lattice,
            surface,
            gap_threshold_mm=gap_threshold_mm,
            max_garnished=max_garnished,
        )
    else:
        report = _empty_garnish_report(
            lattice,
            gap_threshold_mm=gap_threshold_mm,
            max_garnished=max_garnished,
        )
    n_rows = lattice.row_count
    n_stat = lattice.station_count
    polylines: list[list[list[float]]] = []
    usable: list[list[float]] = []
    for row in range(n_rows):
        line: list[list[float]] = []
        for station in range(n_stat):
            point = [
                round(float(lattice.points_mm[row, station, 0]), 4),
                round(float(lattice.points_mm[row, station, 1]), 4),
                round(float(lattice.points_mm[row, station, 2]), 4),
            ]
            line.append(point)
            usable.append(point)
        polylines.append(line)
    payload = dict(cage)
    payload["polylines_mm"] = polylines
    payload["candidates"] = polylines
    payload["nominal_candidates"] = polylines
    payload["points_mm"] = usable
    payload["admissible"] = lattice.admissible.astype(bool).tolist()
    payload["point_count"] = int(len(usable))
    payload["nominal_point_count"] = int(lattice.nominal_count)
    payload["removed_point_count"] = 0
    payload["admissible_fingerprint"] = lattice.fingerprint
    payload["contact_tolerance_mm"] = float(lattice.contact_tolerance_mm)
    payload["centerline_mm"] = cage.get("centerline_mm")
    payload["row_count"] = int(n_rows)
    payload["center_row_index"] = int(lattice.center_row_index)
    payload["row_offsets_mm"] = [float(v) for v in lattice.row_offsets_mm]
    payload["base_row_count"] = int(base_row_count)
    payload["base_row_offsets_mm"] = [float(v) for v in base_offsets[:base_row_count]]
    payload["envelope_locked"] = True
    payload["lattice_garnished"] = bool(garnish)
    payload["garnish"] = report.to_dict()
    return payload


def cage_segment_crosses_exterior(
    lattice: ContactConstraintLattice,
    surface: InteriorSurfaceReference,
) -> bool:
    """True if any drawn cage segment chord goes through the glass."""
    samples: list[NDArray[np.float64]] = []
    for row in range(lattice.row_count):
        for station in range(lattice.station_count - 1):
            if not (
                bool(lattice.admissible[row, station])
                and bool(lattice.admissible[row, station + 1])
            ):
                continue
            a = lattice.points_mm[row, station]
            b = lattice.points_mm[row, station + 1]
            samples.append(0.5 * (a + b))
    if not samples:
        return False
    signed, _unsigned = envelope_contact_metrics(
        surface, np.asarray(samples, dtype=np.float64)
    )
    return bool(np.any(signed < -float(lattice.contact_tolerance_mm)))


def _grid_from_rows(
    rows: Any,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    if not rows:
        raise ValueError("Control cage has no contact samples")
    n_rows = len(rows)
    n_stat = len(rows[0])
    points = np.full((n_rows, n_stat, 3), np.nan, dtype=np.float64)
    present = np.zeros((n_rows, n_stat), dtype=np.bool_)
    for row_index, row in enumerate(rows):
        if len(row) != n_stat:
            raise ValueError("Control cage rows must share one station count")
        for station, value in enumerate(row):
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                continue
            points[row_index, station] = arr
            present[row_index, station] = True
    return points, present


def _lattice_fingerprint(
    admissible: NDArray[np.bool_],
    points: NDArray[np.float64],
    *,
    center: int,
    tolerance: float,
) -> str:
    parts = [f"c={int(center)}", f"t={float(tolerance):.3f}"]
    for row in range(admissible.shape[0]):
        for station in range(admissible.shape[1]):
            if not bool(admissible[row, station]):
                continue
            point = points[row, station]
            parts.append(
                f"{row}:{station}:{point[0]:.3f}:{point[1]:.3f}:{point[2]:.3f}"
            )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def form_a_row_indices(lattice: ContactConstraintLattice) -> tuple[int, ...]:
    return (int(lattice.center_row_index),) * lattice.station_count


def control_points_for_rows(
    lattice: ContactConstraintLattice,
    row_indices: tuple[int, ...],
) -> NDArray[np.float64]:
    if len(row_indices) != lattice.station_count:
        raise ValueError("Row sequence must have one index per station")
    return np.asarray(
        [lattice.points_mm[int(row), i] for i, row in enumerate(row_indices)],
        dtype=np.float64,
    )


def curve_length_mm(points: NDArray[np.float64]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def mean_abs_second_difference(row_indices: tuple[int, ...]) -> float:
    rows = np.asarray(row_indices, dtype=np.int64)
    if len(rows) < 3:
        return 0.0
    second = rows[2:] - 2 * rows[1:-1] + rows[:-2]
    return float(np.mean(np.abs(second)))


def sequence_is_zigzag(row_indices: tuple[int, ...], *, min_flips: int = 4) -> bool:
    deltas = np.diff(np.asarray(row_indices, dtype=np.int64))
    nonzero = deltas[deltas != 0]
    if len(nonzero) < min_flips:
        return False
    signs = np.sign(nonzero)
    flips = int(np.sum(signs[1:] * signs[:-1] < 0))
    return flips >= min_flips and flips >= max(3, len(nonzero) // 2)


def segments_self_intersect(
    points: NDArray[np.float64],
    *,
    min_gap_mm: float = 1.0,
) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    n_seg = len(pts) - 1
    if n_seg < 3:
        return False
    for i in range(n_seg):
        a0, a1 = pts[i], pts[i + 1]
        for j in range(i + 2, n_seg):
            if i == 0 and j == n_seg - 1:
                continue
            if _segment_distance(a0, a1, pts[j], pts[j + 1]) < min_gap_mm:
                return True
    return False


def validate_row_sequence(
    lattice: ContactConstraintLattice,
    row_indices: tuple[int, ...],
    *,
    max_row_step: int = DEFAULT_MAX_ROW_STEP,
) -> tuple[bool, str | None]:
    if len(row_indices) != lattice.station_count:
        return False, "row sequence length must match station count"
    n_rows = lattice.row_count
    for row in row_indices:
        if row < 0 or row >= n_rows:
            return False, "row index outside lattice"
    steps = np.abs(np.diff(np.asarray(row_indices, dtype=np.int64)))
    if len(steps) and int(np.max(steps)) > int(max_row_step):
        return False, f"row step exceeds {max_row_step}"
    if len(row_indices) >= 3:
        second = np.abs(
            np.asarray(row_indices[2:], dtype=np.int64)
            - 2 * np.asarray(row_indices[1:-1], dtype=np.int64)
            + np.asarray(row_indices[:-2], dtype=np.int64)
        )
        if int(np.max(second)) > MAX_SECOND_DIFFERENCE:
            return False, "curvature second difference exceeds 1"
    if sequence_is_zigzag(row_indices):
        return False, "zigzag oscillation"
    points = control_points_for_rows(lattice, row_indices)
    for i, row in enumerate(row_indices):
        if not bool(lattice.admissible[int(row), i]):
            return False, "control point not on InteriorSurfaceReference"
    if segments_self_intersect(points):
        return False, "self-intersection"
    for i, row in enumerate(row_indices):
        if float(lattice.signed_mm[int(row), i]) < -float(lattice.contact_tolerance_mm):
            return False, "control point behind envelope"
    return True, None


def build_candidate(
    lattice: ContactConstraintLattice,
    row_indices: tuple[int, ...],
    *,
    index: int,
    max_row_step: int = DEFAULT_MAX_ROW_STEP,
) -> CandidateShape:
    valid, reason = validate_row_sequence(
        lattice, row_indices, max_row_step=max_row_step
    )
    if (
        len(row_indices) == lattice.station_count
        and all(0 <= int(row) < lattice.row_count for row in row_indices)
    ):
        points = control_points_for_rows(lattice, row_indices)
    else:
        points = np.asarray(
            lattice.points_mm[lattice.center_row_index], dtype=np.float64
        )
    length = curve_length_mm(points)
    rounded = np.round(points, 4)
    fingerprint = _fingerprint(row_indices, rounded)
    candidate_id = REFERENCE_CANDIDATE_ID if index == 0 else f"S{index:04d}"
    shape = ScraperShape(
        fingerprint=fingerprint,
        control_points_mm=tuple(
            (float(p[0]), float(p[1]), float(p[2])) for p in rounded
        ),
        thickness_mm=BLADE_THICKNESS_MM,
        width_mm=BLADE_WIDTH_MM,
        length_mm=length,
        vertices=None,
        faces=None,
    )
    return CandidateShape(
        candidate_id=candidate_id,
        index=index,
        row_indices=tuple(int(v) for v in row_indices),
        shape=shape,
        start_row=int(row_indices[0]) if row_indices else 0,
        end_row=int(row_indices[-1]) if row_indices else 0,
        mean_abs_curvature=mean_abs_second_difference(row_indices),
        family=classify_curve_family(
            tuple(int(v) for v in row_indices),
            center=lattice.center_row_index,
            index=index,
        ),
        valid=valid,
        reason_if_invalid=reason,
    )


def generate_candidate_shapes(
    lattice: ContactConstraintLattice,
    count: int = DEFAULT_CANDIDATE_COUNT,
    *,
    max_row_step: int = DEFAULT_MAX_ROW_STEP,
) -> tuple[CandidateShape, ...]:
    """Deterministic catalog of lightweight curves. Candidate 0 is form A."""
    n = int(np.clip(int(count), 1, MAX_CANDIDATE_SHAPES))
    ordered: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(rows: tuple[int, ...]) -> None:
        if rows in seen or len(ordered) >= n:
            return
        ok, _reason = validate_row_sequence(
            lattice, rows, max_row_step=max_row_step
        )
        if not ok:
            return
        seen.add(rows)
        ordered.append(rows)

    _add(form_a_row_indices(lattice))
    n_stat = lattice.station_count
    n_rows = lattice.row_count
    center = lattice.center_row_index
    for offset in range(1, n_rows):
        left = center - offset
        right = center + offset
        if 0 <= left < n_rows:
            _add((left,) * n_stat)
        if 0 <= right < n_rows:
            _add((right,) * n_stat)
        if len(ordered) >= n:
            break
    for start in range(n_rows):
        for end in range(n_rows):
            _add(_linear_rows(n_stat, start, end, n_rows))
            if len(ordered) >= n:
                break
        if len(ordered) >= n:
            break
    mid = n_stat // 2
    for peak in range(n_rows):
        if peak == center:
            continue
        rise = _linear_rows(mid + 1, center, peak, n_rows)
        fall = _linear_rows(n_stat - mid, peak, center, n_rows)
        _add(rise[:-1] + fall)
        if len(ordered) >= n:
            break
    third = max(2, n_stat // 3)
    for start in (center, max(0, center - 1), min(n_rows - 1, center + 1)):
        for end in range(n_rows):
            if end == start:
                continue
            hold = (start,) * third
            ramp = _linear_rows(n_stat - third + 1, start, end, n_rows)
            _add(hold + ramp[1:])
            ramp_first = _linear_rows(n_stat - third + 1, start, end, n_rows)
            _add(ramp_first[:-1] + ((end,) * third))
            if len(ordered) >= n:
                break
        if len(ordered) >= n:
            break
    if len(ordered) < n:
        for rows in _random_walks(
            lattice,
            needed=n - len(ordered),
            max_row_step=max_row_step,
            seen=seen,
        ):
            _add(rows)
            if len(ordered) >= n:
                break

    return tuple(
        build_candidate(lattice, rows, index=index, max_row_step=max_row_step)
        for index, rows in enumerate(ordered[:n])
    )


def posed_control_points(
    shape: ScraperShape,
    pose: ScraperPose,
) -> NDArray[np.float64]:
    """Apply SE(3) to a copy of the control points. Shape stays unmodified."""
    pts = np.asarray(shape.control_points_mm, dtype=np.float64)
    if len(pts) == 0:
        return pts
    homogeneous = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    posed = (pose_matrix(pose) @ homogeneous.T).T[:, :3]
    return np.asarray(posed, dtype=np.float64)


def _linear_rows(count: int, start: int, end: int, n_rows: int) -> tuple[int, ...]:
    if count <= 1:
        return (int(np.clip(start, 0, n_rows - 1)),)
    seq: list[int] = []
    for i in range(count):
        t = i / (count - 1)
        seq.append(int(round(start + t * (end - start))))
    return tuple(int(np.clip(v, 0, n_rows - 1)) for v in seq)


def _random_walks(
    lattice: ContactConstraintLattice,
    *,
    needed: int,
    max_row_step: int,
    seen: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(_WALK_SEED)
    n_stat = lattice.station_count
    n_rows = lattice.row_count
    found: list[tuple[int, ...]] = []
    attempts = 0
    limit = max(needed * 40, 200)
    while len(found) < needed and attempts < limit:
        attempts += 1
        rows = [int(rng.integers(0, n_rows))]
        prev_delta = 0
        for _ in range(n_stat - 1):
            choices: list[int] = []
            for delta in range(-max_row_step, max_row_step + 1):
                nxt = rows[-1] + delta
                if 0 <= nxt < n_rows and abs(delta - prev_delta) <= MAX_SECOND_DIFFERENCE:
                    choices.append(delta)
            if not choices:
                choices = [0]
            delta = int(choices[int(rng.integers(0, len(choices)))])
            rows.append(rows[-1] + delta)
            prev_delta = delta
        tup = tuple(rows)
        if tup in seen:
            continue
        ok, _reason = validate_row_sequence(
            lattice, tup, max_row_step=max_row_step
        )
        if ok:
            found.append(tup)
            seen.add(tup)
    return found


def _fingerprint(
    row_indices: tuple[int, ...],
    points: NDArray[np.float64],
) -> str:
    payload = (
        "rows="
        + ",".join(str(int(v)) for v in row_indices)
        + "|pts="
        + ",".join(f"{p[0]:.3f}:{p[1]:.3f}:{p[2]:.3f}" for p in points)
        + f"|t={BLADE_THICKNESS_MM}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _segment_distance(
    a0: NDArray[np.float64],
    a1: NDArray[np.float64],
    b0: NDArray[np.float64],
    b1: NDArray[np.float64],
) -> float:
    """Minimum distance between two 3D segments."""
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    uu = float(np.dot(u, u))
    vv = float(np.dot(v, v))
    uv = float(np.dot(u, v))
    uw = float(np.dot(u, w))
    vw = float(np.dot(v, w))
    denom = uu * vv - uv * uv
    if denom < 1e-12:
        t = 0.0
    else:
        t = float(np.clip((uv * vw - vv * uw) / denom, 0.0, 1.0))
    s = float(np.clip((uv * t + vw) / max(vv, 1e-12), 0.0, 1.0)) if vv > 1e-12 else 0.0
    if uu > 1e-12:
        t = float(np.clip((uv * s - uw) / uu, 0.0, 1.0))
    delta = (a0 + u * t) - (b0 + v * s)
    return float(np.linalg.norm(delta))
