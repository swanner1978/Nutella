"""Hard non-penetration constraint: rigid scraper vs interior envelope.

The interior product surface is a physical boundary. A pose is ADMISSIBLE
only if the scraper volume stays on the interior side with
``distance >= clearance_mm``. Penetration is diagnostic, never a valid pose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.envelope_surface_proximity import (
    closest_on_envelope_surface,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import (
    NUMERIC_GAP_MM,
    scraper_length_span,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    EnvelopeContactFrame,
    RigidScraperArtifact,
    apply_rigid_transform,
    contact_frame_from_tip,
    envelope_contact_frame,
    rigid_transform_between_frames,
    transform_points,
)
from nutella_scraper.engines.compute.scraper_transform import pose_matrix

# Tessellation / tangency band. Above this, glass-side volume is a collision.
_CONTACT_EPS_MM = 0.15
# Extra edge/face samples around the wall catch a triangle that crosses
# with all vertices still interior.
_WALL_BAND_EXTRA_MM = 3.0

# Last pose vs collision split (ms). Diagnostic only — not used by the solver.
LAST_POSE_COLLISION_MS: dict[str, float] = {"pose_ms": 0.0, "collision_ms": 0.0}


@dataclass(frozen=True)
class EnvelopeCollisionReport:
    """Volume-aware collision result against the interior envelope."""

    has_collision: bool
    admissible: bool
    min_signed_interior_mm: float
    max_outward_mm: float
    min_unsigned_distance_mm: float
    clearance_mm: float
    vertex_hit: bool
    edge_hit: bool
    face_hit: bool
    clearance_ok: bool = True
    contact_face_ids: frozenset[int] = frozenset()

    @property
    def status(self) -> str:
        return "VALID" if self.admissible else "INVALID"


def _contact_eps_mm(surface_mesh: trimesh.Trimesh) -> float:
    """Numeric contact band from tessellation size, never a physical hole."""
    lengths = np.asarray(getattr(surface_mesh, "edges_unique_length", []), dtype=np.float64)
    if lengths.size == 0:
        return _CONTACT_EPS_MM
    return float(max(_CONTACT_EPS_MM, 0.04 * float(np.median(lengths))))


def _inward_normals(
    surface_mesh: trimesh.Trimesh,
    closest: NDArray[np.float64],
    tri_ids: NDArray[np.int64],
) -> NDArray[np.float64]:
    face_normals = np.asarray(surface_mesh.face_normals, dtype=np.float64)
    normals = face_normals[np.clip(tri_ids, 0, len(face_normals) - 1)].copy()
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-9)
    bounds = np.asarray(surface_mesh.bounds, dtype=np.float64)
    target = np.asarray(
        [0.0, 0.5 * (float(bounds[0, 1]) + float(bounds[1, 1])), 0.0],
        dtype=np.float64,
    )
    to_inside = target[None, :] - closest
    normals[np.sum(normals * to_inside, axis=1) < 0.0] *= -1.0
    return normals


def _proximity(
    surface_mesh: trimesh.Trimesh,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Interior-positive signed distance, unsigned distance, nearest triangle ids."""
    closest, distances, tri_ids = closest_on_envelope_surface(surface_mesh, points)
    closest = np.asarray(closest, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    tri_ids = np.asarray(tri_ids, dtype=np.int64)
    inward = _inward_normals(surface_mesh, closest, tri_ids)
    signed = np.sum((points - closest) * inward, axis=1)
    return signed, distances, tri_ids


def _signed_interior(
    surface_mesh: trimesh.Trimesh,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interior-positive signed distance and unsigned nearest-surface distance."""
    signed, distances, _tri_ids = _proximity(surface_mesh, points)
    return signed, distances


def _empty_points() -> NDArray[np.float64]:
    return np.zeros((0, 3), dtype=np.float64)


def _empty_ids() -> NDArray[np.int64]:
    return np.zeros((0,), dtype=np.int64)


def _edge_and_face_samples(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    unique_edges: NDArray[np.int64],
    vertex_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Edge midpoints + face centroids for features touching ``vertex_mask``."""
    if len(unique_edges) > 0:
        edge_sel = vertex_mask[unique_edges[:, 0]] | vertex_mask[unique_edges[:, 1]]
    else:
        edge_sel = np.zeros((0,), dtype=np.bool_)
    if np.any(edge_sel):
        a = vertices[unique_edges[edge_sel, 0]]
        b = vertices[unique_edges[edge_sel, 1]]
        mids = 0.5 * (a + b)
    else:
        mids = _empty_points()

    if len(faces) > 0:
        face_sel = vertex_mask[np.asarray(faces, dtype=np.int64)].any(axis=1)
    else:
        face_sel = np.zeros((0,), dtype=np.bool_)
    if np.any(face_sel):
        centroids = vertices[faces[face_sel]].mean(axis=1)
    else:
        centroids = _empty_points()
    return mids, centroids, edge_sel, face_sel


def _near_wall_edge_face_samples(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    unique_edges: NDArray[np.int64],
    vertex_distances: NDArray[np.float64],
    band_mm: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Edge midpoints + face centroids whose vertices sit near the envelope."""
    near = np.asarray(vertex_distances, dtype=np.float64) <= float(band_mm)
    mids, centroids, _edge_sel, _face_sel = _edge_and_face_samples(
        vertices, faces, unique_edges, near
    )
    chunks: list[NDArray[np.float64]] = []
    kinds: list[NDArray[np.int64]] = []
    if len(mids) > 0:
        chunks.append(mids)
        kinds.append(np.ones(len(mids), dtype=np.int64))
    if len(centroids) > 0:
        chunks.append(centroids)
        kinds.append(np.full(len(centroids), 2, dtype=np.int64))
    if not chunks:
        return _empty_points(), _empty_ids()
    return np.vstack(chunks), np.concatenate(kinds)


def _useful_corridor_mask(
    points: NDArray[np.float64],
    lower_y: float,
    opening_y: float,
) -> NDArray[np.bool_]:
    """True for samples inside the interior envelope height (floor → opening)."""
    y = np.asarray(points, dtype=np.float64)[:, 1]
    return (y >= float(lower_y) - 1e-3) & (y <= float(opening_y))


def _proximity_vertices_and_wall_extras(
    surface_mesh: trimesh.Trimesh,
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    unique_edges: NDArray[np.int64],
    in_band: NDArray[np.bool_],
    *,
    lower_y: float,
    opening_y: float,
    band_mm: float,
    eps: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    bool,
]:
    """Vertex nearest, then extra nearest only if vertices did not already penetrate.

    Extra samples (edge midpoints / face centroids) are different query points.
    They stay mandatory when vertices stay interior: a triangle can still cross
    the envelope, and extras can add contact-face ids. Concatenating a
    corridor superset into the first ``on_surface`` call is equivalent but
    slower, so the second query is skipped only when it cannot change the
    decision (vertex penetration, or no extra samples in-band).
    """
    signed, distances, tri_ids = _proximity(surface_mesh, vertices[in_band])
    vertex_hit = bool(float(np.max(-signed)) > eps)
    empty_s = np.zeros((0,), dtype=np.float64)
    if vertex_hit:
        return (
            signed,
            distances,
            tri_ids,
            empty_s,
            empty_s,
            _empty_ids(),
            _empty_ids(),
            vertex_hit,
        )

    distances_all = np.full(len(vertices), 1e9, dtype=np.float64)
    distances_all[in_band] = distances
    extra, kinds = _near_wall_edge_face_samples(
        vertices, faces, unique_edges, distances_all, band_mm
    )
    if len(extra) > 0:
        extra_in = _useful_corridor_mask(extra, lower_y, opening_y)
        extra = extra[extra_in]
        kinds = kinds[extra_in]
    if len(extra) == 0:
        return (
            signed,
            distances,
            tri_ids,
            empty_s,
            empty_s,
            _empty_ids(),
            _empty_ids(),
            vertex_hit,
        )

    extra_signed, extra_dist, extra_tri = _proximity(surface_mesh, extra)
    return (
        signed,
        distances,
        tri_ids,
        extra_signed,
        extra_dist,
        extra_tri,
        kinds,
        vertex_hit,
    )


def evaluate_envelope_collision(
    posed_mesh: trimesh.Trimesh,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
    *,
    surface_mesh: trimesh.Trimesh | None = None,
    vertices: NDArray[np.float64] | None = None,
    faces: NDArray[np.int64] | None = None,
    edges_unique: NDArray[np.int64] | None = None,
) -> EnvelopeCollisionReport:
    """
    Hard constraint: scraper volume must stay interior-side of the envelope.

    Collision is evaluated from the interior floor to the opening. Geometry
    above the opening is outside the working volume and is ignored.

    Vertices are a fast reject. Edges and faces near the wall are required
    before a pose is declared VALID — a triangle can cross while all of its
    vertices remain inside. Clearance is unsigned distance in-band.

    Optional ``vertices`` / ``faces`` / ``edges_unique`` apply a rigid pose
    without copying the scraper mesh: topology stays immutable, only the
    vertex array moves.
    """
    verts = np.asarray(
        vertices if vertices is not None else posed_mesh.vertices,
        dtype=np.float64,
    )
    if verts.size == 0:
        raise ValueError("Empty scraper mesh")
    face_idx = np.asarray(
        faces if faces is not None else posed_mesh.faces,
        dtype=np.int64,
    )
    edge_idx = np.asarray(
        edges_unique if edges_unique is not None else posed_mesh.edges_unique,
        dtype=np.int64,
    )

    surface_mesh = surface_mesh if surface_mesh is not None else surface.to_trimesh()
    eps = _contact_eps_mm(surface_mesh)
    opening_y, lower_y, _max_length = scraper_length_span(surface)
    in_band = _useful_corridor_mask(verts, lower_y, opening_y)
    clearance = float(parameters.clearance_mm)
    if not np.any(in_band):
        return EnvelopeCollisionReport(
            has_collision=False,
            admissible=True,
            min_signed_interior_mm=float("inf"),
            max_outward_mm=0.0,
            min_unsigned_distance_mm=float("inf"),
            clearance_mm=clearance,
            vertex_hit=False,
            edge_hit=False,
            face_hit=False,
            clearance_ok=True,
        )

    band = (
        float(parameters.thickness_mm)
        + float(parameters.clearance_mm)
        + _WALL_BAND_EXTRA_MM
    )
    (
        signed,
        distances,
        tri_ids,
        extra_signed,
        extra_dist,
        extra_tri,
        extra_kinds,
        vertex_hit,
    ) = _proximity_vertices_and_wall_extras(
        surface_mesh,
        verts,
        face_idx,
        edge_idx,
        in_band,
        lower_y=lower_y,
        opening_y=opening_y,
        band_mm=band,
        eps=eps,
    )
    max_outward = float(np.max(-signed))
    min_signed = float(np.min(signed))
    min_unsigned = float(np.min(distances))
    edge_hit = False
    face_hit = False
    if len(extra_signed) > 0:
        max_outward = max(max_outward, float(np.max(-extra_signed)))
        min_signed = min(min_signed, float(np.min(extra_signed)))
        min_unsigned = min(min_unsigned, float(np.min(extra_dist)))
        outward_mask = (-extra_signed) > eps
        has_edge_kind = np.any(extra_kinds == 1)
        has_face_kind = np.any(extra_kinds == 2)
        edge_hit = bool(np.any(outward_mask[extra_kinds == 1])) if has_edge_kind else False
        face_hit = bool(np.any(outward_mask[extra_kinds == 2])) if has_face_kind else False

    has_collision = bool(max_outward > eps)
    # Tessellation chords sit slightly inside the true CAD wall; allow the
    # same numeric band used for tangency, never a physical hole.
    required = max(0.0, clearance - NUMERIC_GAP_MM)
    clearance_ok = bool(min_unsigned + eps + 1e-9 >= required)
    admissible = (not has_collision) and clearance_ok
    all_signed = np.concatenate([signed, extra_signed]) if len(extra_signed) else signed
    all_dist = np.concatenate([distances, extra_dist]) if len(extra_dist) else distances
    all_tri = np.concatenate([tri_ids, extra_tri]) if len(extra_tri) else tri_ids
    contact_mask = (all_dist <= eps + 1e-9) & (all_signed >= -eps)
    contact_ids = (
        frozenset(int(v) for v in all_tri[contact_mask])
        if np.any(contact_mask)
        else frozenset()
    )
    return EnvelopeCollisionReport(
        has_collision=has_collision,
        admissible=admissible,
        min_signed_interior_mm=min_signed,
        max_outward_mm=max(0.0, max_outward),
        min_unsigned_distance_mm=min_unsigned,
        clearance_mm=clearance,
        vertex_hit=vertex_hit,
        edge_hit=edge_hit,
        face_hit=face_hit,
        clearance_ok=clearance_ok,
        contact_face_ids=contact_ids,
    )


def _offset_frame(
    base: EnvelopeContactFrame,
    *,
    inward_mm: float = 0.0,
    along_length_mm: float = 0.0,
    along_width_mm: float = 0.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> EnvelopeContactFrame:
    """Perturb a contact frame in its local 6-DOF without reshaping the solid."""
    origin = (
        np.asarray(base.origin_mm, dtype=np.float64)
        + np.asarray(base.inward_normal, dtype=np.float64) * float(inward_mm)
        + np.asarray(base.rotation[:, 1], dtype=np.float64) * float(along_length_mm)
        + np.asarray(base.rotation[:, 2], dtype=np.float64) * float(along_width_mm)
    )
    extra = pose_matrix(
        ScraperPose(
            position_mm=(0.0, 0.0, 0.0),
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg),
        )
    )[:3, :3]
    rotation = np.asarray(base.rotation, dtype=np.float64) @ extra
    normal = rotation[:, 0]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    return contact_frame_from_tip(
        tip_origin_mm=origin,
        inward_normal=normal,
        wall_point_mm=np.asarray(base.wall_point_mm, dtype=np.float64),
        surface_progress_deg=float(base.surface_progress_deg),
    )


def rigid_pose_neighborhood(
    nominal: EnvelopeContactFrame,
) -> tuple[EnvelopeContactFrame, ...]:
    """Small 6-DOF neighbourhood around a nominal envelope pose. Rigid only."""
    return tuple(_candidate_frames(nominal))


def envelope_contact_face_ids(
    posed_mesh: trimesh.Trimesh,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
    *,
    surface_mesh: trimesh.Trimesh | None = None,
) -> frozenset[int]:
    """Interior-triangle ids within the collision contact band, jar-side only.

    Penetrating / glass-side samples are rejected. Uses the same proximity
    primitive as ``evaluate_envelope_collision``.
    """
    if posed_mesh.is_empty or len(posed_mesh.vertices) == 0:
        return frozenset()
    return evaluate_envelope_collision(
        posed_mesh, surface, parameters, surface_mesh=surface_mesh
    ).contact_face_ids


def _candidate_frames(nominal: EnvelopeContactFrame) -> list[EnvelopeContactFrame]:
    """Small SE(3) neighbourhood around the nominal envelope pose."""
    frames = [nominal]
    for inward in (0.5, 1.0, 1.8, 3.0):
        frames.append(_offset_frame(nominal, inward_mm=inward))
    for dy in (-1.5, 1.5):
        frames.append(_offset_frame(nominal, along_length_mm=dy, inward_mm=0.8))
    for dw in (-1.0, 1.0):
        frames.append(_offset_frame(nominal, along_width_mm=dw, inward_mm=0.8))
    for yaw, inward in ((-6.0, 1.0), (6.0, 1.0), (-10.0, 1.6), (10.0, 1.6)):
        frames.append(_offset_frame(nominal, yaw_deg=yaw, inward_mm=inward))
    for pitch in (-4.0, 4.0):
        frames.append(_offset_frame(nominal, pitch_deg=pitch, inward_mm=1.0))
    for roll in (-4.0, 4.0):
        frames.append(_offset_frame(nominal, roll_deg=roll, inward_mm=1.0))
    return frames


@dataclass(frozen=True)
class AdmissiblePoseResult:
    """Outcome of posing a rigid scraper under the hard envelope constraint."""

    posed_mesh: trimesh.Trimesh
    pose: ScraperPose
    transform: NDArray[np.float64]
    frame: EnvelopeContactFrame
    collision: EnvelopeCollisionReport
    status: str  # VALID | INVALID | BLOCKED
    alternative_used: bool
    blocked: bool
    wall_edge_mm: NDArray[np.float64]


def _result(
    *,
    posed_mesh: trimesh.Trimesh,
    frame: EnvelopeContactFrame,
    transform: NDArray[np.float64],
    collision: EnvelopeCollisionReport,
    artifact: RigidScraperArtifact,
    status: str,
    alternative_used: bool,
) -> AdmissiblePoseResult:
    pose = ScraperPose(
        position_mm=(
            float(frame.origin_mm[0]),
            float(frame.origin_mm[1]),
            float(frame.origin_mm[2]),
        ),
        yaw_deg=float(frame.surface_progress_deg),
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    return AdmissiblePoseResult(
        posed_mesh=posed_mesh,
        pose=pose,
        transform=transform,
        frame=frame,
        collision=collision,
        status=status,
        alternative_used=alternative_used,
        blocked=status == "BLOCKED",
        wall_edge_mm=transform_points(artifact.wall_edge_mm, transform),
    )


def pose_rigid_scraper_admissible(
    artifact: RigidScraperArtifact,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
) -> AdmissiblePoseResult:
    """
    Transform the cached rigid solid, then accept only a non-penetrating pose.

    Search order: nominal envelope frame, then a small 6-DOF neighbourhood.
    If none is admissible the last candidate is returned as BLOCKED — the
    solid is never deformed and never forced through the envelope.
    """
    pose_started = time.perf_counter()
    progress = float(parameters.surface_progress_deg)
    if abs(progress) <= 1e-9:
        nominal = artifact.design_frame
        transform = np.eye(4, dtype=np.float64)
        posed = artifact.mesh.copy()
    else:
        nominal = envelope_contact_frame(surface, parameters)
        transform = rigid_transform_between_frames(artifact.design_frame, nominal)
        posed = apply_rigid_transform(artifact.mesh, transform)
    LAST_POSE_COLLISION_MS["pose_ms"] = (time.perf_counter() - pose_started) * 1000.0

    collision_started = time.perf_counter()
    report = evaluate_envelope_collision(posed, surface, parameters)
    if report.admissible:
        LAST_POSE_COLLISION_MS["collision_ms"] = (
            time.perf_counter() - collision_started
        ) * 1000.0
        return _result(
            posed_mesh=posed,
            frame=nominal,
            transform=transform,
            collision=report,
            artifact=artifact,
            status="VALID",
            alternative_used=False,
        )

    for candidate in _candidate_frames(nominal)[1:]:
        try:
            trial_tf = rigid_transform_between_frames(artifact.design_frame, candidate)
        except np.linalg.LinAlgError:
            continue
        trial_mesh = apply_rigid_transform(artifact.mesh, trial_tf)
        trial_report = evaluate_envelope_collision(trial_mesh, surface, parameters)
        if not trial_report.admissible:
            continue
        LAST_POSE_COLLISION_MS["collision_ms"] = (
            time.perf_counter() - collision_started
        ) * 1000.0
        return _result(
            posed_mesh=trial_mesh,
            frame=candidate,
            transform=trial_tf,
            collision=trial_report,
            artifact=artifact,
            status="VALID",
            alternative_used=True,
        )

    LAST_POSE_COLLISION_MS["collision_ms"] = (
        time.perf_counter() - collision_started
    ) * 1000.0
    return _result(
        posed_mesh=posed,
        frame=nominal,
        transform=transform,
        collision=report,
        artifact=artifact,
        status="BLOCKED",
        alternative_used=False,
    )


def collision_payload(
    result: AdmissiblePoseResult,
    parameters: ScraperParameters,
) -> dict[str, object]:
    """Viewer / API diagnostic — not a coverage score."""
    report = result.collision
    collision_yes = bool(report.has_collision) and result.status != "VALID"
    return {
        "surface_progress_deg": float(parameters.surface_progress_deg),
        "rotation_angle_deg": float(parameters.surface_progress_deg),
        "collision": "YES" if collision_yes else "NO",
        "has_collision": collision_yes,
        "pose_status": result.status,
        "admissible": bool(result.status == "VALID"),
        "blocked": bool(result.blocked),
        "alternative_used": bool(result.alternative_used),
        "clearance_mm": float(report.clearance_mm),
        "clearance_min_mm": float(report.min_unsigned_distance_mm),
        "clearance_ok": bool(report.clearance_ok),
        "penetration_mm": float(report.max_outward_mm) if collision_yes else 0.0,
        "vertex_hit": bool(report.vertex_hit) if collision_yes else False,
        "edge_hit": bool(report.edge_hit) if collision_yes else False,
        "face_hit": bool(report.face_hit) if collision_yes else False,
        "distance_min_mm": float(report.min_unsigned_distance_mm),
        "distance_max_mm": float(report.min_unsigned_distance_mm),
    }
