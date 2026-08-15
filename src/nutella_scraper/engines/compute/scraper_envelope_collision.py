"""Hard non-penetration constraint: rigid scraper vs interior envelope.

The interior product surface is a physical boundary. A pose is ADMISSIBLE
only if the scraper volume stays on the interior side with
``distance >= clearance_mm``. Penetration is diagnostic, never a valid pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import NUMERIC_GAP_MM
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
    eps = 0.25
    plus = closest + normals * eps
    minus = closest - normals * eps
    r_plus = np.hypot(plus[:, 0], plus[:, 2])
    r_minus = np.hypot(minus[:, 0], minus[:, 2])
    normals[r_plus > r_minus] *= -1.0
    return normals


def _signed_interior(
    surface_mesh: trimesh.Trimesh,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interior-positive signed distance and unsigned nearest-surface distance."""
    closest, distances, tri_ids = surface_mesh.nearest.on_surface(points)
    closest = np.asarray(closest, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    tri_ids = np.asarray(tri_ids, dtype=np.int64)
    inward = _inward_normals(surface_mesh, closest, tri_ids)
    signed = np.sum((points - closest) * inward, axis=1)
    return signed, distances


def _near_wall_edge_face_samples(
    mesh: trimesh.Trimesh,
    vertex_distances: NDArray[np.float64],
    band_mm: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Edge midpoints + face centroids whose vertices sit near the envelope."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    near = vertex_distances <= float(band_mm)
    chunks: list[NDArray[np.float64]] = []
    kinds: list[NDArray[np.int64]] = []

    unique_edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(unique_edges) > 0:
        edge_near = near[unique_edges[:, 0]] | near[unique_edges[:, 1]]
        if np.any(edge_near):
            a = vertices[unique_edges[edge_near, 0]]
            b = vertices[unique_edges[edge_near, 1]]
            mid = 0.5 * (a + b)
            chunks.append(mid)
            kinds.append(np.ones(len(mid), dtype=np.int64))

    if len(faces) > 0:
        face_near = near[faces].any(axis=1)
        if np.any(face_near):
            centroids = vertices[faces[face_near]].mean(axis=1)
            chunks.append(centroids)
            kinds.append(np.full(int(np.count_nonzero(face_near)), 2, dtype=np.int64))

    if not chunks:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.vstack(chunks), np.concatenate(kinds)


def evaluate_envelope_collision(
    posed_mesh: trimesh.Trimesh,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
) -> EnvelopeCollisionReport:
    """
    Hard constraint: scraper volume must stay interior-side of the envelope.

    Vertices are a fast reject. Edges and faces near the wall are required
    before a pose is declared VALID — a triangle can cross while all of its
    vertices remain inside. Clearance is unsigned distance everywhere.
    """
    if posed_mesh.is_empty or len(posed_mesh.vertices) == 0:
        raise ValueError("Empty scraper mesh")

    surface_mesh = surface.to_trimesh()
    eps = _contact_eps_mm(surface_mesh)
    vertices = np.asarray(posed_mesh.vertices, dtype=np.float64)
    signed, distances = _signed_interior(surface_mesh, vertices)
    max_outward = float(np.max(-signed))
    min_signed = float(np.min(signed))
    min_unsigned = float(np.min(distances))
    vertex_hit = bool(max_outward > eps)
    edge_hit = False
    face_hit = False

    if not vertex_hit:
        band = (
            float(parameters.thickness_mm)
            + float(parameters.clearance_mm)
            + _WALL_BAND_EXTRA_MM
        )
        extra, kinds = _near_wall_edge_face_samples(posed_mesh, distances, band)
        if len(extra) > 0:
            extra_signed, extra_dist = _signed_interior(surface_mesh, extra)
            max_outward = max(max_outward, float(np.max(-extra_signed)))
            min_signed = min(min_signed, float(np.min(extra_signed)))
            min_unsigned = min(min_unsigned, float(np.min(extra_dist)))
            outward_mask = (-extra_signed) > eps
            edge_hit = bool(np.any(outward_mask[kinds == 1])) if np.any(kinds == 1) else False
            face_hit = bool(np.any(outward_mask[kinds == 2])) if np.any(kinds == 2) else False

    clearance = float(parameters.clearance_mm)
    has_collision = bool(max_outward > eps)
    # Tessellation chords sit slightly inside the true CAD wall; allow the
    # same numeric band used for tangency, never a physical hole.
    required = max(0.0, clearance - NUMERIC_GAP_MM)
    clearance_ok = bool(min_unsigned + eps + 1e-9 >= required)
    admissible = (not has_collision) and clearance_ok
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
    progress = float(parameters.surface_progress_deg)
    if abs(progress) <= 1e-9:
        nominal = artifact.design_frame
        transform = np.eye(4, dtype=np.float64)
        posed = artifact.mesh.copy()
    else:
        nominal = envelope_contact_frame(surface, parameters)
        transform = rigid_transform_between_frames(artifact.design_frame, nominal)
        posed = apply_rigid_transform(artifact.mesh, transform)

    report = evaluate_envelope_collision(posed, surface, parameters)
    if report.admissible:
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
        return _result(
            posed_mesh=trial_mesh,
            frame=candidate,
            transform=trial_tf,
            collision=trial_report,
            artifact=artifact,
            status="VALID",
            alternative_used=True,
        )

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
