"""Rigid scraper motion along the interior envelope — pose only, no reshape.

SCRAPER GEOMETRY  ≠  SCRAPER POSE  ≠  SURFACE ENVELOPE

Manufacturing geometry is lofted once at a design approach (progress = 0).
``surface_progress_deg`` only selects a contact frame on the interior surface
and applies a rigid SE(3) transform. The envelope is a reference, never a
deformer of the FDM solid.
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
from nutella_scraper.engines.compute.scraper_envelope_path import (
    NUMERIC_GAP_MM,
    EnvelopeStation,
    ScraperEnvelopePath,
    ScraperEnvelopePathBuilder,
    interior_centroid_mm,
)
from nutella_scraper.engines.compute.scraper_geometry_generator import (
    ScraperGeometryGenerator,
)


@dataclass(frozen=True)
class EnvelopeContactFrame:
    """Local jar-frame basis at one envelope contact location.

    Columns of ``rotation`` are world axes of the scraper local frame:
      column 0 — inward surface normal (thickness / into jar)
      column 1 — length (along height, orthogonalized)
      column 2 — width (along contour)
    ``origin_mm`` is the active-tip mid point (wall + clearance).
    """

    origin_mm: NDArray[np.float64]
    rotation: NDArray[np.float64]
    wall_point_mm: NDArray[np.float64]
    inward_normal: NDArray[np.float64]
    surface_progress_deg: float

    def matrix(self) -> NDArray[np.float64]:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(self.rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(self.origin_mm, dtype=np.float64)
        return transform


@dataclass(frozen=True)
class RigidScraperArtifact:
    """Cached manufacturing solid + design-frame tip edge (progress = 0)."""

    mesh: trimesh.Trimesh
    design_frame: EnvelopeContactFrame
    tip_edge_mm: NDArray[np.float64]
    wall_edge_mm: NDArray[np.float64]
    design_path: ScraperEnvelopePath
    shape_fingerprint: str


def manufacturing_fingerprint(parameters: ScraperParameters, *, model_id: str) -> str:
    """Shape identity — excludes surface_progress / pose-only fields."""
    return (
        f"{model_id}|"
        f"w={parameters.width_mm:.6g}|"
        f"l={parameters.length_mm:.6g}|"
        f"t={parameters.thickness_mm:.6g}|"
        f"z={parameters.position_z_mm:.6g}|"
        f"bevel={parameters.bevel_angle_deg:.6g}|"
        f"relief={parameters.relief_angle_deg:.6g}|"
        f"helix={parameters.helix_rate_deg_per_mm:.6g}|"
        f"clear={parameters.clearance_mm:.6g}"
    )


def design_parameters(parameters: ScraperParameters) -> ScraperParameters:
    """Parameters used to loft the rigid solid (progress forced to 0)."""
    return parameters.with_updates(surface_progress_deg=0.0, rotation_angle_deg=0.0)


def contact_frame_from_tip(
    *,
    tip_origin_mm: NDArray[np.float64],
    inward_normal: NDArray[np.float64],
    wall_point_mm: NDArray[np.float64] | None = None,
    surface_progress_deg: float = 0.0,
) -> EnvelopeContactFrame:
    normal = np.asarray(inward_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    length = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    length = length - normal * float(np.dot(length, normal))
    length_n = float(np.linalg.norm(length))
    if length_n <= 1e-9:
        length = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        length = length / length_n
    width = np.cross(normal, length)
    width = width / max(float(np.linalg.norm(width)), 1e-9)
    length = np.cross(width, normal)
    length = length / max(float(np.linalg.norm(length)), 1e-9)
    origin = np.asarray(tip_origin_mm, dtype=np.float64)
    wall = (
        np.asarray(wall_point_mm, dtype=np.float64)
        if wall_point_mm is not None
        else origin - normal * NUMERIC_GAP_MM
    )
    return EnvelopeContactFrame(
        origin_mm=origin,
        rotation=np.column_stack((normal, length, width)),
        wall_point_mm=wall,
        inward_normal=normal,
        surface_progress_deg=float(surface_progress_deg),
    )


def envelope_contact_frame(
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
    *,
    surface_progress_deg: float | None = None,
    surface_mesh: trimesh.Trimesh | None = None,
) -> EnvelopeContactFrame:
    """
    Contact frame on the interior envelope at a path progress.

    Progress selects where along the horizontal contour the tip mid sits.
    It is NOT a forced spin of the solid about the jar axis.

    ``surface_mesh`` reuses an already-built interior mesh and its spatial
    index. The mesh is not copied and must not be mutated by the caller.
    """
    progress = float(
        parameters.surface_progress_deg
        if surface_progress_deg is None
        else surface_progress_deg
    )
    mesh = surface_mesh if surface_mesh is not None else surface.to_trimesh()
    y_min = float(surface.y_min_mm)
    y_max = float(surface.y_max_mm)
    y_mm = float(np.clip(parameters.position_z_mm, y_min, y_max))
    clearance = float(parameters.clearance_mm) + NUMERIC_GAP_MM

    builder = ScraperEnvelopePathBuilder()
    contour = builder._horizontal_contour(mesh, y_mm)
    yaw = float(np.deg2rad(progress))
    ux = float(np.cos(yaw))
    uz = float(-np.sin(yaw))
    proj = contour[:, 0] * ux + contour[:, 2] * uz
    seed = np.asarray(contour[int(np.argmax(proj))], dtype=np.float64).reshape(1, 3)
    closest, _dist, tri_ids = mesh.nearest.on_surface(seed)
    wall = np.asarray(closest[0], dtype=np.float64)
    normal = builder._inward_normal_at(mesh, wall, int(tri_ids[0]))
    origin = wall + normal * clearance
    return contact_frame_from_tip(
        tip_origin_mm=origin,
        inward_normal=normal,
        wall_point_mm=wall,
        surface_progress_deg=progress,
    )


def rigid_transform_between_frames(
    design_frame: EnvelopeContactFrame,
    target_frame: EnvelopeContactFrame,
) -> NDArray[np.float64]:
    """SE(3) map that takes the design contact frame onto the target frame."""
    return target_frame.matrix() @ np.linalg.inv(design_frame.matrix())


def apply_rigid_transform(
    mesh: trimesh.Trimesh,
    transform: NDArray[np.float64],
) -> trimesh.Trimesh:
    posed = mesh.copy()
    posed.apply_transform(np.asarray(transform, dtype=np.float64))
    return posed


def transform_points(
    points: NDArray[np.float64],
    transform: NDArray[np.float64],
) -> NDArray[np.float64]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    homogeneous = np.concatenate(
        [pts, np.ones((len(pts), 1), dtype=np.float64)],
        axis=1,
    )
    out = homogeneous @ np.asarray(transform, dtype=np.float64).T
    return np.asarray(out[:, :3], dtype=np.float64)


def build_rigid_scraper_artifact(
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
) -> RigidScraperArtifact:
    """Loft the manufacturing solid once at design progress = 0."""
    shape = design_parameters(parameters)
    path = ScraperEnvelopePathBuilder().build(surface, shape)
    mesh = ScraperGeometryGenerator().generate(shape, surface, path=path)
    mid = path.stations[len(path.stations) // 2]
    tip_edge = np.asarray(mid.tip_points_mm, dtype=np.float64)
    wall_edge = np.asarray(mid.wall_points_mm, dtype=np.float64)
    tip_mid = tip_edge[len(tip_edge) // 2]
    normal = np.asarray(mid.inward_normals[len(tip_edge) // 2], dtype=np.float64)
    design_frame = contact_frame_from_tip(
        tip_origin_mm=tip_mid,
        inward_normal=normal,
        wall_point_mm=wall_edge[len(wall_edge) // 2],
        surface_progress_deg=0.0,
    )
    return RigidScraperArtifact(
        mesh=mesh,
        design_frame=design_frame,
        tip_edge_mm=tip_edge,
        wall_edge_mm=wall_edge,
        design_path=path,
        shape_fingerprint=manufacturing_fingerprint(
            shape,
            model_id=surface.model_id,
        ),
    )


def remap_envelope_path_to_wall_curve(
    reference_path: ScraperEnvelopePath,
    wall_curve_mm: NDArray[np.float64],
    surface: InteriorSurfaceReference,
) -> ScraperEnvelopePath:
    """Rigidly map each A-station onto a frozen candidate centreline.

    Stations keep their relative width chord and thickness frame. Only the
    local SE(3) frame at the wall mid-point changes. The candidate curve is
    not resampled and A is not rebuilt.
    """
    curve = np.asarray(wall_curve_mm, dtype=np.float64)
    stations = reference_path.stations
    if curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError("wall curve must be an (N, 3) point list")
    if len(stations) != len(curve):
        raise ValueError(
            "wall curve length must match the reference envelope station count"
        )
    mesh = surface.to_trimesh()
    snapped, _dist, tri_ids = mesh.nearest.on_surface(curve)
    snapped = np.asarray(snapped, dtype=np.float64)
    builder = ScraperEnvelopePathBuilder()
    interior_target = interior_centroid_mm(surface)
    remapped: list[EnvelopeStation] = []
    for index, station in enumerate(stations):
        wall0 = np.asarray(station.wall_points_mm, dtype=np.float64)
        mid = len(wall0) // 2
        src_wall = np.asarray(wall0[mid], dtype=np.float64)
        src_normal = np.asarray(station.inward_normals[mid], dtype=np.float64)
        dst_wall = np.asarray(snapped[index], dtype=np.float64)
        dst_normal = builder._inward_normal_at(
            mesh,
            dst_wall,
            int(tri_ids[index]),
            interior_target=interior_target,
        )
        src_frame = contact_frame_from_tip(
            tip_origin_mm=src_wall,
            inward_normal=src_normal,
            wall_point_mm=src_wall,
        )
        dst_frame = contact_frame_from_tip(
            tip_origin_mm=dst_wall,
            inward_normal=dst_normal,
            wall_point_mm=dst_wall,
        )
        transform = rigid_transform_between_frames(src_frame, dst_frame)
        rotation = np.asarray(transform[:3, :3], dtype=np.float64)
        wall = transform_points(wall0, transform)
        tip = transform_points(station.tip_points_mm, transform)
        normals = np.asarray(station.inward_normals, dtype=np.float64) @ rotation.T
        tangent = rotation @ np.asarray(station.tangent_length, dtype=np.float64)
        remapped.append(
            EnvelopeStation(
                s_mm=float(station.s_mm),
                y_mm=float(wall[mid, 1]),
                tip_points_mm=tip,
                inward_normals=normals,
                tangent_length=tangent,
                wall_points_mm=wall,
            )
        )
    return ScraperEnvelopePath(
        stations=tuple(remapped),
        source=reference_path.source,
    )


def build_rigid_scraper_artifact_from_path(
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
    path: ScraperEnvelopePath,
    *,
    shape_fingerprint: str,
) -> RigidScraperArtifact:
    """Loft a frozen envelope path once. Fingerprint is the caller's identity."""
    shape = design_parameters(parameters)
    mesh = ScraperGeometryGenerator().generate(shape, surface, path=path)
    mid = path.stations[len(path.stations) // 2]
    tip_edge = np.asarray(mid.tip_points_mm, dtype=np.float64)
    wall_edge = np.asarray(mid.wall_points_mm, dtype=np.float64)
    tip_mid = tip_edge[len(tip_edge) // 2]
    normal = np.asarray(mid.inward_normals[len(tip_edge) // 2], dtype=np.float64)
    design_frame = contact_frame_from_tip(
        tip_origin_mm=tip_mid,
        inward_normal=normal,
        wall_point_mm=wall_edge[len(wall_edge) // 2],
        surface_progress_deg=0.0,
    )
    return RigidScraperArtifact(
        mesh=mesh,
        design_frame=design_frame,
        tip_edge_mm=tip_edge,
        wall_edge_mm=wall_edge,
        design_path=path,
        shape_fingerprint=str(shape_fingerprint),
    )


def pose_rigid_scraper(
    artifact: RigidScraperArtifact,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
) -> tuple[trimesh.Trimesh, ScraperPose, NDArray[np.float64], EnvelopeContactFrame]:
    """
    Apply a free pose along the envelope to a cached rigid solid.

    Returns posed mesh, pose metadata, transform, and target contact frame.
    At progress = 0 the manufacturing mesh is returned unchanged (identity).
    """
    progress = float(parameters.surface_progress_deg)
    if abs(progress) <= 1e-9:
        posed = artifact.mesh.copy()
        pose = ScraperPose(
            position_mm=(
                float(artifact.design_frame.origin_mm[0]),
                float(artifact.design_frame.origin_mm[1]),
                float(artifact.design_frame.origin_mm[2]),
            ),
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
        return posed, pose, np.eye(4, dtype=np.float64), artifact.design_frame

    target = envelope_contact_frame(surface, parameters)
    transform = rigid_transform_between_frames(artifact.design_frame, target)
    posed = apply_rigid_transform(artifact.mesh, transform)
    pose = ScraperPose(
        position_mm=(
            float(target.origin_mm[0]),
            float(target.origin_mm[1]),
            float(target.origin_mm[2]),
        ),
        yaw_deg=progress,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    return posed, pose, transform, target
