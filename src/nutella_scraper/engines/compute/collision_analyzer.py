"""Mesh intersection and penetration analysis."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.domain.models.contact import CollisionPoint3D, CollisionResult


def analyze_collision(
    jar_mesh: trimesh.Trimesh,
    scraper_mesh: trimesh.Trimesh,
    *,
    mesh_tolerance_mm: float,
) -> CollisionResult:
    """
    Detect geometric interpenetration between jar inner wall and scraper volume.

    Distinct from proximity contact: a collision indicates a physically invalid
    configuration that optimization should reject.
    """
    jar_samples = jar_mesh.triangles_center
    jar_radial = _radial_distance(jar_samples)
    inner_radius = float(np.median(jar_radial))

    scraper_vertices = scraper_mesh.vertices
    scraper_radial = _radial_distance(scraper_vertices)
    outward_penetration = scraper_radial - inner_radius
    penetrating_vertices = outward_penetration > mesh_tolerance_mm

    collision_points: list[CollisionPoint3D] = []
    colliding_faces: set[int] = set()
    for face_id, sample in enumerate(jar_samples):
        sample_radius = float(_radial_distance(sample[None, :])[0])
        if sample_radius + mesh_tolerance_mm < inner_radius:
            continue
        local_penetration = float(scraper_radial.max() - sample_radius)
        if local_penetration <= mesh_tolerance_mm:
            continue
        colliding_faces.add(face_id)
        collision_points.append(
            CollisionPoint3D(
                position_mm=tuple(float(v) for v in sample),
                jar_face_id=face_id,
                penetration_depth_mm=local_penetration,
            )
        )

    penetration_depth = 0.0
    if np.any(penetrating_vertices):
        penetration_depth = float(outward_penetration[penetrating_vertices].max())

    has_collision = bool(np.any(penetrating_vertices) or colliding_faces)
    return CollisionResult(
        has_collision=has_collision,
        penetration_depth_mm=penetration_depth,
        collision_points=tuple(collision_points),
        colliding_face_ids=frozenset(colliding_faces),
    )


def merge_collisions(current: CollisionResult, incoming: CollisionResult) -> CollisionResult:
    if not incoming.has_collision:
        return current
    if not current.has_collision:
        return incoming

    collision_points = current.collision_points + incoming.collision_points
    return CollisionResult(
        has_collision=True,
        penetration_depth_mm=max(
            current.penetration_depth_mm,
            incoming.penetration_depth_mm,
        ),
        collision_points=collision_points,
        colliding_face_ids=current.colliding_face_ids | incoming.colliding_face_ids,
    )


def _radial_distance(points: np.ndarray) -> np.ndarray:
    return np.sqrt(points[:, 0] ** 2 + points[:, 2] ** 2)
