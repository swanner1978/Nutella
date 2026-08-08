"""Per-face contact analysis between jar and scraper meshes."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import trimesh

from nutella_scraper.domain.models.contact import ContactPoint3D


def analyze_contact(
    jar_mesh: trimesh.Trimesh,
    scraper_mesh: trimesh.Trimesh,
    *,
    contact_threshold_mm: float,
    timing_callback: Callable[[str, float], None] | None = None,
) -> tuple[np.ndarray, tuple[ContactPoint3D, ...]]:
    """
    Compute minimum distance from each jar face to the scraper volume.

    Returns one distance per jar face and contact points where the distance
    falls below the configured threshold.
    """
    distance_started = time.perf_counter()
    sample_points = _face_sample_points(jar_mesh)
    closest, distances, _ = trimesh.proximity.closest_point(scraper_mesh, sample_points)

    face_count = len(jar_mesh.faces)
    min_distances = np.full(face_count, np.inf, dtype=np.float64)
    for face_id, distance in enumerate(distances):
        min_distances[face_id] = min(min_distances[face_id], float(distance))
    if timing_callback is not None:
        timing_callback(
            "distance_calculation",
            (time.perf_counter() - distance_started) * 1000.0,
        )

    contact_started = time.perf_counter()
    contact_points: list[ContactPoint3D] = []
    for face_id, distance in enumerate(distances):
        if float(distance) <= contact_threshold_mm:
            contact_points.append(
                ContactPoint3D(
                    position_mm=tuple(float(v) for v in closest[face_id]),
                    jar_face_id=face_id,
                    distance_mm=float(distance),
                )
            )
    if timing_callback is not None:
        timing_callback(
            "contact_calculation",
            (time.perf_counter() - contact_started) * 1000.0,
        )

    return min_distances, tuple(contact_points)


def merge_face_distances(current: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    return np.minimum(current, incoming)


def _face_sample_points(jar_mesh: trimesh.Trimesh) -> np.ndarray:
    triangles = jar_mesh.vertices[jar_mesh.faces]
    return triangles.mean(axis=1)
