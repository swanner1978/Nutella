"""Contact analysis against the interior surface representation."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import trimesh

from nutella_scraper.domain.models.contact import ContactPoint3D
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.engines.compute.internal_jar_surface_builder import internal_mesh_to_trimesh


def analyze_interior_contact(
    internal: InternalJarSurface,
    scraper_mesh: trimesh.Trimesh,
    *,
    contact_threshold_mm: float,
    timing_callback: Callable[[str, float], None] | None = None,
) -> tuple[np.ndarray, tuple[ContactPoint3D, ...]]:
    """
    Compute contact using InternalJarSurface samples against the Scraper3D volume.
    """
    jar_mesh = internal_mesh_to_trimesh(internal)
    sample_points = np.asarray(internal.sample_points_mm, dtype=np.float64)
    distance_started = time.perf_counter()
    closest, distances, _ = trimesh.proximity.closest_point(scraper_mesh, sample_points)
    if timing_callback is not None:
        timing_callback(
            "distance_calculation",
            (time.perf_counter() - distance_started) * 1000.0,
        )

    contact_started = time.perf_counter()
    face_centers = jar_mesh.triangles_center
    face_count = len(jar_mesh.faces)
    min_distances = np.full(face_count, np.inf, dtype=np.float64)
    contact_points: list[ContactPoint3D] = []

    for sample_index, (sample_point, distance) in enumerate(
        zip(sample_points, distances, strict=True)
    ):
        face_id = int(np.argmin(np.linalg.norm(face_centers - sample_point, axis=1)))
        distance_value = float(distance)
        min_distances[face_id] = min(min_distances[face_id], distance_value)
        if distance_value <= contact_threshold_mm:
            contact_points.append(
                ContactPoint3D(
                    position_mm=tuple(float(value) for value in closest[sample_index]),
                    jar_face_id=face_id,
                    distance_mm=distance_value,
                )
            )

    if timing_callback is not None:
        timing_callback(
            "contact_calculation",
            (time.perf_counter() - contact_started) * 1000.0,
        )

    return min_distances, tuple(contact_points)
