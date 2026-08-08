"""Jar inner-surface mesh generation for contact simulation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import (
    CanonicalModel3D,
    GeometricMetadata,
    JarCanonicalModel,
    JarProfilePoint,
    RigidTransform,
)
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.engines.compute.internal_jar_surface_builder import internal_mesh_to_trimesh
from nutella_scraper.engines.compute.mesh_utils import (
    bounding_box_from_mesh,
    mesh_data_to_trimesh,
    trimesh_to_mesh_data,
)


class JarMeshBuilder:
    """Builds jar inner-wall meshes from profile data or InternalJarSurface."""

    def from_canonical_raw(self, jar: CanonicalModel3D) -> trimesh.Trimesh:
        """Raw imported tessellation — CAD import and archival only."""
        return mesh_data_to_trimesh(jar.mesh)

    def from_internal(self, surface: InternalJarSurface) -> trimesh.Trimesh:
        """Interior cavity mesh — use for all simulation and 2D projections."""
        return internal_mesh_to_trimesh(surface)

    def from_canonical(self, jar: CanonicalModel3D) -> trimesh.Trimesh:
        """
        Deprecated path: builds InternalJarSurface on the fly.

        Prefer ModelStore.get_internal() after import.
        """
        from nutella_scraper.engines.compute.internal_jar_surface_builder import (
            InternalJarSurfaceBuilder,
        )

        return internal_mesh_to_trimesh(InternalJarSurfaceBuilder().from_canonical(jar))

    def from_profile(
        self,
        jar: JarCanonicalModel,
        *,
        theta_segments: int = 48,
    ) -> trimesh.Trimesh:
        if jar.mesh is not None:
            return mesh_data_to_trimesh(jar.mesh)

        profile = _densify_profile(jar.meridian_profile)
        thetas = np.linspace(0.0, 2.0 * np.pi, theta_segments, endpoint=False)
        ring_vertices: list[list[np.ndarray]] = []

        for point in profile:
            y = point.z_mm
            radius = point.r_mm
            ring = np.column_stack(
                (
                    radius * np.cos(thetas),
                    np.full(theta_segments, y),
                    radius * np.sin(thetas),
                )
            )
            ring_vertices.append(ring)

        vertices = np.vstack(ring_vertices)
        faces: list[list[int]] = []
        ring_count = len(profile)
        for ring_idx in range(ring_count - 1):
            for seg in range(theta_segments):
                next_seg = (seg + 1) % theta_segments
                v0 = ring_idx * theta_segments + seg
                v1 = ring_idx * theta_segments + next_seg
                v2 = (ring_idx + 1) * theta_segments + seg
                v3 = (ring_idx + 1) * theta_segments + next_seg
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])

        return trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )

    def to_canonical(
        self,
        jar: JarCanonicalModel,
        *,
        theta_segments: int = 48,
    ) -> CanonicalModel3D:
        mesh = self.from_profile(jar, theta_segments=theta_segments)
        mesh_data = trimesh_to_mesh_data(mesh)
        bounds = bounding_box_from_mesh(mesh)
        dimensions = (
            bounds.max_x - bounds.min_x,
            bounds.max_y - bounds.min_y,
            bounds.max_z - bounds.min_z,
        )
        digest = hashlib.sha256(mesh_data.vertices.__repr__().encode()).hexdigest()
        geometry = GeometricMetadata(
            bounding_box=bounds,
            dimensions_mm=dimensions,
            center_mm=(
                (bounds.min_x + bounds.max_x) / 2.0,
                (bounds.min_y + bounds.max_y) / 2.0,
                (bounds.min_z + bounds.max_z) / 2.0,
            ),
            principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            volume_mm3=None,
            is_watertight=bool(mesh.is_watertight),
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
        )
        return CanonicalModel3D(
            id=jar.id,
            source_hash=digest,
            format="stl",
            source_path=Path(f"generated://jars/{jar.id}.stl"),
            mesh=mesh_data,
            bounds=bounds,
            geometry=geometry,
            frame=RigidTransform(),
        )


def _densify_profile(profile: tuple[JarProfilePoint, ...]) -> tuple[JarProfilePoint, ...]:
    if len(profile) >= 8:
        return profile
    dense: list[JarProfilePoint] = []
    for left, right in zip(profile, profile[1:], strict=False):
        dense.append(left)
        steps = 4
        for step in range(1, steps):
            t = step / steps
            dense.append(
                JarProfilePoint(
                    z_mm=left.z_mm + (right.z_mm - left.z_mm) * t,
                    r_mm=left.r_mm + (right.r_mm - left.r_mm) * t,
                )
            )
    dense.append(profile[-1])
    return tuple(dense)
