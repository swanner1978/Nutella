"""Build InternalJarSurface — the single interior cavity representation."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import CanonicalModel3D, MeshData
from nutella_scraper.domain.models.internal_jar_surface import (
    InternalJarSurface,
    InternalJarSurfaceSlice,
)
from nutella_scraper.engines.compute.mesh_utils import mesh_data_to_trimesh, trimesh_to_mesh_data


class InternalJarSurfaceBuilder:
    """
    Derive the interior cavity mesh and sampling grid from CanonicalModel3D.

    Raw CAD tessellations may include outer walls; only the innermost shell
    per (height, angle) bin is retained.
    """

    def __init__(self, *, jar_mesh_builder: object | None = None) -> None:
        self._jar_mesh_builder = jar_mesh_builder

    def from_canonical(
        self,
        jar: CanonicalModel3D,
        *,
        slice_count: int = 48,
        angular_bins: int = 72,
        max_samples: int = 4096,
        radial_tolerance_mm: float = 0.35,
    ) -> InternalJarSurface:
        raw_mesh = _canonical_raw_mesh(jar, self._jar_mesh_builder)
        inner_mesh, inner_face_indices = self._extract_inner_shell(
            raw_mesh,
            angular_bins=angular_bins,
            radial_tolerance_mm=radial_tolerance_mm,
        )
        centers = np.asarray(inner_mesh.triangles_center, dtype=np.float64)
        vertices = np.asarray(inner_mesh.vertices, dtype=np.float64)
        y_min = float(vertices[:, 1].min())
        y_max = float(vertices[:, 1].max())
        height = max(y_max - y_min, 1e-6)

        slices = self._build_slices(
            vertices=vertices,
            y_min=y_min,
            y_max=y_max,
            slice_count=slice_count,
        )
        sample_points, sample_areas = self._build_samples(
            mesh=inner_mesh,
            y_min=y_min,
            y_max=y_max,
            slice_count=slice_count,
            angular_bins=angular_bins,
            max_samples=max_samples,
        )

        return InternalJarSurface(
            jar_id=jar.id,
            canonical_mesh_sha256=_mesh_sha256(jar.mesh),
            mesh=trimesh_to_mesh_data(inner_mesh),
            y_min_mm=y_min,
            y_max_mm=y_max,
            slices=tuple(slices),
            sample_points_mm=sample_points,
            sample_areas_mm2=sample_areas,
            source_face_count=len(raw_mesh.faces),
            metadata={
                "inner_face_count": len(inner_mesh.faces),
                "excluded_face_count": len(raw_mesh.faces) - len(inner_face_indices),
                "slice_count": len(slices),
                "sample_count": len(sample_points),
                "angular_bins": angular_bins,
                "radial_tolerance_mm": radial_tolerance_mm,
                "builder": "InternalJarSurfaceBuilder",
            },
        )

    @staticmethod
    def _extract_inner_shell(
        mesh: trimesh.Trimesh,
        *,
        angular_bins: int,
        radial_tolerance_mm: float,
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        centers = np.asarray(mesh.triangles_center, dtype=np.float64)
        if len(centers) == 0:
            return mesh, np.arange(len(mesh.faces), dtype=np.int64)

        radial = np.sqrt(centers[:, 0] ** 2 + centers[:, 2] ** 2)
        y_values = centers[:, 1]
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        y_bins = np.linspace(y_min, y_max, max(angular_bins, 8))
        y_index = np.clip(
            np.digitize(y_values, y_bins, right=False) - 1,
            0,
            len(y_bins) - 1,
        )
        theta = np.arctan2(centers[:, 2], centers[:, 0])
        theta_bins = np.linspace(-math.pi, math.pi, angular_bins, endpoint=False)
        theta_index = np.clip(
            np.digitize(theta, theta_bins, right=False) - 1,
            0,
            len(theta_bins) - 1,
        )

        keep = np.zeros(len(mesh.faces), dtype=bool)
        for y_bin in range(len(y_bins)):
            for theta_bin in range(len(theta_bins)):
                mask = (y_index == y_bin) & (theta_index == theta_bin)
                if not np.any(mask):
                    continue
                candidates = np.flatnonzero(mask)
                min_radial = float(np.min(radial[candidates]))
                tolerance = max(radial_tolerance_mm, 0.015 * min_radial)
                inner = candidates[radial[candidates] <= min_radial + tolerance]
                keep[inner] = True

        if not np.any(keep):
            keep[:] = True

        face_indices = np.flatnonzero(keep).astype(np.int64)
        submesh = mesh.submesh([face_indices], append=True, only_watertight=False)
        if isinstance(submesh, trimesh.Scene):
            submesh = trimesh.util.concatenate(tuple(submesh.geometry.values()))
        return submesh, face_indices

    @staticmethod
    def _build_slices(
        *,
        vertices: np.ndarray,
        y_min: float,
        y_max: float,
        slice_count: int,
    ) -> list[InternalJarSurfaceSlice]:
        height = max(y_max - y_min, 1e-6)
        radial = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 2] ** 2)
        y_bins = np.linspace(y_min, y_max, max(slice_count, 2))
        bin_half = height / max(slice_count * 2, 4)
        slices: list[InternalJarSurfaceSlice] = []
        for y_value in y_bins:
            mask = np.abs(vertices[:, 1] - y_value) <= bin_half
            if not np.any(mask):
                continue
            slices.append(
                InternalJarSurfaceSlice(
                    y_mm=float(y_value),
                    inner_radius_mm=float(np.min(radial[mask])),
                )
            )
        if not slices:
            slices.append(InternalJarSurfaceSlice(y_mm=y_min, inner_radius_mm=0.0))
        return slices

    @staticmethod
    def _build_samples(
        *,
        mesh: trimesh.Trimesh,
        y_min: float,
        y_max: float,
        slice_count: int,
        angular_bins: int,
        max_samples: int,
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, ...]]:
        centers = np.asarray(mesh.triangles_center, dtype=np.float64)
        radial = np.sqrt(centers[:, 0] ** 2 + centers[:, 2] ** 2)
        y_bins = np.linspace(y_min, y_max, max(slice_count, 2))
        y_index = np.clip(
            np.digitize(centers[:, 1], y_bins, right=False) - 1,
            0,
            len(y_bins) - 1,
        )
        theta = np.arctan2(centers[:, 2], centers[:, 0])
        theta_bins = np.linspace(-math.pi, math.pi, angular_bins, endpoint=False)
        theta_index = np.clip(
            np.digitize(theta, theta_bins, right=False) - 1,
            0,
            len(theta_bins) - 1,
        )

        sample_indices: list[int] = []
        for y_bin in range(len(y_bins)):
            for theta_bin in range(len(theta_bins)):
                mask = (y_index == y_bin) & (theta_index == theta_bin)
                if not np.any(mask):
                    continue
                candidates = np.flatnonzero(mask)
                innermost = candidates[int(np.argmin(radial[candidates]))]
                sample_indices.append(int(innermost))

        if not sample_indices:
            sample_indices = list(range(len(centers)))

        if len(sample_indices) > max_samples:
            pick = np.linspace(0, len(sample_indices) - 1, max_samples, dtype=int)
            sample_indices = [sample_indices[int(index)] for index in pick]

        sample_points = tuple(
            tuple(float(value) for value in centers[index]) for index in sample_indices
        )
        sample_areas = tuple(float(mesh.area_faces[index]) for index in sample_indices)
        return sample_points, sample_areas


def _canonical_raw_mesh(jar: CanonicalModel3D, jar_mesh_builder: object | None) -> trimesh.Trimesh:
    if jar_mesh_builder is not None:
        return jar_mesh_builder.from_canonical_raw(jar)  # type: ignore[union-attr]
    from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder

    return JarMeshBuilder().from_canonical_raw(jar)


def resolve_internal_jar_surface(
    jar: CanonicalModel3D,
    *,
    cached: InternalJarSurface | None = None,
    builder: InternalJarSurfaceBuilder | None = None,
    model_store: object | None = None,
) -> InternalJarSurface:
    """Return the single interior cavity representation for a canonical jar."""
    if cached is not None:
        return cached
    if model_store is not None:
        from nutella_scraper.cad_import.model_store import ModelStore

        if isinstance(model_store, ModelStore):
            return model_store.get_internal(jar.id)
    surface_builder = builder or InternalJarSurfaceBuilder()
    return surface_builder.from_canonical(jar)


def internal_mesh_to_trimesh(surface: InternalJarSurface) -> trimesh.Trimesh:
    """Convert InternalJarSurface.mesh to trimesh for compute/visualization."""
    return mesh_data_to_trimesh(surface.mesh)


def _mesh_sha256(mesh: MeshData) -> str:
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype="<f8")
    faces = np.asarray(mesh.faces, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"CanonicalModel3D.mesh.v1\0")
    digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes())
    digest.update(vertices.tobytes(order="C"))
    digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
    digest.update(faces.tobytes(order="C"))
    return digest.hexdigest()
