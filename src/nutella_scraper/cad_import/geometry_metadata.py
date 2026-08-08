"""Compute geometric metadata from a validated mesh."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import BoundingBox, GeometricMetadata, MeshData


def mesh_to_mesh_data(mesh: trimesh.Trimesh) -> MeshData:
    """Convert trimesh to portable MeshData."""
    vertices = tuple(tuple(float(v) for v in row) for row in mesh.vertices)
    faces = tuple(tuple(int(i) for i in row) for row in mesh.faces)
    return MeshData(
        vertices=vertices,
        faces=faces,
        metadata={"source_units": "mm"},
    )


def bounds_from_trimesh(mesh: trimesh.Trimesh) -> BoundingBox:
    """Extract axis-aligned bounding box in millimeters."""
    lo, hi = mesh.bounds
    return BoundingBox(
        min_x=float(lo[0]),
        min_y=float(lo[1]),
        min_z=float(lo[2]),
        max_x=float(hi[0]),
        max_y=float(hi[1]),
        max_z=float(hi[2]),
    )


def compute_geometric_metadata(mesh: trimesh.Trimesh) -> GeometricMetadata:
    """Compute essential metadata from a validated trimesh mesh."""
    bbox = bounds_from_trimesh(mesh)
    dimensions = (
        bbox.max_x - bbox.min_x,
        bbox.max_y - bbox.min_y,
        bbox.max_z - bbox.min_z,
    )
    center = tuple(float(c) for c in mesh.centroid)
    principal_axes = _compute_principal_axes(np.asarray(mesh.vertices, dtype=np.float64))

    volume: float | None = None
    if mesh.is_watertight:
        try:
            volume = float(mesh.volume)
        except Exception:
            volume = None

    return GeometricMetadata(
        bounding_box=bbox,
        dimensions_mm=dimensions,
        center_mm=center,
        principal_axes=principal_axes,
        volume_mm3=volume,
        is_watertight=bool(mesh.is_watertight),
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
    )


def _compute_principal_axes(
    vertices: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Principal axes via eigendecomposition of the vertex covariance matrix."""
    if len(vertices) < 2:
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        return identity

    centered = vertices - vertices.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.shape != (3, 3):
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        return identity

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].T

    result: list[tuple[float, float, float]] = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm
        result.append((float(axis[0]), float(axis[1]), float(axis[2])))

    while len(result) < 3:
        result.append((0.0, 0.0, 1.0))

    return (result[0], result[1], result[2])
