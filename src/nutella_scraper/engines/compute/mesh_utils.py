"""Mesh conversion helpers for the compute engine."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import BoundingBox, MeshData


def mesh_data_to_trimesh(mesh_data: MeshData) -> trimesh.Trimesh:
    vertices = np.asarray(mesh_data.vertices, dtype=np.float64)
    faces = np.asarray(mesh_data.faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def trimesh_to_mesh_data(mesh: trimesh.Trimesh) -> MeshData:
    vertices = tuple(tuple(float(v) for v in point) for point in mesh.vertices)
    faces = tuple(tuple(int(i) for i in face) for face in mesh.faces)
    return MeshData(vertices=vertices, faces=faces)


def bounding_box_from_mesh(mesh: trimesh.Trimesh) -> BoundingBox:
    bounds = mesh.bounds
    return BoundingBox(
        min_x=float(bounds[0, 0]),
        min_y=float(bounds[0, 1]),
        min_z=float(bounds[0, 2]),
        max_x=float(bounds[1, 0]),
        max_y=float(bounds[1, 1]),
        max_z=float(bounds[1, 2]),
    )


def face_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.area_faces, dtype=np.float64)


def sample_face_points(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return one representative point per face (triangle centroid)."""
    triangles = mesh.vertices[mesh.faces]
    return triangles.mean(axis=1)
