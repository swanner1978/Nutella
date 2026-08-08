"""Orthographic projection helpers — visualization only."""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray

# Must match scripts/visualization_helpers.py and demo ViewProjectionConfig
VIEW_WIDTH = 900
VIEW_HEIGHT = 650
VIEW_PADDING = 48

PLANE_PROFILE = "PROFILE"
PLANE_TOP_XZ = "TOP_XZ"
# Mesh overlay planes for Y-up jar: profile = XY, top = XZ
PLANE_SIDE = "XY"
PLANE_TOP = "XZ"


def project_vertices(
    vertices: NDArray[np.float64],
    plane: str,
) -> tuple[NDArray[np.float64], tuple[int, int], int]:
    """Project 3D vertices onto a 2D plane (same convention as the CAD viewer)."""
    # Plane math is fixed: XZ → (X,Z) looking ±Y; XY → (X,Y) looking ±Z.
    # PLANE_SIDE / PLANE_TOP only select which plane overlays use.
    if plane in ("XZ", PLANE_TOP_XZ):
        indices = (0, 2)
        view_axis = 1
    elif plane == "XY":
        indices = (0, 1)
        view_axis = 2
    elif plane == PLANE_PROFILE:
        raise ValueError("PROFILE plane uses analytical projection, not mesh vertices")
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")
    coords = vertices[:, indices].astype(np.float64, copy=False)
    return coords, indices, view_axis


def fit_to_viewport(
    coords: NDArray[np.float64],
    *,
    width: int = VIEW_WIDTH,
    height: int = VIEW_HEIGHT,
    padding: int = VIEW_PADDING,
) -> tuple[float, NDArray[np.float64]]:
    """Scale and offset 2D coordinates to the standard viewer viewport."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    extent = hi - lo
    extent = np.where(extent == 0, 1.0, extent)
    draw_w = width - 2 * padding
    draw_h = height - 2 * padding
    scale = min(draw_w / extent[0], draw_h / extent[1])
    scaled_extent = extent * scale
    offset = np.array(
        [
            padding + (draw_w - scaled_extent[0]) / 2 - lo[0] * scale,
            padding + (draw_h - scaled_extent[1]) / 2 - lo[1] * scale,
        ],
        dtype=np.float64,
    )
    return float(scale), offset


def canonical_to_trimesh(model_mesh: object) -> trimesh.Trimesh:
    """Build a trimesh from CanonicalModel3D.mesh without mutating the source."""
    from nutella_scraper.domain.models.canonical import MeshData

    if not isinstance(model_mesh, MeshData):
        raise TypeError(f"Expected MeshData, got {type(model_mesh)!r}")
    vertices = np.asarray(model_mesh.vertices, dtype=np.float64)
    faces = np.asarray(model_mesh.faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def world_to_svg(
    point: tuple[float, float, float],
    *,
    plane: str,
    scale: float,
    offset: NDArray[np.float64],
) -> tuple[float, float]:
    """Convert one 3D point to SVG coordinates for the given orthographic plane."""
    vertex = np.asarray([point], dtype=np.float64)
    coords, _, _ = project_vertices(vertex, plane)
    projected = coords[0] * scale + offset
    return float(projected[0]), float(projected[1])


def silhouette_edges(mesh: trimesh.Trimesh, view_axis: int) -> NDArray[np.int64]:
    """Return boundary edges for an orthographic view along ``view_axis``."""
    edges_sorted = np.asarray(mesh.edges_sorted, dtype=np.int64)
    unique_edges, inverse = np.unique(edges_sorted, axis=0, return_inverse=True)
    face_ids = np.repeat(np.arange(len(mesh.faces), dtype=np.int64), 3)
    front_facing = np.asarray(mesh.face_normals[:, view_axis] > 1e-9, dtype=np.int8)

    counts = np.bincount(inverse, minlength=len(unique_edges))
    has_front = np.zeros(len(unique_edges), dtype=np.int8)
    has_other = np.zeros(len(unique_edges), dtype=np.int8)
    np.maximum.at(has_front, inverse, front_facing[face_ids])
    np.maximum.at(has_other, inverse, 1 - front_facing[face_ids])

    is_contour = (counts == 1) | ((has_front == 1) & (has_other == 1))
    return unique_edges[is_contour]


def segments_path(
    segments: NDArray[np.float64],
    *,
    scale: float,
    offset: NDArray[np.float64],
    css_class: str = "",
) -> str:
    """Build an SVG path from projected 2D segment endpoints."""
    if segments.size == 0:
        return ""
    parts: list[str] = []
    for (x0, y0), (x1, y1) in segments:
        sx0 = x0 * scale + offset[0]
        sy0 = y0 * scale + offset[1]
        sx1 = x1 * scale + offset[0]
        sy1 = y1 * scale + offset[1]
        parts.append(f"M{sx0:.2f},{sy0:.2f}L{sx1:.2f},{sy1:.2f}")
    class_attr = f' class="{css_class}"' if css_class else ""
    return f'<path{class_attr} fill="none" d="{" ".join(parts)}"/>'


def distance_to_color(distance_mm: float, *, max_distance_mm: float) -> str:
    """Map a contact distance to an RGB color (far=blue, near=red)."""
    if not np.isfinite(distance_mm):
        return "#2a2a2a"
    if max_distance_mm <= 0:
        return "#4488ff"
    ratio = min(max(float(distance_mm) / max_distance_mm, 0.0), 1.0)
    red = int(40 + 215 * ratio)
    green = int(180 * (1.0 - abs(ratio - 0.5) * 2.0))
    blue = int(220 * (1.0 - ratio))
    return f"rgb({red},{green},{blue})"
