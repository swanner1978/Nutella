"""Canonical 3D model — single source of truth for all computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nutella_scraper.domain.models.common import ModelFormat, Provenance


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in millimeters."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform (rotation + translation) in canonical frame."""

    matrix: tuple[tuple[float, float, float, float], ...] = field(
        default=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


@dataclass(frozen=True)
class MeshData:
    """Portable mesh representation (vertices, faces) for computation."""

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometricMetadata:
    """
    Essential geometric metadata computed from the canonical 3D mesh.

    Used by ComputeEngine — never derived from 2D view projections.
    """

    bounding_box: BoundingBox
    dimensions_mm: tuple[float, float, float]
    center_mm: tuple[float, float, float]
    principal_axes: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    volume_mm3: float | None
    is_watertight: bool
    vertex_count: int
    face_count: int


@dataclass(frozen=True)
class CanonicalModel3D:
    """
    Normalized 3D model from SolidWorks export (STEP/STL).

    Used exclusively by ComputeEngine and OptimizationEngine.
    Must never be derived from ViewProjectionCache.
    """

    id: str
    source_hash: str
    format: ModelFormat
    source_path: Path
    mesh: MeshData
    bounds: BoundingBox
    geometry: GeometricMetadata
    frame: RigidTransform
    provenance: Provenance = "canonical_3d"


@dataclass(frozen=True)
class JarProfilePoint:
    """Meridian profile sample for jar inner wall."""

    z_mm: float
    r_mm: float


@dataclass(frozen=True)
class JarCanonicalModel:
    """
    3D canonical representation of Nutella jar inner geometry.

    Generated from profile JSON or imported STEP/STL — used for contact simulation.
    """

    id: str
    version: str
    meridian_profile: tuple[JarProfilePoint, ...]
    neck_inner_diameter_mm: float
    total_height_mm: float
    mesh: MeshData | None = None
    bounds: BoundingBox | None = None
    provenance: Provenance = "canonical_3d"
