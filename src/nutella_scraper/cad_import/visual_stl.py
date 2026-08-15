"""Visualization-only jar tessellation — never used by compute or collision.

``visual.stl`` is a dedicated display mesh tessellated from ``reference.step``.
It must not replace ``canonical.stl`` or ``internal.stl``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from nutella_scraper.cad_import.exceptions import CadImportError
from nutella_scraper.cad_import.geometry_metadata import bounds_from_trimesh
from nutella_scraper.cad_import.geometry_validator import GeometryValidator
from nutella_scraper.cad_import.trimesh_loader import TrimeshLoader
from nutella_scraper.domain.models.canonical import CanonicalModel3D

_LOG = logging.getLogger(__name__)

VISUAL_STL_NAME = "visual.stl"
VISUAL_META_NAME = "visual.json"
VISUAL_ROLE = "visualization_only"

# Independent of TrimeshLoader compute defaults (0.01 mm / 0.1 rad).
# Chord ~0.15 mm and ~2.9° keep jar fillets smooth without a giant mesh.
DEFAULT_VISUAL_TOL_LINEAR_MM = 0.15
DEFAULT_VISUAL_TOL_ANGULAR_RAD = 0.05
DEFAULT_VISUAL_TOL_RELATIVE = False
MAX_VISUAL_FACES = 750_000
MAX_VISUAL_VERTICES = 750_000


class VisualStlError(CadImportError):
    """Raised when the visualization STL cannot be produced or validated."""


@dataclass(frozen=True)
class VisualTessellationConfig:
    """Tessellation used only for ``visual.stl`` (not CanonicalModel3D.mesh)."""

    tol_linear_mm: float = DEFAULT_VISUAL_TOL_LINEAR_MM
    tol_angular_rad: float = DEFAULT_VISUAL_TOL_ANGULAR_RAD
    tol_relative: bool = DEFAULT_VISUAL_TOL_RELATIVE
    max_faces: int = MAX_VISUAL_FACES
    max_vertices: int = MAX_VISUAL_VERTICES
    method: str = "trimesh+cascadio"

    def __post_init__(self) -> None:
        if self.tol_linear_mm <= 0:
            raise ValueError("tol_linear_mm must be positive")
        if self.tol_angular_rad <= 0:
            raise ValueError("tol_angular_rad must be positive")


@dataclass(frozen=True)
class VisualFrameComparison:
    """AABB / center comparison between visual.stl and CanonicalModel3D.mesh."""

    bbox_delta_mm: tuple[float, float, float, float, float, float]
    center_delta_mm: tuple[float, float, float]
    dimensions_delta_mm: tuple[float, float, float]
    max_frame_delta_mm: float
    abs_tolerance_mm: float

    @property
    def frame_aligned(self) -> bool:
        return self.max_frame_delta_mm <= self.abs_tolerance_mm


def tessellate_visual_mesh(
    step_path: Path,
    config: VisualTessellationConfig | None = None,
) -> trimesh.Trimesh:
    """Tessellate the full jar from STEP with visualization-only tolerances."""
    cfg = config or VisualTessellationConfig()
    loader = TrimeshLoader(
        step_tol_linear_mm=cfg.tol_linear_mm,
        step_tol_angular_rad=cfg.tol_angular_rad,
        step_tol_relative=cfg.tol_relative,
    )
    _LOG.info(
        "[visual.stl] tessellate STEP=%s | linear=%.3f mm | angular=%.3f rad | relative=%s",
        step_path,
        cfg.tol_linear_mm,
        cfg.tol_angular_rad,
        cfg.tol_relative,
    )
    mesh = loader.load(step_path)
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def compare_visual_frame(
    visual: trimesh.Trimesh,
    canonical: CanonicalModel3D,
    *,
    abs_tolerance_mm: float,
) -> VisualFrameComparison:
    """Compare AABB and AABB-center; tessellation chord error is allowed."""
    vis_lo, vis_hi = np.asarray(visual.bounds, dtype=np.float64)
    can_lo = np.array(
        [canonical.bounds.min_x, canonical.bounds.min_y, canonical.bounds.min_z],
        dtype=np.float64,
    )
    can_hi = np.array(
        [canonical.bounds.max_x, canonical.bounds.max_y, canonical.bounds.max_z],
        dtype=np.float64,
    )
    bbox_delta = np.concatenate([vis_lo - can_lo, vis_hi - can_hi])
    vis_center = 0.5 * (vis_lo + vis_hi)
    can_center = 0.5 * (can_lo + can_hi)
    center_delta = vis_center - can_center
    vis_dim = vis_hi - vis_lo
    can_dim = can_hi - can_lo
    dim_delta = vis_dim - can_dim
    max_delta = float(
        max(
            np.max(np.abs(bbox_delta)),
            np.max(np.abs(center_delta)),
            np.max(np.abs(dim_delta)),
        )
    )
    return VisualFrameComparison(
        bbox_delta_mm=tuple(float(v) for v in bbox_delta),
        center_delta_mm=tuple(float(v) for v in center_delta),
        dimensions_delta_mm=tuple(float(v) for v in dim_delta),
        max_frame_delta_mm=max_delta,
        abs_tolerance_mm=float(abs_tolerance_mm),
    )


def validate_visual_mesh(
    visual: trimesh.Trimesh,
    canonical: CanonicalModel3D,
    config: VisualTessellationConfig,
    *,
    source_path: str,
) -> VisualFrameComparison:
    """Reload-time checks: valid mesh, same frame as compute mesh, display density."""
    GeometryValidator().validate(visual, source_path)
    vertex_count = int(len(visual.vertices))
    face_count = int(len(visual.faces))
    if vertex_count > config.max_vertices or face_count > config.max_faces:
        raise VisualStlError(
            f"visual.stl too dense for display "
            f"(vertices={vertex_count}, faces={face_count}; "
            f"max vertices={config.max_vertices}, max faces={config.max_faces})"
        )

    abs_tol = max(2.0 * config.tol_linear_mm, 0.25)
    comparison = compare_visual_frame(visual, canonical, abs_tolerance_mm=abs_tol)
    if not comparison.frame_aligned:
        raise VisualStlError(
            "visual.stl frame does not match CanonicalModel3D.mesh "
            f"(max_delta={comparison.max_frame_delta_mm:.4f} mm, "
            f"tol={comparison.abs_tolerance_mm:.4f} mm)"
        )

    _assert_display_density(visual, canonical, config)
    return comparison


def _assert_display_density(
    visual: trimesh.Trimesh,
    canonical: CanonicalModel3D,
    config: VisualTessellationConfig,
) -> None:
    """Keep fillets visually smooth without a gigantic mesh."""
    extents = np.asarray(visual.extents, dtype=np.float64)
    span = float(np.max(extents)) if extents.size else 0.0
    if span <= 1.0:
        return
    lengths = np.asarray(getattr(visual, "edges_unique_length", []), dtype=np.float64)
    if lengths.size == 0:
        return
    median_edge = float(np.median(lengths))
    max_edge = float(np.max(lengths))
    # Planar boxes have long edges; skip chord checks when the compute mesh
    # is already a handful of triangles (no curved fillets to resolve).
    if canonical.geometry.face_count <= 24 and max_edge > 0.15 * span:
        return
    max_allowed_edge = max(6.0 * config.tol_linear_mm, 0.06 * span)
    if median_edge > max_allowed_edge:
        raise VisualStlError(
            "visual.stl tessellation is too coarse for a smooth display "
            f"(median_edge={median_edge:.3f} mm, limit={max_allowed_edge:.3f} mm)"
        )


def visual_metadata_payload(
    *,
    model_id: str,
    stl_path: Path,
    step_path: Path,
    source_hash: str,
    mesh: trimesh.Trimesh,
    config: VisualTessellationConfig,
    comparison: VisualFrameComparison,
) -> dict[str, Any]:
    bbox = bounds_from_trimesh(mesh)
    lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
    center = 0.5 * (lo + hi)
    data = stl_path.read_bytes()
    return {
        "model_id": model_id,
        "filename": VISUAL_STL_NAME,
        "path": str(stl_path),
        "role": VISUAL_ROLE,
        "source_step": str(step_path),
        "source_hash": source_hash,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "units": "millimeters",
        "bounding_box": {
            "min_x": bbox.min_x,
            "min_y": bbox.min_y,
            "min_z": bbox.min_z,
            "max_x": bbox.max_x,
            "max_y": bbox.max_y,
            "max_z": bbox.max_z,
        },
        "center_mm": [float(center[0]), float(center[1]), float(center[2])],
        "dimensions_mm": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])],
        "tessellation": {
            "method": config.method,
            "tol_linear_mm": config.tol_linear_mm,
            "tol_angular_rad": config.tol_angular_rad,
            "tol_relative": config.tol_relative,
        },
        "frame_check": {
            "bbox_delta_mm": list(comparison.bbox_delta_mm),
            "center_delta_mm": list(comparison.center_delta_mm),
            "dimensions_delta_mm": list(comparison.dimensions_delta_mm),
            "max_frame_delta_mm": comparison.max_frame_delta_mm,
            "abs_tolerance_mm": comparison.abs_tolerance_mm,
            "aligned": comparison.frame_aligned,
        },
    }


def reload_visual_stl(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise VisualStlError(f"Expected triangle mesh when reloading {path}")
    if loaded.is_empty:
        raise VisualStlError(f"Reloaded visual.stl is empty: {path}")
    return trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )
