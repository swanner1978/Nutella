"""Geometry validation for imported meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from nutella_scraper.cad_import.exceptions import InvalidGeometryError

_MIN_DIMENSION_MM = 1e-6
_MIN_VERTICES = 3
_MIN_FACES = 1


@dataclass(frozen=True)
class ValidationConfig:
    """Validation strictness for imported geometry."""

    require_watertight: bool = False
    min_dimension_mm: float = _MIN_DIMENSION_MM


class GeometryValidator:
    """Validates mesh integrity before CanonicalModel3D construction."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self._config = config or ValidationConfig()

    def validate(self, mesh: trimesh.Trimesh, source_path: str) -> None:
        violations = self.collect_violations(mesh)
        if violations:
            raise InvalidGeometryError(source_path, tuple(violations))

    def collect_violations(self, mesh: trimesh.Trimesh) -> list[str]:
        violations: list[str] = []

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        if vertices.size == 0:
            violations.append("Mesh has no vertices")
            return violations

        if faces.size == 0:
            violations.append("Mesh has no faces")
            return violations

        if len(vertices) < _MIN_VERTICES:
            violations.append(f"Mesh has fewer than {_MIN_VERTICES} vertices")

        if len(faces) < _MIN_FACES:
            violations.append(f"Mesh has fewer than {_MIN_FACES} faces")

        if not np.all(np.isfinite(vertices)):
            violations.append("Vertices contain NaN or Inf values")

        if faces.size > 0 and (int(faces.max()) >= len(vertices) or int(faces.min()) < 0):
            violations.append("Face indices out of vertex range")

        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        if np.any(extents <= self._config.min_dimension_mm):
            violations.append(
                f"Bounding box degenerate (min dimension <= {self._config.min_dimension_mm} mm)"
            )

        if self._config.require_watertight and not mesh.is_watertight:
            violations.append("Mesh is not watertight")

        return violations
