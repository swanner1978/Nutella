"""Geometry normalizer — STEP/STL to CanonicalModel3D."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import trimesh

from nutella_scraper.cad_import.exceptions import UnsupportedFormatError
from nutella_scraper.cad_import.geometry_metadata import (
    bounds_from_trimesh,
    compute_geometric_metadata,
    mesh_to_mesh_data,
)
from nutella_scraper.cad_import.geometry_validator import GeometryValidator
from nutella_scraper.cad_import.mesh_loader import IMeshLoader
from nutella_scraper.cad_import.trimesh_loader import TrimeshLoader
from nutella_scraper.domain.models.canonical import CanonicalModel3D, RigidTransform
from nutella_scraper.domain.models.common import ModelFormat
from nutella_scraper.domain.protocols.import_pipeline import ExportPaths


class GeometryNormalizer:
    """Normalizes STEP/STL files into CanonicalModel3D."""

    def __init__(
        self,
        loader: IMeshLoader | None = None,
        validator: GeometryValidator | None = None,
    ) -> None:
        self._loader = loader or TrimeshLoader()
        self._validator = validator or GeometryValidator()

    def normalize_from_step(
        self,
        step_path: Path,
        model_id: str | None = None,
    ) -> CanonicalModel3D:
        """Build CanonicalModel3D from a STEP file (primary import path)."""
        step_path = step_path.resolve()
        suffix = step_path.suffix.lower()
        if suffix not in (".step", ".stp"):
            raise UnsupportedFormatError(str(step_path), (".step", ".stp"))

        mesh = self._loader.load(step_path)
        assert isinstance(mesh, trimesh.Trimesh)
        return self._build_canonical(
            mesh=mesh,
            source_path=step_path,
            model_format="step",
            model_id=model_id,
        )

    def normalize_from_stl(
        self,
        stl_path: Path,
        model_id: str | None = None,
    ) -> CanonicalModel3D:
        """Build CanonicalModel3D from an STL file."""
        stl_path = stl_path.resolve()
        mesh = self._loader.load(stl_path)
        assert isinstance(mesh, trimesh.Trimesh)
        return self._build_canonical(
            mesh=mesh,
            source_path=stl_path,
            model_format="stl",
            model_id=model_id,
        )

    def normalize(
        self,
        paths: ExportPaths,
        model_id: str | None = None,
    ) -> CanonicalModel3D:
        """
        Build CanonicalModel3D from export paths.

        Prefers STEP when available; falls back to STL.
        """
        if paths.step_path is not None:
            return self.normalize_from_step(paths.step_path, model_id=model_id)
        return self.normalize_from_stl(paths.stl_path, model_id=model_id)

    def _build_canonical(
        self,
        mesh: trimesh.Trimesh,
        source_path: Path,
        model_format: ModelFormat,
        model_id: str | None,
    ) -> CanonicalModel3D:
        self._validator.validate(mesh, str(source_path))
        geometry = compute_geometric_metadata(mesh)
        bounds = bounds_from_trimesh(mesh)

        return CanonicalModel3D(
            id=model_id or self._new_id(),
            source_hash=self._compute_hash(source_path),
            format=model_format,
            source_path=source_path,
            mesh=mesh_to_mesh_data(mesh),
            bounds=bounds,
            geometry=geometry,
            frame=RigidTransform(),
        )

    def _compute_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _new_id(self) -> str:
        return str(uuid.uuid4())
