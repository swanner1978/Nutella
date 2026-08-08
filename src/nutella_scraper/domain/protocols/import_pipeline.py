"""Import pipeline protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nutella_scraper.domain.models.canonical import CanonicalModel3D


@dataclass(frozen=True)
class ExportPaths:
    """Paths to exported STEP and STL files."""

    step_path: Path | None
    stl_path: Path


@dataclass(frozen=True)
class ImportResult:
    """Result of CAD import pipeline."""

    model_id: str
    canonical: CanonicalModel3D
    views_id: str | None = None


class ISolidWorksExporter(Protocol):
    """Exports SLDPRT to STEP/STL."""

    def export_to_step_stl(self, sldprt_path: Path) -> ExportPaths:
        """Convert SolidWorks part to STEP and STL."""
        ...


class IGeometryNormalizer(Protocol):
    """Normalizes STEP/STL to CanonicalModel3D."""

    def normalize(self, paths: ExportPaths, model_id: str | None = None) -> CanonicalModel3D:
        """Produce canonical 3D model from exported files."""
        ...


class IModelStore(Protocol):
    """Persists and retrieves canonical models."""

    def persist(self, model: CanonicalModel3D) -> str:
        """Store model and return model_id."""
        ...

    def get(self, model_id: str) -> CanonicalModel3D:
        """Retrieve model by ID."""
        ...

    def delete(self, model_id: str) -> None:
        """Remove model from store."""
        ...


class IImportPipeline(Protocol):
    """Full CAD import pipeline orchestrator."""

    def import_step(
        self,
        step_path: Path,
        *,
        generate_views: bool = True,
    ) -> ImportResult:
        """Import from STEP file — primary import path."""
        ...

    def import_stl(
        self,
        stl_path: Path,
        *,
        generate_views: bool = True,
    ) -> ImportResult:
        """Import from STL file."""
        ...

    def import_sldprt(self, sldprt_path: Path) -> ImportResult:
        """Import from SolidWorks part file."""
        ...

    def import_step_stl(self, step_path: Path | None, stl_path: Path) -> ImportResult:
        """Import from pre-exported STEP/STL (manual fallback)."""
        ...
