"""CAD file exporters."""

from __future__ import annotations

from pathlib import Path

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.common import ExportFormat


class STLExporter:
    def export(self, model: CanonicalModel3D, output_path: Path) -> Path:
        raise NotImplementedError("STLExporter.export not implemented")


class STEPExporter:
    def export(self, model: CanonicalModel3D, output_path: Path) -> Path:
        raise NotImplementedError("STEPExporter.export not implemented")


class ModelExporter:
    """Facade for model export."""

    def export(
        self,
        model: CanonicalModel3D,
        output_path: Path,
        format: ExportFormat,
    ) -> Path:
        if format == "stl":
            return STLExporter().export(model, output_path)
        if format == "step":
            return STEPExporter().export(model, output_path)
        raise ValueError(f"Unsupported export format: {format}")
