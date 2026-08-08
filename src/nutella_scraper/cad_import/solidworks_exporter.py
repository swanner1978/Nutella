"""SolidWorks COM exporter — SLDPRT to STEP/STL."""

from __future__ import annotations

from pathlib import Path

from nutella_scraper.domain.protocols.import_pipeline import ExportPaths


class SolidWorksExporter:
    """
    Exports SolidWorks parts via COM automation (Windows) or validates pre-exported files.
    """

    def export_to_step_stl(self, sldprt_path: Path) -> ExportPaths:
        raise NotImplementedError("SolidWorksExporter.export_to_step_stl not implemented")

    def validate_pre_exported(self, step_path: Path | None, stl_path: Path) -> ExportPaths:
        """Validate manually exported STEP/STL files."""
        if not stl_path.exists():
            raise FileNotFoundError(f"STL not found: {stl_path}")
        if step_path is not None and not step_path.exists():
            raise FileNotFoundError(f"STEP not found: {step_path}")
        return ExportPaths(step_path=step_path, stl_path=stl_path)
