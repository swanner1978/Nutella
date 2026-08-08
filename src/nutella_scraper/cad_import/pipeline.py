"""CAD import pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer
from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.cad_import.solidworks_exporter import SolidWorksExporter
from nutella_scraper.cad_import.view_cache_store import ViewCacheStore
from nutella_scraper.cad_import.view_projection_generator import ViewProjectionGenerator
from nutella_scraper.domain.protocols.import_pipeline import ExportPaths, ImportResult

_LOG = logging.getLogger(__name__)


class ImportPipeline:
    """
    Orchestrates STEP → CanonicalModel3D → optional visualization views.

    Computational path:
        STEP → GeometryNormalizer → CanonicalModel3D → ModelStore

    Visualization path (parallel, never feeds computation):
        CanonicalModel3D → ViewProjectionGenerator → ViewProjectionCache → ViewCacheStore
    """

    def __init__(
        self,
        normalizer: GeometryNormalizer,
        model_store: ModelStore,
        view_generator: ViewProjectionGenerator | None = None,
        view_cache_store: ViewCacheStore | None = None,
        exporter: SolidWorksExporter | None = None,
    ) -> None:
        self._normalizer = normalizer
        self._model_store = model_store
        self._view_generator = view_generator or ViewProjectionGenerator()
        self._view_cache_store = view_cache_store
        self._exporter = exporter or SolidWorksExporter()

    def import_step(
        self,
        step_path: Path,
        *,
        generate_views: bool = True,
    ) -> ImportResult:
        """
        Import a STEP file and build CanonicalModel3D.

        Args:
            step_path: Path to .step or .stp file.
            generate_views: If True, generate visualization-only profile/top views.

        Returns:
            ImportResult with canonical model and optional views_id.
        """
        canonical = self._normalizer.normalize_from_step(step_path)
        return self._finalize(canonical, generate_views=generate_views)

    def import_stl(
        self,
        stl_path: Path,
        *,
        generate_views: bool = True,
    ) -> ImportResult:
        """Import an STL file (secondary format)."""
        canonical = self._normalizer.normalize_from_stl(stl_path)
        return self._finalize(canonical, generate_views=generate_views)

    def import_step_stl(
        self,
        step_path: Path | None,
        stl_path: Path,
        *,
        generate_views: bool = True,
    ) -> ImportResult:
        """Import from pre-exported STEP/STL; prefers STEP when available."""
        if step_path is not None:
            return self.import_step(step_path, generate_views=generate_views)
        return self.import_stl(stl_path, generate_views=generate_views)

    def import_sldprt(self, sldprt_path: Path) -> ImportResult:
        """Import via SolidWorks export (SLDPRT support — not yet implemented)."""
        paths = self._exporter.export_to_step_stl(sldprt_path)
        assert isinstance(paths, ExportPaths)
        return self.import_step_stl(
            paths.step_path,
            paths.stl_path,
            generate_views=True,
        )

    def _finalize(self, canonical: object, *, generate_views: bool) -> ImportResult:
        from nutella_scraper.domain.models.canonical import CanonicalModel3D

        assert isinstance(canonical, CanonicalModel3D)
        model_id = self._model_store.persist(canonical)
        self._persist_cad_reference_if_step(canonical, model_id)

        views_id: str | None = None
        if generate_views:
            views = self._view_generator.generate(canonical)
            if self._view_cache_store is not None:
                views_id = self._view_cache_store.save(views)

        return ImportResult(
            model_id=model_id,
            canonical=canonical,
            views_id=views_id,
        )

    def _persist_cad_reference_if_step(self, canonical: object, model_id: str) -> None:
        from nutella_scraper.domain.models.canonical import CanonicalModel3D

        assert isinstance(canonical, CanonicalModel3D)
        if canonical.format != "step":
            return

        try:
            from nutella_scraper.cad_import.cad_reference_builder import CadReferenceGeometryBuilder

            geometry = CadReferenceGeometryBuilder().from_step(
                canonical.source_path,
                model_id=model_id,
            )
            self._model_store.persist_cad_reference(
                model_id,
                geometry,
                step_path=canonical.source_path,
            )
            _LOG.info(
                "[cad_reference] persisted model=%s inner_faces=%d profile_edges=%d top_edges=%d",
                model_id,
                geometry.inner_face_count,
                geometry.profile_contour.edge_count if geometry.profile_contour else 0,
                geometry.top_contour.edge_count if geometry.top_contour else 0,
            )
        except Exception as exc:
            _LOG.warning("[cad_reference] failed for model=%s: %s", model_id, exc)
