"""CAD import pipeline integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer
from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.cad_import.pipeline import ImportPipeline
from nutella_scraper.cad_import.view_cache_store import ViewCacheStore
from nutella_scraper.cad_import.view_projection_generator import ViewProjectionGenerator


class TestImportPipeline:
    def test_import_stl_full_pipeline(self, box_stl_path: Path, tmp_path: Path) -> None:
        pipeline = ImportPipeline(
            normalizer=GeometryNormalizer(),
            model_store=ModelStore(tmp_path / "models"),
            view_generator=ViewProjectionGenerator(),
            view_cache_store=ViewCacheStore(tmp_path / "views"),
        )
        result = pipeline.import_stl(box_stl_path)

        assert result.model_id == result.canonical.id
        assert result.canonical.provenance == "canonical_3d"
        assert result.views_id is not None
        assert result.canonical.geometry.volume_mm3 == pytest.approx(6000.0, rel=0.01)

    def test_import_without_views(self, box_stl_path: Path, tmp_path: Path) -> None:
        pipeline = ImportPipeline(
            normalizer=GeometryNormalizer(),
            model_store=ModelStore(tmp_path / "models"),
        )
        result = pipeline.import_stl(box_stl_path, generate_views=False)
        assert result.views_id is None

    def test_import_sldprt_not_implemented(self, tmp_path: Path) -> None:
        pipeline = ImportPipeline(
            normalizer=GeometryNormalizer(),
            model_store=ModelStore(tmp_path / "models"),
        )
        with pytest.raises(NotImplementedError):
            pipeline.import_sldprt(tmp_path / "part.sldprt")

    def test_canonical_not_from_views(self, box_stl_path: Path, tmp_path: Path) -> None:
        """Ensure computational model is built from 3D file, not from views."""
        pipeline = ImportPipeline(
            normalizer=GeometryNormalizer(),
            model_store=ModelStore(tmp_path / "models"),
            view_generator=ViewProjectionGenerator(),
            view_cache_store=ViewCacheStore(tmp_path / "views"),
        )
        result = pipeline.import_stl(box_stl_path)

        assert result.canonical.mesh.vertices
        assert result.canonical.geometry.face_count > 0
        views = ViewCacheStore(tmp_path / "views").get(result.views_id or "")
        assert views is not None
        assert views.provenance == "visualization_projection"
