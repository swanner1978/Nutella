"""View cache store tests."""

from __future__ import annotations

from pathlib import Path

from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer
from nutella_scraper.cad_import.view_cache_store import ViewCacheStore
from nutella_scraper.cad_import.view_projection_generator import ViewProjectionGenerator


class TestViewCacheStore:
    def test_save_and_get(self, box_stl_path: Path, tmp_path: Path) -> None:
        model = GeometryNormalizer().normalize_from_stl(box_stl_path)
        cache = ViewProjectionGenerator().generate(model)

        store = ViewCacheStore(tmp_path / "views")
        views_id = store.save(cache)
        loaded = store.get(views_id)

        assert loaded is not None
        assert loaded.model_id == model.id
        assert loaded.provenance == "visualization_projection"
        assert loaded.profile_view.svg_content is not None
        assert loaded.top_view.svg_content is not None
