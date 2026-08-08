"""View projection generator tests."""

from __future__ import annotations

from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer
from nutella_scraper.cad_import.view_projection_generator import ViewProjectionGenerator


class TestViewProjectionGenerator:
    def test_generates_profile_and_top_views(self, box_stl_path: object) -> None:
        from pathlib import Path

        model = GeometryNormalizer().normalize_from_stl(Path(str(box_stl_path)))
        cache = ViewProjectionGenerator().generate(model)

        assert cache.provenance == "visualization_projection"
        assert cache.profile_view.plane == "XZ"
        assert cache.top_view.plane == "XY"
        assert cache.profile_view.svg_content is not None
        assert "<svg" in cache.profile_view.svg_content
        assert cache.projection_metadata.get("visualization_only") is True

    def test_views_are_not_canonical_models(self, box_stl_path: object) -> None:
        from pathlib import Path

        model = GeometryNormalizer().normalize_from_stl(Path(str(box_stl_path)))
        cache = ViewProjectionGenerator().generate(model)

        assert not hasattr(cache, "mesh")
        assert not hasattr(cache, "geometry")
        assert cache.provenance != "canonical_3d"
