"""Geometry normalizer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nutella_scraper.cad_import.exceptions import UnsupportedFormatError
from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer


class TestGeometryNormalizer:
    def test_normalize_from_stl(self, box_stl_path: Path) -> None:
        normalizer = GeometryNormalizer()
        model = normalizer.normalize_from_stl(box_stl_path)

        assert model.format == "stl"
        assert model.provenance == "canonical_3d"
        assert model.geometry.vertex_count == 8
        assert model.geometry.volume_mm3 == pytest.approx(6000.0, rel=0.01)
        assert model.bounds.max_z - model.bounds.min_z == pytest.approx(30.0, abs=0.01)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "model.obj"
        bad.write_text("dummy")
        normalizer = GeometryNormalizer()
        with pytest.raises(UnsupportedFormatError):
            normalizer.normalize_from_stl(bad)

    def test_step_wrong_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "model.txt"
        bad.write_text("not step")
        normalizer = GeometryNormalizer()
        with pytest.raises(UnsupportedFormatError):
            normalizer.normalize_from_step(bad)
