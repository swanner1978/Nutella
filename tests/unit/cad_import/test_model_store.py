"""Model store tests."""

from __future__ import annotations

from pathlib import Path

from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer
from nutella_scraper.cad_import.model_store import ModelStore


class TestModelStore:
    def test_persist_and_get(self, box_stl_path: Path, tmp_path: Path) -> None:
        store = ModelStore(tmp_path / "models")
        model = GeometryNormalizer().normalize_from_stl(box_stl_path)
        model_id = store.persist(model)

        loaded = store.get(model_id)
        assert loaded.id == model.id
        assert loaded.source_hash == model.source_hash
        assert loaded.geometry.dimensions_mm == model.geometry.dimensions_mm
        assert len(loaded.mesh.vertices) == len(model.mesh.vertices)

    def test_delete(self, box_stl_path: Path, tmp_path: Path) -> None:
        store = ModelStore(tmp_path / "models")
        model = GeometryNormalizer().normalize_from_stl(box_stl_path)
        model_id = store.persist(model)
        store.delete(model_id)
        assert not (tmp_path / "models" / model_id).exists()
