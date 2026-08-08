"""Trimesh loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from nutella_scraper.cad_import.exceptions import UnsupportedFormatError
from nutella_scraper.cad_import.trimesh_loader import TrimeshLoader


class TestTrimeshLoader:
    def test_load_stl(self, box_stl_path: Path) -> None:
        mesh = TrimeshLoader().load(box_stl_path)
        assert len(mesh.vertices) == 8

    def test_unsupported_format(self, tmp_path: Path) -> None:
        obj = tmp_path / "x.obj"
        obj.write_text("o")
        with pytest.raises(UnsupportedFormatError):
            TrimeshLoader().load(obj)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TrimeshLoader().load(tmp_path / "missing.stl")

    def test_step_uses_precise_tessellation_and_converts_to_mm(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        step_path = tmp_path / "part.step"
        step_path.write_bytes(b"STEP")
        captured: dict[str, object] = {}

        def fake_load(path: str, **kwargs: object) -> trimesh.Trimesh:
            captured["path"] = path
            captured.update(kwargs)
            mesh = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
            mesh.units = "meters"
            return mesh

        monkeypatch.setattr(trimesh, "load", fake_load)
        loader = TrimeshLoader(
            step_tol_linear_mm=0.01,
            step_tol_angular_rad=0.1,
        )

        mesh = loader.load(step_path)

        assert captured["path"] == str(step_path.resolve())
        assert captured["tol_linear"] == 0.01
        assert captured["tol_angular"] == 0.1
        assert captured["tol_relative"] is False
        assert mesh.units == "millimeters"
        assert mesh.extents.tolist() == pytest.approx([100.0, 200.0, 300.0])
