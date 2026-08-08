"""Geometry validator tests."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from nutella_scraper.cad_import.exceptions import InvalidGeometryError
from nutella_scraper.cad_import.geometry_validator import GeometryValidator, ValidationConfig


class TestGeometryValidator:
    def test_valid_box_passes(self, box_mesh: trimesh.Trimesh) -> None:
        GeometryValidator().validate(box_mesh, "box.stl")

    def test_empty_mesh_fails(self) -> None:
        mesh = trimesh.Trimesh(vertices=[], faces=[])
        with pytest.raises(InvalidGeometryError) as exc:
            GeometryValidator().validate(mesh, "empty.stl")
        assert "no vertices" in exc.value.violations[0].lower()

    def test_nan_vertices_fail(self, box_mesh: trimesh.Trimesh) -> None:
        mesh = box_mesh.copy()
        mesh.vertices[0, 0] = float("nan")
        with pytest.raises(InvalidGeometryError) as exc:
            GeometryValidator().validate(mesh, "nan.stl")
        assert any("NaN" in v for v in exc.value.violations)

    def test_degenerate_bounds_fail(self) -> None:
        vertices = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        with pytest.raises(InvalidGeometryError) as exc:
            GeometryValidator().validate(mesh, "flat.stl")
        assert any("degenerate" in v.lower() for v in exc.value.violations)

    def test_watertight_required(self, box_mesh: trimesh.Trimesh) -> None:
        validator = GeometryValidator(ValidationConfig(require_watertight=True))
        validator.validate(box_mesh, "box.stl")

        open_mesh = trimesh.creation.box(extents=(5, 5, 5))
        open_mesh.faces = open_mesh.faces[:-2]
        with pytest.raises(InvalidGeometryError):
            validator.validate(open_mesh, "open.stl")
