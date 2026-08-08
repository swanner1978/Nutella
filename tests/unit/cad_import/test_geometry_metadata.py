"""Geometry metadata tests."""

from __future__ import annotations

import pytest
import trimesh

from nutella_scraper.cad_import.geometry_metadata import compute_geometric_metadata


class TestGeometryMetadata:
    def test_box_metadata(self, box_mesh: trimesh.Trimesh) -> None:
        meta = compute_geometric_metadata(box_mesh)
        assert meta.vertex_count == 8
        assert meta.face_count == 12
        assert meta.is_watertight is True
        assert meta.volume_mm3 == pytest.approx(6000.0, rel=0.01)
        assert meta.dimensions_mm == pytest.approx((10.0, 20.0, 30.0), abs=0.01)
        assert meta.bounding_box.max_x - meta.bounding_box.min_x == pytest.approx(10.0, abs=0.01)

    def test_principal_axes_unit_length(self, box_mesh: trimesh.Trimesh) -> None:
        meta = compute_geometric_metadata(box_mesh)
        for axis in meta.principal_axes:
            length = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
            assert length == pytest.approx(1.0, abs=0.01)

    def test_center_near_origin_for_centered_box(self, box_mesh: trimesh.Trimesh) -> None:
        meta = compute_geometric_metadata(box_mesh)
        assert meta.center_mm == pytest.approx((0.0, 0.0, 0.0), abs=0.01)
