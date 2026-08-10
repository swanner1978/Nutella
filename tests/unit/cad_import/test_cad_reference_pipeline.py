"""CAD B-Rep reference geometry pipeline tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("OCP")

from nutella_scraper.cad_import.brep_contour_extractor import PLANE_PROFILE, PLANE_TOP_XZ
from nutella_scraper.cad_import.cad_reference_builder import CadReferenceGeometryBuilder
from nutella_scraper.cad_import.model_store import CadReferenceNotAvailableError, ModelStore
from nutella_scraper.engines.visualization.envelope_projector import EnvelopeProjector
from scripts.visualization_helpers import build_cad_reference_projection_svg


class TestCadReferencePipeline:
    def test_views_generated_from_brep(self, cad_reference_geometry) -> None:
        assert cad_reference_geometry.metadata.get("source") == "opencascade_brep"
        assert cad_reference_geometry.profile_contour is not None
        assert cad_reference_geometry.profile_contour.source == "opencascade_brep_section_xy"
        assert cad_reference_geometry.top_contour is not None
        assert cad_reference_geometry.top_contour.source == "opencascade_brep_rim_xz"

    def test_brep_pipeline_does_not_use_trimesh_for_contours(self, jar_step_path: Path) -> None:
        """CAD reference builder must not depend on mesh tessellation."""
        with patch(
            "nutella_scraper.cad_import.cad_reference_builder.load_step_shape",
            wraps=__import__(
                "nutella_scraper.cad_import.step_brep_loader",
                fromlist=["load_step_shape"],
            ).load_step_shape,
        ) as load_shape:
            geometry = CadReferenceGeometryBuilder().from_step(jar_step_path, model_id="no_mesh")
        load_shape.assert_called_once()
        assert geometry.inner_face_count > 0

    def test_inner_faces_only(self, cad_reference_geometry) -> None:
        assert cad_reference_geometry.inner_face_count > 0
        assert cad_reference_geometry.outer_face_count > cad_reference_geometry.inner_face_count

    def test_profile_and_top_use_view_planes(self, cad_reference_geometry) -> None:
        profile = cad_reference_geometry.profile_contour
        top = cad_reference_geometry.top_contour
        assert profile is not None and top is not None
        assert profile.plane == PLANE_PROFILE == "XY"
        assert profile.view_axis == "Z"
        assert top.plane == PLANE_TOP_XZ == "XZ"
        assert top.view_axis == "Y"

    def test_profile_has_pot_like_extent(self, cad_reference_geometry) -> None:
        profile = cad_reference_geometry.profile_contour
        assert profile is not None
        points = [point for polyline in profile.polylines for point in polyline.points_mm]
        xs = [x for x, _y in points]
        heights = [y for _x, y in points]
        assert max(abs(x) for x in xs) > 20.0
        assert min(xs) < 0.0 < max(xs)
        assert max(heights) - min(heights) > 50.0
        assert min(heights) < 0.0
        assert max(heights) > 80.0

    def test_top_contour_is_rim_opening(self, cad_reference_geometry) -> None:
        top = cad_reference_geometry.top_contour
        assert top is not None
        assert top.edge_count >= 1
        assert len(top.polylines) >= 1
        assert top.polylines[0].is_closed
        points = [point for polyline in top.polylines for point in polyline.points_mm]
        z_values = [z for _x, z in points]
        assert max(z_values) - min(z_values) > 10.0
        assert sum(len(polyline.points_mm) for polyline in top.polylines) >= 4

    def test_no_theoretical_circle_in_svg(self, cad_reference_geometry) -> None:
        svg = build_cad_reference_projection_svg(
            cad_reference_geometry,
            plane=PLANE_TOP_XZ,
            model_id="jar_test",
            canonical_mesh_sha256="test",
        )
        assert "<circle" not in svg.lower()
        assert "path" in svg

    def test_mesh_density_does_not_affect_cad_reference(self, jar_step_path: Path) -> None:
        first = CadReferenceGeometryBuilder().from_step(jar_step_path, model_id="a")
        second = CadReferenceGeometryBuilder().from_step(jar_step_path, model_id="b")
        assert first.profile_contour == second.profile_contour
        assert first.top_contour == second.top_contour

    def test_envelope_and_contour_share_cad_source(self, cad_reference_geometry) -> None:
        contour_projection = EnvelopeProjector().project_geometry(cad_reference_geometry)
        assert contour_projection.profile_layers
        assert contour_projection.top_layers
        assert "#a855f7" in contour_projection.profile_layers[0].svg_fragment

    def test_cad_reference_persisted_in_model_store(
        self,
        tmp_path: Path,
        jar_step_path: Path,
        cad_reference_geometry,
    ) -> None:
        store = ModelStore(tmp_path / "models")
        store.persist_cad_reference(
            "jar_test",
            cad_reference_geometry,
            step_path=jar_step_path,
        )
        loaded = store.get_cad_reference("jar_test")
        assert loaded.step_sha256 == cad_reference_geometry.step_sha256
        assert loaded.profile_contour == cad_reference_geometry.profile_contour

    def test_stl_model_has_no_cad_reference(self, tmp_path: Path, cylindrical_jar_canonical) -> None:
        store = ModelStore(tmp_path / "models")
        store.persist(cylindrical_jar_canonical)
        with pytest.raises(CadReferenceNotAvailableError):
            store.get_cad_reference(cylindrical_jar_canonical.id)
