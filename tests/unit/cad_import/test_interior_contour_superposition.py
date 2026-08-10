"""Interior contour must be the B-Rep frontier in the same plane/viewport as mesh views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")

from nutella_scraper.cad_import import GeometryNormalizer, ImportPipeline, ModelStore
from nutella_scraper.cad_import.brep_contour_extractor import PLANE_PROFILE, PLANE_TOP_XZ
from nutella_scraper.cad_import.cad_reference_builder import CadReferenceGeometryBuilder
from nutella_scraper.cad_import.trimesh_loader import TrimeshLoader
from nutella_scraper.engines.visualization.cad_reference_projector import contour_to_svg_fragment
from nutella_scraper.engines.visualization.projection_math import fit_to_viewport, project_vertices
from scripts.visualization_helpers import VIEW_CONVENTIONS


@pytest.fixture
def jar_step_path() -> Path:
    path = Path(__file__).resolve().parents[3] / "Solidworks" / "jar.STEP"
    if not path.exists():
        pytest.skip("Solidworks/jar.STEP fixture missing")
    return path


def test_interior_contours_use_same_planes_as_mesh_views(jar_step_path: Path) -> None:
    geometry = CadReferenceGeometryBuilder().from_step(jar_step_path, model_id="planes")
    assert geometry.profile_contour is not None
    assert geometry.top_contour is not None
    assert geometry.profile_contour.plane == VIEW_CONVENTIONS["side"]["plane"] == "XY"
    assert geometry.top_contour.plane == VIEW_CONVENTIONS["top"]["plane"] == "XZ"
    assert geometry.profile_contour.source == "opencascade_brep_section_xy"
    assert geometry.top_contour.source == "opencascade_brep_rim_xz"


def test_interior_contour_svg_matches_mesh_viewport_transform(
    jar_step_path: Path,
    tmp_path: Path,
) -> None:
    """
    Contour points transformed with the jar-mesh viewport must equal SVG coords
    of the same mm points (exact frontier, not a re-fitted shape).
    """
    pipeline = ImportPipeline(
        normalizer=GeometryNormalizer(loader=TrimeshLoader()),
        model_store=ModelStore(tmp_path / "models"),
    )
    result = pipeline.import_step(jar_step_path, generate_views=False)
    model = result.canonical
    geometry = CadReferenceGeometryBuilder().from_step(jar_step_path, model_id=model.id)
    jar_vertices = np.asarray(model.mesh.vertices, dtype=np.float64)

    for contour, plane in (
        (geometry.profile_contour, PLANE_PROFILE),
        (geometry.top_contour, PLANE_TOP_XZ),
    ):
        assert contour is not None
        assert len(contour.polylines) >= 1
        polyline = max(contour.polylines, key=lambda item: len(item.points_mm))
        points = np.asarray(polyline.points_mm, dtype=np.float64)
        jar_coords, _, _ = project_vertices(jar_vertices, plane)
        scale, offset = fit_to_viewport(jar_coords)
        expected = points * scale + offset

        fragment = contour_to_svg_fragment(contour, reference_coords=jar_coords)
        assert fragment
        # Parse first path's M/L coordinates and compare to expected transform.
        path_start = fragment.index('d="') + 3
        path_end = fragment.index('"', path_start)
        commands = fragment[path_start:path_end].replace("Z", " ").split()
        parsed: list[tuple[float, float]] = []
        for token in commands:
            if token.startswith(("M", "L")):
                x_str, y_str = token[1:].split(",")
                parsed.append((float(x_str), float(y_str)))

        assert len(parsed) == len(expected)
        delta = np.asarray(parsed, dtype=np.float64) - expected
        assert float(np.max(np.abs(delta))) <= 0.02

        # Must NOT be fitted on the contour bbox alone (that was the old bug).
        own_scale, own_offset = fit_to_viewport(points)
        own_projected = points * own_scale + own_offset
        # For a rim/section smaller than the full jar, own fit differs from jar fit.
        assert float(np.max(np.abs(own_projected - expected))) > 1.0


def test_profile_contour_is_xy_section_not_radial_approximation(jar_step_path: Path) -> None:
    geometry = CadReferenceGeometryBuilder().from_step(jar_step_path, model_id="xy")
    profile = geometry.profile_contour
    assert profile is not None
    xs = [x for polyline in profile.polylines for x, _y in polyline.points_mm]
    assert min(xs) < -20.0
    assert max(xs) > 20.0
    # Old R×Y contour stored only r ≥ 0 then mirrored; raw section has both signs.
    assert any(x < 0.0 for x in xs) and any(x > 0.0 for x in xs)
