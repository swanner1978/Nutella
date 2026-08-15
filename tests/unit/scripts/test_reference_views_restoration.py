"""Reference 2D views must use CanonicalModel3D.mesh + build_projection_svg."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import pytest

from nutella_scraper.cad_import import GeometryNormalizer, ImportPipeline, ModelStore
from nutella_scraper.cad_import.trimesh_loader import TrimeshLoader
from nutella_scraper.engines.visualization.projection_math import (
    PLANE_LEFT,
    PLANE_RIGHT,
    PLANE_SIDE,
    PLANE_TOP,
)
from scripts.demo_import import _save_views
from scripts.visualization_helpers import VIEW_CONVENTIONS, build_projection_svg


@pytest.fixture
def jar_step_path() -> Path:
    path = Path(__file__).resolve().parents[3] / "Solidworks" / "jar.STEP"
    if not path.exists():
        pytest.skip("Solidworks/jar.STEP fixture missing")
    return path


def test_view_conventions_assign_xy_to_profile_and_xz_to_top() -> None:
    """Panel mapping: profil = XY, dessus = XZ (Y-up jar)."""
    assert VIEW_CONVENTIONS["side"]["plane"] == "XY"
    assert VIEW_CONVENTIONS["side"]["view_axis"] == "Z"
    assert VIEW_CONVENTIONS["side"]["label_fr"] == "Vue de profil"
    assert VIEW_CONVENTIONS["top"]["plane"] == "X-Z"
    assert VIEW_CONVENTIONS["top"]["view_axis"] == "Y"
    assert VIEW_CONVENTIONS["top"]["label_fr"] == "Vue de dessus"
    assert VIEW_CONVENTIONS["bottom"]["plane"] == "XZ"
    assert VIEW_CONVENTIONS["bottom"]["view_axis"] == "-Y"
    assert VIEW_CONVENTIONS["bottom"]["label_fr"] == "Vue de dessous"
    assert PLANE_SIDE == VIEW_CONVENTIONS["side"]["plane"] == "XY"
    assert PLANE_TOP == VIEW_CONVENTIONS["top"]["plane"] == "X-Z"


def test_view_conventions_include_opposite_left_right() -> None:
    assert VIEW_CONVENTIONS["left"]["plane"] == PLANE_LEFT == "ZY"
    assert VIEW_CONVENTIONS["left"]["view_axis"] == "X"
    assert VIEW_CONVENTIONS["left"]["label_fr"] == "Vue gauche"
    assert VIEW_CONVENTIONS["right"]["plane"] == PLANE_RIGHT == "-ZY"
    assert VIEW_CONVENTIONS["right"]["view_axis"] == "-X"
    assert VIEW_CONVENTIONS["right"]["label_fr"] == "Vue droite"


def test_save_views_assigns_projections_to_correct_panels(
    jar_step_path: Path,
    tmp_path: Path,
) -> None:
    pipeline = ImportPipeline(
        normalizer=GeometryNormalizer(loader=TrimeshLoader()),
        model_store=ModelStore(tmp_path / "models"),
    )
    result = pipeline.import_step(jar_step_path, generate_views=False)
    model = result.canonical

    with patch("scripts.demo_import.build_projection_svg", wraps=build_projection_svg) as mocked:
        view_dir = _save_views(
            model,
            tmp_path / "views",
            canonical_mesh_hash="test_hash",
            tessellation={"applies_to": "STEP"},
        )

    assert mocked.call_count == 5
    planes_by_call = [call.kwargs["plane"] for call in mocked.call_args_list]
    assert set(planes_by_call) == {"X-Z", "XZ", "XY", "ZY", "-ZY"}
    for call in mocked.call_args_list:
        mesh = call.args[0]
        assert len(mesh.vertices) == model.geometry.vertex_count
        assert len(mesh.faces) == model.geometry.face_count

    side_svg = (view_dir / "side_composite.svg").read_text(encoding="utf-8")
    top_svg = (view_dir / "top_composite.svg").read_text(encoding="utf-8")
    bottom_svg = (view_dir / "bottom_composite.svg").read_text(encoding="utf-8")
    left_svg = (view_dir / "left_composite.svg").read_text(encoding="utf-8")
    right_svg = (view_dir / "right_composite.svg").read_text(encoding="utf-8")
    side_root = ElementTree.fromstring(side_svg)
    top_root = ElementTree.fromstring(top_svg)
    bottom_root = ElementTree.fromstring(bottom_svg)
    left_root = ElementTree.fromstring(left_svg)
    right_root = ElementTree.fromstring(right_svg)
    assert side_root.attrib["data-plane"] == "XY"
    assert side_root.attrib["data-view-axis"] == "Z"
    assert top_root.attrib["data-plane"] == "X-Z"
    assert top_root.attrib["data-view-axis"] == "Y"
    assert bottom_root.attrib["data-plane"] == "XZ"
    assert bottom_root.attrib["data-view-axis"] == "-Y"
    assert left_root.attrib["data-plane"] == "ZY"
    assert left_root.attrib["data-view-axis"] == "X"
    assert right_root.attrib["data-plane"] == "-ZY"
    assert right_root.attrib["data-view-axis"] == "-X"
    assert "path" in side_svg and "path" in top_svg
    assert "path" in left_svg and "path" in right_svg
    assert "path" in bottom_svg

    metadata = (view_dir / "metadata.json").read_text(encoding="utf-8")
    for view_name in ("side", "top", "left", "right", "bottom"):
        assert f'"view_name": "{view_name}"' in metadata
    assert "PROFILE" not in metadata
    assert "TOP_XZ" not in metadata
