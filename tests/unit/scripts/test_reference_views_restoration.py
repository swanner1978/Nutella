"""Reference 2D views must use CanonicalModel3D.mesh + build_projection_svg (XZ/XY)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import pytest

from nutella_scraper.cad_import import GeometryNormalizer, ImportPipeline, ModelStore
from nutella_scraper.cad_import.trimesh_loader import TrimeshLoader
from scripts.demo_import import _save_views
from scripts.visualization_helpers import VIEW_CONVENTIONS, build_projection_svg


@pytest.fixture
def jar_step_path() -> Path:
    path = Path(__file__).resolve().parents[3] / "Solidworks" / "jar.STEP"
    if not path.exists():
        pytest.skip("Solidworks/jar.STEP fixture missing")
    return path


def test_view_conventions_are_xz_profile_and_xy_top() -> None:
    assert VIEW_CONVENTIONS["side"]["plane"] == "XZ"
    assert VIEW_CONVENTIONS["side"]["view_axis"] == "Y"
    assert VIEW_CONVENTIONS["side"]["label_fr"] == "Vue de profil"
    assert VIEW_CONVENTIONS["top"]["plane"] == "XY"
    assert VIEW_CONVENTIONS["top"]["view_axis"] == "Z"
    assert VIEW_CONVENTIONS["top"]["label_fr"] == "Vue de dessus"


def test_save_views_calls_build_projection_svg_with_canonical_mesh(
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

    assert mocked.call_count == 2
    planes = {call.kwargs["plane"] for call in mocked.call_args_list}
    assert planes == {"XZ", "XY"}
    for call in mocked.call_args_list:
        mesh = call.args[0]
        assert len(mesh.vertices) == model.geometry.vertex_count
        assert len(mesh.faces) == model.geometry.face_count

    side_svg = (view_dir / "side_composite.svg").read_text(encoding="utf-8")
    top_svg = (view_dir / "top_composite.svg").read_text(encoding="utf-8")
    side_root = ElementTree.fromstring(side_svg)
    top_root = ElementTree.fromstring(top_svg)
    assert side_root.attrib["data-plane"] == "XZ"
    assert side_root.attrib["data-view-axis"] == "Y"
    assert top_root.attrib["data-plane"] == "XY"
    assert top_root.attrib["data-view-axis"] == "Z"
    assert "path" in side_svg
    assert "path" in top_svg

    metadata = (view_dir / "metadata.json").read_text(encoding="utf-8")
    assert '"plane": "XZ"' in metadata
    assert '"plane": "XY"' in metadata
    assert "PROFILE" not in metadata
    assert "TOP_XZ" not in metadata
