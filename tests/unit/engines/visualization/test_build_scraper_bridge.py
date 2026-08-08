"""Build scraper API tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nutella_scraper.engines.visualization.viewer_bridge import (
    build_interior_contour_response,
    build_scraper_visualization_response,
)


def test_build_scraper_response_contains_scraper_overlays_only(
    viewer_dir: Path,
) -> None:
    models_root = viewer_dir.parent.parent / "models"
    payload = build_scraper_visualization_response(
        view_dir=viewer_dir,
        models_root=models_root,
    )

    assert payload["scraper"]["vertex_count"] > 0
    assert "scraper-volume" in payload["overlays"]["side"]
    assert "contact-covered" not in payload["overlays"]["side"]
    assert payload["pose"]["height_mm"] is not None


def test_build_interior_contour_response(viewer_dir: Path) -> None:
    models_root = viewer_dir.parent.parent / "models"
    if not (models_root / viewer_dir.name / "cad_reference.json").exists():
        pytest.skip("cad_reference.json required for interior contour overlay")
    payload = build_interior_contour_response(
        view_dir=viewer_dir,
        models_root=models_root,
    )

    assert "interior-envelope" in payload["overlays"]["side"]
    assert "#a855f7" in payload["overlays"]["side"]["interior-envelope"]
    cad_reference = payload["cad_reference"]
    assert cad_reference["source"] == "opencascade_brep"
    assert cad_reference["inner_face_count"] > 0
    assert cad_reference["profile_edge_count"] > 0
