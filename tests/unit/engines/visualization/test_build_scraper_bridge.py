"""Build scraper / interior contour API tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nutella_scraper.cad_import.model_store import ModelStore
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


def test_build_interior_contour_response_uses_step_face_colors(viewer_dir: Path) -> None:
    models_root = viewer_dir.parent.parent / "models"
    colored = Path(__file__).resolve().parents[4] / "Solidworks" / "jar_color-jar.step"
    if not colored.exists():
        pytest.skip("Solidworks/jar_color-jar.step missing")

    model_dir = models_root / viewer_dir.name
    shutil.copy2(colored, model_dir / ModelStore.REFERENCE_STEP)

    payload = build_interior_contour_response(
        view_dir=viewer_dir,
        models_root=models_root,
    )

    assert payload["layer"] == "interior-envelope"
    assert payload["source"] == "step_face_color_rgb_85_255_255"
    assert payload["interior_colored_faces"] == 13
    assert payload["target_rgb_255"] == [85, 255, 255]
    assert "interior-envelope" in payload["overlays"]["side"]
    assert "rgb(85,255,255)" in payload["overlays"]["side"]["interior-envelope"]
    assert "cad_reference" not in payload
