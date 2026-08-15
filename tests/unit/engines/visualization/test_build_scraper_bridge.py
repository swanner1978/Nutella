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
    colored = Path(__file__).resolve().parents[4] / "Solidworks" / "jar_color-jar.step"
    if not colored.exists():
        pytest.skip("Solidworks/jar_color-jar.step missing")

    model_dir = models_root / viewer_dir.name
    shutil.copy2(colored, model_dir / ModelStore.REFERENCE_STEP)
    cache = model_dir / "interior_product_surface.npz"
    if cache.exists():
        cache.unlink()
    legacy = model_dir / "interior_rgb_85_255_255.npz"
    if legacy.exists():
        legacy.unlink()

    payload = build_scraper_visualization_response(
        view_dir=viewer_dir,
        models_root=models_root,
    )

    assert payload["scraper"]["vertex_count"] > 0
    assert payload["scraper"]["provenance"] == "parametric_v1_rigid_pose"
    assert payload["scraper"]["interior_source"] == "interior_product_surface_mesh"
    assert payload["scraper"]["rigid_geometry"] is True
    assert "parameters" in payload
    assert payload["parameters"]["width_mm"] > 0
    assert "surface_progress_deg" in payload["parameters"]
    assert payload["scraper_pipeline"]["stages"]["envelope_path"]["source"] == (
        "interior_product_surface_mesh"
    )
    assert payload["scraper_pipeline"]["stages"]["envelope_path"]["sampling"] == (
        "rigid_pose_along_envelope"
    )
    assert "scraper-volume" in payload["overlays"]["side"]
    assert "scraper-volume" in payload["overlays"]["top"]
    assert "scraper-volume" in payload["overlays"]["bottom"]
    assert payload["scraper_transform"]
    assert len(payload["scraper_transform"]) == 4
    assert "scraper_geometry" in payload
    assert len(payload["scraper_geometry"]["vertices"]) == payload["scraper"]["vertex_count"]
    assert len(payload["scraper_geometry"]["faces"]) == payload["scraper"]["face_count"]
    assert "scraper-trajectory" in payload["overlays"]["top"]
    assert payload["active_edge"]["point_count"] >= 2
    assert payload["active_edge"]["source"] == "rigid_pose_tip_edge"
    assert "validation" in payload
    assert payload["validation"]["rotation_angle_deg"] == pytest.approx(
        payload["parameters"]["rotation_angle_deg"]
    )
    assert payload["validation"]["surface_progress_deg"] == pytest.approx(
        payload["parameters"]["surface_progress_deg"]
    )
    assert "pose_status" in payload["validation"]
    assert payload["validation"]["pose_status"] in {"VALID", "INVALID", "BLOCKED"}
    assert "collision" in payload["validation"]
    if payload["validation"]["pose_status"] == "VALID":
        assert payload["validation"]["has_collision"] is False
        assert payload["validation"]["admissible"] is True
    assert payload["collision"]["pose_status"] == payload["validation"]["pose_status"]
    assert payload["validation"]["distance_min_mm"] >= 0.0
    assert payload["validation"]["penetration_mm"] >= 0.0
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
