"""Unit tests for STEP face-color diagnostic helpers (no production pipeline changes)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("OCP")

from nutella_scraper.cad_import.step_face_color_diagnostics import (
    TARGET_RGB_255,
    diagnose_step_face_colors,
    format_step_face_color_report,
    normalized_srgb_to_rgb255,
    rgb255_matches,
)


def test_rgb255_matches_target_cyan() -> None:
    assert TARGET_RGB_255 == (85, 255, 255)
    assert rgb255_matches((85, 255, 255))
    assert rgb255_matches((84, 255, 254), tolerance=2)
    assert not rgb255_matches((23, 255, 255), tolerance=2)


def test_normalized_srgb_to_rgb255() -> None:
    assert normalized_srgb_to_rgb255((0.33333, 1.0, 1.0)) == (85, 255, 255)


def test_diagnose_freecad_colored_jar_step() -> None:
    path = Path(__file__).resolve().parents[3] / "Solidworks" / "jar_color-jar.step"
    if not path.exists():
        pytest.skip("Solidworks/jar_color-jar.step missing")

    diagnostic = diagnose_step_face_colors(path)
    report = format_step_face_color_report(diagnostic)

    assert diagnostic.color_information_available
    assert "COLOR INFORMATION AVAILABLE" in report
    assert diagnostic.total_faces > 0
    assert diagnostic.faces_with_readable_color > 0
    assert diagnostic.matching_face_count > 0
    assert diagnostic.total_target_area_mm2 > 0.0
    assert diagnostic.loader.startswith("STEPCAFControl_Reader")
    assert TARGET_RGB_255 in diagnostic.unique_colors_255 or any(
        rgb255_matches(color) for color in diagnostic.unique_colors_255
    )
