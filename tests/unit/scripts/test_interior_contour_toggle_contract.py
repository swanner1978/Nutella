"""Frontend contract: Contour intérieur toggle is visibility-only over a cached overlay."""

from __future__ import annotations

from pathlib import Path


def test_demo_viewer_caches_interior_faces_and_toggles_visibility_only() -> None:
    html = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "templates"
        / "demo_viewer.html"
    ).read_text(encoding="utf-8")

    assert 'data-visibility-toggle="envelope"' in html
    assert "Contour intérieur" in html
    assert "fetchInteriorContour().catch" in html
    assert "once per STEP load" in html

    start = html.index("envelopeToggle.addEventListener")
    chunk = html[start : start + 280]
    assert "applyLayerVisibility" in chunk
    assert "fetchInteriorContour" not in chunk
    assert "simulate-contact" not in chunk
    assert "import-step" not in chunk

    assert "interiorContourOverlays" in html
    assert "toggle-target-face-colors" not in html
    assert "interiorContour:" in html
    assert 'id="toolbar-toggle-envelope"' not in html
