"""Frontend contract: rotation PLAY uses a viewer-frame cache + loop replay."""

from __future__ import annotations

from pathlib import Path


def _html() -> str:
    return (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "templates"
        / "demo_viewer.html"
    ).read_text(encoding="utf-8")


def test_demo_viewer_rotation_play_caches_viewer_frames() -> None:
    html = _html()

    assert "const rotationCache = new Map()" in html
    assert "ROTATION_STEP_DEFAULT_DEG = 5" in html
    assert "prepareRotationCache" in html
    assert "rotationReplayTick" in html
    assert "applyCachedRotationFrame" in html
    assert "isPlayCacheComplete" in html
    assert "clearRotationCache" in html
    assert "geometryFingerprint" in html
    assert "Calcul des positions" in html
    assert "scraper-cache-status" in html
    assert "animationFrames" in html
    assert "syncFrameNavigator" in html
    assert "showFrameAtIndex" in html
    # Replay must not call the geometry API once the cache is complete.
    start = html.index("function rotationReplayTick")
    chunk = html[start : start + 900]
    assert "applyCachedRotationFrame" in chunk
    assert "buildScraperOnly" not in chunk
    assert "API.buildScraper" not in chunk


def test_demo_viewer_rotation_step_control_preserves_cache() -> None:
    html = _html()

    assert 'id="scraper-step-minus"' in html
    assert 'id="scraper-step-plus"' in html
    assert 'id="scraper-step-display"' in html
    assert "scraper-rotation-minus" not in html
    assert "scraper-rotation-plus" not in html
    assert "setRotationStepDeg" in html
    assert "ROTATION_STEP_MIN_DEG = 1" in html
    assert "ROTATION_STEP_MAX_DEG = 30" in html

    start = html.index("function setRotationStepDeg")
    chunk = html[start : start + 900]
    assert "clearRotationCache" not in chunk
    assert "rotationCache.clear" not in chunk
    assert "updateRotationCacheStatus" in chunk


def test_demo_viewer_rotation_slider_is_frame_index_navigator() -> None:
    html = _html()

    assert 'id="scraper-frame-status"' in html
    assert "Frame" in html

    slider = html.index('scraper-rotation-slider")?.addEventListener("input"')
    slider_chunk = html[slider : slider + 700]
    assert "showFrameAtIndex" in slider_chunk
    assert "allowCompute: false" in slider_chunk
    assert "clearRotationCache" not in slider_chunk
    assert "setRotationAngle(Number(event.target.value))" not in slider_chunk

    navigator = html.index("function syncFrameNavigator")
    navigator_chunk = html[navigator : navigator + 1400]
    assert 'slider.max = String(Math.max(0, frames.length - 1))' in navigator_chunk
    assert 'slider.step = "1"' in navigator_chunk
    assert "scraper-frame-status" in navigator_chunk
    assert "Frame ${index + 1}" in navigator_chunk


def test_demo_viewer_has_top_bottom_and_isometric_views() -> None:
    html = _html()
    assert 'option value="top"' in html
    assert 'option value="bottom"' in html
    assert 'option value="iso"' in html
    assert "Vue de dessus" in html
    assert "Vue de dessous" in html
    assert "Vue isométrique" in html
    assert "CAMERA_VIEW_NAMES" in html
    assert 'const CAMERA_VIEW_NAMES = ["top", "bottom", "side", "left", "right", "iso"]' in html
    assert "const SVG_VIEW_ORDER = []" in html
    assert 'option value="side"' in html
    assert 'option value="left"' in html
    assert 'option value="right"' in html
    assert "function isLayerEnabled" in html
    assert "function collectSilhouetteEdges" in html
    assert 'data-layer-toggle="contour"' in html
    assert 'data-layer-toggle="wireframe"' in html
    assert "/api/viewer-scene" in html
    assert "function cameraFromPreset" in html
    assert "function drawScene3D" in html
    assert "function cameraOppositionDebug" in html
    assert "function renderIsometricView" in html
    assert "function updateIsoFromBuildResult" in html
    assert "function renderScene3DView" in html
    assert "scraper_transform" in html
    assert "SVG_VIEW_ORDER" in html
    assert html.index("function switchView") > 0
    switch_start = html.index("function switchView")
    switch_chunk = html[switch_start : switch_start + 500]
    assert "wireframe" not in switch_chunk
    assert "collectTriangles(scene3d.jar" in html
    assert "scene3d.canvas && scene3d.canvas.isConnected" in html


def test_demo_viewer_hard_constraint_diagnostics_and_blocked_play() -> None:
    html = _html()
    assert "Collision :" in html
    assert 'id="val-collision"' in html
    assert "Clearance minimale" in html
    assert 'id="val-clearance-min"' in html
    assert 'id="val-pose-status"' in html
    assert "MOUVEMENT BLOQUÉ" in html
    assert "function frameIsBlocked" in html
    start = html.index("function rotationReplayTick")
    chunk = html[start : start + 2500]
    assert "frameIsBlocked" in chunk
    assert "stopRotationPlay" in chunk
    assert "buildScraperOnly" not in chunk
