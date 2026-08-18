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
    assert 'id="toggle-wireframe"' in html
    assert 'id="toggle-wireframe" data-layer-toggle="wireframe" checked' not in html
    assert "function ensureWireframeDefaultOff" in html
    assert "function recordRotationPerf" in html
    assert "timings_ms" in html
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
    assert "collectTriangles(scene3d.jar" not in html
    assert "function ensureMeshEdges" in html
    assert "include_svg_overlays: false" in html
    assert "scene3d.canvas && scene3d.canvas.isConnected" in html


def test_demo_viewer_switch_view_only_swaps_camera() -> None:
    html = _html()
    switch_start = html.index("function switchView")
    switch_end = html.index("function displayMetadata")
    switch_chunk = html[switch_start:switch_end]
    assert "renderActiveView" in switch_chunk
    assert "buildScraperOnly" not in switch_chunk
    assert "API.buildScraper" not in switch_chunk
    assert "loadViewerScene" not in switch_chunk
    assert "wireframe" not in switch_chunk
    assert "flattenMesh" not in switch_chunk

    render_start = html.index("function renderActiveView")
    render_chunk = html[render_start:switch_start]
    assert "drawScene3D" in render_chunk
    assert "buildScraperOnly" not in render_chunk
    assert "API.buildScraper" not in render_chunk
    assert "wireframe" not in render_chunk
    assert "flattenMesh" not in render_chunk
    assert "Camera swap only" in render_chunk


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


def test_demo_viewer_play_path_does_not_request_svg_overlays() -> None:
    html = _html()
    start = html.index("async function buildScraperOnly")
    chunk = html[start : start + 4200]
    assert "include_svg_overlays: false" in chunk
    assert "function prepareRotationCache" in html
    assert "function rotationReplayTick" in html
    assert "function applyCachedRotationFrame" in html


def test_demo_viewer_wireframe_off_skips_full_mesh_edges() -> None:
    html = _html()
    assert "function ensureMeshEdges" in html
    assert "if (!isSilhouette) continue" in html
    assert "if (!isBoundary && !isSilhouette) continue" not in html

    draw_start = html.index("function drawScene3D")
    draw_end = html.index("async function loadViewerScene")
    draw_chunk = html[draw_start:draw_end]
    assert "collectTriangles(scene3d.jar" not in draw_chunk
    assert "collectTriangles(scene3d.interior" not in draw_chunk
    assert "collectSilhouetteEdges(scene3d.jar" in draw_chunk
    assert "collectTriangles(posed" in draw_chunk
    assert "ensureMeshEdges(scene3d.jar)" in draw_chunk
    assert "collectEdges(scene3d.jar" in draw_chunk
    jar_edges_at = draw_chunk.index("collectEdges(scene3d.jar")
    wire_at = draw_chunk.rfind("if (showWireframe)", 0, jar_edges_at)
    assert wire_at != -1
    assert "ensureMeshEdges(scene3d.jar)" in draw_chunk[wire_at:jar_edges_at]


def test_demo_viewer_hides_legacy_viewport_debug_labels() -> None:
    html = _html()
    assert "function drawCameraDebugOverlay" not in html
    assert "drawCameraDebugOverlay(" not in html
    assert 'id="active-view-label" hidden' in html
    assert "Vue de dessus (Top View) — œil +Y" not in html
    assert 'option value="top" selected>Vue de dessus' in html
    assert 'option value="bottom">Vue de dessous' in html
    assert 'option value="side">Vue de profil' in html
    assert 'option value="left">Vue gauche' in html
    assert 'option value="right">Vue droite' in html
    assert 'option value="iso">Vue isométrique' in html


def test_demo_viewer_iso_orbit_is_frontend_camera_only() -> None:
    html = _html()
    assert "function applyIsoZoom" in html
    assert "function applyIsoOrbitDelta" in html
    assert "function isoCameraPreset" in html
    assert "function resetIsoOrbitFromScene" in html
    assert "ISO_ELEVATION_LIMIT" in html
    assert "isoOrbit.minDistance" in html
    assert "isoOrbit.maxDistance" in html
    assert "ctrlKey" in html

    zoom = html[html.index("function applyIsoZoom") : html.index("function applyIsoOrbitDelta")]
    orbit = html[
        html.index("function applyIsoOrbitDelta") : html.index("function updateViewHint")
    ]
    switch = html[html.index("function switchView") : html.index("function displayMetadata")]
    for chunk in (zoom, orbit):
        assert "buildScraperOnly" not in chunk
        assert "API.buildScraper" not in chunk
        assert "fetch(" not in chunk
        assert "scraperRest.vertices" not in chunk
        assert "scene3d.jar.vertices" not in chunk
    assert "resetIsoOrbitFromScene" not in switch
    assert 'currentViewName !== "iso"' in html
    handlers = html[html.index("mainStage.addEventListener") :]
    assert "applyIsoZoom" in handlers
    assert "applyIsoOrbitDelta" in handlers
    assert "buildScraperOnly" not in handlers[:2500]


def test_demo_viewer_waits_for_scene_before_declaring_camera_ready() -> None:
    html = _html()
    start = html.index("async function loadModel")
    end = html.index("async function fetchSvg")
    chunk = html[start:end]
    assert "Initialisation de la vue 3D…" in chunk
    assert "await loadViewerScene" in chunk
    assert chunk.index("await loadViewerScene") < chunk.index("scene3d.ready = true")
    assert chunk.index("scene3d.ready = true") < chunk.rindex("renderActiveView()")
    before_scene = chunk[: chunk.index("await loadViewerScene")]
    assert "renderActiveView()" not in before_scene
    draw = html[
        html.index("function drawScene3D") : html.index("async function loadViewerScene")
    ]
    assert "Initialisation de la vue 3D…" in draw
    assert "isViewerSceneReady" in html


def test_demo_viewer_viewcube_is_frontend_camera_only() -> None:
    html = _html()
    assert 'id="viewcube-canvas"' in html
    assert "function drawViewCube" in html
    assert "function applyViewCubeHit" in html
    assert "function pickViewCube" in html
    assert "function applyViewCubeArrow" in html
    assert "function ensureIsoOrbitFromCurrentView" in html
    assert "VIEWCUBE_ARROW_STEP" in html
    assert "VIEWCUBE_CHAMFER" in html
    assert "function ensureOutwardWinding" in html
    assert "function inflatePoly" in html
    assert "const arrowSize = 19" in html
    assert "for (const piece of faces) paint(piece)" in html
    assert "for (const piece of edges) paint(piece)" in html
    assert "for (const piece of corners) paint(piece)" in html
    assert 'label: "TOP"' in html
    assert 'label: "BOTTOM"' in html
    assert 'label: "FRONT"' in html
    assert 'label: "BACK"' in html
    assert 'label: "LEFT"' in html
    assert 'label: "RIGHT"' in html
    assert 'id: "arrow-up"' in html
    assert 'id: "arrow-down"' in html
    assert 'id: "arrow-left"' in html
    assert 'id: "arrow-right"' in html
    assert 'kind: "iso"' in html
    assert "function applyIsoOrbitDirection" in html
    assert "function fitIsoOrbitDistance" in html
    assert "function projectedAabbSpan" in html
    assert "function cameraFitOrthoScale" in html
    assert "function cameraFitPerspectiveDistance" in html
    assert "CAMERA_FIT_VIEWPORT_FACTOR" in html
    assert "CAMERA_FIT_HALF_SPAN" in html
    hit = html[html.index("function applyViewCubeHit") : html.index("function isCameraView")]
    assert "buildScraperOnly" not in hit
    assert "API.buildScraper" not in hit
    assert "fetch(" not in hit
    assert "switchView" in hit
    assert "applyIsoOrbitDirection" in hit
    assert "applyViewCubeArrow" in hit
    assert "resetIsoOrbitFromScene" in hit
    assert "api: false" in hit
    arrow = html[
        html.index("function applyViewCubeArrow") : html.index("function applyViewCubeHit")
    ]
    assert "fetch(" not in arrow
    assert "buildScraperOnly" not in arrow
    assert "VIEWCUBE_ARROW_STEP" in arrow
    assert 'switchView("iso")' in arrow
    assert "fitIsoOrbitDistance" not in arrow
    direction = html[
        html.index("function applyIsoOrbitDirection") : html.index("function projectPoint")
    ]
    assert "fitIsoOrbitDistance" in direction
    assert "fetch(" not in direction
    reset = html[
        html.index("function resetIsoOrbitFromScene") : html.index("function isoCameraPreset")
    ]
    assert "fitIsoOrbitDistance" in reset
    preset = html[
        html.index("function cameraFromPreset") : html.index("function isoOrbitLimitsFromBounds")
    ]
    assert "cameraFitOrthoScale" in preset
    assert "CAMERA_PERSPECTIVE_FOCAL_FACTOR" in preset


def test_demo_viewer_scene_scraper_is_visibility_and_color_only() -> None:
    html = _html()
    assert 'id="toggle-scene-scraper"' in html
    assert 'id="toggle-coordinate-frame"' in html
    assert "collectSilhouetteEdges(scene3d.jar, \"#d4d4d4\"" in html
    assert "collectSilhouetteEdges(scene3d.jar, \"#ffffff\"" not in html
    assert "collectEdges(scene3d.interior, \"#55ffff\"" in html
    assert "scene3d.collision && !scraperSoloMode ? \"#ff6b6b\" : \"#ffe600\"" in html
    sync = html[
        html.index("function syncScraperVisibilityToggles") : html.index(
            "function toggleScraperPanel"
        )
    ]
    assert "scraperFrameToggle" in sync
    assert "buildScraperOnly" not in sync
    assert "fetch(" not in sync
    assert "API.buildScraper" not in sync
    draw = html[html.index("function drawScene3D") : html.index("async function loadViewerScene")]
    assert "scraperFrameToggle" in draw
    assert "buildScraperOnly" not in draw
    assert "API.buildScraper" not in draw


def test_demo_viewer_clamps_length_and_shows_warning() -> None:
    html = _html()
    assert 'id="scraper-length-warning"' in html
    assert "function applyLengthLimit" in html
    assert "length_limit" in html
    assert "applyLengthLimit(result)" in html
    assert "writeScraperParametersToForm(result.parameters)" in html


def test_demo_viewer_scraper_button_opens_solo_view_with_cage() -> None:
    html = _html()
    assert "const SCRAPER_A_REFERENCE" in html
    assert "width_mm: 2.5" in html
    assert "thickness_mm: 2.5" in html
    assert "function enterScraperSoloView" in html
    assert "function exitScraperSoloView" in html
    assert "function drawControlCageOverlay" in html
    assert "control_cage" in html
    assert "enterScraperSoloView()" in html
    assert 'toolbarScraperPanel?.addEventListener("click", toggleScraperPanel)' not in html
    enter = html[
        html.index("async function enterScraperSoloView") : html.index(
            "async function buildScraperOnly"
        )
    ]
    assert "gizmo" not in enter.lower()
    assert "Sculpt" not in enter
    assert "SCRAPER_A_REFERENCE" in enter
    overlay = html[
        html.index("function drawControlCageOverlay") : html.index(
            "function cameraOppositionDebug"
        )
    ]
    assert "polylines_mm" in overlay
    assert "center_row_index" in overlay
    assert "addEventListener" not in overlay
    draw = html[html.index("function drawScene3D") : html.index("async function loadViewerScene")]
    assert "drawControlCageOverlay" in draw
    assert "scraperSoloMode" in draw
    assert "buildScraperOnly" not in draw
    assert "API.buildScraper" not in draw
