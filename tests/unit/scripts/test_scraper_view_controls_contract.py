"""UI contract: Charger STEP / Scraper toolbar — Pot, Repère, Scraper, Nuage."""

from __future__ import annotations

from pathlib import Path

HTML_SRC = Path("scripts/templates/demo_viewer.html")


def _html() -> str:
    return HTML_SRC.read_text(encoding="utf-8")


def _slice(html: str, start: str, end: str) -> str:
    return html[html.index(start) : html.index(end)]


def test_load_step_toolbar_is_pot_frame_scraper() -> None:
    html = _html()
    toolbar = _slice(html, 'class="view-toolbar"', 'class="view-column"')
    assert 'id="toggle-toolbar-pot"' in toolbar
    assert 'id="toggle-coordinate-frame"' in toolbar
    assert 'id="toggle-scene-scraper"' in toolbar
    assert toolbar.index('id="toggle-toolbar-pot"') < toolbar.index(
        'id="toggle-toolbar-wireframe"'
    )
    assert toolbar.index('id="toggle-toolbar-wireframe"') < toolbar.index(
        'id="toggle-coordinate-frame"'
    )
    assert toolbar.index('id="toggle-coordinate-frame"') < toolbar.index(
        'id="toggle-scene-scraper"'
    )
    assert "Wireframe" in toolbar
    assert "Repères" not in toolbar
    assert "Pot de Nutella" not in html
    assert "Référence A0" not in html


def test_scraper_view_toolbar_has_pot_frame_scraper_and_points() -> None:
    html = _html()
    toolbar = _slice(html, 'class="view-toolbar"', 'class="view-column"')
    assert 'id="scraper-view-controls"' in toolbar
    assert 'id="toggle-scraper-points"' in toolbar
    assert 'id="toggle-toolbar-wireframe"' in toolbar
    assert "Nuage de points" in toolbar
    assert "Nuages de points" not in html
    assert 'id="toggle-toolbar-wireframe-label" hidden' in toolbar
    assert 'id="toggle-scraper-cues"' not in html
    assert 'id="toggle-scraper-reference-a0"' not in html
    assert 'id="toggle-scraper-pot"' not in html
    assert "checked" not in html[
        html.index('id="toggle-scraper-points"') : html.index(
            'id="toggle-scraper-points"'
        )
        + 80
    ]


def test_entering_scraper_view_sets_default_toggles() -> None:
    html = _html()
    enter = _slice(
        html,
        "async function enterScraperSoloView",
        "function cacheReferenceCandidate",
    )
    assert "setScraperViewChrome(true)" in enter
    assert "loadShapeCandidateCatalog({ resetToBest: true })" in enter
    assert "ensureVisualA0" in enter
    chrome = _slice(
        html,
        "function setScraperViewChrome",
        "async function ensureVisualA0",
    )
    assert "toggle-toolbar-pot" in chrome
    assert "toggle-toolbar-wireframe-label" in chrome
    assert "wireframeLabel.hidden = !active" in chrome
    assert "potToggle.checked = true" in chrome
    assert "scraperToggle.checked = true" in chrome
    assert "pointsToggle.checked = false" in chrome
    assert "API.buildScraper" not in chrome


def test_smoothing_and_pot_toggles_do_not_rebuild() -> None:
    html = _html()
    draw = _slice(html, "function drawCandidateContactCurve", "function cameraOppositionDebug")
    assert "smoothContactCurve(curve)" in draw
    assert "for (const point of curve)" in draw
    assert "candidateCache" not in draw
    assert "API.buildScraper" not in draw
    assert "scraperShapeCandidates" not in draw

    cage = _slice(html, "function drawControlCageOverlay", "function pchipSlopes")
    assert "smoothContactCurve" not in cage
    assert "polylines_mm" in cage

    smooth = _slice(html, "function smoothContactCurve", "function drawCandidateContactCurve")
    assert "src[i][0] =" not in smooth
    assert "curve[i]" not in smooth
    assert "candidateCache" not in smooth
    assert "controlCage" not in smooth

    handler = _slice(
        html,
        "function onScraperViewDisplayToggle",
        "function exitScraperSoloView",
    )
    assert "redrawCameraView" in handler
    assert "buildScraperOnly" not in handler
    assert "loadShapeCandidateCatalog" not in handler
    assert "API.buildScraper" not in handler
    assert "scraperShapeCandidates" not in handler
    assert 'getElementById("toggle-toolbar-pot")' in html
    assert 'getElementById("toggle-toolbar-wireframe")' in html
    assert 'getElementById("toggle-scene-scraper")' in html
    assert 'getElementById("toggle-scraper-points")' in html


def test_candidate_navigation_ignores_smoothing_toggle() -> None:
    html = _html()
    step = _slice(
        html,
        "async function stepShapeCandidate",
        "async function buildScraperOnly",
    )
    assert "toggle-scraper-smoothing" not in step
    assert "smoothToggle" not in step
    apply = _slice(html, "function applyCachedCandidate", "async function stepShapeCandidate")
    assert "toggle-scraper-smoothing" not in apply
    assert "candidateCache.get" in apply
    assert "referenceA0Mesh" in apply
    assert "candidateBladeMesh" not in apply
    assert "scraperSoloRestMesh" not in apply
    assert "is_reference_a &&" not in apply
