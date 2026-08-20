"""UI contract: Scraper view chrome — Repères / pot / lissage."""

from __future__ import annotations

from pathlib import Path

HTML_SRC = Path("scripts/templates/demo_viewer.html")


def _html() -> str:
    return HTML_SRC.read_text(encoding="utf-8")


def _slice(html: str, start: str, end: str) -> str:
    return html[html.index(start) : html.index(end)]


def test_scraper_view_toolbar_has_cues_pot_and_smoothing() -> None:
    html = _html()
    toolbar = _slice(html, 'class="view-toolbar"', 'class="view-column"')
    assert 'id="scraper-view-controls"' in toolbar
    assert 'id="toggle-scraper-cues"' in toolbar
    assert 'id="toggle-scraper-pot"' in toolbar
    assert 'id="toggle-scraper-smoothing"' in toolbar
    assert "Pot de Nutella" in toolbar
    assert "Lissage" in toolbar
    assert "Repères" in toolbar
    assert 'id="toggle-scraper-cues" checked' in toolbar
    assert 'id="toggle-scraper-pot"' in toolbar
    assert "checked" not in html[
        html.index('id="toggle-scraper-pot"') : html.index('id="toggle-scraper-pot"') + 80
    ]
    assert "checked" not in html[
        html.index('id="toggle-scraper-smoothing"') : html.index(
            'id="toggle-scraper-smoothing"'
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
    assert "loadShapeCandidateCatalog" not in enter
    chrome = _slice(html, "function setScraperViewChrome", "function exitScraperSoloView")
    assert "toggle-scraper-cues" in chrome
    assert "cuesToggle.checked = true" in chrome
    assert "potToggle.checked = false" in chrome
    assert "smoothToggle.checked = false" in chrome
    assert "API.buildScraper" not in chrome
    assert "loadShapeCandidateCatalog" not in chrome
    assert "buildScraperOnly" not in chrome


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
    assert "onScraperViewDisplayToggle" in html
    assert 'getElementById("toggle-scraper-pot")' in html
    assert 'getElementById("toggle-scraper-smoothing")' in html
    assert 'getElementById("toggle-scraper-cues")' in html


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
