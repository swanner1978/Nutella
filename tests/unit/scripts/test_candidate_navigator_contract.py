"""UI contract: scraper solo view shows A0, then ←/→ navigate the catalog."""

from __future__ import annotations

from pathlib import Path

HTML_SRC = Path("scripts/templates/demo_viewer.html")


def _html() -> str:
    return HTML_SRC.read_text(encoding="utf-8")


def _slice(html: str, start: str, end: str) -> str:
    return html[html.index(start) : html.index(end)]


def test_candidate_navigator_lives_in_view_toolbar() -> None:
    html = _html()
    toolbar = _slice(html, 'class="view-toolbar"', 'class="view-column"')
    assert 'id="candidate-navigator"' in toolbar
    assert 'id="candidate-prev"' in toolbar
    assert 'id="candidate-next"' in toolbar
    assert 'id="candidate-label"' in toolbar
    assert "Candidat A0 · 1 / 1" in toolbar
    frame = _slice(html, 'class="view-frame"', 'id="viewcube-canvas"')
    assert "candidate-navigator" not in frame


def test_entering_scraper_view_shows_a0_without_catalog() -> None:
    html = _html()
    enter = _slice(
        html,
        "async function enterScraperSoloView",
        "function cacheReferenceCandidate",
    )
    assert "loadShapeCandidateCatalog" not in enter
    assert "API.scraperShapeCandidates" not in enter
    assert "nav.hidden = false" in enter
    assert "cacheReferenceCandidate" in enter
    assert "applyCachedCandidate(0)" in enter
    assert "SCRAPER_A_REFERENCE" in enter

    cache = _slice(html, "function cacheReferenceCandidate", "async function loadShapeCandidateCatalog")
    assert 'candidate_id: "A0"' in cache
    assert "is_reference_a: true" in cache
    assert 'family: "A0"' in cache
    assert "candidateCache.set(0" in cache


def test_next_loads_catalog_once_then_prev_uses_cache() -> None:
    html = _html()
    step = _slice(
        html,
        "async function stepShapeCandidate",
        "async function buildScraperOnly",
    )
    assert "loadShapeCandidateCatalog" in step
    assert "applyCachedCandidate(target)" in step
    assert "candidateCache" not in step or "candidateCache.clear" not in step

    load = _slice(
        html,
        "async function loadShapeCandidateCatalog",
        "function formatCandidateFamily",
    )
    assert "API.scraperShapeCandidates" in load
    assert "candidateCache.set" in load
    assert "if (!candidateCache.has(index)) candidateCache.set(index, item)" in load

    listeners = html[html.index('toolbarScraperPanel?.addEventListener("click"') :]
    prev_click = listeners[
        listeners.index('getElementById("candidate-prev")') : listeners.index(
            'getElementById("candidate-next")'
        )
    ]
    next_click = listeners[
        listeners.index('getElementById("candidate-next")') : listeners.index(
            "scraperRecalculateButton"
        )
    ]
    assert "stepShapeCandidate(-1)" in prev_click
    assert "loadShapeCandidateCatalog" not in prev_click
    assert "stepShapeCandidate(1)" in next_click


def test_navigator_label_includes_id_position_and_family() -> None:
    html = _html()
    sync = _slice(html, "function syncCandidateNavigator", "function applyCachedCandidate")
    assert "cached.candidate_id" in sync
    assert "formatCandidateFamily(cached.family)" in sync
    assert "Candidat ${id}" in sync
    assert "${candidateIndex + 1} / ${total}" in sync
    assert "CANDIDATE_FAMILY_LABELS" in html
    assert "référence" in html
    assert "parallèle" in html
    assert "inclinée" in html
    assert "asymétrique" in html
