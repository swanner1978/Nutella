"""Load saved coverage-100 rankings without invoking CoverageSimulator."""

from __future__ import annotations

from pathlib import Path

import pytest

from nutella_scraper.engines.visualization.coverage_rank_catalog import (
    COVERAGE_PLAY_REQUIRED_FIELDS,
    clamp_rank_index,
    coverage_play_status,
    filter_ranked_prefix,
    load_coverage_rank_csv,
    load_coverage_rank_json,
    neighbor_rank_rows,
    ranked_viewer_rows,
    shapes_by_id_from_candidates,
    step_rank_index,
    walk_rank_indices,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_JSON = FIXTURE_DIR / "candidate_coverage_sample.json"
SAMPLE_CSV = FIXTURE_DIR / "candidate_coverage_sample.csv"
SAVED_JSON = Path("output/coverage/candidate_coverage_100.json")
SAVED_CSV = Path("output/coverage/candidate_coverage_100.csv")
SRC = Path("src/nutella_scraper/engines/visualization/coverage_rank_catalog.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")


def test_rank_catalog_module_never_imports_coverage_simulator() -> None:
    text = SRC.read_text(encoding="utf-8")
    assert "from nutella_scraper.engines.compute.coverage_simulator" not in text
    assert "import CoverageSimulator" not in text
    assert "evaluate_candidate(" not in text


def test_sample_json_and_csv_agree_and_play_is_unavailable() -> None:
    payload = load_coverage_rank_json(SAMPLE_JSON)
    csv_rows = load_coverage_rank_csv(SAMPLE_CSV)
    json_ids = [row["candidate_id"] for row in payload["ranked"]]
    csv_ids = [row["candidate_id"] for row in csv_rows]
    assert json_ids == csv_ids == ["S0008", "A0"]
    play = coverage_play_status(payload["ranked"][0])
    assert play["available"] is False
    for field in COVERAGE_PLAY_REQUIRED_FIELDS:
        assert field in play["missing_fields"]


def test_ranked_rows_join_curves_without_mutating_geometry_fields() -> None:
    payload = load_coverage_rank_json(SAMPLE_JSON)
    shapes = shapes_by_id_from_candidates(
        [
            {
                "candidate_id": "S0008",
                "control_points_mm": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "family": "parallel",
                "thickness_mm": 2.5,
                "shape_fingerprint": "fixture-s0008",
            },
            {
                "candidate_id": "A0",
                "control_points_mm": [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                "family": "A0",
                "thickness_mm": 2.5,
                "is_reference_a": True,
                "shape_fingerprint": "fixture-a0",
            },
        ]
    )
    rows = ranked_viewer_rows(payload, shapes_by_id=shapes)
    assert [row["candidate_id"] for row in rows] == ["S0008", "A0"]
    assert rows[0]["rank"] == 1
    assert rows[0]["coverage_percent"] == 66.25
    assert rows[1]["is_reference_a"] is True
    assert rows[0]["is_best"] is True
    assert rows[1]["is_best"] is False
    assert rows[0]["metrics_source"] == "saved_json"
    assert rows[0]["thickness_mm"] == 2.5
    assert rows[0]["curve_available"] is True
    assert rows[0]["coverage_play"]["available"] is False
    fingerprint_before = shapes["S0008"]["shape_fingerprint"]
    points_before = [list(p) for p in shapes["S0008"]["control_points_mm"]]
    rows[0]["control_points_mm"][0][0] = 99.0
    rows[0]["rank"] = 99
    assert shapes["S0008"]["shape_fingerprint"] == fingerprint_before
    assert shapes["S0008"]["control_points_mm"] == points_before


@pytest.mark.skipif(not SAVED_JSON.is_file(), reason="saved coverage-100 JSON absent")
def test_saved_s0008_is_rank_1_and_a0_is_rank_5() -> None:
    payload = load_coverage_rank_json(SAVED_JSON)
    ranked = payload["ranked"]
    assert ranked[0]["candidate_id"] == "S0008"
    assert ranked[0]["rank"] == 1
    assert float(ranked[0]["coverage_percent"]) == pytest.approx(66.25, abs=1e-4)
    a0 = next(row for row in ranked if row["candidate_id"] == "A0")
    assert a0["rank"] == 5
    assert float(a0["coverage_percent"]) == pytest.approx(63.3333, abs=1e-4)
    s0010 = next(row for row in ranked if row["candidate_id"] == "S0010")
    assert float(s0010["coverage_percent"]) == pytest.approx(65.0, abs=1e-4)
    rows = ranked_viewer_rows(payload)
    assert rows[0]["is_best"] is True
    assert rows[4]["candidate_id"] == "A0"
    assert rows[4]["is_reference_a"] is True
    csv_rows = load_coverage_rank_csv(SAVED_CSV)
    assert csv_rows[0]["candidate_id"] == "S0008"
    assert csv_rows[4]["candidate_id"] == "A0"


@pytest.mark.skipif(not SAVED_JSON.is_file(), reason="saved coverage-100 JSON absent")
def test_navigation_1_to_100_and_back_does_not_recompute() -> None:
    payload = load_coverage_rank_json(SAVED_JSON)
    rows = ranked_viewer_rows(payload)
    assert len(rows) == 100
    assert [int(row["rank"]) for row in rows] == list(range(1, 101))
    path = walk_rank_indices(100)
    assert path[0] == 0
    assert path[99] == 99
    assert path[-1] == 0
    ids = []
    index = 0
    for _ in range(99):
        index = step_rank_index(index, 1, 100)
        ids.append(rows[index]["candidate_id"])
    assert index == 99
    assert rows[index]["rank"] == 100
    for _ in range(99):
        index = step_rank_index(index, -1, 100)
    assert index == 0
    assert rows[index]["candidate_id"] == "S0008"
    assert step_rank_index(0, -1, 100) == 0
    assert step_rank_index(99, 1, 100) == 99
    assert clamp_rank_index(-3, 100) == 0
    top10 = filter_ranked_prefix(rows, 10)
    top20 = filter_ranked_prefix(rows, 20)
    assert [row["candidate_id"] for row in top10] == [
        row["candidate_id"] for row in rows[:10]
    ]
    assert len(top20) == 20
    assert "A0" in {row["candidate_id"] for row in top10}
    neighbors = neighbor_rank_rows(rows, 0)
    assert neighbors["previous"] is None
    assert neighbors["next"]["candidate_id"] == rows[1]["candidate_id"]
    src = SRC.read_text(encoding="utf-8")
    assert "from nutella_scraper.engines.compute.coverage_simulator" not in src
    assert "evaluate_candidate(" not in src


@pytest.mark.skipif(not SAVED_JSON.is_file(), reason="saved coverage-100 JSON absent")
def test_saved_coverage_100_has_a0_and_no_play_poses() -> None:
    payload = load_coverage_rank_json(SAVED_JSON)
    ranked = payload["ranked"]
    assert len(ranked) == 100
    ids = [row["candidate_id"] for row in ranked]
    assert "A0" in ids
    assert ranked[0]["candidate_id"] == "S0008"
    assert ranked[ids.index("A0")]["family"] == "A0"
    assert "best_pose_by_angle" not in ranked[0]
    assert "touched_face_ids_by_angle" not in ranked[0]
    csv_rows = load_coverage_rank_csv(SAVED_CSV)
    assert [row["candidate_id"] for row in csv_rows] == ids


def test_viewer_html_loads_saved_rank_and_never_calls_simulator() -> None:
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "API.coverageRankCatalog" in html
    assert "/api/coverage-rank-catalog" in html
    assert "tryLoadCoverageRankCatalog" in html
    assert 'id="coverage-play"' in html
    assert 'id="coverage-play-bar"' in html
    assert 'id="coverage-pause"' in html
    assert 'id="coverage-reset"' in html
    assert "ArrowLeft" in html
    assert "ArrowRight" in html
    assert "Couverture :" in html
    assert 'id="candidate-top-filter"' in html
    assert 'option value="10" selected' in html
    assert "let candidateTopLimit = 10" in html
    assert "candidateBladeMesh" in html
    assert "unionCoveredFacesUntil" in html
    assert "showCoverageEnvelopeAt45" not in html
    assert "toggle-scene-scraper" in html
    assert "Top 20" in html
    assert "Meilleur" in html
    assert "Référence" in html
    assert "play.disabled = !canDebugPlay" in html
    assert "onCandidateTopFilterChange" in html
    assert "evaluate_candidate(" not in html
    sync = html[
        html.index("function syncCandidateNavigator") : html.index(
            "function applyCachedCandidate"
        )
    ]
    assert "catalogTotal" in sync
    assert "neighborSummary" in sync
    assert "metrics.hidden = true" in sync
    assert "play.disabled = !canDebugPlay" in sync
    assert "CoverageSimulator" not in html
    enter = html[
        html.index("async function enterScraperSoloView") : html.index(
            "function cacheReferenceCandidate"
        )
    ]
    assert "loadShapeCandidateCatalog({ resetToBest: true })" in enter
    assert "coverageRankCatalog" in html
    play_click = html[
        html.index("document.getElementById(\"coverage-play\")?.addEventListener") : html.index(
            'document.addEventListener("keydown"'
        )
    ]
    assert "buildScraperOnly" not in play_click
    assert "simulate-contact" not in play_click
    assert "evaluate_candidate(" not in play_click
    draw = html[html.index("function drawScene3D") : html.index("async function loadViewerScene")]
    assert "coverageTarget.points_mm" in draw
    assert "toggle-scraper-points" in draw
    assert "coverageTarget" in draw
    assert "showEvaluationEnvelope" not in draw
    assert "candidateBladeMesh" in html
    assert "DISPLAY_BLADE_SCALE = 4.0" in html
    assert "toggle-scene-scraper" in html
    assert '"#ffffff"' in draw


def test_blade_display_mesh_uses_physical_2_5mm_and_visual_scale_four() -> None:
    from nutella_scraper.engines.visualization.coverage_rank_catalog import (
        DISPLAY_BLADE_SCALE,
        blade_display_mesh_from_curve,
    )

    assert DISPLAY_BLADE_SCALE == 4.0
    curve = ((50.0, 40.0, 0.0), (50.0, 60.0, 0.0), (50.0, 80.0, 0.0))
    mesh = blade_display_mesh_from_curve(curve, width_mm=2.5, thickness_mm=2.5)
    assert mesh is not None
    assert mesh["display_blade_scale"] == 4.0
    assert mesh["width_mm"] == 2.5
    assert mesh["thickness_mm"] == 2.5
    assert mesh["visual_only"] is True
    assert len(mesh["vertices"]) >= 8
    assert len(mesh["faces"]) >= 8
    xs = [v[0] for v in mesh["vertices"]]
    assert max(xs) - min(xs) == pytest.approx(10.0, abs=1e-6)


def test_evaluation_interior_envelope_stays_on_cylinder() -> None:
    from nutella_scraper.engines.visualization.coverage_rank_catalog import (
        evaluation_interior_envelope_payload,
    )

    payload = evaluation_interior_envelope_payload()
    assert payload["simulator_invoked"] is False
    assert payload["on_interior_surface"] is True
    radius = float(payload["radius_mm"])
    for x, _y, z in payload["vertices"]:
        r = (float(x) ** 2 + float(z) ** 2) ** 0.5
        assert abs(r - radius) < 1e-5
    assert payload["max_vertex_radius_mm"] == pytest.approx(radius, abs=1e-6)
    assert payload["min_vertex_radius_mm"] == pytest.approx(radius, abs=1e-6)
    assert len(payload["sector_face_ids"]) > 0
    covered_ok = all(
        0 <= int(i) < len(payload["faces"]) for i in payload["sector_face_ids"]
    )
    assert covered_ok
