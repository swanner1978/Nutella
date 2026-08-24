"""Union-not-sum checks for coverage angle audit. No CoverageSimulator physics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.engines.compute.coverage_angle_audit import (
    CoverageUnionMismatchError,
    area_of_faces,
    build_angle_audit,
    rest_azimuth_deg,
    union_face_ids,
    verify_coverage_is_union,
)
from nutella_scraper.engines.compute.coverage_simulator import CoverageResult

SRC = Path("src/nutella_scraper/engines/compute/coverage_angle_audit.py")
SIM = Path("src/nutella_scraper/engines/compute/coverage_simulator.py")
HTML = Path("scripts/templates/demo_viewer.html")


def _result(*, faces_by_angle, covered, area) -> CoverageResult:
    angles = tuple(float(a) for a, _ids in faces_by_angle)
    return CoverageResult(
        candidate_id="S0008",
        coverage_percent=66.25,
        covered_area_mm2=float(area),
        target_area_mm2=100.0,
        angle_start_deg=0.0,
        angle_end_deg=45.0,
        angle_step_deg=2.0,
        evaluated_angles=angles,
        covered_face_ids=frozenset(covered),
        best_pose_by_angle=tuple(
            (float(a), ScraperPose(position_mm=(float(i), 0.0, 0.0)))
            for i, (a, _ids) in enumerate(faces_by_angle)
        ),
        touched_face_ids_by_angle=tuple(
            (float(a), tuple(ids)) for a, ids in faces_by_angle
        ),
        shape_fingerprint="fixture",
    )


def test_audit_module_does_not_edit_simulator_source() -> None:
    sim = SIM.read_text(encoding="utf-8")
    audit = SRC.read_text(encoding="utf-8")
    assert "Does not choose poses" in audit
    assert "never the sum" in audit
    assert "union of touched faces" in sim


def test_union_is_not_the_sum_of_per_angle_areas() -> None:
    areas = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    faces_by_angle = ((0.0, (0, 1)), (2.0, (1, 2)))
    union = union_face_ids(faces_by_angle)
    assert union == frozenset({0, 1, 2})
    union_area = area_of_faces(union, areas)
    summed = area_of_faces((0, 1), areas) + area_of_faces((1, 2), areas)
    assert union_area == 30.0
    assert summed == 40.0
    result = _result(faces_by_angle=faces_by_angle, covered={0, 1, 2}, area=30.0)
    checks = verify_coverage_is_union(result, areas)
    assert checks["union_matches_covered_face_ids"] is True
    assert checks["union_area_matches_covered_area_mm2"] is True
    assert checks["sum_differs_from_union"] is True
    assert checks["coverage_is_union_not_sum"] is True


def test_union_mismatch_raises() -> None:
    areas = np.array([10.0, 10.0], dtype=np.float64)
    result = _result(faces_by_angle=((0.0, (0,)),), covered={0, 1}, area=20.0)
    with pytest.raises(CoverageUnionMismatchError):
        verify_coverage_is_union(result, areas)


def test_build_angle_audit_records_independent_poses() -> None:
    areas = np.array([10.0, 5.0], dtype=np.float64)
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    result = _result(
        faces_by_angle=((0.0, (0,)), (2.0, (0, 1))),
        covered={0, 1},
        area=15.0,
    )
    payload = build_angle_audit(
        result,
        areas=areas,
        centroids=centroids,
        control_points_mm=((0.0, 10.0, 0.0), (0.0, 20.0, 0.0)),
        rest_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        family="parallel",
        saved_row={"coverage_percent": 66.25, "covered_area_mm2": 15.0},
    )
    assert payload["replay_only"] is True
    assert payload["angles"][1]["pose_changed_from_previous"] is True
    assert payload["pose_jumps"]["independent_pose_per_angle"] is True
    assert payload["control_y_span_mm"] == 10.0
    assert payload["union_checks"]["sum_differs_from_union"] is True
    assert payload["angles"][0]["se3_matrix_4x4"] is not None
    assert len(payload["angles"][0]["se3_matrix_4x4"]) == 4
    assert payload["angles"][0]["position_xyz_mm"] == [0.0, 0.0, 0.0]
    assert payload["angles"][1]["position_xyz_mm"] == [1.0, 0.0, 0.0]


def test_rest_azimuth_uses_progress_convention() -> None:
    assert abs(float(rest_azimuth_deg(((50.0, 40.0, 0.0),))) - 0.0) < 1e-6
    assert abs(float(rest_azimuth_deg(((0.0, 40.0, -50.0),))) - 90.0) < 1e-6


def test_viewer_debug_play_uses_dump_api_not_simulator() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "API.coverageAngleAudit" in html
    assert "/api/coverage-angle-audit" in html
    assert "startCoverageAuditPlay" in html
    assert "showCoverageAuditFrame" in html
    assert "pauseCoverageAuditPlay" in html
    assert "resetCoverageAuditPlay" in html
    replay = html[
        html.index("function unionCoveredFacesUntil") : html.index(
            "async function buildScraperOnly"
        )
    ]
    assert "evaluate_candidate" not in replay
    assert "CoverageSimulator" not in replay
    assert "buildScraper" not in replay
    assert "showCoverageEnvelopeAt45" not in replay
    assert "unionCoveredFacesUntil" in replay
    assert "coverageAuditFrame + 1" in replay
    assert "covered_face_ids" in replay
    assert "startCoverageAuditPlay" in replay
    assert "pauseCoverageAuditPlay" in replay
    assert "resetCoverageAuditPlay" in replay
    click = html[
        html.index('getElementById("coverage-play")?.addEventListener') : html.index(
            'document.addEventListener("keydown"'
        )
    ]
    assert "startCoverageAuditPlay" in click
    assert "pauseCoverageAuditPlay" in click
    assert "resetCoverageAuditPlay" in click
    assert "evaluate_candidate" not in click


def test_viewer_bridge_serves_dump_without_simulator() -> None:
    text = Path("src/nutella_scraper/engines/visualization/viewer_bridge.py").read_text(
        encoding="utf-8"
    )
    fn = text[
        text.index("def build_coverage_angle_audit_response") : text.index(
            "def _compact_mesh_payload"
        )
    ]
    assert "Does not run CoverageSimulator" in fn
    assert "evaluate_candidate" not in fn
    assert "json.loads" in fn
    assert "evaluation_interior_envelope_payload" in fn
    assert "replay_mode" in fn
    draw = HTML.read_text(encoding="utf-8")
    scene = draw[
        draw.index("function drawScene3D") : draw.index("async function loadViewerScene")
    ]
    assert "coverageTarget.points_mm" in scene
    assert "toggle-scraper-points" in scene
    assert "showEvaluationEnvelope" not in scene

