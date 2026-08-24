"""Rigid coverage simulator — 0–45° sector, InteriorSurfaceReference only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from tests.unit.engines.compute.test_scraper_parametric_v1 import (
    _profile_a,
    _reference_from_profile,
)

from nutella_scraper.engines.compute.coverage_simulator import (
    ANGLE_END_DEG,
    ANGLE_START_DEG,
    ANGLE_STEP_DEG,
    REFERENCE_CANDIDATE_ID,
    CoverageSimulator,
    coverage_angle_samples_deg,
    unique_edge_lengths_mm,
)
from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    RigidScraperArtifact,
    build_rigid_scraper_artifact,
)

COMPUTE_SRC = Path("src/nutella_scraper/engines/compute/coverage_simulator.py")
COLLISION_SRC = Path(
    "src/nutella_scraper/engines/compute/scraper_envelope_collision.py"
)
HTML_SRC = Path("scripts/templates/demo_viewer.html")


def _fast_surface():
    return _reference_from_profile(
        radius_at_y=lambda _y: 50.0,
        y_min=0.0,
        y_max=80.0,
        y_count=21,
        angular_count=48,
    )


def _a0_parameters(surface):
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    return _profile_a(
        width_mm=2.5,
        thickness_mm=2.5,
        length_mm=min(40.0, max_length),
        clearance_mm=0.0,
        position_z_mm=float(0.5 * (surface.y_min_mm + surface.y_max_mm)),
    )


def _blocked_artifact(good: RigidScraperArtifact) -> RigidScraperArtifact:
    huge = trimesh.creation.box(extents=(120.0, 120.0, 120.0))
    return RigidScraperArtifact(
        mesh=huge,
        design_frame=good.design_frame,
        tip_edge_mm=good.tip_edge_mm,
        wall_edge_mm=good.wall_edge_mm,
        design_path=good.design_path,
        shape_fingerprint="blocked-volume",
    )


@pytest.fixture(scope="module")
def a0_bundle():
    surface = _fast_surface()
    params = _a0_parameters(surface)
    artifact = build_rigid_scraper_artifact(surface, params)
    vertices_before = np.asarray(artifact.mesh.vertices, dtype=np.float64).copy()
    edges_before = unique_edge_lengths_mm(artifact.mesh)
    simulator = CoverageSimulator(surface, parameters=params)
    simulator.register(REFERENCE_CANDIDATE_ID, artifact)
    simulator.register("BAD", _blocked_artifact(artifact))
    result = simulator.evaluate_candidate("A0")
    bad = simulator.evaluate_candidate("BAD")
    return {
        "surface": surface,
        "params": params,
        "artifact": artifact,
        "simulator": simulator,
        "result": result,
        "bad": bad,
        "vertices_before": vertices_before,
        "edges_before": edges_before,
    }


def test_evaluated_angles_are_exactly_zero_to_forty_five_by_two(a0_bundle) -> None:
    expected = tuple([float(deg) for deg in range(0, 45, 2)] + [45.0])
    assert coverage_angle_samples_deg() == expected
    assert expected[0] == ANGLE_START_DEG
    assert expected[-1] == ANGLE_END_DEG
    assert ANGLE_STEP_DEG == 2.0
    result = a0_bundle["result"]
    assert result.evaluated_angles == expected
    assert result.angle_start_deg == 0.0
    assert result.angle_end_deg == 45.0
    assert result.angle_step_deg == 2.0


def test_a0_is_evaluated_without_modifying_its_geometry(a0_bundle) -> None:
    artifact = a0_bundle["artifact"]
    result = a0_bundle["result"]
    assert result.candidate_id == "A0"
    assert result.shape_fingerprint == artifact.shape_fingerprint
    assert np.allclose(
        a0_bundle["edges_before"], unique_edge_lengths_mm(artifact.mesh), atol=1e-6
    )
    assert np.allclose(
        a0_bundle["vertices_before"],
        np.asarray(artifact.mesh.vertices),
        atol=1e-9,
    )


def test_se3_pose_does_not_change_intrinsic_edge_lengths(a0_bundle) -> None:
    result = a0_bundle["result"]
    artifact = a0_bundle["artifact"]
    posed_count = sum(1 for _angle, pose in result.best_pose_by_angle if pose is not None)
    assert posed_count >= 1
    assert np.allclose(
        a0_bundle["edges_before"], unique_edge_lengths_mm(artifact.mesh), atol=1e-6
    )


def test_touched_faces_are_unioned_not_summed(a0_bundle) -> None:
    result = a0_bundle["result"]
    stacked: list[int] = []
    for _angle, face_ids in result.touched_face_ids_by_angle:
        stacked.extend(face_ids)
    assert set(stacked) == set(result.covered_face_ids)
    assert len(result.covered_face_ids) <= len(stacked)


def test_coverage_percent_is_between_zero_and_one_hundred(a0_bundle) -> None:
    result = a0_bundle["result"]
    assert 0.0 <= result.coverage_percent <= 100.0
    if result.target_area_mm2 > 0.0:
        expected = 100.0 * result.covered_area_mm2 / result.target_area_mm2
        assert result.coverage_percent == pytest.approx(expected, abs=1e-6)
    assert result.symmetry_multiplier_applied is False
    assert result.uses_visual_stl is False


def test_blocked_shape_does_not_get_artificial_coverage(a0_bundle) -> None:
    bad = a0_bundle["bad"]
    assert bad.coverage_percent == 0.0
    assert bad.covered_area_mm2 == 0.0
    assert bad.covered_face_ids == frozenset()


def test_a0_evaluation_is_deterministic(a0_bundle) -> None:
    first = a0_bundle["result"]
    second = a0_bundle["simulator"].evaluate_candidate("A0")
    assert first.coverage_percent == second.coverage_percent
    assert first.covered_face_ids == second.covered_face_ids
    assert first.covered_area_mm2 == second.covered_area_mm2
    assert first.evaluated_angles == second.evaluated_angles


def test_coverage_never_reads_visual_stl_or_viewer_caches() -> None:
    text = COMPUTE_SRC.read_text(encoding="utf-8")
    assert "visual.stl" not in text
    assert "rotationCache" not in text
    assert "candidateCache" not in text
    assert "engines.visualization" not in text
    assert "InteriorSurfaceReference" in text
    collision = COLLISION_SRC.read_text(encoding="utf-8")
    assert "coverage_simulator" not in collision
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "const rotationCache = new Map()" in html


A0_BASELINE_COVERAGE_PERCENT = 63.33333333333344
A0_BASELINE_COVERED_AREA_MM2 = 1988.2551285963507
A0_BASELINE_TARGET_AREA_MM2 = 3139.3502030468644
A0_BASELINE_FINGERPRINT = (
    "synthetic-interior|w=2.5|l=40|t=2.5|z=40|bevel=0|relief=0|helix=0|clear=0"
)
A0_BASELINE_FACE_IDS = frozenset(
    {
        420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
        516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527,
        612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623,
        708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719,
        804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815,
        900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911,
        996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
        1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103,
        1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199,
        1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295,
        1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391,
        1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487,
        1582, 1583, 1678, 1679, 1774, 1775, 1870, 1871,
    }
)


def test_evaluate_candidates_keeps_a0_and_sorts_by_coverage(a0_bundle) -> None:
    ranked = a0_bundle["simulator"].evaluate_candidates(["A0", "BAD"])
    assert [item.candidate_id for item in ranked] == ["A0", "BAD"]
    assert ranked[0].coverage_percent >= ranked[1].coverage_percent
    assert any(item.candidate_id == "A0" for item in ranked)


def test_a0_covers_the_useful_zero_to_forty_five_sector(a0_bundle) -> None:
    result = a0_bundle["result"]
    assert result.target_area_mm2 > 0.0
    assert result.coverage_percent > 0.0
    assert len(result.covered_face_ids) >= 1
    assert result.symmetry_multiplier_applied is False
    q0, q1, _q2, _q3 = result.quadrant_areas_mm2
    assert q0 > 0.0
    if min(q0, q1) > 0.0:
        assert max(q0, q1) / min(q0, q1) < 1.5


def test_a0_physical_result_uses_ninety_degree_quadrant(a0_bundle) -> None:
    """Live target is the 90° quadrant. Saved 0–45° campaign JSON stays frozen."""
    result = a0_bundle["result"]
    n_valid = sum(1 for _angle, pose in result.best_pose_by_angle if pose is not None)
    assert result.coverage_target_surface == "interior_product_surface"
    assert result.coverage_target_region == "interior_matrix_a0_left_45"
    assert result.coverage_target_azimuth_range[0] == pytest.approx(0.0)
    assert result.coverage_target_azimuth_range[1] == pytest.approx(45.0)
    assert result.target_area_mm2 == pytest.approx(A0_BASELINE_TARGET_AREA_MM2, abs=1e-2)
    assert result.symmetry_multiplier_applied is False
    assert n_valid == 24
    assert len(result.evaluated_angles) == 24
    assert result.shape_fingerprint == A0_BASELINE_FINGERPRINT
    assert result.shape_fingerprint == a0_bundle["artifact"].shape_fingerprint


def test_a0_does_not_mutate_interior_reference_mesh(a0_bundle) -> None:
    surface = a0_bundle["surface"]
    mesh = a0_bundle["simulator"]._surface_mesh
    assert np.allclose(np.asarray(mesh.vertices), np.asarray(surface.vertices), atol=1e-12)
    assert np.array_equal(np.asarray(mesh.faces), np.asarray(surface.faces))
    assert np.allclose(
        a0_bundle["vertices_before"],
        np.asarray(a0_bundle["artifact"].mesh.vertices),
        atol=1e-9,
    )


def test_a0_batch_matches_evaluate_candidate(a0_bundle) -> None:
    simulator = a0_bundle["simulator"]
    first = a0_bundle["result"]
    artifact = a0_bundle["artifact"]
    vertices_before = np.asarray(artifact.mesh.vertices, dtype=np.float64).copy()
    simulator._results.clear()
    batched = simulator.evaluate_candidates_batch(["A0"])
    assert len(batched) == 1
    second = batched[0]
    n_valid_first = sum(1 for _angle, pose in first.best_pose_by_angle if pose is not None)
    n_valid_second = sum(
        1 for _angle, pose in second.best_pose_by_angle if pose is not None
    )
    assert second.coverage_percent == pytest.approx(first.coverage_percent, abs=1e-6)
    assert second.covered_area_mm2 == pytest.approx(first.covered_area_mm2, abs=1e-2)
    assert second.covered_face_ids == first.covered_face_ids
    assert n_valid_second == n_valid_first
    assert second.shape_fingerprint == first.shape_fingerprint
    assert second.evaluated_angles == first.evaluated_angles
    assert [
        pose is None for _angle, pose in second.best_pose_by_angle
    ] == [pose is None for _angle, pose in first.best_pose_by_angle]
    assert np.allclose(artifact.mesh.vertices, vertices_before, atol=1e-9)
