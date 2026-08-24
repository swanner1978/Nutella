"""Catalog → rigid coverage ranking. Compute only; no viewer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from tests.unit.engines.compute.coverage_catalog_fixtures import load_generated_catalog
from tests.unit.engines.compute.test_coverage_simulator import (
    A0_BASELINE_COVERAGE_PERCENT,
    A0_BASELINE_COVERED_AREA_MM2,
    A0_BASELINE_FACE_IDS,
    A0_BASELINE_FINGERPRINT,
    A0_BASELINE_TARGET_AREA_MM2,
)

from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.engines.compute.candidate_coverage import (
    A0_BASELINE_TOUCHED_FACE_COUNT,
    A0_BASELINE_VALID_POSE_COUNT,
    COVERAGE_BATCH_SIZE,
    A0BaselineRegressionError,
    CandidateCoverageResult,
    RankedCoverageBatch,
    a0_matches_baseline,
    assert_a0_matches_baseline,
    candidate_result_from_coverage,
    evaluate_rigid_candidate_batch,
    format_coverage_rank_report,
    geometry_unchanged,
    rank_candidate_coverage,
    select_and_materialize_catalog,
    select_coverage_catalog,
    snapshot_artifact_geometry,
)
from nutella_scraper.engines.compute.coverage_simulator import (
    REFERENCE_CANDIDATE_ID,
    CoverageResult,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    EnvelopeContactFrame,
    RigidScraperArtifact,
)

COMPUTE_SRC = Path("src/nutella_scraper/engines/compute/candidate_coverage.py")
SIM_SRC = Path("src/nutella_scraper/engines/compute/coverage_simulator.py")
PROX_SRC = Path("src/nutella_scraper/engines/compute/envelope_surface_proximity.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")


@pytest.fixture(scope="module")
def catalog_bundle():
    return load_generated_catalog(count=1000)


@dataclass(frozen=True)
class _FakeCandidate:
    candidate_id: str
    family: str
    valid: bool
    shape_fingerprint: str
    control_points_mm: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)


def _result(
    candidate_id: str,
    family: str,
    coverage: float,
    area: float = 10.0,
    faces: int = 1,
    valid: int = 24,
    fingerprint: str = "fp",
    elapsed: float = 1.0,
) -> CandidateCoverageResult:
    return CandidateCoverageResult(
        candidate_id=candidate_id,
        family=family,
        coverage_percent=coverage,
        covered_area_mm2=area,
        useful_area_mm2=A0_BASELINE_TARGET_AREA_MM2,
        touched_face_count=faces,
        valid_pose_count=valid,
        total_pose_count=24,
        elapsed_seconds=elapsed,
        shape_fingerprint=fingerprint,
    )


def _a0_coverage_result() -> CoverageResult:
    poses = tuple((float(deg), ScraperPose()) for deg in list(range(0, 45, 2)) + [45])
    return CoverageResult(
        candidate_id=REFERENCE_CANDIDATE_ID,
        coverage_percent=A0_BASELINE_COVERAGE_PERCENT,
        covered_area_mm2=A0_BASELINE_COVERED_AREA_MM2,
        target_area_mm2=A0_BASELINE_TARGET_AREA_MM2,
        angle_start_deg=0.0,
        angle_end_deg=45.0,
        angle_step_deg=2.0,
        evaluated_angles=tuple(angle for angle, _pose in poses),
        covered_face_ids=A0_BASELINE_FACE_IDS,
        best_pose_by_angle=poses,
        touched_face_ids_by_angle=tuple((angle, ()) for angle, _pose in poses),
        shape_fingerprint=A0_BASELINE_FINGERPRINT,
    )


def _dummy_frame() -> EnvelopeContactFrame:
    return EnvelopeContactFrame(
        origin_mm=np.zeros(3, dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        wall_point_mm=np.zeros(3, dtype=np.float64),
        inward_normal=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        surface_progress_deg=0.0,
    )


def _dummy_artifact(fingerprint: str) -> RigidScraperArtifact:
    mesh = trimesh.creation.box(extents=(2.5, 40.0, 2.5))
    frame = _dummy_frame()
    return RigidScraperArtifact(
        mesh=mesh,
        design_frame=frame,
        tip_edge_mm=np.zeros((3, 3), dtype=np.float64),
        wall_edge_mm=np.zeros((3, 3), dtype=np.float64),
        design_path=None,  # type: ignore[arg-type]
        shape_fingerprint=fingerprint,
    )


class _StubSimulator:
    def __init__(self, results: dict[str, CoverageResult]) -> None:
        self.results = results
        self.registered: dict[str, RigidScraperArtifact] = {}

    def register(self, candidate_id: str, artifact: RigidScraperArtifact) -> None:
        self.registered[str(candidate_id)] = artifact

    def evaluate_candidate(self, candidate_id: str) -> CoverageResult:
        return self.results[str(candidate_id)]


def test_compute_batch_does_not_import_visualization_or_touch_simulator() -> None:
    text = COMPUTE_SRC.read_text(encoding="utf-8")
    assert "engines.visualization" not in text
    assert "demo_viewer" not in text
    sim = SIM_SRC.read_text(encoding="utf-8")
    assert "candidate_coverage" not in sim
    prox = PROX_SRC.read_text(encoding="utf-8")
    assert "candidate_coverage" not in prox
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "candidateCoverageResult" not in html
    assert "Rank | Candidate | Family" not in html


def test_a0_baseline_gate_accepts_golden_numbers() -> None:
    coverage = _a0_coverage_result()
    item = candidate_result_from_coverage(coverage, family="A0", elapsed_seconds=1.0)
    assert item.touched_face_count == A0_BASELINE_TOUCHED_FACE_COUNT
    assert item.valid_pose_count == A0_BASELINE_VALID_POSE_COUNT
    assert a0_matches_baseline(coverage)
    assert a0_matches_baseline(item)
    assert_a0_matches_baseline(item)


def test_a0_baseline_gate_stops_when_coverage_drifts() -> None:
    drifted = _result(
        "A0",
        "A0",
        50.0,
        area=A0_BASELINE_COVERED_AREA_MM2,
        faces=A0_BASELINE_TOUCHED_FACE_COUNT,
        fingerprint=A0_BASELINE_FINGERPRINT,
    )
    assert a0_matches_baseline(drifted) is False
    with pytest.raises(A0BaselineRegressionError, match="baseline"):
        assert_a0_matches_baseline(drifted)


def test_select_coverage_catalog_keeps_a0_first_valid_and_distinct() -> None:
    catalog = [
        _FakeCandidate("A0", "A0", True, "fp-a0"),
        _FakeCandidate("S0001", "parallel", True, "fp-par"),
        _FakeCandidate("S0002", "parallel", True, "fp-par"),
        _FakeCandidate("S0003", "inclined", True, "fp-inc"),
        _FakeCandidate("S0004", "asymmetric", True, "fp-asy"),
        _FakeCandidate("S0005", "progressive", True, "fp-prg"),
        _FakeCandidate("S0006", "s_curve", True, "fp-s"),
        _FakeCandidate("S0007", "combined", True, "fp-cmb"),
        _FakeCandidate("S0008", "inclined", False, "fp-bad"),
        _FakeCandidate("S0009", "parallel", True, "fp-par2"),
        _FakeCandidate("S0010", "inclined", True, "fp-inc2"),
        _FakeCandidate("S0011", "asymmetric", True, "fp-asy2"),
        _FakeCandidate("S0012", "progressive", True, "fp-prg2"),
    ]
    selected = select_coverage_catalog(catalog, count=10)
    assert len(selected) == COVERAGE_BATCH_SIZE
    assert selected[0].candidate_id == "A0"
    assert selected[0].family == "A0"
    assert all(item.valid for item in selected)
    fingerprints = [item.shape_fingerprint for item in selected]
    assert len(set(fingerprints)) == 10
    families = {item.family for item in selected[1:]}
    assert "parallel" in families
    assert "inclined" in families
    assert "asymmetric" in families
    assert "progressive" in families
    assert "s_curve" in families


def test_generated_catalog_batch_is_valid_distinct_and_representative(catalog_bundle) -> None:
    _surface, _params, _reference, catalog = catalog_bundle
    selected = select_coverage_catalog(catalog, count=10)
    assert catalog[0].candidate_id == "A0"
    assert selected[0].candidate_id == "A0"
    assert selected[0].family == "A0"
    others = selected[1:]
    assert len(others) == 9
    assert all(item.valid for item in others)
    assert len({item.candidate_id for item in selected}) == 10
    assert len({item.shape_fingerprint for item in selected}) == 10
    families = {item.family for item in others}
    assert "parallel" in families
    assert "inclined" in families
    assert "asymmetric" in families
    assert "progressive" in families
    assert len(families) >= 4


def test_materialized_candidates_are_frozen_and_distinct(catalog_bundle) -> None:
    surface, params, reference, catalog = catalog_bundle
    selected, entries = select_and_materialize_catalog(
        catalog,
        surface=surface,
        parameters=params,
        reference=reference,
        count=10,
    )
    assert selected[0].candidate_id == "A0"
    assert entries[0][2] is reference
    assert entries[0][2].shape_fingerprint == A0_BASELINE_FINGERPRINT
    fingerprints = [artifact.shape_fingerprint for _cid, _fam, artifact in entries]
    assert len(set(fingerprints)) == 10
    snapshots = [snapshot_artifact_geometry(artifact) for _cid, _fam, artifact in entries]
    for (_cid, _fam, artifact), snap in zip(entries, snapshots, strict=True):
        assert geometry_unchanged(artifact, snap)
        control = np.asarray(selected[0].control_points_mm, dtype=np.float64)
        assert control.shape[1] == 3


def test_rank_candidate_coverage_uses_documented_tie_breaks() -> None:
    a = _result("S0002", "parallel", 50.0, area=100.0, valid=20)
    b = _result("S0001", "inclined", 50.0, area=100.0, valid=20)
    c = _result("S0003", "asymmetric", 50.0, area=120.0, valid=10)
    d = _result("A0", "A0", 63.33, area=1988.0, valid=24)
    e = _result("S0004", "progressive", 50.0, area=100.0, valid=24)
    ranked = rank_candidate_coverage([a, b, c, d, e])
    assert [item.candidate_id for item in ranked] == [
        "A0",
        "S0003",
        "S0004",
        "S0001",
        "S0002",
    ]
    again = rank_candidate_coverage([e, d, c, b, a])
    assert [item.candidate_id for item in again] == [item.candidate_id for item in ranked]


def test_evaluate_batch_keeps_fingerprints_and_stops_if_a0_drifts() -> None:
    a0 = _dummy_artifact(A0_BASELINE_FINGERPRINT)
    other = _dummy_artifact("curve-other")
    a0_cov = _a0_coverage_result()
    other_cov = replace(
        a0_cov,
        candidate_id="S0001",
        coverage_percent=40.0,
        covered_area_mm2=1000.0,
        covered_face_ids=frozenset({1}),
        shape_fingerprint="curve-other",
        best_pose_by_angle=a0_cov.best_pose_by_angle[:20] + tuple(
            (angle, None) for angle, _pose in a0_cov.best_pose_by_angle[20:]
        ),
    )
    simulator = _StubSimulator({"A0": a0_cov, "S0001": other_cov})
    verts_a0 = np.asarray(a0.mesh.vertices, dtype=np.float64).copy()
    verts_other = np.asarray(other.mesh.vertices, dtype=np.float64).copy()
    batch = evaluate_rigid_candidate_batch(
        simulator,  # type: ignore[arg-type]
        (("A0", "A0", a0), ("S0001", "parallel", other)),
    )
    assert batch.fingerprints_unchanged is True
    assert batch.ranked[0].candidate_id == "A0"
    assert np.allclose(a0.mesh.vertices, verts_a0, atol=1e-9)
    assert np.allclose(other.mesh.vertices, verts_other, atol=1e-9)
    drifted = replace(a0_cov, coverage_percent=1.0)
    with pytest.raises(A0BaselineRegressionError):
        evaluate_rigid_candidate_batch(
            _StubSimulator({"A0": drifted}),  # type: ignore[arg-type]
            (("A0", "A0", a0),),
        )


def test_deformed_mesh_is_rejected_after_evaluation() -> None:
    artifact = _dummy_artifact(A0_BASELINE_FINGERPRINT)
    coverage = _a0_coverage_result()

    class MutatingSimulator(_StubSimulator):
        def evaluate_candidate(self, candidate_id: str) -> CoverageResult:
            verts = np.asarray(artifact.mesh.vertices, dtype=np.float64).copy()
            verts[0] += 10.0
            artifact.mesh.vertices = verts
            return super().evaluate_candidate(candidate_id)

    with pytest.raises(ValueError, match="deformed"):
        evaluate_rigid_candidate_batch(
            MutatingSimulator({"A0": coverage}),  # type: ignore[arg-type]
            (("A0", "A0", artifact),),
        )


def test_rank_report_lists_ten_rows_in_rank_order() -> None:
    evaluated = tuple(
        _result(f"S{i:04d}", "parallel", 10.0 + i, fingerprint=f"fp{i}")
        for i in range(10)
    )
    ranked = rank_candidate_coverage(evaluated)
    report = format_coverage_rank_report(
        RankedCoverageBatch(
            evaluated=evaluated,
            ranked=ranked,
            total_elapsed_seconds=1.0,
            fingerprints_unchanged=True,
        )
    )
    assert report.splitlines()[0].startswith("Rank | Candidate | Family")
    assert "S0009" in report.splitlines()[2]
    assert "S0000" in report.splitlines()[-1]
