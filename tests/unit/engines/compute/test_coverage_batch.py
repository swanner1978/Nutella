"""Batch coverage invariants — same physics, jar-level work done once."""

from __future__ import annotations

from pathlib import Path

from tests.unit.engines.compute.test_coverage_simulator import (
    _a0_parameters,
    _fast_surface,
)

from nutella_scraper.engines.compute import coverage_simulator as cov_mod
from nutella_scraper.engines.compute.coverage_simulator import (
    CoverageSimulator,
    rank_coverage_results,
)
from nutella_scraper.engines.compute.scraper_envelope_collision import (
    EnvelopeCollisionReport,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    build_rigid_scraper_artifact,
)

COMPUTE_SRC = Path("src/nutella_scraper/engines/compute/coverage_simulator.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")
PROX_SRC = Path("src/nutella_scraper/engines/compute/envelope_surface_proximity.py")


def _admissible_report() -> EnvelopeCollisionReport:
    return EnvelopeCollisionReport(
        has_collision=False,
        admissible=True,
        min_signed_interior_mm=1.0,
        max_outward_mm=0.0,
        min_unsigned_distance_mm=1.0,
        clearance_mm=0.0,
        vertex_hit=False,
        edge_hit=False,
        face_hit=False,
        clearance_ok=True,
        contact_face_ids=frozenset(),
    )


def test_batch_module_does_not_touch_viewer_or_proximity() -> None:
    text = COMPUTE_SRC.read_text(encoding="utf-8")
    assert "evaluate_candidates_batch" in text
    assert "engines.visualization" not in text
    assert "rotationCache" not in text
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "evaluate_candidates_batch" not in html
    prox = PROX_SRC.read_text(encoding="utf-8")
    assert "evaluate_candidates_batch" not in prox


def test_batch_prepares_envelope_frames_once_per_nonzero_angle(monkeypatch) -> None:
    surface = _fast_surface()
    params = _a0_parameters(surface)
    artifact = build_rigid_scraper_artifact(surface, params)
    calls = {"n": 0}
    orig = cov_mod.envelope_contact_frame

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(cov_mod, "envelope_contact_frame", counted)
    monkeypatch.setattr(
        cov_mod, "evaluate_envelope_collision", lambda *args, **kwargs: _admissible_report()
    )

    sequential = CoverageSimulator(surface, parameters=params)
    sequential.register("A0", artifact)
    sequential.register("B", artifact)
    sequential.evaluate_candidate("A0")
    sequential.evaluate_candidate("B")
    sequential_calls = calls["n"]
    assert sequential._batch_invariants is None

    calls["n"] = 0
    batched = CoverageSimulator(surface, parameters=params)
    batched.register("A0", artifact)
    batched.register("B", artifact)
    ranked = batched.evaluate_candidates_batch(["A0", "B"])
    batch_calls = calls["n"]

    nonzero_angles = sum(1 for angle in sequential._angles if abs(float(angle)) > 1e-9)
    assert sequential_calls == nonzero_angles * 2
    assert batch_calls == nonzero_angles
    assert batched._batch_invariants is not None
    assert [item.candidate_id for item in ranked] == ["A0", "B"]


def test_rank_coverage_results_matches_documented_tie_breaks() -> None:
    from nutella_scraper.domain.models.scraper import ScraperPose
    from nutella_scraper.engines.compute.coverage_simulator import CoverageResult

    def make(candidate_id: str, coverage: float, area: float, valid: int) -> CoverageResult:
        poses = tuple(
            (float(i), ScraperPose() if i < valid else None) for i in range(24)
        )
        return CoverageResult(
            candidate_id=candidate_id,
            coverage_percent=coverage,
            covered_area_mm2=area,
            target_area_mm2=100.0,
            angle_start_deg=0.0,
            angle_end_deg=45.0,
            angle_step_deg=2.0,
            evaluated_angles=tuple(float(i) for i in range(24)),
            covered_face_ids=frozenset(),
            best_pose_by_angle=poses,
            touched_face_ids_by_angle=tuple((float(i), ()) for i in range(24)),
            shape_fingerprint=candidate_id,
        )

    ranked = rank_coverage_results(
        [
            make("S0002", 50.0, 100.0, 20),
            make("S0001", 50.0, 100.0, 20),
            make("S0003", 50.0, 120.0, 10),
            make("A0", 63.33, 1988.0, 24),
            make("S0004", 50.0, 100.0, 24),
        ]
    )
    assert [item.candidate_id for item in ranked] == [
        "A0",
        "S0003",
        "S0004",
        "S0001",
        "S0002",
    ]
