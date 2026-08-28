"""Heuristic shape search on interior_matrix_a0_0_90 — no CoverageSimulator.evaluate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.unit.engines.compute.test_coverage_simulator import _fast_surface
from tests.unit.engines.compute.test_trajectory_search import _tiny_reference_matrix

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
    LEGACY_A0_QUADRANT_REGION,
)
from nutella_scraper.engines.compute.pose_contact_cache import (
    PoseContactEntry,
    pose_cache_from_entries,
)
from nutella_scraper.engines.compute.pose_space import PoseSamplingConfig
from nutella_scraper.engines.compute.shape_families import (
    BLADE_THICKNESS_MM,
    BLADE_WIDTH_MM,
    PRELIMINARY_FAMILY_IDS,
)
from nutella_scraper.engines.compute.shape_result import (
    DISCLAIMER,
    OPTIMIZATION_LABEL,
    ShapeCandidate,
)
from nutella_scraper.engines.compute.shape_search import (
    ShapeSearchConfig,
    a0_gain_summary,
    comparison_rows,
    descending_contact_trajectory,
    length_pretest_config,
    length_sweep_config,
    preliminary_config,
    rank_shape_candidates,
    search_scraper_shapes,
    validation_config,
)
from nutella_scraper.engines.compute.trajectory_search import (
    index_reference_matrix,
)

SEARCH_SRC = Path("src/nutella_scraper/engines/compute/shape_search.py")
FAM_SRC = Path("src/nutella_scraper/engines/compute/shape_families.py")
MAT_SRC = Path("src/nutella_scraper/engines/compute/shape_materialize.py")
SIM_SRC = Path("src/nutella_scraper/engines/compute/coverage_simulator.py")


def _fake_physics(surface, matrix, specs, *, artifact=None, parameters=None):
    import math

    n_points = int(matrix.point_count)
    entries = []
    for spec in specs:
        yaw = float(spec.azimuth_deg)
        ox = 50.0 * math.cos(math.radians(yaw))
        oz = -50.0 * math.sin(math.radians(yaw))
        mask = 1 << (int(spec.pose_id) % max(n_points, 1))
        entries.append(
            PoseContactEntry(
                pose_id=int(spec.pose_id),
                y_mm=float(spec.y_mm),
                azimuth_deg=yaw,
                origin_mm=(ox, float(spec.y_mm), oz),
                yaw_deg=yaw,
                length_axis=(0.0, -1.0, 0.0),
                admissible=True,
                neighborhood_used=False,
                covered_mask=mask,
                covered_count=mask.bit_count(),
                physics_queries=1,
            )
        )
    return pose_cache_from_entries(tuple(entries), n_points=n_points)


def _stub(
    *,
    candidate_id: str,
    family_id: str,
    covered: int,
    n_params: int = 2,
    mean_err: float = 1.0,
    path_mm: float = 10.0,
    turns: int = 1,
) -> ShapeCandidate:
    return ShapeCandidate(
        candidate_id=candidate_id,
        family_id=family_id,
        parameters=(0.0,) * n_params,
        n_parameters=n_params,
        coverage_percent=100.0 * covered / 4.0,
        covered_points=covered,
        total_points=4,
        covered_point_indices=tuple(range(covered)),
        untouched_point_indices=tuple(range(covered, 4)),
        mean_geometric_error_mm=mean_err,
        max_geometric_error_mm=mean_err * 2,
        scraper_length_mm=40.0,
        min_curvature_radius_mm=10.0,
        trajectory_steps=3,
        trajectory_length_mm=path_mm,
        lateral_changes=1,
        direction_changes=turns,
        geometric_valid=True,
        physical_valid=True,
        geometric_reasons=(),
        profile_points_mm=np.zeros((4, 3), dtype=np.float64),
    )


def test_modules_do_not_call_coverage_simulator_or_a0_grid() -> None:
    for src in (SEARCH_SRC, FAM_SRC, MAT_SRC):
        text = src.read_text(encoding="utf-8")
        assert "evaluate_candidate(" not in text
        assert "from nutella_scraper.engines.compute.coverage_simulator" not in text
        assert "engines.visualization" not in text
        assert "engines.optimization" not in text
    search = SEARCH_SRC.read_text(encoding="utf-8")
    assert "interior_matrix_a0_0_90" in search
    assert "HEURISTIC" in search
    assert "POSE_GRAPH" in search
    assert "MAX_DOWNWARD_STEP" not in search
    assert "forme optimale" not in search.lower()
    assert BLADE_THICKNESS_MM == 2.0
    assert BLADE_WIDTH_MM == 2.0


def test_rank_uses_coverage_first_without_blending() -> None:
    worse_cov = _stub(
        candidate_id="S1",
        family_id="bezier_4",
        covered=3,
        n_params=4,
        mean_err=0.01,
        path_mm=1.0,
    )
    better_cov = _stub(
        candidate_id="S2",
        family_id="straight",
        covered=4,
        n_params=2,
        mean_err=9.0,
        path_mm=99.0,
        turns=9,
    )
    ranked = rank_shape_candidates((worse_cov, better_cov))
    assert ranked[0].candidate_id == "S2"
    blob = str(comparison_rows(ranked))
    assert "weighted" not in blob.lower()


def test_union_coverage_is_not_a_sum() -> None:
    first = (1 << 0) | (1 << 1)
    second = (1 << 1) | (1 << 2)
    combined = first | second
    assert combined.bit_count() == 3
    assert combined.bit_count() != first.bit_count() + second.bit_count()


def test_validation_search_uses_matrix_not_legacy_a0() -> None:
    surface = _fast_surface()
    matrix = _tiny_reference_matrix()
    report = search_scraper_shapes(
        surface,
        matrix=matrix,
        config=validation_config(),
        physics_builder=_fake_physics,
    )
    assert report.grid.target_definition == COVERAGE_TARGET_REGION
    assert report.grid.target_definition != LEGACY_A0_QUADRANT_REGION
    assert report.grid.uses_legacy_a0_point_matrix is False
    assert report.optimization_label == OPTIMIZATION_LABEL
    assert "optimum" in DISCLAIMER.lower()
    assert report.a0_reference is not None
    assert report.a0_reference.candidate_id == "A0"
    assert report.a0_reference.total_points == 4
    families = {item.family_id for item in report.candidates}
    assert "straight" in families
    assert "bezier_4" in families
    assert report.stats.cache_misses >= 1
    assert report.a0_reference.trajectory_model == "POSE_GRAPH"
    summary = a0_gain_summary(report.a0_reference, report.candidates[0])
    assert "gain_points" in summary
    assert summary["optimization_label"] == "HEURISTIC"


def test_tiny_real_physics_does_not_teleport() -> None:
    surface = _fast_surface()
    matrix = _tiny_reference_matrix()
    report = search_scraper_shapes(
        surface,
        matrix=matrix,
        config=ShapeSearchConfig(
            max_shape_evaluations=1,
            family_ids=("straight",),
            beam_width=8,
            top_k_per_family=1,
            sample_count=12,
            run_a0_reference=True,
            pose_sampling=PoseSamplingConfig(height_step_mm=40.0, azimuth_step_deg=45.0),
        ),
    )
    grid = index_reference_matrix(matrix, surface=surface)
    assert grid.target_definition == COVERAGE_TARGET_REGION
    for item in (report.a0_reference, *report.candidates):
        if item is None or not item.physical_valid:
            continue
        assert item.covered_points == len(item.covered_point_indices)
        assert item.covered_points + len(item.untouched_point_indices) == item.total_points
        assert item.optimization_label == "HEURISTIC"
        assert item.trajectory_model == "POSE_GRAPH"
        ys = [y for y, _az in item.trajectory_poses]
        assert all(nxt <= prev + 1e-6 for prev, nxt in zip(ys[:-1], ys[1:], strict=True))
        if item.family_id != "A0":
            assert item.thickness_mm == 2.0
    assert "evaluate_candidate" in SIM_SRC.read_text(encoding="utf-8")


def test_preliminary_config_is_five_simple_families() -> None:
    config = preliminary_config()
    assert config.family_ids == PRELIMINARY_FAMILY_IDS
    assert config.max_shape_evaluations == 1
    assert config.stage_label == "PRELIMINARY"
    assert config.scraper_lengths_mm == (40.0,)
    assert "fourier_5" not in config.family_ids
    assert "bezier_10" not in config.family_ids
    assert "poly_2" not in config.family_ids


def test_length_pretest_is_not_a_35_form_campaign() -> None:
    pretest = length_pretest_config()
    assert pretest.stage_label == "LENGTH_PRETEST"
    assert pretest.shape_specs is not None
    assert len(pretest.iter_shape_specs()) == 6
    sweep = length_sweep_config()
    assert len(sweep.iter_shape_specs()) == 35
    assert sweep.stage_label == "LENGTH_SWEEP"


def test_rank_uses_shorter_path_when_coverage_ties() -> None:
    long_path = _stub(
        candidate_id="long",
        family_id="straight",
        covered=4,
        path_mm=90.0,
    )
    short_path = _stub(
        candidate_id="short",
        family_id="concave",
        covered=4,
        n_params=3,
        path_mm=10.0,
    )
    ranked = rank_shape_candidates((long_path, short_path))
    assert ranked[0].candidate_id == "short"


def test_descending_contact_requires_motion_and_union() -> None:
    from dataclasses import replace

    empty = _stub(candidate_id="z", family_id="straight", covered=0)
    assert descending_contact_trajectory(empty) is False
    moving = replace(
        _stub(candidate_id="m", family_id="straight", covered=2),
        trajectory_poses=((80.0, 0.0), (40.0, 5.0)),
    )
    assert descending_contact_trajectory(moving) is True
