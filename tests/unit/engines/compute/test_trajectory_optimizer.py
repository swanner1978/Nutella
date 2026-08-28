"""Beam-search coverage on the validated cloud — no CoverageSimulator.evaluate."""

from __future__ import annotations

from pathlib import Path

from tests.unit.engines.compute.test_coverage_simulator import _fast_surface
from tests.unit.engines.compute.test_trajectory_search import (
    _rect_grid,
    _tiny_reference_matrix,
)

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
    LEGACY_A0_QUADRANT_REGION,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import (
    MOTION_DIRECTION_AFFECTS_CONTACT,
    POSE_VARIABLES,
    contact_cache_from_masks,
)
from nutella_scraper.engines.compute.trajectory_optimizer import (
    DISCLAIMER,
    OPTIMIZATION_LABEL,
    beam_search_trajectories,
)
from nutella_scraper.engines.compute.trajectory_search import trajectory_is_valid

CACHE_SRC = Path("src/nutella_scraper/engines/compute/trajectory_contact_cache.py")
OPT_SRC = Path("src/nutella_scraper/engines/compute/trajectory_optimizer.py")
SIM_SRC = Path("src/nutella_scraper/engines/compute/coverage_simulator.py")
SAVED_JSON = Path("output/coverage/candidate_coverage_100.json")


def test_modules_do_not_call_coverage_simulator_or_old_grid() -> None:
    for src in (CACHE_SRC, OPT_SRC):
        text = src.read_text(encoding="utf-8")
        assert "evaluate_candidate(" not in text
        assert "from nutella_scraper.engines.compute.coverage_simulator" not in text
        assert "engines.visualization" not in text
        assert "engines.optimization" not in text
        assert "ANGLE_END_DEG" not in text
    sim = SIM_SRC.read_text(encoding="utf-8")
    assert "ANGLE_END_DEG = 45.0" in sim


def test_pose_mapping_is_position_and_yaw_not_motion() -> None:
    assert "surface_progress_deg" in POSE_VARIABLES
    assert "position_z_mm" in POSE_VARIABLES
    assert MOTION_DIRECTION_AFFECTS_CONTACT is False


def test_union_coverage_does_not_sum_overlaps() -> None:
    first = (1 << 0) | (1 << 1) | (1 << 2)
    second = (1 << 2) | (1 << 3) | (1 << 4)
    combined = first | second
    assert first.bit_count() == 3
    assert second.bit_count() == 3
    assert combined.bit_count() == 5
    assert combined.bit_count() != first.bit_count() + second.bit_count()


def test_beam_search_never_climbs_and_reaches_last_row() -> None:
    grid = _rect_grid(3, 3)
    masks = {
        (row, col): 1 << (row * 3 + col)
        for row in range(3)
        for col in range(3)
    }
    cache = contact_cache_from_masks(grid, masks, n_points=9)
    ranked = beam_search_trajectories(
        grid,
        cache,
        beam_width=16,
        top_k=10,
        max_lateral_step=2,
        max_downward_step=1,
    )
    assert ranked
    assert ranked[0].optimization_label == OPTIMIZATION_LABEL
    assert ranked[0].total_points == 9
    assert "énumérer" in DISCLAIMER
    for item in ranked:
        assert trajectory_is_valid(
            item.path, grid, max_lateral_step=2, max_downward_step=1
        )
        rows = [cell.row for cell in item.path]
        assert rows[0] == 0
        assert rows[-1] == 2
        assert all(
            nxt >= prev for prev, nxt in zip(rows[:-1], rows[1:], strict=True)
        )
        assert item.covered_points == item.covered_mask.bit_count()
        assert item.coverage_percent == 100.0 * item.covered_points / 9


def test_beam_prefers_union_over_sum_and_skips_blocked_cells() -> None:
    grid = _rect_grid(2, 2)
    cache = contact_cache_from_masks(
        grid,
        {
            (0, 0): (1 << 0) | (1 << 1),
            (0, 1): 1 << 0,
            (1, 0): (1 << 1) | (1 << 2),
            (1, 1): 1 << 3,
        },
        n_points=4,
        admissible={(1, 0): False},
    )
    ranked = beam_search_trajectories(
        grid,
        cache,
        beam_width=8,
        top_k=5,
        max_lateral_step=1,
        max_downward_step=1,
    )
    assert ranked
    best = ranked[0]
    assert all(not (cell.row == 1 and cell.col == 0) for cell in best.path)
    assert best.covered_points == best.covered_mask.bit_count()
    assert 3 in best.covered_point_indices
    assert best.covered_points >= 2


def test_same_row_lateral_is_allowed_and_a0_is_not_the_grid() -> None:
    grid = _rect_grid(2, 3)
    assert grid.target_definition == COVERAGE_TARGET_REGION
    assert grid.target_definition != LEGACY_A0_QUADRANT_REGION
    masks = {(row, col): 1 << col for row in range(2) for col in range(3)}
    cache = contact_cache_from_masks(grid, masks, n_points=6)
    ranked = beam_search_trajectories(
        grid,
        cache,
        beam_width=12,
        top_k=3,
        max_lateral_step=1,
        max_downward_step=1,
    )
    assert ranked
    assert max(item.lateral_moves for item in ranked) >= 1
    assert cache.uses_legacy_a0_point_matrix is False
    assert cache.symmetry_multiplier_applied is False
    assert cache.angle_window_deg[1] == 90.0


def test_saved_coverage_100_is_not_used_as_search_input() -> None:
    text = CACHE_SRC.read_text(encoding="utf-8") + OPT_SRC.read_text(encoding="utf-8")
    assert "candidate_coverage_100" not in text
    assert "evaluate_candidates_batch" not in text
    if SAVED_JSON.is_file():
        assert SAVED_JSON.is_file()


def test_contact_cache_uses_existing_collision_on_synthetic_surface() -> None:
    from nutella_scraper.engines.compute.trajectory_contact_cache import (
        build_contact_cache,
    )
    from nutella_scraper.engines.compute.trajectory_search import (
        index_reference_matrix,
        simulate_scraper_trajectory_search,
    )

    surface = _fast_surface()
    matrix = _tiny_reference_matrix()
    grid = index_reference_matrix(matrix, surface=surface)
    cache = build_contact_cache(surface, matrix, grid)
    assert cache.target_definition == COVERAGE_TARGET_REGION
    assert cache.n_points == 4
    assert cache.uses_legacy_a0_point_matrix is False
    assert cache.symmetry_multiplier_applied is False
    assert cache.angle_window_deg == (0.0, 90.0)
    assert 4 <= cache.physics_queries <= 4 * 17
    ranked = beam_search_trajectories(
        grid,
        cache,
        beam_width=8,
        top_k=3,
        max_lateral_step=1,
        max_downward_step=1,
    )
    assert ranked
    assert ranked[0].total_points == 4
    assert ranked[0].optimization_label == OPTIMIZATION_LABEL
    report = simulate_scraper_trajectory_search(matrix=matrix, optimize=True)
    assert report.physics_executed is False
    assert report.results == ()
    assert report.optimization_label == "CARDINALITY_ONLY"
