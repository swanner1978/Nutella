"""TRAJECTORY_MODEL_DIAGNOSTIC — graph cut / mapping. No physics campaign."""

from __future__ import annotations

from pathlib import Path

from tests.unit.engines.compute.test_trajectory_search import _cell, _rect_grid

from nutella_scraper.engines.compute.trajectory_contact_cache import (
    contact_cache_from_masks,
)
from nutella_scraper.engines.compute.trajectory_model_diagnostic import (
    CELL_EQUALS_POSE,
    DIAGNOSTIC_LABEL,
    MAPPING_STEPS,
    MAX_DOWNWARD_STEP_MEANS,
    analyze_shape_graph,
    height_monotone_pose_chain,
    mask_indices,
    pose_contact_spans,
    union_indices,
)
from nutella_scraper.engines.compute.trajectory_optimizer import beam_search_trajectories
from nutella_scraper.engines.compute.trajectory_search import (
    MAX_DOWNWARD_STEP,
    MAX_LATERAL_STEP,
    trajectory_grid_from_cells,
)

SRC = Path("src/nutella_scraper/engines/compute/trajectory_model_diagnostic.py")
CACHE_SRC = Path("src/nutella_scraper/engines/compute/trajectory_contact_cache.py")
OPT_SRC = Path("src/nutella_scraper/engines/compute/trajectory_optimizer.py")


def test_label_and_mapping_are_documented() -> None:
    assert DIAGNOSTIC_LABEL == "TRAJECTORY_MODEL_DIAGNOSTIC"
    assert CELL_EQUALS_POSE is True
    assert "yaw = azimuth" in " ".join(MAPPING_STEPS)
    assert "rangée d'index" in MAX_DOWNWARD_STEP_MEANS
    assert MAX_DOWNWARD_STEP == 1
    assert MAX_LATERAL_STEP == 2
    for src in (SRC, CACHE_SRC, OPT_SRC):
        text = src.read_text(encoding="utf-8")
        assert "evaluate_candidate(" not in text
        assert "engines.visualization" not in text


def test_last_row_empty_cuts_the_graph() -> None:
    grid = _rect_grid(4, 3)
    masks = {}
    admissible = {}
    for row in range(4):
        for col in range(3):
            masks[(row, col)] = 1 << (row * 3 + col)
            admissible[(row, col)] = row < 3
    cache = contact_cache_from_masks(
        grid, masks, n_points=12, admissible=admissible
    )
    ranked = beam_search_trajectories(grid, cache, beam_width=8, top_k=3)
    report = analyze_shape_graph("A0", grid, cache, beam_complete_paths=len(ranked))
    assert report.first_row_without_admissible == 3
    assert report.last_row_admissible == 0
    assert report.max_reachable_row == 2
    assert report.n_starts == 3
    assert report.extinction.kind == "downward_cut"
    assert report.extinction.row == 3
    assert ranked == ()


def test_opening_blocked_is_a_start_cut() -> None:
    grid = _rect_grid(3, 2)
    masks = {(row, col): 1 << (row * 2 + col) for row in range(3) for col in range(2)}
    admissible = {(row, col): row > 0 for row in range(3) for col in range(2)}
    cache = contact_cache_from_masks(
        grid, masks, n_points=6, admissible=admissible
    )
    report = analyze_shape_graph("straight", grid, cache, beam_complete_paths=0)
    assert report.n_starts == 0
    assert report.extinction.kind == "no_opening_start"
    assert report.first_row_without_admissible == 0


def test_one_pose_mask_covers_several_rows() -> None:
    cells = tuple(_cell(row, 0, n_rows=4, n_cols=1) for row in range(4))
    grid = trajectory_grid_from_cells(cells)
    wide = (1 << 0) | (1 << 1) | (1 << 2)
    masks = {(0, 0): wide, (1, 0): 1 << 1, (2, 0): 1 << 2, (3, 0): 1 << 3}
    cache = contact_cache_from_masks(grid, masks, n_points=4)
    spans = pose_contact_spans(grid, cache)
    top = next(item for item in spans if item.row == 0)
    assert top.covered_count == 3
    assert top.n_rows_covered == 3
    assert 0 in top.covered_rows and 2 in top.covered_rows
    assert mask_indices(wide, 4) == (0, 1, 2)


def test_height_chain_covers_without_visiting_every_row() -> None:
    cells = tuple(_cell(row, 0, n_rows=5, n_cols=1) for row in range(5))
    grid = trajectory_grid_from_cells(cells)
    admissible = {(row, 0): row in {0, 2, 4} for row in range(5)}
    masks = {
        (0, 0): 1 << 0,
        (1, 0): 0,
        (2, 0): 1 << 2,
        (3, 0): 0,
        (4, 0): 1 << 4,
    }
    cache = contact_cache_from_masks(
        grid, masks, n_points=5, admissible=admissible
    )
    report = analyze_shape_graph("A0", grid, cache, beam_complete_paths=0)
    assert report.n_successive_row_transitions == 0
    assert report.extinction.kind == "downward_cut"
    spans = pose_contact_spans(grid, cache)
    chain = height_monotone_pose_chain(spans, max_poses=8, min_drop_mm=0.5)
    covered = union_indices(chain)
    assert len(chain) >= 2
    assert len(covered) >= 2
    ranked = beam_search_trajectories(grid, cache, beam_width=8, top_k=1)
    assert ranked == ()
