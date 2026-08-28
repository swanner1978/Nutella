"""Trajectory search on the validated interior cloud — cardinality only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tests.unit.engines.compute.test_coverage_simulator import _fast_surface

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
    LEGACY_A0_QUADRANT_REGION,
    MATRIX_SPACING_MM,
    build_coverage_reference_matrix,
)
from nutella_scraper.engines.compute.trajectory_search import (
    MAX_DOWNWARD_STEP,
    MAX_LATERAL_STEP,
    PHYSICS_ENUMERATION_LIMIT,
    TrajectoryCell,
    count_valid_trajectories,
    index_reference_matrix,
    simulate_scraper_trajectory_search,
    trajectory_grid_from_cells,
    trajectory_is_valid,
    transition_allowed,
    write_trajectory_search_results,
)

SRC = Path("src/nutella_scraper/engines/compute/trajectory_search.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")
SAVED_JSON = Path("output/coverage/candidate_coverage_100.json")


def _cell(
    row: int,
    col: int,
    *,
    n_rows: int,
    n_cols: int = 8,
) -> TrajectoryCell:
    last = n_rows - 1
    return TrajectoryCell(
        index=row * n_cols + col,
        row=row,
        col=col,
        x_mm=float(col),
        y_mm=float(-row),
        z_mm=0.0,
        azimuth_deg=float(col),
        is_top=row == 0,
        is_bottom=row == last,
    )


def _rect_grid(n_rows: int, n_cols: int):
    cells = tuple(
        _cell(row, col, n_rows=n_rows, n_cols=n_cols)
        for row in range(n_rows)
        for col in range(n_cols)
    )
    return trajectory_grid_from_cells(cells)


def _brute_count(
    grid,
    *,
    max_lateral_step: int = MAX_LATERAL_STEP,
    max_downward_step: int = MAX_DOWNWARD_STEP,
) -> int:
    occ = {
        (cell.row, cell.col): cell
        for cell in grid.cells
    }
    last_row = grid.n_rows - 1

    def rec(row: int, col: int, seen: frozenset[int]) -> int:
        total = 1 if row == last_row else 0
        for ncol in range(grid.n_cols):
            if ncol == col or ncol in seen:
                continue
            dst = occ.get((row, ncol))
            if dst is None:
                continue
            src = occ[(row, col)]
            if not transition_allowed(
                src,
                dst,
                max_lateral_step=max_lateral_step,
                max_downward_step=max_downward_step,
            ):
                continue
            total += rec(row, ncol, seen | {ncol})
        for drow in range(1, int(max_downward_step) + 1):
            nxt_row = row + drow
            if nxt_row >= grid.n_rows:
                break
            src = occ[(row, col)]
            for ncol in range(grid.n_cols):
                dst = occ.get((nxt_row, ncol))
                if dst is None:
                    continue
                if not transition_allowed(
                    src,
                    dst,
                    max_lateral_step=max_lateral_step,
                    max_downward_step=max_downward_step,
                ):
                    continue
                total += rec(nxt_row, ncol, frozenset({ncol}))
        return total

    return sum(
        rec(cell.row, cell.col, frozenset({cell.col}))
        for cell in grid.cells
        if cell.is_top
    )


def test_module_is_search_layer_only() -> None:
    text = SRC.read_text(encoding="utf-8")
    assert "evaluate_candidate(" not in text
    assert "from nutella_scraper.engines.compute.coverage_simulator" not in text
    assert "import CoverageSimulator" not in text
    assert "engines.visualization" not in text
    assert "engines.optimization" not in text
    assert "bind_envelope_proximity" not in text
    assert "scraper_envelope_collision" not in text
    assert "def count_valid_trajectories(" in text
    assert "def simulate_scraper_trajectory_search(" in text
    assert "MAX_LATERAL_STEP = 2" in text


def test_transition_never_climbs_and_respects_lateral_step() -> None:
    grid = _rect_grid(3, 4)
    top = grid.cell_at(0, 1)
    mid = grid.cell_at(1, 1)
    low = grid.cell_at(2, 1)
    same = grid.cell_at(1, 2)
    far = grid.cell_at(1, 3)
    assert top is not None and mid is not None and low is not None
    assert same is not None and far is not None
    assert transition_allowed(mid, top) is False
    assert transition_allowed(low, mid) is False
    assert transition_allowed(mid, same, max_lateral_step=1) is True
    assert transition_allowed(mid, far, max_lateral_step=1) is False
    assert transition_allowed(mid, far, max_lateral_step=2) is True
    assert transition_allowed(top, low, max_downward_step=1) is False
    assert transition_allowed(top, low, max_downward_step=2) is True
    assert transition_allowed(mid, mid) is False


def test_valid_trajectory_may_stay_on_row_but_must_reach_last_row() -> None:
    grid = _rect_grid(3, 4)
    stay = (
        grid.cell_at(0, 1),
        grid.cell_at(0, 2),
        grid.cell_at(1, 2),
        grid.cell_at(1, 1),
        grid.cell_at(2, 1),
    )
    assert all(cell is not None for cell in stay)
    assert trajectory_is_valid(stay, grid, max_lateral_step=1, max_downward_step=1)
    reverse_same_row = (
        grid.cell_at(0, 2),
        grid.cell_at(0, 3),
        grid.cell_at(0, 1),
        grid.cell_at(1, 1),
        grid.cell_at(2, 1),
    )
    assert trajectory_is_valid(
        reverse_same_row,
        grid,
        max_lateral_step=2,
        max_downward_step=1,
    )
    revisit = (
        grid.cell_at(0, 1),
        grid.cell_at(0, 2),
        grid.cell_at(0, 1),
        grid.cell_at(1, 1),
        grid.cell_at(2, 1),
    )
    assert trajectory_is_valid(revisit, grid) is False
    climb = (
        grid.cell_at(0, 1),
        grid.cell_at(1, 1),
        grid.cell_at(0, 1),
        grid.cell_at(1, 1),
        grid.cell_at(2, 1),
    )
    assert trajectory_is_valid(climb, grid) is False
    incomplete = (grid.cell_at(0, 1), grid.cell_at(1, 1))
    assert trajectory_is_valid(incomplete, grid) is False


def test_cardinality_matches_brute_force_on_tiny_grids() -> None:
    for n_rows, n_cols, lateral, down in (
        (2, 2, 1, 1),
        (3, 2, 1, 1),
        (3, 3, 1, 1),
        (3, 3, 2, 1),
    ):
        grid = _rect_grid(n_rows, n_cols)
        exact = count_valid_trajectories(
            grid,
            max_lateral_step=lateral,
            max_downward_step=down,
        )
        brute = _brute_count(
            grid,
            max_lateral_step=lateral,
            max_downward_step=down,
        )
        assert exact == brute
        assert exact == count_valid_trajectories(
            grid,
            max_lateral_step=lateral,
            max_downward_step=down,
        )


def test_ragged_grid_must_finish_on_global_last_row() -> None:
    cells = (
        _cell(0, 0, n_rows=3),
        _cell(1, 0, n_rows=3),
        _cell(2, 0, n_rows=3),
        TrajectoryCell(
            index=10,
            row=0,
            col=1,
            x_mm=1.0,
            y_mm=0.0,
            z_mm=0.0,
            azimuth_deg=1.0,
            is_top=True,
            is_bottom=False,
        ),
        TrajectoryCell(
            index=11,
            row=1,
            col=1,
            x_mm=1.0,
            y_mm=-1.0,
            z_mm=0.0,
            azimuth_deg=1.0,
            is_top=False,
            is_bottom=True,
        ),
    )
    grid = trajectory_grid_from_cells(cells)
    exact = count_valid_trajectories(
        grid,
        max_lateral_step=1,
        max_downward_step=1,
    )
    assert exact == _brute_count(grid, max_lateral_step=1, max_downward_step=1)
    short_stop = (grid.cell_at(0, 1), grid.cell_at(1, 1))
    assert trajectory_is_valid(short_stop, grid) is False
    reach_floor = (
        grid.cell_at(0, 1),
        grid.cell_at(1, 1),
        grid.cell_at(2, 0),
    )
    assert trajectory_is_valid(reach_floor, grid, max_lateral_step=1, max_downward_step=1)


def _tiny_reference_matrix():
    from nutella_scraper.engines.compute.coverage_reference_matrix import (
        AZIMUTH_NEGATIVE_SENSE,
        AZIMUTH_POSITIVE_SENSE,
        REFERENCE_ZONE_SIDE,
        CoverageReferenceMatrix,
    )

    points = (
        (50.0, 10.0, 0.0),
        (0.0, 10.0, -50.0),
        (50.0, 0.0, 0.0),
        (0.0, 0.0, -50.0),
    )
    return CoverageReferenceMatrix(
        coverage_target_surface="interior_product_surface",
        coverage_target_region=COVERAGE_TARGET_REGION,
        coverage_target_azimuth_range=(0.0, 90.0),
        a0_azimuth_deg=0.0,
        azimuth_span_deg=90.0,
        azimuth_positive_sense=AZIMUTH_POSITIVE_SENSE,
        azimuth_negative_sense=AZIMUTH_NEGATIVE_SENSE,
        reference_zone_side=REFERENCE_ZONE_SIDE,
        spacing_mm=MATRIX_SPACING_MM,
        points_mm=points,
        point_count=len(points),
        y_min_mm=0.0,
        y_max_mm=10.0,
        azimuth_min_deg=0.0,
        azimuth_max_deg=90.0,
        target_face_ids=(),
        target_area_mm2=0.0,
        face_count=0,
        fingerprint="tiny-test",
        source="test",
        mean_vertical_spacing_mm=5.0,
        mean_tangential_spacing_mm=5.0,
        neighbor_min_mm=5.0,
        neighbor_max_mm=5.0,
        bbox_min_mm=(0.0, 0.0, -50.0),
        bbox_max_mm=(50.0, 10.0, 0.0),
        on_interior_envelope=True,
        max_distance_to_interior_mm=0.0,
        any_point_outside_envelope=False,
        uses_legacy_a0_point_matrix=False,
        simulator_invoked=False,
    )


def test_indexes_validated_cloud_not_a0_and_does_not_run_physics() -> None:
    surface = _fast_surface()
    matrix = build_coverage_reference_matrix(surface)
    assert matrix.coverage_target_region == COVERAGE_TARGET_REGION
    assert matrix.coverage_target_region != LEGACY_A0_QUADRANT_REGION
    assert matrix.uses_legacy_a0_point_matrix is False
    assert matrix.spacing_mm == MATRIX_SPACING_MM
    grid = index_reference_matrix(matrix, surface=surface)
    assert grid.target_definition == COVERAGE_TARGET_REGION
    assert grid.uses_legacy_a0_point_matrix is False
    assert grid.spacing_mm == MATRIX_SPACING_MM
    assert grid.angle_range_deg == pytest.approx((0.0, 90.0))
    assert grid.n_rows >= 2
    assert grid.n_cols >= 2
    assert len(grid.cells) == matrix.point_count
    wall = [
        cell
        for cell in grid.cells
        if (cell.x_mm**2 + cell.z_mm**2) ** 0.5 >= 5.0
    ]
    assert wall
    assert max(cell.azimuth_deg for cell in wall) <= 90.0 + 1.0
    assert min(cell.azimuth_deg for cell in wall) >= -1.0
    tops = [cell for cell in grid.cells if cell.is_top]
    assert len(tops) == grid.n_cols
    assert all(cell.row == 0 for cell in tops)
    assert all(
        cell.y_mm >= max(c.y_mm for c in grid.cells if c.col == cell.col) - 1e-9
        for cell in tops
    )
    report = simulate_scraper_trajectory_search(
        matrix=_tiny_reference_matrix(),
        physics_limit=PHYSICS_ENUMERATION_LIMIT,
    )
    assert report.physics_executed is False
    assert report.results == ()
    assert report.allow_upward_motion is False
    assert report.monotonic_downward is True
    assert report.max_lateral_step == MAX_LATERAL_STEP
    assert report.grid.uses_legacy_a0_point_matrix is False
    assert report.grid.target_definition == COVERAGE_TARGET_REGION
    assert report.total_valid_trajectories == count_valid_trajectories(report.grid)


def test_write_refuses_to_overwrite_saved_coverage_100(tmp_path: Path) -> None:
    grid = _rect_grid(2, 2)
    from nutella_scraper.engines.compute.trajectory_search import (
        TrajectorySearchReport,
    )

    payload_report = TrajectorySearchReport(
        grid=grid,
        max_lateral_step=1,
        max_downward_step=1,
        monotonic_downward=True,
        allow_same_row_motion=True,
        allow_upward_motion=False,
        total_valid_trajectories=4,
        physics_executed=False,
        enumeration_mode="empty",
        physics_limit=10,
        results=(),
        transition_rules=("start_on_top_row",),
    )
    with pytest.raises(ValueError, match="candidate_coverage_100"):
        write_trajectory_search_results(
            payload_report,
            Path("output/coverage/candidate_coverage_100.json"),
        )
    with pytest.raises(ValueError, match="candidate_coverage_100"):
        write_trajectory_search_results(
            payload_report,
            tmp_path / "candidate_coverage_100.csv",
        )
    target = tmp_path / "trajectory_search_results.json"
    write_trajectory_search_results(payload_report, target)
    body = json.loads(target.read_text(encoding="utf-8"))
    assert body["search"]["total_valid_trajectories"] == 4
    assert body["results"] == []
    assert body["grid"]["spacing_mm"] == MATRIX_SPACING_MM
    assert body["search"]["physics_executed"] is False


@pytest.mark.skipif(not SAVED_JSON.is_file(), reason="saved coverage-100 JSON absent")
def test_saved_ranking_is_not_rewritten() -> None:
    before = hashlib.sha256(SAVED_JSON.read_bytes()).hexdigest()
    payload = json.loads(SAVED_JSON.read_text(encoding="utf-8"))
    ranked = payload["ranked"]
    a0 = next(row for row in ranked if row["candidate_id"] == "A0")
    s8 = next(row for row in ranked if row["candidate_id"] == "S0008")
    assert float(a0["coverage_percent"]) == pytest.approx(63.3333, abs=1e-4)
    assert float(s8["coverage_percent"]) == pytest.approx(66.25, abs=1e-4)
    assert hashlib.sha256(SAVED_JSON.read_bytes()).hexdigest() == before


def test_simuler_button_calls_trajectory_search_not_a0_grid() -> None:
    html = HTML_SRC.read_text(encoding="utf-8")
    assert html.count('id="simulate-contact"') == 1
    assert ">Simuler<" in html
    sim = html[
        html.index("async function simulateContact") : html.index(
            "async function cancelSimulation"
        )
    ]
    assert "API.trajectorySearch" in sim
    assert "API.simulateContact" not in sim
    assert "/api/trajectory-search" in html
    assert 'simulateContact: "/api/simulate-contact"' in html
    assert "drawSearchTrajectory" in html
    draw = html[html.index("function drawScene3D") : html.index("async function loadViewerScene")]
    assert "drawSearchTrajectory" in draw
    assert "selectedTrajectory" in draw
    assert '"#ffffff"' in draw
    assert "Référence A0" not in html
    assert "evaluate_candidate(" not in html


def test_simulate_requires_matrix_or_surface() -> None:
    with pytest.raises(ValueError, match="surface or matrix"):
        simulate_scraper_trajectory_search()
