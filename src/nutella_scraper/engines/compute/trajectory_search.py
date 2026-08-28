"""Search layer: valid scraper trajectories on the validated interior point cloud.

The white 5 mm coverage reference matrix is the exclusive waypoint grid.
This module does not rebuild that cloud, does not call the coverage simulator,
and does not import collision / visualization / optimization.

A0 is only the azimuth origin of the existing matrix (0°), not a search grid.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
    LEGACY_A0_QUADRANT_REGION,
    MATRIX_SPACING_MM,
    WALL_RADIUS_MIN_MM,
    CoverageReferenceMatrix,
    azimuths_deg,
    build_coverage_reference_matrix,
    surface_axis_xz,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)

# Few columns of the 5 mm azimuth lattice. Independent of the blade-shape
# lattice DEFAULT_MAX_ROW_STEP (scraper_shape_space) which is a different space.
MAX_LATERAL_STEP = 2
MAX_DOWNWARD_STEP = 1
PHYSICS_ENUMERATION_LIMIT = 10_000
MAX_BITMASK_COLUMNS = 22
SAVED_COVERAGE_100 = Path("output/coverage/candidate_coverage_100.json")
DEFAULT_RESULTS_PATH = Path("output/coverage/trajectory_search_results.json")


@dataclass(frozen=True)
class TrajectoryCell:
    """One waypoint of the validated interior cloud, indexed as (row, col)."""

    index: int
    row: int
    col: int
    x_mm: float
    y_mm: float
    z_mm: float
    azimuth_deg: float
    is_top: bool
    is_bottom: bool
    on_interior_envelope: bool = True


@dataclass(frozen=True)
class TrajectoryGrid:
    """Existing cloud indexed as meridians (columns) × stations from the opening."""

    cells: tuple[TrajectoryCell, ...]
    n_rows: int
    n_cols: int
    points_per_row: tuple[int, ...]
    spacing_mm: float
    angle_range_deg: tuple[float, float]
    target_definition: str
    fingerprint: str
    uses_legacy_a0_point_matrix: bool = False

    def cell_at(self, row: int, col: int) -> TrajectoryCell | None:
        key = (int(row), int(col))
        for cell in self.cells:
            if (cell.row, cell.col) == key:
                return cell
        return None


@dataclass(frozen=True)
class TrajectorySearchReport:
    """Cardinality-first search report. Physics is gated on N."""

    grid: TrajectoryGrid
    max_lateral_step: int
    max_downward_step: int
    monotonic_downward: bool
    allow_same_row_motion: bool
    allow_upward_motion: bool
    total_valid_trajectories: int
    physics_executed: bool
    enumeration_mode: str
    physics_limit: int
    results: tuple[dict[str, Any], ...]
    transition_rules: tuple[str, ...]
    optimization_method: str = "none"
    optimization_label: str = "CARDINALITY_ONLY"
    beam_width: int = 0
    physics_pose_count: int = 0
    disclaimer: str = ""


def index_reference_matrix(
    matrix: CoverageReferenceMatrix,
    *,
    axis_xz: NDArray[np.float64] | None = None,
    surface: InteriorSurfaceReference | None = None,
) -> TrajectoryGrid:
    """Assign (row, col) to the existing cloud. Does not move or resample points."""
    points = np.asarray(matrix.points_mm, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("Coverage reference matrix is empty")
    if axis_xz is not None:
        axis = np.asarray(axis_xz, dtype=np.float64)
    elif surface is not None:
        axis = surface_axis_xz(np.asarray(surface.vertices, dtype=np.float64))
    else:
        axis = surface_axis_xz(points)
    az = azimuths_deg(points, axis)
    radii = np.hypot(points[:, 0] - float(axis[0]), points[:, 2] - float(axis[1]))
    wall = radii >= WALL_RADIUS_MIN_MM
    if not np.any(wall):
        wall = np.ones(len(points), dtype=np.bool_)
    col_az = np.unique(np.round(az[wall], 1))
    if len(col_az) == 0:
        raise ValueError("Coverage reference matrix has no azimuth columns")
    col_id = np.argmin(np.abs(az[:, None] - col_az[None, :]), axis=1)
    columns: list[list[int]] = [[] for _ in col_az]
    for i, c in enumerate(col_id):
        columns[int(c)].append(i)
    cells: list[TrajectoryCell] = []
    lengths: list[int] = []
    for col, idxs in enumerate(columns):
        if not idxs:
            lengths.append(0)
            continue
        order = sorted(
            idxs,
            key=lambda i: (-float(points[i, 1]), -float(radii[i])),
        )
        lengths.append(len(order))
        last = len(order) - 1
        for row, i in enumerate(order):
            cells.append(
                TrajectoryCell(
                    index=int(i),
                    row=int(row),
                    col=int(col),
                    x_mm=float(points[i, 0]),
                    y_mm=float(points[i, 1]),
                    z_mm=float(points[i, 2]),
                    azimuth_deg=float(az[i]),
                    is_top=row == 0,
                    is_bottom=row == last,
                    on_interior_envelope=bool(matrix.on_interior_envelope),
                )
            )
    n_rows = max(lengths) if lengths else 0
    per_row = []
    for row in range(n_rows):
        per_row.append(sum(1 for cell in cells if cell.row == row))
    payload = np.round(points, 4).tobytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    return TrajectoryGrid(
        cells=tuple(cells),
        n_rows=int(n_rows),
        n_cols=int(len(col_az)),
        points_per_row=tuple(int(v) for v in per_row),
        spacing_mm=float(matrix.spacing_mm),
        angle_range_deg=tuple(matrix.coverage_target_azimuth_range),
        target_definition=str(matrix.coverage_target_region),
        fingerprint=fingerprint,
        uses_legacy_a0_point_matrix=bool(matrix.uses_legacy_a0_point_matrix),
    )


def trajectory_grid_from_cells(
    cells: tuple[TrajectoryCell, ...],
    *,
    spacing_mm: float = MATRIX_SPACING_MM,
    angle_range_deg: tuple[float, float] = (0.0, 90.0),
    target_definition: str = COVERAGE_TARGET_REGION,
) -> TrajectoryGrid:
    """Test helper: wrap an explicit cell set as a grid."""
    if not cells:
        raise ValueError("Trajectory grid is empty")
    n_rows = max(cell.row for cell in cells) + 1
    n_cols = max(cell.col for cell in cells) + 1
    per_row = tuple(
        sum(1 for cell in cells if cell.row == row) for row in range(n_rows)
    )
    blob = np.asarray(
        [(c.row, c.col, c.x_mm, c.y_mm, c.z_mm) for c in cells],
        dtype=np.float64,
    )
    fingerprint = hashlib.sha256(np.round(blob, 4).tobytes()).hexdigest()
    return TrajectoryGrid(
        cells=cells,
        n_rows=n_rows,
        n_cols=n_cols,
        points_per_row=per_row,
        spacing_mm=float(spacing_mm),
        angle_range_deg=angle_range_deg,
        target_definition=str(target_definition),
        fingerprint=fingerprint,
        uses_legacy_a0_point_matrix=False,
    )


def _occupation(grid: TrajectoryGrid) -> NDArray[np.bool_]:
    occ = np.zeros((grid.n_rows, grid.n_cols), dtype=np.bool_)
    for cell in grid.cells:
        occ[cell.row, cell.col] = True
    return occ


def transition_allowed(
    src: TrajectoryCell,
    dst: TrajectoryCell,
    *,
    max_lateral_step: int = MAX_LATERAL_STEP,
    max_downward_step: int = MAX_DOWNWARD_STEP,
) -> bool:
    """One legal step: same row (lateral) or strictly downward. Never up."""
    d_row = int(dst.row) - int(src.row)
    d_col = abs(int(dst.col) - int(src.col))
    if d_row < 0:
        return False
    if d_col > int(max_lateral_step):
        return False
    if d_row == 0:
        return d_col > 0
    return 1 <= d_row <= int(max_downward_step)


def trajectory_is_valid(
    cells: Sequence[TrajectoryCell],
    grid: TrajectoryGrid,
    *,
    max_lateral_step: int = MAX_LATERAL_STEP,
    max_downward_step: int = MAX_DOWNWARD_STEP,
) -> bool:
    """True iff the sequence is a monotone-down path from a top cell to a bottom cell."""
    if not cells:
        return False
    if not cells[0].is_top:
        return False
    if int(cells[-1].row) != int(grid.n_rows) - 1:
        return False
    if not cells[-1].is_bottom:
        return False
    seen_on_row: dict[int, set[int]] = {}
    for prev, nxt in zip(cells[:-1], cells[1:], strict=True):
        if not transition_allowed(
            prev,
            nxt,
            max_lateral_step=max_lateral_step,
            max_downward_step=max_downward_step,
        ):
            return False
        seen_on_row.setdefault(prev.row, set()).add(prev.col)
        if nxt.row == prev.row and nxt.col in seen_on_row[prev.row]:
            return False
        if nxt.row != prev.row:
            seen_on_row[nxt.row] = {nxt.col}
    return True


def count_valid_trajectories(
    grid: TrajectoryGrid,
    *,
    max_lateral_step: int = MAX_LATERAL_STEP,
    max_downward_step: int = MAX_DOWNWARD_STEP,
) -> int:
    """Exact number of simple, monotone-down trajectories that reach a floor cell.

    Same-row motion may reverse column as long as a column is not revisited on
    that row. Rows only increase, so a path never climbs. Python ints are used
    so the count stays exact when N is huge.
    """
    n_rows = int(grid.n_rows)
    n_cols = int(grid.n_cols)
    if n_rows < 1 or n_cols < 1:
        return 0
    if n_cols > MAX_BITMASK_COLUMNS:
        raise ValueError(
            f"Exact cardinality needs a {n_cols}-bit mask "
            f"(limit {MAX_BITMASK_COLUMNS})"
        )
    occ = _occupation(grid)
    lateral = int(max_lateral_step)
    down = int(max_downward_step)
    laterals: list[list[list[int]]] = [
        [[] for _ in range(n_cols)] for _ in range(n_rows)
    ]
    for row in range(n_rows):
        for col in range(n_cols):
            if not occ[row, col]:
                continue
            for ncol in range(n_cols):
                if ncol == col or not occ[row, ncol]:
                    continue
                if abs(ncol - col) <= lateral:
                    laterals[row][col].append(ncol)
    arrivals: list[list[int]] = [[0] * n_cols for _ in range(n_rows)]
    for cell in grid.cells:
        if cell.is_top:
            arrivals[cell.row][cell.col] = 1
    total = 0
    for row in range(n_rows):
        states: list[dict[int, int]] = [dict() for _ in range(n_cols)]
        for col in range(n_cols):
            if arrivals[row][col] and occ[row, col]:
                states[col][1 << col] = arrivals[row][col]
        for k in range(n_cols):
            for col in range(n_cols):
                items = [
                    (mask, ways)
                    for mask, ways in states[col].items()
                    if mask.bit_count() == k
                ]
                for mask, ways in items:
                    for ncol in laterals[row][col]:
                        bit = 1 << ncol
                        if mask & bit:
                            continue
                        nmask = mask | bit
                        states[ncol][nmask] = states[ncol].get(nmask, 0) + ways
        sitting = [int(sum(states[col].values())) for col in range(n_cols)]
        if row == n_rows - 1:
            for col in range(n_cols):
                if occ[row, col]:
                    total += sitting[col]
        if down < 1:
            continue
        for col in range(n_cols):
            ways = sitting[col]
            if ways == 0:
                continue
            for drow in range(1, down + 1):
                nxt_row = row + drow
                if nxt_row >= n_rows:
                    break
                for ncol in range(n_cols):
                    if not occ[nxt_row, ncol]:
                        continue
                    if abs(ncol - col) > lateral:
                        continue
                    arrivals[nxt_row][ncol] += ways
    return int(total)


def simulate_scraper_trajectory_search(
    surface: InteriorSurfaceReference | None = None,
    *,
    matrix: CoverageReferenceMatrix | None = None,
    max_lateral_step: int = MAX_LATERAL_STEP,
    max_downward_step: int = MAX_DOWNWARD_STEP,
    physics_limit: int = PHYSICS_ENUMERATION_LIMIT,
    output_path: Path | None = None,
    optimize: bool = False,
) -> TrajectorySearchReport:
    """Index the cloud, count exact N, optionally beam-search with a contact cache.

    Does not call CoverageSimulator.evaluate_candidate. Never enumerates N.
    """
    if matrix is None:
        if surface is None:
            raise ValueError("surface or matrix is required")
        matrix = build_coverage_reference_matrix(surface)
    if matrix.uses_legacy_a0_point_matrix:
        raise ValueError("Legacy A0 point matrix cannot be used as the search grid")
    if str(matrix.coverage_target_region) == LEGACY_A0_QUADRANT_REGION:
        raise ValueError("Legacy A0 quadrant region cannot be used as the search grid")
    axis = None
    if surface is not None:
        axis = surface_axis_xz(np.asarray(surface.vertices, dtype=np.float64))
    grid = index_reference_matrix(matrix, axis_xz=axis, surface=surface)
    n_valid = count_valid_trajectories(
        grid,
        max_lateral_step=max_lateral_step,
        max_downward_step=max_downward_step,
    )
    physics_executed = False
    optimization_method = "none"
    optimization_label = "CARDINALITY_ONLY"
    beam_width = 0
    physics_pose_count = 0
    disclaimer = (
        "Nous ne pouvons pas énumérer toutes les trajectoires. "
        "Cardinalité exacte seulement ; pas d'optimum de couverture."
    )
    ranked_payload: tuple[dict[str, Any], ...] = ()
    if n_valid == 0:
        enumeration_mode = "empty"
    elif n_valid > int(physics_limit):
        enumeration_mode = "cardinality_only"
    else:
        enumeration_mode = "cardinality_only_below_physics_limit"
    if optimize and surface is not None and n_valid > 0:
        from nutella_scraper.engines.compute.trajectory_contact_cache import (
            build_contact_cache,
        )
        from nutella_scraper.engines.compute.trajectory_optimizer import (
            BEAM_WIDTH,
            DISCLAIMER,
            OPTIMIZATION_LABEL,
            OPTIMIZATION_METHOD,
            beam_search_trajectories,
        )

        cache = build_contact_cache(surface, matrix, grid)
        ranked = beam_search_trajectories(
            grid,
            cache,
            max_lateral_step=max_lateral_step,
            max_downward_step=max_downward_step,
        )
        ranked_payload = tuple(item.to_payload() for item in ranked)
        physics_executed = True
        enumeration_mode = "heuristic_beam_search"
        optimization_method = OPTIMIZATION_METHOD
        optimization_label = OPTIMIZATION_LABEL
        beam_width = int(BEAM_WIDTH)
        physics_pose_count = int(cache.physics_queries)
        disclaimer = DISCLAIMER
    rules = (
        "start_on_top_row",
        "end_on_global_last_row",
        "row_index_non_decreasing",
        "same_row_motion_allowed",
        "no_upward_row_step",
        "no_column_revisit_on_the_same_row",
        f"|Δcol| <= {int(max_lateral_step)}",
        f"1 <= Δrow <= {int(max_downward_step)} when leaving a row",
    )
    report = TrajectorySearchReport(
        grid=grid,
        max_lateral_step=int(max_lateral_step),
        max_downward_step=int(max_downward_step),
        monotonic_downward=True,
        allow_same_row_motion=True,
        allow_upward_motion=False,
        total_valid_trajectories=int(n_valid),
        physics_executed=physics_executed,
        enumeration_mode=enumeration_mode,
        physics_limit=int(physics_limit),
        results=ranked_payload,
        transition_rules=rules,
        optimization_method=optimization_method,
        optimization_label=optimization_label,
        beam_width=beam_width,
        physics_pose_count=physics_pose_count,
        disclaimer=disclaimer,
    )
    if output_path is not None:
        write_trajectory_search_results(report, output_path)
    return report


def report_to_payload(report: TrajectorySearchReport) -> dict[str, Any]:
    n_valid = int(report.total_valid_trajectories)
    results = list(report.results)
    best = results[0] if results else None
    exhaustive_ok = n_valid <= report.physics_limit
    return {
        "grid": {
            "rows": report.grid.n_rows,
            "cols": report.grid.n_cols,
            "points": len(report.grid.cells),
            "points_per_row": list(report.grid.points_per_row),
            "spacing_mm": report.grid.spacing_mm,
            "angle_range_deg": list(report.grid.angle_range_deg),
            "target_definition": report.grid.target_definition,
            "fingerprint": report.grid.fingerprint,
            "uses_legacy_a0_point_matrix": report.grid.uses_legacy_a0_point_matrix,
        },
        "search": {
            "max_lateral_step": report.max_lateral_step,
            "max_downward_step": report.max_downward_step,
            "monotonic_downward": report.monotonic_downward,
            "allow_same_row_motion": True,
            "allow_upward_motion": False,
            "total_valid_trajectories": n_valid,
            "total_valid_trajectories_decimal": str(n_valid),
            "physics_executed": report.physics_executed,
            "enumeration_mode": report.enumeration_mode,
            "physics_limit": report.physics_limit,
            "transition_rules": list(report.transition_rules),
            "optimization_method": report.optimization_method,
            "optimization_label": report.optimization_label,
            "beam_width": report.beam_width,
            "physics_pose_count": report.physics_pose_count,
            "physics_blocked_reason": (
                None
                if exhaustive_ok
                else (
                    "Espace trop grand pour l'exhaustif physique "
                    f"({n_valid} > {report.physics_limit})"
                )
            ),
        },
        "results": results,
        "best": best,
        "disclaimer": report.disclaimer,
        "message": (
            f"Espace de recherche = {n_valid} trajectoires. "
            + report.disclaimer
        ),
    }


def write_trajectory_search_results(
    report: TrajectorySearchReport,
    path: Path,
) -> None:
    target = Path(path)
    if target.resolve() == SAVED_COVERAGE_100.resolve():
        raise ValueError("Refus d'écraser candidate_coverage_100.json")
    if target.name.startswith("candidate_coverage_100"):
        raise ValueError("Refus d'écraser les résultats candidate_coverage_100")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report_to_payload(report), indent=2),
        encoding="utf-8",
    )
