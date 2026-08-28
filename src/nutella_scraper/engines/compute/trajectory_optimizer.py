"""Heuristic beam search of scraper trajectories on the contact cache.

We cannot enumerate the ~10^176 valid trajectories. This module searches
the waypoint DAG (monotone downward on the validated cloud) and ranks
paths by UNION coverage of the 608 matrix points.

Label: HEURISTIC. Not a mathematical optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import (
    TrajectoryContactCache,
)
from nutella_scraper.engines.compute.trajectory_search import (
    MAX_DOWNWARD_STEP,
    MAX_LATERAL_STEP,
    TrajectoryCell,
    TrajectoryGrid,
    transition_allowed,
)

BEAM_WIDTH = 48
TOP_K = 10
OPTIMIZATION_METHOD = "beam_search"
OPTIMIZATION_LABEL = "HEURISTIC"
DISCLAIMER = (
    "Nous ne pouvons pas énumérer toutes les trajectoires. "
    "Nous cherchons donc une très bonne trajectoire par optimisation du graphe."
)


@dataclass(frozen=True)
class BeamState:
    cell: TrajectoryCell
    visited_on_row: frozenset[int]
    path: tuple[TrajectoryCell, ...]
    covered_mask: int
    covered_count: int
    path_length_mm: float
    lateral_moves: int
    downward_moves: int
    direction_changes: int
    last_move: str


@dataclass(frozen=True)
class TrajectoryCandidate:
    rank: int
    coverage_percent: float
    covered_points: int
    total_points: int
    covered_mask: int
    covered_point_indices: tuple[int, ...]
    trajectory_id: str
    position_count: int
    downward_moves: int
    lateral_moves: int
    direction_changes: int
    path_length_mm: float
    path: tuple[TrajectoryCell, ...]
    optimization_label: str = OPTIMIZATION_LABEL
    optimization_method: str = OPTIMIZATION_METHOD
    beam_trajectories_explored: int = 0

    def to_payload(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "coverage_percent": self.coverage_percent,
            "covered_points": self.covered_points,
            "total_points": self.total_points,
            "covered_point_indices": list(self.covered_point_indices),
            "trajectory_id": self.trajectory_id,
            "position_count": self.position_count,
            "downward_moves": self.downward_moves,
            "lateral_moves": self.lateral_moves,
            "direction_changes": self.direction_changes,
            "path_length_mm": self.path_length_mm,
            "sequence_rows_cols": [[cell.row, cell.col] for cell in self.path],
            "sequence_point_indices": [cell.index for cell in self.path],
            "points_mm": [[cell.x_mm, cell.y_mm, cell.z_mm] for cell in self.path],
            "optimization_label": self.optimization_label,
            "optimization_method": self.optimization_method,
            "beam_trajectories_explored": int(self.beam_trajectories_explored),
        }


def _move_kind(src: TrajectoryCell, dst: TrajectoryCell) -> str:
    if dst.row == src.row:
        if dst.col < src.col:
            return "left"
        return "right"
    return "down"


def _distance_mm(src: TrajectoryCell, dst: TrajectoryCell) -> float:
    dx = float(dst.x_mm) - float(src.x_mm)
    dy = float(dst.y_mm) - float(src.y_mm)
    dz = float(dst.z_mm) - float(src.z_mm)
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _neighbors(
    cell: TrajectoryCell,
    visited_on_row: frozenset[int],
    grid: TrajectoryGrid,
    *,
    max_lateral_step: int,
    max_downward_step: int,
) -> list[tuple[TrajectoryCell, frozenset[int]]]:
    out: list[tuple[TrajectoryCell, frozenset[int]]] = []
    for ncol in range(grid.n_cols):
        if ncol == cell.col or ncol in visited_on_row:
            continue
        dst = grid.cell_at(cell.row, ncol)
        if dst is None:
            continue
        if not transition_allowed(
            cell,
            dst,
            max_lateral_step=max_lateral_step,
            max_downward_step=max_downward_step,
        ):
            continue
        out.append((dst, visited_on_row | {ncol}))
    for drow in range(1, int(max_downward_step) + 1):
        nxt_row = cell.row + drow
        if nxt_row >= grid.n_rows:
            break
        for ncol in range(grid.n_cols):
            dst = grid.cell_at(nxt_row, ncol)
            if dst is None:
                continue
            if not transition_allowed(
                cell,
                dst,
                max_lateral_step=max_lateral_step,
                max_downward_step=max_downward_step,
            ):
                continue
            out.append((dst, frozenset({ncol})))
    return out


def _expand(
    state: BeamState,
    dst: TrajectoryCell,
    next_visited: frozenset[int],
    cache: TrajectoryContactCache,
) -> BeamState:
    kind = _move_kind(state.cell, dst)
    covered = int(state.covered_mask) | int(cache.mask_for(dst))
    return BeamState(
        cell=dst,
        visited_on_row=next_visited,
        path=state.path + (dst,),
        covered_mask=covered,
        covered_count=covered.bit_count(),
        path_length_mm=state.path_length_mm + _distance_mm(state.cell, dst),
        lateral_moves=state.lateral_moves + (0 if kind == "down" else 1),
        downward_moves=state.downward_moves + (1 if kind == "down" else 0),
        direction_changes=state.direction_changes
        + (0 if state.last_move in {"start", kind} else 1),
        last_move=kind,
    )


def _rank_key(state: BeamState) -> tuple[int, int, float, int]:
    """Coverage first. Length and turns are secondary, never mixed as a weight."""
    return (
        int(state.covered_count),
        -int(state.downward_moves),
        -float(state.path_length_mm),
        -int(state.direction_changes),
    )


def _indices_from_mask(mask: int, n_points: int) -> tuple[int, ...]:
    return tuple(index for index in range(n_points) if mask & (1 << index))


def _candidate_from_state(
    rank: int,
    state: BeamState,
    n_points: int,
    *,
    beam_trajectories_explored: int = 0,
) -> TrajectoryCandidate:
    total = int(n_points)
    covered = int(state.covered_count)
    percent = 0.0 if total <= 0 else 100.0 * covered / total
    path_id = "-".join(f"{cell.row}:{cell.col}" for cell in state.path)
    return TrajectoryCandidate(
        rank=int(rank),
        coverage_percent=float(percent),
        covered_points=covered,
        total_points=total,
        covered_mask=int(state.covered_mask),
        covered_point_indices=_indices_from_mask(int(state.covered_mask), total),
        trajectory_id=f"T{rank:02d}:{path_id}",
        position_count=len(state.path),
        downward_moves=int(state.downward_moves),
        lateral_moves=int(state.lateral_moves),
        direction_changes=int(state.direction_changes),
        path_length_mm=float(state.path_length_mm),
        path=state.path,
        beam_trajectories_explored=int(beam_trajectories_explored),
    )


def beam_search_trajectories(
    grid: TrajectoryGrid,
    cache: TrajectoryContactCache,
    *,
    beam_width: int = BEAM_WIDTH,
    top_k: int = TOP_K,
    max_lateral_step: int = MAX_LATERAL_STEP,
    max_downward_step: int = MAX_DOWNWARD_STEP,
) -> tuple[TrajectoryCandidate, ...]:
    """Keep the best beam_width partial paths. Does not enumerate N."""
    if grid.target_definition != COVERAGE_TARGET_REGION:
        raise ValueError("Beam search target must be the validated interior matrix")
    if cache.target_definition != COVERAGE_TARGET_REGION:
        raise ValueError("Contact cache target must be the validated interior matrix")
    if len(cache.entries) != len(grid.cells):
        raise ValueError("Contact cache entries must match the grid cells")
    if int(cache.n_points) < 1:
        raise ValueError("Contact cache has no target points")
    if cache.uses_legacy_a0_point_matrix or grid.uses_legacy_a0_point_matrix:
        raise ValueError("A0 point matrix cannot be used as the search grid")
    if cache.symmetry_multiplier_applied:
        raise ValueError("Symmetry multipliers are forbidden")
    last_row = int(grid.n_rows) - 1
    starts: list[BeamState] = []
    for cell in grid.cells:
        if not cell.is_top:
            continue
        if cache.entry_at(cell.row, cell.col) is not None:
            entry = cache.entry_at(cell.row, cell.col)
            if entry is not None and not entry.admissible:
                continue
        mask = int(cache.mask_for(cell))
        starts.append(
            BeamState(
                cell=cell,
                visited_on_row=frozenset({cell.col}),
                path=(cell,),
                covered_mask=mask,
                covered_count=mask.bit_count(),
                path_length_mm=0.0,
                lateral_moves=0,
                downward_moves=0,
                direction_changes=0,
                last_move="start",
            )
        )
    if not starts:
        return ()
    width = max(1, int(beam_width))
    beam = sorted(starts, key=_rank_key, reverse=True)[:width]
    completed: list[BeamState] = []
    seen: set[tuple[int, int, frozenset[int], int]] = set()
    max_steps = int(grid.n_rows) * max(1, int(grid.n_cols)) + 2
    for _ in range(max_steps):
        nxt: list[BeamState] = []
        for state in beam:
            if state.cell.row == last_row:
                completed.append(state)
            for dst, visited in _neighbors(
                state.cell,
                state.visited_on_row,
                grid,
                max_lateral_step=max_lateral_step,
                max_downward_step=max_downward_step,
            ):
                entry = cache.entry_at(dst.row, dst.col)
                if entry is not None and not entry.admissible:
                    continue
                child = _expand(state, dst, visited, cache)
                key = (
                    child.cell.row,
                    child.cell.col,
                    child.visited_on_row,
                    child.covered_mask,
                )
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(child)
        if not nxt:
            break
        beam = sorted(nxt, key=_rank_key, reverse=True)[:width]
    unique: dict[tuple[int, ...], BeamState] = {}
    for state in completed:
        key = tuple(cell.index for cell in state.path)
        prev = unique.get(key)
        if prev is None or _rank_key(state) > _rank_key(prev):
            unique[key] = state
    ranked = sorted(unique.values(), key=_rank_key, reverse=True)[: max(1, int(top_k))]
    n_points = int(cache.n_points)
    explored = len(unique)
    return tuple(
        _candidate_from_state(
            rank,
            state,
            n_points,
            beam_trajectories_explored=explored,
        )
        for rank, state in enumerate(ranked, start=1)
    )
