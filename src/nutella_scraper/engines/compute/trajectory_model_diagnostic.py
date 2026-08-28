"""TRAJECTORY_MODEL_DIAGNOSTIC — graph / mapping audit. No campaign.

Does not call CoverageSimulator. Does not change collision, jar, or the cloud.
Label: TRAJECTORY_MODEL_DIAGNOSTIC.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from nutella_scraper.engines.compute.trajectory_contact_cache import (
    CellContactEntry,
    TrajectoryContactCache,
)
from nutella_scraper.engines.compute.trajectory_optimizer import _neighbors
from nutella_scraper.engines.compute.trajectory_search import (
    MAX_DOWNWARD_STEP,
    MAX_LATERAL_STEP,
    TrajectoryCell,
    TrajectoryGrid,
)

DIAGNOSTIC_LABEL = "TRAJECTORY_MODEL_DIAGNOSTIC"

CELL_EQUALS_POSE = True
MAPPING_STEPS = (
    "cellule cible (point du nuage, indexé row/col)",
    "pose SE(3) : yaw = azimuth de la cellule, Y = y_mm de la cellule",
    "voisinage SE(3) si la pose nominale est bloquée",
    "contact / collision à cette pose",
    "masque UNION des points du nuage dont la face est en contact",
    "recherche de chemin : succession de cellules ADMISSIBLES",
)

MAX_DOWNWARD_STEP_MEANS = (
    "le racloir ne peut descendre que d'une rangée d'index du nuage 5 mm "
    "entre deux poses-cellules (d_row in [1, MAX_DOWNWARD_STEP])"
)


def mask_indices(mask: int, n_points: int) -> tuple[int, ...]:
    return tuple(index for index in range(int(n_points)) if mask & (1 << index))


def _cell_key(cell: TrajectoryCell) -> tuple[int, int]:
    return (int(cell.row), int(cell.col))


@dataclass(frozen=True)
class RowOccupancy:
    row: int
    n_cells: int
    n_admissible: int
    n_with_contact: int
    y_min_mm: float
    y_max_mm: float


@dataclass(frozen=True)
class GraphCut:
    kind: str
    row: int
    detail: str
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ShapeGraphReport:
    family_id: str
    n_admissible: int
    n_with_contact: int
    admissible_per_row: tuple[RowOccupancy, ...]
    first_row_without_admissible: int | None
    n_successive_row_transitions: int
    n_starts: int
    n_reachable_from_opening: int
    max_reachable_row: int | None
    last_row: int
    last_row_admissible: int
    extinction: GraphCut
    beam_complete_paths: int
    mapping_is_cell_equals_pose: bool = CELL_EQUALS_POSE
    max_downward_step: int = MAX_DOWNWARD_STEP
    max_lateral_step: int = MAX_LATERAL_STEP
    max_downward_step_means: str = MAX_DOWNWARD_STEP_MEANS


@dataclass(frozen=True)
class PoseContactSpan:
    row: int
    col: int
    origin_mm: tuple[float, float, float]
    yaw_deg: float
    covered_count: int
    covered_indices: tuple[int, ...]
    covered_rows: tuple[int, ...]
    touched_y_min_mm: float
    touched_y_max_mm: float
    n_rows_covered: int


def occupancy_by_row(
    grid: TrajectoryGrid,
    cache: TrajectoryContactCache,
) -> tuple[RowOccupancy, ...]:
    by_row: dict[int, list[TrajectoryCell]] = {}
    for cell in grid.cells:
        by_row.setdefault(int(cell.row), []).append(cell)
    rows: list[RowOccupancy] = []
    for row in range(int(grid.n_rows)):
        cells = by_row.get(row, [])
        n_adm = 0
        n_touch = 0
        ys: list[float] = []
        for cell in cells:
            entry = cache.entry_at(cell.row, cell.col)
            ys.append(float(cell.y_mm))
            if entry is not None and entry.admissible:
                n_adm += 1
                if int(entry.covered_count) > 0:
                    n_touch += 1
        rows.append(
            RowOccupancy(
                row=row,
                n_cells=len(cells),
                n_admissible=n_adm,
                n_with_contact=n_touch,
                y_min_mm=min(ys) if ys else 0.0,
                y_max_mm=max(ys) if ys else 0.0,
            )
        )
    return tuple(rows)


def successive_row_transitions(
    grid: TrajectoryGrid,
    cache: TrajectoryContactCache,
) -> int:
    """Count admissible→admissible edges with d_row == 1 (the MAX_DOWNWARD_STEP hop)."""
    count = 0
    for src in grid.cells:
        src_entry = cache.entry_at(src.row, src.col)
        if src_entry is None or not src_entry.admissible:
            continue
        for dst, _visited in _neighbors(
            src,
            frozenset({src.col}),
            grid,
            max_lateral_step=MAX_LATERAL_STEP,
            max_downward_step=MAX_DOWNWARD_STEP,
        ):
            if dst.row != src.row + 1:
                continue
            dst_entry = cache.entry_at(dst.row, dst.col)
            if dst_entry is not None and dst_entry.admissible:
                count += 1
    return count


def reachable_from_opening(
    grid: TrajectoryGrid,
    cache: TrajectoryContactCache,
) -> tuple[set[tuple[int, int]], tuple[TrajectoryCell, ...]]:
    starts = [
        cell
        for cell in grid.cells
        if cell.is_top
        and (entry := cache.entry_at(cell.row, cell.col)) is not None
        and entry.admissible
    ]
    seen: set[tuple[int, int]] = set()
    queue: deque[TrajectoryCell] = deque()
    for cell in starts:
        key = _cell_key(cell)
        if key in seen:
            continue
        seen.add(key)
        queue.append(cell)
    while queue:
        src = queue.popleft()
        for dst, _visited in _neighbors(
            src,
            frozenset({src.col}),
            grid,
            max_lateral_step=MAX_LATERAL_STEP,
            max_downward_step=MAX_DOWNWARD_STEP,
        ):
            entry = cache.entry_at(dst.row, dst.col)
            if entry is None or not entry.admissible:
                continue
            key = _cell_key(dst)
            if key in seen:
                continue
            seen.add(key)
            queue.append(dst)
    return seen, tuple(starts)


def _extinction(
    grid: TrajectoryGrid,
    cache: TrajectoryContactCache,
    occupancy: tuple[RowOccupancy, ...],
    reachable: set[tuple[int, int]],
    starts: tuple[TrajectoryCell, ...],
) -> GraphCut:
    last_row = int(grid.n_rows) - 1
    if not starts:
        blocked = tuple(
            (cell.row, cell.col) for cell in grid.cells if cell.is_top
        )
        return GraphCut(
            kind="no_opening_start",
            row=0,
            detail=(
                "aucune cellule admissible sur la rangée d'ouverture (row=0). "
                "Le beam ne peut pas démarrer."
            ),
            cells=blocked[:12],
        )
    if not reachable:
        return GraphCut(
            kind="empty_reachable",
            row=0,
            detail="départs ouverture présents mais ensemble atteignable vide",
            cells=tuple(_cell_key(cell) for cell in starts[:12]),
        )
    max_row = max(row for row, _col in reachable)
    if max_row >= last_row and any(
        key[0] == last_row for key in reachable
    ):
        return GraphCut(
            kind="none",
            row=last_row,
            detail="la dernière rangée globale est atteignable",
            cells=(),
        )
    frontier = [
        cell
        for cell in grid.cells
        if _cell_key(cell) in reachable and cell.row == max_row
    ]
    attempted: list[tuple[int, int]] = []
    for src in frontier:
        for dst, _visited in _neighbors(
            src,
            frozenset({src.col}),
            grid,
            max_lateral_step=MAX_LATERAL_STEP,
            max_downward_step=MAX_DOWNWARD_STEP,
        ):
            if dst.row <= src.row:
                continue
            attempted.append((dst.row, dst.col))
    next_row = max_row + 1
    n_next = occupancy[next_row].n_admissible if next_row < len(occupancy) else 0
    return GraphCut(
        kind="downward_cut",
        row=next_row,
        detail=(
            f"chemins ouverts jusqu'à row={max_row}. "
            f"Aucune transition admissible vers row>={next_row} "
            f"(MAX_DOWNWARD_STEP={MAX_DOWNWARD_STEP}, "
            f"cellules admissibles row {next_row}={n_next}). "
            "Le beam exige de finir sur la dernière rangée globale."
        ),
        cells=tuple(_cell_key(cell) for cell in frontier[:12]),
    )


def analyze_shape_graph(
    family_id: str,
    grid: TrajectoryGrid,
    cache: TrajectoryContactCache,
    *,
    beam_complete_paths: int,
) -> ShapeGraphReport:
    occupancy = occupancy_by_row(grid, cache)
    first_empty = next(
        (item.row for item in occupancy if item.n_admissible == 0),
        None,
    )
    reachable, starts = reachable_from_opening(grid, cache)
    max_row = max((row for row, _col in reachable), default=None)
    last_row = int(grid.n_rows) - 1
    last_adm = occupancy[last_row].n_admissible if occupancy else 0
    n_adm = sum(item.n_admissible for item in occupancy)
    n_touch = sum(item.n_with_contact for item in occupancy)
    return ShapeGraphReport(
        family_id=str(family_id),
        n_admissible=n_adm,
        n_with_contact=n_touch,
        admissible_per_row=occupancy,
        first_row_without_admissible=first_empty,
        n_successive_row_transitions=successive_row_transitions(grid, cache),
        n_starts=len(starts),
        n_reachable_from_opening=len(reachable),
        max_reachable_row=max_row,
        last_row=last_row,
        last_row_admissible=last_adm,
        extinction=_extinction(grid, cache, occupancy, reachable, starts),
        beam_complete_paths=int(beam_complete_paths),
    )


def pose_contact_spans(
    grid: TrajectoryGrid,
    cache: TrajectoryContactCache,
    *,
    limit: int | None = None,
) -> tuple[PoseContactSpan, ...]:
    index_to_cell = {int(cell.index): cell for cell in grid.cells}
    spans: list[PoseContactSpan] = []
    for cell in grid.cells:
        entry = cache.entry_at(cell.row, cell.col)
        if entry is None or not entry.admissible or int(entry.covered_count) < 1:
            continue
        indices = mask_indices(int(entry.covered_mask), int(cache.n_points))
        covered_cells = [
            index_to_cell[index] for index in indices if index in index_to_cell
        ]
        rows = tuple(sorted({int(item.row) for item in covered_cells}))
        ys = [float(item.y_mm) for item in covered_cells]
        spans.append(
            PoseContactSpan(
                row=int(cell.row),
                col=int(cell.col),
                origin_mm=tuple(float(v) for v in entry.origin_mm),
                yaw_deg=float(entry.yaw_deg),
                covered_count=int(entry.covered_count),
                covered_indices=indices,
                covered_rows=rows,
                touched_y_min_mm=min(ys) if ys else float(cell.y_mm),
                touched_y_max_mm=max(ys) if ys else float(cell.y_mm),
                n_rows_covered=len(rows),
            )
        )
    spans.sort(key=lambda item: (-item.covered_count, item.row, item.col))
    if limit is not None:
        spans = spans[: int(limit)]
    return tuple(spans)


def height_monotone_pose_chain(
    spans: tuple[PoseContactSpan, ...],
    *,
    max_poses: int = 24,
    min_drop_mm: float = 1.0,
) -> tuple[PoseContactSpan, ...]:
    """Chain a few high-contact poses by decreasing origin Y. Not a cell-row DAG."""
    if not spans:
        return ()
    ranked = sorted(spans, key=lambda item: -item.covered_count)
    seed = max(ranked[: min(8, len(ranked))], key=lambda item: item.origin_mm[1])
    chain = [seed]
    remaining = [item for item in ranked if item is not seed]
    remaining.sort(key=lambda item: -item.origin_mm[1])
    while remaining and len(chain) < int(max_poses):
        prev_y = chain[-1].origin_mm[1]
        nxt = next(
            (
                item
                for item in remaining
                if item.origin_mm[1] <= prev_y - float(min_drop_mm)
            ),
            None,
        )
        if nxt is None:
            break
        chain.append(nxt)
        remaining = [item for item in remaining if item is not nxt]
    return tuple(chain)


def union_indices(spans: tuple[PoseContactSpan, ...]) -> tuple[int, ...]:
    found: set[int] = set()
    for item in spans:
        found.update(item.covered_indices)
    return tuple(sorted(found))


def report_to_payload(report: ShapeGraphReport) -> dict[str, Any]:
    cut = report.extinction
    return {
        "forme": report.family_id,
        "cellules_admissibles": report.n_admissible,
        "cellules_avec_contact": report.n_with_contact,
        "admissibles_par_rangee": [
            {
                "row": item.row,
                "n_cells": item.n_cells,
                "n_admissible": item.n_admissible,
                "n_with_contact": item.n_with_contact,
                "y_min_mm": item.y_min_mm,
                "y_max_mm": item.y_max_mm,
            }
            for item in report.admissible_per_row
        ],
        "premiere_rangee_sans_admissible": report.first_row_without_admissible,
        "transitions_rangees_successives": report.n_successive_row_transitions,
        "departs_ouverture": report.n_starts,
        "chemins_partiels_atteignables_cellules": report.n_reachable_from_opening,
        "profondeur_max_row": report.max_reachable_row,
        "derniere_rangee": report.last_row,
        "admissibles_derniere_rangee": report.last_row_admissible,
        "extinction": {
            "kind": cut.kind,
            "row": cut.row,
            "detail": cut.detail,
            "cells": [list(item) for item in cut.cells],
        },
        "trajets_beam_complets": report.beam_complete_paths,
        "mapping_cellule_egale_pose": report.mapping_is_cell_equals_pose,
        "max_downward_step": report.max_downward_step,
        "max_lateral_step": report.max_lateral_step,
        "max_downward_step_means": report.max_downward_step_means,
    }


def cache_entries_payload(
    grid: TrajectoryGrid,
    cache: TrajectoryContactCache,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in grid.cells:
        entry = cache.entry_at(cell.row, cell.col)
        if entry is None:
            continue
        rows.append(_entry_payload(cell, entry, int(cache.n_points)))
    return rows


def _entry_payload(
    cell: TrajectoryCell,
    entry: CellContactEntry,
    n_points: int,
) -> dict[str, Any]:
    indices = (
        mask_indices(int(entry.covered_mask), n_points) if entry.admissible else ()
    )
    return {
        "index": int(cell.index),
        "row": int(cell.row),
        "col": int(cell.col),
        "x_mm": float(cell.x_mm),
        "y_mm": float(cell.y_mm),
        "z_mm": float(cell.z_mm),
        "azimuth_deg": float(cell.azimuth_deg),
        "is_top": bool(cell.is_top),
        "admissible": bool(entry.admissible),
        "covered_count": int(entry.covered_count),
        "covered_indices": list(indices),
        "origin_mm": [float(v) for v in entry.origin_mm],
        "yaw_deg": float(entry.yaw_deg),
        "neighborhood_used": bool(entry.neighborhood_used),
        "physics_queries": int(entry.physics_queries),
        "covered_mask": int(entry.covered_mask),
    }


def cache_from_payload(
    grid: TrajectoryGrid,
    rows: list[dict[str, Any]],
    *,
    n_points: int,
    fingerprint: str,
) -> TrajectoryContactCache:
    by_key = {(int(item["row"]), int(item["col"])): item for item in rows}
    entries: list[CellContactEntry] = []
    queries = 0
    for cell in grid.cells:
        item = by_key.get((cell.row, cell.col))
        if item is None:
            entries.append(
                CellContactEntry(
                    point_index=int(cell.index),
                    row=int(cell.row),
                    col=int(cell.col),
                    yaw_deg=float(cell.azimuth_deg),
                    origin_mm=(float(cell.x_mm), float(cell.y_mm), float(cell.z_mm)),
                    admissible=False,
                    neighborhood_used=False,
                    covered_mask=0,
                    covered_count=0,
                    physics_queries=0,
                )
            )
            continue
        mask = int(item.get("covered_mask", 0))
        if not mask and item.get("covered_indices"):
            mask = 0
            for index in item["covered_indices"]:
                mask |= 1 << int(index)
        queries += int(item.get("physics_queries", 0))
        origin = item.get("origin_mm") or [cell.x_mm, cell.y_mm, cell.z_mm]
        entries.append(
            CellContactEntry(
                point_index=int(cell.index),
                row=int(cell.row),
                col=int(cell.col),
                yaw_deg=float(item.get("yaw_deg", cell.azimuth_deg)),
                origin_mm=(float(origin[0]), float(origin[1]), float(origin[2])),
                admissible=bool(item.get("admissible", False)),
                neighborhood_used=bool(item.get("neighborhood_used", False)),
                covered_mask=int(mask),
                covered_count=int(item.get("covered_count", int(mask).bit_count())),
                physics_queries=int(item.get("physics_queries", 0)),
            )
        )
    return TrajectoryContactCache(
        target_definition=str(grid.target_definition),
        n_points=int(n_points),
        point_face_ids=tuple(0 for _ in range(int(n_points))),
        entries=tuple(entries),
        scraper_fingerprint=str(fingerprint),
        physics_queries=int(queries),
        uses_legacy_a0_point_matrix=False,
        angle_window_deg=tuple(grid.angle_range_deg),
        symmetry_multiplier_applied=False,
    )
