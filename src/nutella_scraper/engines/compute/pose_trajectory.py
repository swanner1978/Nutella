"""Heuristic beam search on a pose graph. Not a cloud-row DAG.

Each node is a scraper pose. Each edge is a physically admissible motion
(mm / deg). Coverage is UNION of target-point masks along the path.

Label: HEURISTIC. Result = meilleure trajectoire trouvée, never a global optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
)
from nutella_scraper.engines.compute.pose_contact_cache import (
    PoseContactCache,
    PoseContactEntry,
    mask_indices,
)
from nutella_scraper.engines.compute.pose_space import (
    TARGET_MATRIX,
    TRAJECTORY_MODEL,
    PoseMotionLimits,
    transition_allowed,
)

BEAM_WIDTH = 48
TOP_K = 10
OPTIMIZATION_METHOD = "beam_search"
OPTIMIZATION_LABEL = "HEURISTIC"
DISCLAIMER = (
    "Nous ne pouvons pas énumérer toutes les trajectoires. "
    "Résultat : meilleure trajectoire trouvée (HEURISTIC), pas un optimum global."
)


@dataclass(frozen=True)
class PoseBeamState:
    pose_id: int
    path: tuple[int, ...]
    covered_mask: int
    covered_count: int
    path_length_mm: float
    downward_moves: int
    lateral_moves: int
    rotation_changes: int
    last_kind: str


@dataclass(frozen=True)
class PoseTrajectory:
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
    rotation_changes: int
    path_length_mm: float
    max_depth_reached_mm: float
    start_y_mm: float
    path: tuple[PoseContactEntry, ...]
    n_edges: int
    optimization_label: str = OPTIMIZATION_LABEL
    optimization_method: str = OPTIMIZATION_METHOD
    trajectory_model: str = TRAJECTORY_MODEL
    target_matrix: str = TARGET_MATRIX
    beam_trajectories_explored: int = 0
    termination_reason: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "coverage_percent": self.coverage_percent,
            "covered_points": self.covered_points,
            "total_points": self.total_points,
            "covered_point_indices": list(self.covered_point_indices),
            "trajectory_id": self.trajectory_id,
            "number_of_poses": self.position_count,
            "downward_moves": self.downward_moves,
            "lateral_moves": self.lateral_moves,
            "rotation_changes": self.rotation_changes,
            "trajectory_length_mm": self.path_length_mm,
            "max_depth_reached_mm": self.max_depth_reached_mm,
            "start_y_mm": self.start_y_mm,
            "optimization_label": self.optimization_label,
            "optimization_method": self.optimization_method,
            "trajectory_model": self.trajectory_model,
            "target_matrix": self.target_matrix,
            "termination_reason": self.termination_reason,
            "disclaimer": DISCLAIMER,
            "poses": [
                {
                    "pose_id": int(item.pose_id),
                    "y_mm": float(item.y_mm),
                    "azimuth_deg": float(item.azimuth_deg),
                    "origin_mm": list(item.origin_mm),
                    "yaw_deg": float(item.yaw_deg),
                    "covered_count": int(item.covered_count),
                }
                for item in self.path
            ],
        }


def _distance_mm(src: PoseContactEntry, dst: PoseContactEntry) -> float:
    dx = float(dst.origin_mm[0]) - float(src.origin_mm[0])
    dy = float(dst.origin_mm[1]) - float(src.origin_mm[1])
    dz = float(dst.origin_mm[2]) - float(src.origin_mm[2])
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _move_kind(src: PoseContactEntry, dst: PoseContactEntry) -> str:
    if dst.y_mm < src.y_mm - 1e-9:
        return "down"
    return "lateral"


def pose_transition_allowed(
    src: PoseContactEntry,
    dst: PoseContactEntry,
    limits: PoseMotionLimits,
) -> bool:
    if src.pose_id == dst.pose_id:
        return False
    if not src.admissible or not dst.admissible:
        return False
    return transition_allowed(
        src.y_mm,
        src.azimuth_deg,
        src.origin_mm,
        dst.y_mm,
        dst.azimuth_deg,
        dst.origin_mm,
        limits,
    )


def build_pose_edges(
    cache: PoseContactCache,
    limits: PoseMotionLimits,
) -> tuple[tuple[int, int], ...]:
    """Admissible pose→pose edges. Cloud row indices are not consulted."""
    admissible = cache.admissible_entries()
    edges: list[tuple[int, int]] = []
    for src in admissible:
        for dst in admissible:
            if pose_transition_allowed(src, dst, limits):
                edges.append((int(src.pose_id), int(dst.pose_id)))
    return tuple(edges)


def opening_pose_ids(
    cache: PoseContactCache,
    limits: PoseMotionLimits,
) -> tuple[int, ...]:
    return tuple(
        int(item.pose_id)
        for item in cache.admissible_entries()
        if limits.is_opening_pose(item.y_mm)
    )


def _neighbors_map(
    cache: PoseContactCache,
    edges: tuple[tuple[int, int], ...],
) -> dict[int, tuple[int, ...]]:
    buckets: dict[int, list[int]] = {int(item.pose_id): [] for item in cache.entries}
    for src, dst in edges:
        buckets.setdefault(int(src), []).append(int(dst))
    return {key: tuple(values) for key, values in buckets.items()}


def _is_terminal(
    pose: PoseContactEntry,
    neighbors: tuple[int, ...],
    cache: PoseContactCache,
    limits: PoseMotionLimits,
) -> bool:
    if limits.reached_useful_depth(pose.y_mm):
        return True
    for nid in neighbors:
        nxt = cache.entry_at(nid)
        if nxt is not None and nxt.y_mm < pose.y_mm - 1e-9:
            return False
    return True


def _rank_key(state: PoseBeamState, cache: PoseContactCache) -> tuple[int, float, int]:
    """Coverage first. Depth and pose count are tie-breaks, not a weighted mix."""
    pose = cache.entry_at(state.pose_id)
    depth = float(pose.y_mm) if pose is not None else 0.0
    return (int(state.covered_count), -depth, -int(len(state.path)))


def _expand(
    state: PoseBeamState,
    dst: PoseContactEntry,
    src: PoseContactEntry,
) -> PoseBeamState:
    kind = _move_kind(src, dst)
    covered = int(state.covered_mask) | int(dst.covered_mask)
    rotation = 0
    if kind != state.last_kind and state.last_kind not in {"start"}:
        rotation = 1
    return PoseBeamState(
        pose_id=int(dst.pose_id),
        path=state.path + (int(dst.pose_id),),
        covered_mask=covered,
        covered_count=covered.bit_count(),
        path_length_mm=state.path_length_mm + _distance_mm(src, dst),
        downward_moves=state.downward_moves + (1 if kind == "down" else 0),
        lateral_moves=state.lateral_moves + (0 if kind == "down" else 1),
        rotation_changes=state.rotation_changes + rotation,
        last_kind=kind,
    )


def beam_search_pose_trajectories(
    cache: PoseContactCache,
    limits: PoseMotionLimits,
    *,
    edges: tuple[tuple[int, int], ...] | None = None,
    beam_width: int = BEAM_WIDTH,
    top_k: int = TOP_K,
) -> tuple[PoseTrajectory, ...]:
    """Keep the best beam_width partial pose paths. HEURISTIC, not an optimum."""
    if cache.target_definition != COVERAGE_TARGET_REGION:
        raise ValueError("Pose beam target must be the validated interior matrix")
    if cache.uses_legacy_a0_point_matrix:
        raise ValueError("A0 point matrix cannot be used as the target cloud")
    if cache.symmetry_multiplier_applied:
        raise ValueError("Symmetry multipliers are forbidden")
    if int(cache.n_points) < 1:
        raise ValueError("Contact cache has no target points")
    graph = edges if edges is not None else build_pose_edges(cache, limits)
    neighbors = _neighbors_map(cache, graph)
    starts = opening_pose_ids(cache, limits)
    if not starts:
        return ()
    beam: list[PoseBeamState] = []
    for pose_id in starts:
        entry = cache.entry_at(pose_id)
        if entry is None or not entry.admissible:
            continue
        mask = int(entry.covered_mask)
        beam.append(
            PoseBeamState(
                pose_id=int(pose_id),
                path=(int(pose_id),),
                covered_mask=mask,
                covered_count=mask.bit_count(),
                path_length_mm=0.0,
                downward_moves=0,
                lateral_moves=0,
                rotation_changes=0,
                last_kind="start",
            )
        )
    if not beam:
        return ()
    width = max(1, int(beam_width))
    beam = sorted(beam, key=lambda state: _rank_key(state, cache), reverse=True)[:width]
    completed: list[PoseBeamState] = []
    seen: set[tuple[int, int]] = set()
    max_steps = len(cache.admissible_entries()) + 2
    for _ in range(max_steps):
        nxt: list[PoseBeamState] = []
        progressed = False
        for state in beam:
            pose = cache.entry_at(state.pose_id)
            if pose is None:
                continue
            dests = neighbors.get(state.pose_id, ())
            if _is_terminal(pose, dests, cache, limits):
                completed.append(state)
            visited = set(state.path)
            for nid in dests:
                if nid in visited:
                    continue
                dst = cache.entry_at(nid)
                if dst is None or not dst.admissible:
                    continue
                child = _expand(state, dst, pose)
                key = (child.pose_id, child.covered_mask)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(child)
                progressed = True
        if not nxt:
            for state in beam:
                if state not in completed:
                    completed.append(state)
            break
        beam = sorted(nxt, key=lambda state: _rank_key(state, cache), reverse=True)[:width]
        if not progressed:
            break
    unique: dict[tuple[int, ...], PoseBeamState] = {}
    for state in completed:
        prev = unique.get(state.path)
        if prev is None or _rank_key(state, cache) > _rank_key(prev, cache):
            unique[state.path] = state
    ranked_states = sorted(
        unique.values(),
        key=lambda state: _rank_key(state, cache),
        reverse=True,
    )[: max(1, int(top_k))]
    n_points = int(cache.n_points)
    explored = len(unique)
    out: list[PoseTrajectory] = []
    for rank, state in enumerate(ranked_states, start=1):
        path = tuple(
            cache.entry_at(pid)  # type: ignore[misc]
            for pid in state.path
            if cache.entry_at(pid) is not None
        )
        path = tuple(item for item in path if item is not None)
        last_y = float(path[-1].y_mm) if path else 0.0
        start_y = float(path[0].y_mm) if path else 0.0
        covered = int(state.covered_count)
        percent = 0.0 if n_points <= 0 else 100.0 * covered / n_points
        path_id = "-".join(str(pid) for pid in state.path)
        out.append(
            PoseTrajectory(
                rank=int(rank),
                coverage_percent=float(percent),
                covered_points=covered,
                total_points=n_points,
                covered_mask=int(state.covered_mask),
                covered_point_indices=mask_indices(int(state.covered_mask), n_points),
                trajectory_id=f"T{rank:02d}:{path_id}",
                position_count=len(state.path),
                downward_moves=int(state.downward_moves),
                lateral_moves=int(state.lateral_moves),
                rotation_changes=int(state.rotation_changes),
                path_length_mm=float(state.path_length_mm),
                max_depth_reached_mm=last_y,
                start_y_mm=start_y,
                path=path,
                n_edges=len(graph),
                beam_trajectories_explored=int(explored),
                termination_reason=_termination_reason(path, neighbors, cache, limits),
            )
        )
    return tuple(out)


def reachable_pose_ids(
    cache: PoseContactCache,
    limits: PoseMotionLimits,
    *,
    edges: tuple[tuple[int, int], ...] | None = None,
) -> tuple[int, ...]:
    starts = opening_pose_ids(cache, limits)
    if not starts:
        return ()
    graph = edges if edges is not None else build_pose_edges(cache, limits)
    neighbors = _neighbors_map(cache, graph)
    seen: set[int] = set(starts)
    stack = list(starts)
    while stack:
        current = stack.pop()
        for nid in neighbors.get(current, ()):
            if nid in seen:
                continue
            seen.add(int(nid))
            stack.append(int(nid))
    return tuple(sorted(seen))


def _termination_reason(
    path: tuple[PoseContactEntry, ...],
    neighbors: dict[int, tuple[int, ...]],
    cache: PoseContactCache,
    limits: PoseMotionLimits,
) -> str:
    if not path:
        return "no_opening_start"
    last = path[-1]
    dests = neighbors.get(int(last.pose_id), ())
    if limits.reached_useful_depth(last.y_mm):
        return "reached_useful_depth"
    has_down = False
    for nid in dests:
        nxt = cache.entry_at(nid)
        if nxt is not None and nxt.y_mm < last.y_mm - 1e-9:
            has_down = True
            break
    if not has_down:
        return "no_downward_successor"
    return "beam_exhausted"


def assert_path_physical(
    path: tuple[PoseContactEntry, ...],
    limits: PoseMotionLimits,
) -> None:
    if not path:
        raise ValueError("empty pose path")
    if not limits.is_opening_pose(path[0].y_mm):
        raise ValueError("trajectory does not start in the opening band")
    for src, dst in zip(path[:-1], path[1:], strict=True):
        if not pose_transition_allowed(src, dst, limits):
            raise ValueError(
                f"non-physical step {src.pose_id}->{dst.pose_id} "
                f"dy={dst.y_mm - src.y_mm:.3f}"
            )
        if dst.y_mm > src.y_mm + 1e-9:
            raise ValueError("trajectory climbs")
