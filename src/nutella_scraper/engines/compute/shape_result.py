"""Immutable shape-search results. Metrics stay separate — no weighted blend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

OPTIMIZATION_METHOD = "differential_evolution_multistart"
OPTIMIZATION_LABEL = "HEURISTIC"
DISCLAIMER = (
    "HEURISTIC. Nous ne pouvons pas garantir un optimum mathématique. "
    "Meilleure trajectoire trouvée dans le budget, pas un optimum global."
)


@dataclass(frozen=True)
class ShapeCandidate:
    candidate_id: str
    family_id: str
    parameters: tuple[float, ...]
    n_parameters: int
    coverage_percent: float
    covered_points: int
    total_points: int
    covered_point_indices: tuple[int, ...]
    untouched_point_indices: tuple[int, ...]
    mean_geometric_error_mm: float
    max_geometric_error_mm: float
    scraper_length_mm: float
    min_curvature_radius_mm: float
    trajectory_steps: int
    trajectory_length_mm: float
    lateral_changes: int
    direction_changes: int
    geometric_valid: bool
    physical_valid: bool
    geometric_reasons: tuple[str, ...]
    profile_points_mm: NDArray[np.float64]
    optimization_label: str = OPTIMIZATION_LABEL
    optimization_method: str = OPTIMIZATION_METHOD
    trajectory_id: str = ""
    shape_fingerprint: str = ""
    scraper_fingerprint: str = ""
    poses_evaluated: int = 0
    beam_trajectories_explored: int = 0
    elapsed_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    trajectory_rows_cols: tuple[tuple[int, int], ...] = ()
    trajectory_poses: tuple[tuple[float, float], ...] = ()
    trajectory_origins: tuple[tuple[float, float, float], ...] = ()
    trajectory_model: str = "POSE_GRAPH"
    thickness_mm: float = 2.0
    width_mm: float = 2.0
    max_depth_reached_mm: float = 0.0
    n_pose_candidates: int = 0
    n_admissible_poses: int = 0
    n_contacting_poses: int = 0
    n_reachable_poses: int = 0
    trajectory_found: bool = False
    opening_start_available: bool = False
    floor_reached: bool = False
    termination_reason: str = ""

    def rank_tuple(self) -> tuple[Any, ...]:
        """Coverage UNION first. Path length then turns then complexity. No blend."""
        return (
            -int(self.covered_points),
            float(self.trajectory_length_mm),
            int(self.direction_changes),
            int(self.n_parameters),
            str(self.family_id),
            str(self.candidate_id),
        )


@dataclass
class SearchStats:
    shapes_generated: int = 0
    physics_simulations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    time_per_family_s: dict[str, float] = field(default_factory=dict)
    total_time_s: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "shapes_generated": int(self.shapes_generated),
            "physics_simulations": int(self.physics_simulations),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "time_per_family_s": dict(self.time_per_family_s),
            "total_time_s": float(self.total_time_s),
        }
