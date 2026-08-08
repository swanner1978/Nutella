"""Contact simulation domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.domain.models.common import Provenance


@dataclass(frozen=True)
class CollisionPoint3D:
    """Single penetration sample on the jar inner surface."""

    position_mm: tuple[float, float, float]
    jar_face_id: int
    penetration_depth_mm: float


@dataclass(frozen=True)
class CollisionResult:
    """
    Geometric interpenetration between scraper volume and jar inner walls.

    Distinct from ContactResult: used to reject physically invalid designs.
    """

    has_collision: bool
    penetration_depth_mm: float
    collision_points: tuple[CollisionPoint3D, ...]
    colliding_face_ids: frozenset[int]


@dataclass(frozen=True)
class ContactPoint3D:
    """Single contact sample on the jar inner surface."""

    position_mm: tuple[float, float, float]
    jar_face_id: int
    distance_mm: float


@dataclass(frozen=True)
class ContactOverlayData:
    """
    Visualization-ready contact payload derived from 3D simulation.

    Must be projected read-only by the visualization engine — never used as
    input to optimization or contact recomputation.
    """

    contact_points: tuple[ContactPoint3D, ...]
    face_coverage: tuple[bool, ...]
    min_distance_per_face_mm: tuple[float, ...]
    scraper_pose_count: int


@dataclass(frozen=True)
class TrajectoryConfig:
    """Scraping trajectory parameters."""

    type: str = "rotational_vertical"
    angular_step_deg: float = 5.0
    vertical_step_mm: float = 2.0


@dataclass(frozen=True)
class ContactSimulationConfig:
    """Configuration for contact simulation between scraper and jar."""

    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    contact_threshold_mm: float = 0.5
    clearance_mm: float = 0.15
    mesh_tolerance_mm: float = 0.1


@dataclass(frozen=True)
class ContactResult:
    """
    Result of 3D contact simulation — computed metric only.

    Must not be derived from 2D view projections.
    """

    model_id: str
    jar_id: str
    coverage_score: float
    touched_face_ids: frozenset[int]
    untouched_face_ids: frozenset[int]
    contact_distance_map: NDArray[np.float64]
    trajectory_pose_count: int = 0
    overlay: ContactOverlayData | None = None
    collision: CollisionResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = "computed_metric"


@dataclass(frozen=True)
class Violation:
    """Constraint violation detected during evaluation."""

    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class EvaluationResult:
    """Full evaluation output from ComputeEngine."""

    contact: ContactResult
    metrics: dict[str, float]
    feasible: bool
    violations: tuple[Violation, ...] = ()
    provenance: Provenance = "computed_metric"
