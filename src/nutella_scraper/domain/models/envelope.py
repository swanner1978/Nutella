"""Interior envelope and pose constraint domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PoseRejectionReason(StrEnum):
    """Why a scraper pose was rejected before contact simulation."""

    OUT_OF_ENVELOPE = "out_of_envelope"
    INITIAL_COLLISION = "initial_collision"
    INVALID_ORIENTATION = "invalid_orientation"
    OUTSIDE_JAR = "outside_jar"
    ENTIRELY_OUTSIDE = "entirely_outside"
    CANNOT_INSERT = "cannot_insert"
    CROSSES_WALL = "crosses_wall"


@dataclass(frozen=True)
class EnvelopeSlice:
    """Accessible radial limit at a given jar height."""

    y_mm: float
    max_radial_mm: float


@dataclass(frozen=True)
class InteriorEnvelope:
    """
    Usable interior volume derived from CanonicalModel3D.

    Represents the domain where a scraper may be placed without crossing walls.
    """

    jar_id: str
    y_min_mm: float
    y_max_mm: float
    neck_radius_mm: float
    clearance_mm: float
    slices: tuple[EnvelopeSlice, ...]

    def max_radial_at(self, y_mm: float) -> float:
        if not self.slices:
            return 0.0
        if y_mm <= self.slices[0].y_mm:
            return self.slices[0].max_radial_mm
        if y_mm >= self.slices[-1].y_mm:
            return self.slices[-1].max_radial_mm
        for left, right in zip(self.slices, self.slices[1:], strict=False):
            if left.y_mm <= y_mm <= right.y_mm:
                span = right.y_mm - left.y_mm
                if span <= 1e-9:
                    return left.max_radial_mm
                t = (y_mm - left.y_mm) / span
                return left.max_radial_mm + t * (right.max_radial_mm - left.max_radial_mm)
        return self.slices[-1].max_radial_mm


@dataclass(frozen=True)
class PoseRejection:
    """One rejected trajectory pose with diagnostic context."""

    pose_index: int
    reason: PoseRejectionReason
    detail: str = ""


@dataclass(frozen=True)
class PoseValidationResult:
    """Outcome of validating a single scraper placement."""

    is_valid: bool
    reason: PoseRejectionReason | None = None
    detail: str = ""


@dataclass(frozen=True)
class PoseGenerationResult:
    """Constrained pose generation output for profiling and simulation."""

    accepted_transforms: tuple[Any, ...]
    rejections: tuple[PoseRejection, ...] = ()
    candidate_count: int = 0

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_transforms)

    @property
    def rejected_count(self) -> int:
        return len(self.rejections)


@dataclass(frozen=True)
class PoseConstraintDiagnostics:
    """Aggregated pose constraint profiling for simulation diagnostics."""

    envelope_duration_ms: float = 0.0
    pose_generation_duration_ms: float = 0.0
    candidate_pose_count: int = 0
    accepted_pose_count: int = 0
    rejected_pose_count: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    rejections: tuple[PoseRejection, ...] = ()
