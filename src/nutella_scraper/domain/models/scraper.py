"""Parametric 3D scraper model — solid volume, not a 2D curve."""

from __future__ import annotations

from dataclasses import dataclass, field

from nutella_scraper.domain.models.common import Provenance


@dataclass(frozen=True)
class ScraperGeometry:
    """
    Intrinsic racloir shape — independent of placement in the jar.

    Describes a true 3D solid volume, never a polyline or 2D curve.
    """

    width_mm: float
    length_mm: float
    thickness_mm: float
    tip_radius_mm: float = 1.5
    curvature_radius_mm: float | None = None
    bend_angle_deg: float = 0.0
    id: str = "scraper"
    provenance: Provenance = "canonical_3d"
    metadata: dict[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.length_mm <= 0 or self.thickness_mm <= 0:
            raise ValueError("width_mm, length_mm and thickness_mm must be positive")
        if self.tip_radius_mm < 0:
            raise ValueError("tip_radius_mm must be non-negative")
        if self.curvature_radius_mm is not None and self.curvature_radius_mm <= 0:
            raise ValueError("curvature_radius_mm must be positive when set")


@dataclass(frozen=True)
class ScraperPose:
    """Placement of a scraper geometry inside the jar frame."""

    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
