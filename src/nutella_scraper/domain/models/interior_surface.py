"""Interior surface representation for contact simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InteriorSurfaceSlice:
    """Inner radius profile at one jar height."""

    y_mm: float
    inner_radius_mm: float


@dataclass(frozen=True)
class InteriorSurface:
    """
    Optimized inner-wall sampling derived from CanonicalModel3D.

    Used exclusively by ContactSimulationEngine — never from SVG projections.
    """

    jar_id: str
    y_min_mm: float
    y_max_mm: float
    slices: tuple[InteriorSurfaceSlice, ...]
    sample_points_mm: tuple[tuple[float, float, float], ...]
    sample_areas_mm2: tuple[float, ...]
    source_face_count: int
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.sample_points_mm)

    def inner_radius_at(self, y_mm: float) -> float:
        if not self.slices:
            return 0.0
        if y_mm <= self.slices[0].y_mm:
            return self.slices[0].inner_radius_mm
        if y_mm >= self.slices[-1].y_mm:
            return self.slices[-1].inner_radius_mm
        for left, right in zip(self.slices, self.slices[1:], strict=False):
            if left.y_mm <= y_mm <= right.y_mm:
                span = right.y_mm - left.y_mm
                if span <= 1e-9:
                    return left.inner_radius_mm
                t = (y_mm - left.y_mm) / span
                return left.inner_radius_mm + t * (right.inner_radius_mm - left.inner_radius_mm)
        return self.slices[-1].inner_radius_mm
