"""Single source of truth for the jar interior cavity geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from nutella_scraper.domain.models.canonical import MeshData


@dataclass(frozen=True)
class InternalJarSurfaceSlice:
    """Inner meridian radius at one jar height (Y axis)."""

    y_mm: float
    inner_radius_mm: float


@dataclass(frozen=True)
class InternalJarSurface:
    """
    Exclusive representation of the interior cavity accessible to the scraper.

    Built once after STEP/STL import. All 2D projections, contact simulation,
    envelope computation, and coverage scoring must consume this object — never
    the raw imported tessellation directly.
    """

    jar_id: str
    canonical_mesh_sha256: str
    mesh: MeshData
    y_min_mm: float
    y_max_mm: float
    slices: tuple[InternalJarSurfaceSlice, ...]
    sample_points_mm: tuple[tuple[float, float, float], ...]
    sample_areas_mm2: tuple[float, ...]
    source_face_count: int
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.sample_points_mm)

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces)

    @property
    def vertex_count(self) -> int:
        return len(self.mesh.vertices)

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

    def wall_distance_along_direction(
        self,
        y_mm: float,
        direction_xz: tuple[float, float],
        *,
        angular_tolerance_deg: float = 25.0,
        half_band_mm: float | None = None,
    ) -> float:
        """
        Distance from the Y axis to the interior wall along a horizontal direction.

        Uses the real cavity mesh (not the cylindrical min-radius slices), so an
        elliptical jar returns the meridian extent in ``direction_xz``.
        """
        dx, dz = float(direction_xz[0]), float(direction_xz[1])
        norm = math.hypot(dx, dz)
        if norm <= 1e-12:
            raise ValueError("direction_xz must be non-zero")
        ux, uz = dx / norm, dz / norm

        vertices = self.mesh.vertices
        if not vertices:
            return float(self.inner_radius_at(y_mm))

        height_span = max(float(self.y_max_mm) - float(self.y_min_mm), 1.0)
        band = float(half_band_mm) if half_band_mm is not None else max(1.5, 0.03 * height_span)

        xs: list[float] = []
        zs: list[float] = []
        for x, y, z in vertices:
            if abs(float(y) - y_mm) <= band:
                xs.append(float(x))
                zs.append(float(z))
        if not xs:
            nearest_y = min(vertices, key=lambda v: abs(float(v[1]) - y_mm))[1]
            for x, y, z in vertices:
                if abs(float(y) - float(nearest_y)) <= band:
                    xs.append(float(x))
                    zs.append(float(z))
        if not xs:
            return float(self.inner_radius_at(y_mm))

        target_angle = math.atan2(uz, ux)
        tol = math.radians(max(angular_tolerance_deg, 1.0))
        projections: list[float] = []
        for x_i, z_i in zip(xs, zs, strict=True):
            angle = math.atan2(z_i, x_i)
            delta = abs((angle - target_angle + math.pi) % (2.0 * math.pi) - math.pi)
            if delta <= tol:
                projections.append(x_i * ux + z_i * uz)
        if not projections:
            projections = [x_i * ux + z_i * uz for x_i, z_i in zip(xs, zs, strict=True)]
        return max(float(self.inner_radius_at(y_mm)), max(projections))
