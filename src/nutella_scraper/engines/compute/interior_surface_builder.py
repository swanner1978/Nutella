"""Deprecated adapter — delegates to InternalJarSurfaceBuilder."""

from __future__ import annotations

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.interior_surface import InteriorSurface, InteriorSurfaceSlice
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.engines.compute.internal_jar_surface_builder import InternalJarSurfaceBuilder
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder


class InteriorSurfaceBuilder:
    """
    Legacy adapter around InternalJarSurfaceBuilder.

    New code must consume InternalJarSurface directly.
    """

    def __init__(self, *, jar_mesh_builder: JarMeshBuilder | None = None) -> None:
        self._internal_builder = InternalJarSurfaceBuilder(
            jar_mesh_builder=jar_mesh_builder or JarMeshBuilder()
        )

    def from_canonical(
        self,
        jar: CanonicalModel3D,
        *,
        slice_count: int = 48,
        angular_samples: int = 72,
        max_samples: int = 4096,
    ) -> InteriorSurface:
        internal = self._internal_builder.from_canonical(
            jar,
            slice_count=slice_count,
            angular_bins=angular_samples,
            max_samples=max_samples,
        )
        return _to_interior_surface(internal)

    def from_internal(self, surface: InternalJarSurface) -> InteriorSurface:
        return _to_interior_surface(surface)


def _to_interior_surface(surface: InternalJarSurface) -> InteriorSurface:
    return InteriorSurface(
        jar_id=surface.jar_id,
        y_min_mm=surface.y_min_mm,
        y_max_mm=surface.y_max_mm,
        slices=tuple(
            InteriorSurfaceSlice(
                y_mm=slice_.y_mm,
                inner_radius_mm=slice_.inner_radius_mm,
            )
            for slice_ in surface.slices
        ),
        sample_points_mm=surface.sample_points_mm,
        sample_areas_mm2=surface.sample_areas_mm2,
        source_face_count=surface.source_face_count,
        metadata={
            **{str(key): value for key, value in surface.metadata.items()},
            "source": "InternalJarSurface",
        },
    )
