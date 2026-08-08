"""Interior envelope computation from InternalJarSurface."""

from __future__ import annotations

import numpy as np

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.envelope import EnvelopeSlice, InteriorEnvelope
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.engines.compute.internal_jar_surface_builder import (
    InternalJarSurfaceBuilder,
    internal_mesh_to_trimesh,
)


class EnvelopeBuilder:
    """Derives the usable interior envelope exclusively from InternalJarSurface."""

    def from_internal(
        self,
        surface: InternalJarSurface,
        *,
        clearance_mm: float,
        slice_count: int = 48,
    ) -> InteriorEnvelope:
        mesh = internal_mesh_to_trimesh(surface)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        y_min = surface.y_min_mm
        y_max = surface.y_max_mm
        height = max(y_max - y_min, 1e-6)
        radial = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 2] ** 2)

        neck_threshold = y_max - 0.08 * height
        neck_mask = vertices[:, 1] >= neck_threshold
        if np.any(neck_mask):
            neck_radius = float(np.min(radial[neck_mask]))
        else:
            neck_radius = float(np.min(radial))

        y_samples = np.linspace(y_min, y_max, max(slice_count, 2))
        bin_half = height / max(slice_count * 2, 4)
        slices: list[EnvelopeSlice] = []
        for y_value in y_samples:
            mask = np.abs(vertices[:, 1] - y_value) <= bin_half
            if not np.any(mask):
                continue
            inner_radius = float(np.min(radial[mask]))
            accessible = max(inner_radius - clearance_mm, 0.0)
            slices.append(
                EnvelopeSlice(
                    y_mm=float(y_value),
                    max_radial_mm=accessible,
                )
            )

        if not slices:
            slices.append(EnvelopeSlice(y_mm=y_min, max_radial_mm=0.0))

        return InteriorEnvelope(
            jar_id=surface.jar_id,
            y_min_mm=y_min,
            y_max_mm=y_max,
            neck_radius_mm=max(neck_radius - clearance_mm, 0.0),
            clearance_mm=clearance_mm,
            slices=tuple(slices),
        )

    def from_canonical(
        self,
        jar: CanonicalModel3D,
        *,
        clearance_mm: float,
        slice_count: int = 48,
    ) -> InteriorEnvelope:
        surface = InternalJarSurfaceBuilder().from_canonical(jar)
        return self.from_internal(
            surface,
            clearance_mm=clearance_mm,
            slice_count=slice_count,
        )
