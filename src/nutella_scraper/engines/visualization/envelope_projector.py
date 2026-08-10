"""Orthographic projection of the CAD B-Rep interior reference geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.cad_import.brep_contour_extractor import (
    PLANE_PROFILE,
    PLANE_TOP_XZ,
    offset_contour,
)
from nutella_scraper.domain.models.cad_reference_geometry import CadReferenceGeometry
from nutella_scraper.domain.models.envelope import InteriorEnvelope
from nutella_scraper.domain.models.views import SvgLayer
from nutella_scraper.engines.visualization.cad_reference_projector import (
    LAYER_INTERIOR_PROFILE,
    contour_layers_for_plane,
)
from nutella_scraper.engines.visualization.projection_math import project_vertices

LAYER_ENVELOPE = LAYER_INTERIOR_PROFILE


@dataclass(frozen=True)
class EnvelopeProjection:
    profile_layers: tuple[SvgLayer, ...]
    top_layers: tuple[SvgLayer, ...]


class EnvelopeProjector:
    """Project the CAD B-Rep inner cavity contour (with optional clearance offset)."""

    def project(
        self,
        envelope: InteriorEnvelope,
        geometry: CadReferenceGeometry,
        *,
        jar_vertices: NDArray[np.float64] | None = None,
    ) -> EnvelopeProjection:
        return self._project_geometry(
            geometry,
            clearance_mm=envelope.clearance_mm,
            jar_vertices=jar_vertices,
        )

    def project_geometry(
        self,
        geometry: CadReferenceGeometry,
        *,
        jar_vertices: NDArray[np.float64] | None = None,
    ) -> EnvelopeProjection:
        """Project the cavity contour without clearance offset."""
        return self._project_geometry(
            geometry,
            clearance_mm=0.0,
            jar_vertices=jar_vertices,
        )

    def _project_geometry(
        self,
        geometry: CadReferenceGeometry,
        *,
        clearance_mm: float,
        jar_vertices: NDArray[np.float64] | None,
    ) -> EnvelopeProjection:
        profile = geometry.profile_contour
        top = geometry.top_contour
        if profile is None or top is None:
            return EnvelopeProjection(profile_layers=(), top_layers=())

        if clearance_mm > 0.0:
            profile = offset_contour(profile, clearance_mm)
            top = offset_contour(top, clearance_mm)

        profile_ref: NDArray[np.float64] | None = None
        top_ref: NDArray[np.float64] | None = None
        if jar_vertices is not None and len(jar_vertices) > 0:
            profile_ref, _, _ = project_vertices(jar_vertices, PLANE_PROFILE)
            top_ref, _, _ = project_vertices(jar_vertices, PLANE_TOP_XZ)

        return EnvelopeProjection(
            profile_layers=contour_layers_for_plane(
                profile,
                view_key="profile",
                reference_coords=profile_ref,
            ),
            top_layers=contour_layers_for_plane(
                top,
                view_key="top",
                reference_coords=top_ref,
            ),
        )
