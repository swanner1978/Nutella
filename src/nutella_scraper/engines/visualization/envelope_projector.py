"""Orthographic projection of the CAD B-Rep interior reference geometry."""

from __future__ import annotations

from dataclasses import dataclass

from nutella_scraper.cad_import.brep_contour_extractor import offset_contour
from nutella_scraper.domain.models.cad_reference_geometry import CadReferenceGeometry
from nutella_scraper.domain.models.envelope import InteriorEnvelope
from nutella_scraper.domain.models.views import SvgLayer
from nutella_scraper.engines.visualization.cad_reference_projector import (
    LAYER_INTERIOR_PROFILE,
    contour_layers_for_plane,
)

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
    ) -> EnvelopeProjection:
        return self._project_geometry(geometry, clearance_mm=envelope.clearance_mm)

    def project_geometry(self, geometry: CadReferenceGeometry) -> EnvelopeProjection:
        """Project the cavity contour without clearance offset."""
        return self._project_geometry(geometry, clearance_mm=0.0)

    def _project_geometry(
        self,
        geometry: CadReferenceGeometry,
        *,
        clearance_mm: float,
    ) -> EnvelopeProjection:
        profile = geometry.profile_contour
        top = geometry.top_contour
        if profile is None or top is None:
            return EnvelopeProjection(profile_layers=(), top_layers=())

        if clearance_mm > 0.0:
            profile = offset_contour(profile, clearance_mm)
            top = offset_contour(top, clearance_mm)

        return EnvelopeProjection(
            profile_layers=contour_layers_for_plane(profile, view_key="profile"),
            top_layers=contour_layers_for_plane(top, view_key="top"),
        )
