"""Visualization engine protocol."""

from __future__ import annotations

from typing import Protocol

from nutella_scraper.domain.models.canonical import CanonicalModel3D, JarCanonicalModel
from nutella_scraper.domain.models.contact import ContactResult
from nutella_scraper.domain.models.views import (
    RenderedFrame,
    ViewOverlayPayload,
    ViewProjectionCache,
)


class IVisualizationEngine(Protocol):
    """Contract for visualization — projection only, no metric computation."""

    def generate_views(self, model: CanonicalModel3D) -> ViewProjectionCache:
        """Generate profile and top views from canonical 3D model."""
        ...

    def project_contact(
        self,
        contact: ContactResult,
        views: ViewProjectionCache,
        jar: JarCanonicalModel,
    ) -> ViewOverlayPayload:
        """Project 3D contact result onto 2D views — read-only display."""
        ...

    def render_frame(
        self,
        views: ViewProjectionCache,
        overlay: ViewOverlayPayload,
    ) -> RenderedFrame:
        """Compose final rendered frame for UI."""
        ...
