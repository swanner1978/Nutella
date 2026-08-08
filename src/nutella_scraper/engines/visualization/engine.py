"""Visualization engine facade."""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from dataclasses import replace
from typing import Any

from nutella_scraper.domain.models.canonical import CanonicalModel3D, JarCanonicalModel
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.domain.models.contact import ContactResult
from nutella_scraper.domain.models.views import (
    RenderedFrame,
    ViewOverlayPayload,
    ViewProjectionCache,
)
from nutella_scraper.engines.visualization.contact_metrics_panel import ContactMetricsPanel
from nutella_scraper.engines.visualization.contact_result_projector import ContactResultProjector
from nutella_scraper.engines.visualization.overlay_renderer import OverlayRenderer
from nutella_scraper.engines.visualization.view_projection_generator import ViewProjectionGenerator


class VisualizationEngine:
    """
    Visualization-only engine — projects 3D results to 2D views.

    Never computes contact metrics or optimization scores.
    """

    def __init__(
        self,
        view_generator: ViewProjectionGenerator | None = None,
        contact_projector: ContactResultProjector | None = None,
        overlay_renderer: OverlayRenderer | None = None,
    ) -> None:
        self._view_generator = view_generator or ViewProjectionGenerator()
        self._contact_projector = contact_projector or ContactResultProjector()
        self._overlay_renderer = overlay_renderer or OverlayRenderer()

    def generate_views(self, model: CanonicalModel3D) -> ViewProjectionCache:
        return self._view_generator.generate(model)

    def project_contact(
        self,
        contact: ContactResult,
        views: ViewProjectionCache,
        jar: JarCanonicalModel | CanonicalModel3D,
        *,
        internal: InternalJarSurface | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> ViewOverlayPayload:
        return self._contact_projector.project(
            contact,
            views,
            jar,
            internal=internal,
            profile=profile,
        )

    def render_frame(
        self,
        views: ViewProjectionCache,
        overlay: ViewOverlayPayload,
    ) -> RenderedFrame:
        return self._overlay_renderer.render(views, overlay)

    def build_contact_visualization(
        self,
        contact: ContactResult,
        views: ViewProjectionCache,
        jar: JarCanonicalModel | CanonicalModel3D,
        *,
        internal: InternalJarSurface | None = None,
        simulation_duration_ms: float | None = None,
        overlay_profile: MutableMapping[str, Any] | None = None,
    ) -> tuple[ViewOverlayPayload, ContactMetricsPanel, dict[str, dict[str, str]]]:
        """
        Project contact overlays and extract the metrics panel in one call.

        Optional ``simulation_duration_ms`` is stored in a copied ContactResult
        diagnostics dict so the panel remains sourced from ContactResult only.
        """
        contact_for_panel = contact
        if simulation_duration_ms is not None:
            contact_for_panel = replace(
                contact,
                diagnostics={
                    **contact.diagnostics,
                    "simulation_duration_ms": simulation_duration_ms,
                },
            )

        overlay = self.project_contact(
            contact,
            views,
            jar,
            internal=internal,
            profile=overlay_profile,
        )
        metrics = ContactMetricsPanel.from_contact_result(contact_for_panel)
        wrapping_started = time.perf_counter()
        fragments = self._overlay_renderer.layer_fragments(overlay)
        if overlay_profile is not None:
            overlay_profile["fragment_wrapping_ms"] = (
                time.perf_counter() - wrapping_started
            ) * 1000.0
            overlay_profile["payload_bytes"] = sum(
                len(fragment.encode("utf-8"))
                for view_fragments in fragments.values()
                for fragment in view_fragments.values()
            )
        return overlay, metrics, fragments
