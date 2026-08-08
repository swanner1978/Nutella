"""Composes SVG/canvas overlay layers for UI."""

from __future__ import annotations

import re

from nutella_scraper.domain.models.views import (
    RenderedFrame,
    SvgLayer,
    ViewOverlayPayload,
    ViewProjectionCache,
)
from nutella_scraper.engines.visualization.contact_result_projector import (
    LAYER_COLLISION_FACES,
    LAYER_COLLISION_POINTS,
    LAYER_CONTACT_COVERED,
    LAYER_CONTACT_POINTS,
    LAYER_CONTACT_UNCOVERED,
    LAYER_DISTANCE_MAP,
)

_CONTACT_LAYER_TYPES = frozenset(
    {
        LAYER_CONTACT_COVERED,
        LAYER_CONTACT_UNCOVERED,
        LAYER_DISTANCE_MAP,
        LAYER_CONTACT_POINTS,
        LAYER_COLLISION_FACES,
        LAYER_COLLISION_POINTS,
    }
)
_LAYER_GROUP_PATTERN = re.compile(
    r'<g[^>]*data-layer="(?:'
    + "|".join(re.escape(layer) for layer in _CONTACT_LAYER_TYPES)
    + r')"[^>]*>.*?</g>',
    re.DOTALL,
)


class OverlayRenderer:
    """Renders composed frames with black-background-ready SVG layers."""

    def render(
        self,
        views: ViewProjectionCache,
        overlay: ViewOverlayPayload,
    ) -> RenderedFrame:
        profile_svg = self._compose_view(
            base_svg=views.profile_view.svg_content,
            layers=overlay.profile_layers,
        )
        top_svg = self._compose_view(
            base_svg=views.top_view.svg_content,
            layers=overlay.top_layers,
        )
        return RenderedFrame(
            profile_svg=profile_svg,
            top_svg=top_svg,
            coverage_score_display=overlay.coverage_score_display,
        )

    def layer_fragments(self, overlay: ViewOverlayPayload) -> dict[str, dict[str, str]]:
        """Return injectable overlay fragments keyed by view and layer type."""
        return {
            "profile": {
                layer.layer_type: self._wrap_layer(layer)
                for layer in overlay.profile_layers
            },
            "top": {
                layer.layer_type: self._wrap_layer(layer)
                for layer in overlay.top_layers
            },
        }

    def _compose_view(
        self,
        *,
        base_svg: str | None,
        layers: tuple[SvgLayer, ...],
    ) -> str:
        if base_svg is None:
            raise ValueError("ViewProjectionCache view is missing svg_content")
        cleaned = _strip_existing_contact_layers(base_svg)
        if not layers:
            return cleaned
        injection = "".join(
            self._wrap_layer(layer) for layer in sorted(layers, key=lambda item: item.z_index)
        )
        return _inject_before_closing_svg(cleaned, injection)

    @staticmethod
    def _wrap_layer(layer: SvgLayer) -> str:
        return (
            f'<g id="{layer.id}" data-layer="{layer.layer_type}" '
            f'data-z-index="{layer.z_index}">{layer.svg_fragment}</g>'
        )


def _strip_existing_contact_layers(svg_content: str) -> str:
    return _LAYER_GROUP_PATTERN.sub("", svg_content)


def _inject_before_closing_svg(svg_content: str, injection: str) -> str:
    closing = svg_content.rfind("</svg>")
    if closing == -1:
        raise ValueError("Invalid SVG document: missing </svg>")
    return svg_content[:closing] + injection + svg_content[closing:]
