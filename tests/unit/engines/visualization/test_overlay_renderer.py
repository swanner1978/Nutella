"""Overlay renderer tests."""

from __future__ import annotations

import pytest

from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.domain.models.views import SvgLayer
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine
from nutella_scraper.engines.visualization.contact_result_projector import (
    LAYER_CONTACT_COVERED,
    ContactResultProjector,
)
from nutella_scraper.engines.visualization.overlay_renderer import OverlayRenderer


class TestOverlayRenderer:
    def test_render_injects_contact_layers(
        self,
        cylindrical_jar_canonical: object,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
        coarse_simulation_config: ContactSimulationConfig,
        sample_view_cache: object,
    ) -> None:
        contact = ContactSimulationEngine().simulate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
            coarse_simulation_config,
        )
        overlay = ContactResultProjector().project(
            contact,
            sample_view_cache,
            cylindrical_jar_canonical,
        )
        frame = OverlayRenderer().render(sample_view_cache, overlay)

        assert f'data-layer="{LAYER_CONTACT_COVERED}"' in frame.profile_svg
        assert f'data-layer="{LAYER_CONTACT_COVERED}"' in frame.top_svg
        assert frame.coverage_score_display == contact.coverage_score

    def test_render_replaces_existing_contact_layers(
        self,
        sample_view_cache: object,
    ) -> None:
        layer = SvgLayer(
            id="profile-contact-covered",
            z_index=10,
            svg_fragment='<path d="M0,0 L1,1"/>',
            layer_type=LAYER_CONTACT_COVERED,
        )
        overlay = type(
            "OverlayStub",
            (),
            {
                "profile_layers": (layer,),
                "top_layers": (layer,),
                "coverage_score_display": 0.5,
            },
        )()
        renderer = OverlayRenderer()
        first = renderer.render(sample_view_cache, overlay)
        second = renderer.render(sample_view_cache, overlay)

        assert first.profile_svg.count(f'data-layer="{LAYER_CONTACT_COVERED}"') == 1
        assert second.profile_svg.count(f'data-layer="{LAYER_CONTACT_COVERED}"') == 1

    def test_render_requires_base_svg(self, sample_view_cache: object) -> None:
        from dataclasses import replace

        broken_cache = replace(
            sample_view_cache,
            profile_view=replace(sample_view_cache.profile_view, svg_content=None),
        )
        overlay = type(
            "OverlayStub",
            (),
            {"profile_layers": (), "top_layers": (), "coverage_score_display": 0.0},
        )()
        with pytest.raises(ValueError, match="svg_content"):
            OverlayRenderer().render(broken_cache, overlay)
