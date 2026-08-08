"""End-to-end contact visualization integration tests."""

from __future__ import annotations

import pytest

from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine
from nutella_scraper.engines.visualization.contact_result_projector import (
    ContactResultProjector,
)
from nutella_scraper.engines.visualization.engine import VisualizationEngine


class TestContactVisualizationIntegration:
    def test_simulate_project_render_without_reloading_model(
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
        engine = VisualizationEngine()
        overlay, metrics, fragments = engine.build_contact_visualization(
            contact,
            sample_view_cache,
            cylindrical_jar_canonical,
            simulation_duration_ms=25.0,
        )
        frame = engine.render_frame(sample_view_cache, overlay)

        assert metrics.coverage_score_percent == pytest.approx(contact.coverage_score * 100.0)
        assert metrics.simulation_duration_ms == pytest.approx(25.0)
        assert "contact-covered" in fragments["profile"]
        assert "contact-covered" in fragments["top"]
        assert 'data-layer="contact-covered"' in frame.profile_svg
        assert frame.coverage_score_display == contact.coverage_score

    def test_build_projects_overlays_exactly_once(
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

        class CountingProjector(ContactResultProjector):
            calls = 0

            def project(self, *args: object, **kwargs: object):
                self.calls += 1
                return super().project(*args, **kwargs)

        projector = CountingProjector()
        profile: dict = {}
        VisualizationEngine(contact_projector=projector).build_contact_visualization(
            contact,
            sample_view_cache,
            cylindrical_jar_canonical,
            overlay_profile=profile,
        )

        assert projector.calls == 1
        assert profile["construction_ms"] >= 0.0
        assert profile["fragment_wrapping_ms"] >= 0.0
        assert profile["payload_bytes"] > 0
