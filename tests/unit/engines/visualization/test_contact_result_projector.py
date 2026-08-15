"""Contact result projector tests."""

from __future__ import annotations

import pytest

from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine
from nutella_scraper.engines.visualization.contact_result_projector import (
    LAYER_COLLISION_FACES,
    LAYER_CONTACT_COVERED,
    LAYER_CONTACT_POINTS,
    LAYER_CONTACT_UNCOVERED,
    LAYER_DISTANCE_MAP,
    ContactResultProjector,
)


class TestContactResultProjector:
    def test_project_builds_all_overlay_layers(
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
        payload = ContactResultProjector().project(
            contact,
            sample_view_cache,
            cylindrical_jar_canonical,
        )

        assert payload.coverage_score_display == contact.coverage_score
        profile_types = {layer.layer_type for layer in payload.profile_layers}
        top_types = {layer.layer_type for layer in payload.top_layers}
        expected = {
            LAYER_CONTACT_COVERED,
            LAYER_CONTACT_UNCOVERED,
            LAYER_DISTANCE_MAP,
            LAYER_CONTACT_POINTS,
            LAYER_COLLISION_FACES,
        }
        assert expected.issubset(profile_types)
        assert expected.issubset(top_types)

    def test_project_requires_overlay(
        self,
        sample_view_cache: object,
        cylindrical_jar_canonical: object,
    ) -> None:
        import numpy as np

        from nutella_scraper.domain.models.contact import ContactResult

        bare = ContactResult(
            model_id="x",
            jar_id="y",
            coverage_score=0.0,
            touched_face_ids=frozenset(),
            untouched_face_ids=frozenset({0}),
            contact_distance_map=np.array([np.inf]),
            overlay=None,
        )
        with pytest.raises(ValueError, match="overlay"):
            ContactResultProjector().project(bare, sample_view_cache, cylindrical_jar_canonical)

    def test_profile_reports_cardinality_and_compacted_graphics(
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
        profile: dict = {}

        payload = ContactResultProjector().project(
            contact,
            sample_view_cache,
            cylindrical_jar_canonical,
            profile=profile,
        )

        face_count = len(contact.overlay.face_coverage)
        assert profile["face_count"] == face_count
        assert profile["face_projections_processed"] == face_count * 5
        assert profile["contact_point_count"] == len(contact.overlay.contact_points)
        assert profile["construction_ms"] >= 0.0
        assert profile["graphic_element_count"] < face_count * 8
        assert profile["svg_bytes"] > 0
        for layers in (
            payload.profile_layers,
            payload.top_layers,
            payload.left_layers,
            payload.right_layers,
            payload.bottom_layers,
        ):
            covered = next(
                layer for layer in layers if layer.layer_type == LAYER_CONTACT_COVERED
            )
            uncovered = next(
                layer for layer in layers if layer.layer_type == LAYER_CONTACT_UNCOVERED
            )
            assert covered.svg_fragment.count("<image") <= 1
            assert uncovered.svg_fragment.count("<image") <= 1

    def test_covered_and_uncovered_layers_are_disjoint(
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
        payload = ContactResultProjector().project(
            contact,
            sample_view_cache,
            cylindrical_jar_canonical,
        )
        covered = next(
            layer for layer in payload.profile_layers if layer.layer_type == LAYER_CONTACT_COVERED
        )
        uncovered = next(
            layer for layer in payload.profile_layers if layer.layer_type == LAYER_CONTACT_UNCOVERED
        )
        distance = next(
            layer for layer in payload.profile_layers if layer.layer_type == LAYER_DISTANCE_MAP
        )

        assert "<image" in covered.svg_fragment
        assert "<image" in uncovered.svg_fragment
        assert "<image" in distance.svg_fragment
        assert covered.svg_fragment != uncovered.svg_fragment
