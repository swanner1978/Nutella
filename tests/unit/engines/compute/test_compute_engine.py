"""Compute engine tests."""

from __future__ import annotations

import pytest

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine
from nutella_scraper.engines.compute.coverage_scorer import CoverageScorer
from nutella_scraper.engines.compute.engine import ComputeEngine
from nutella_scraper.engines.compute.objective_functions import ObjectiveFunctions


class TestCoverageScorer:
    def test_from_contact_result_returns_score(self) -> None:
        import numpy as np

        from nutella_scraper.domain.models.contact import ContactResult

        result = ContactResult(
            model_id="m",
            jar_id="j",
            coverage_score=0.87,
            touched_face_ids=frozenset({1, 2}),
            untouched_face_ids=frozenset({3}),
            contact_distance_map=np.array([0.1]),
        )
        scorer = CoverageScorer()
        assert scorer.from_contact_result(result) == 0.87


class TestObjectiveFunctions:
    def test_compute_vector_not_implemented(self) -> None:
        from nutella_scraper.domain.models.design_space import ObjectiveSpec

        obj = ObjectiveFunctions(
            specs=(ObjectiveSpec(name="coverage_score", weight=1.0, direction="maximize"),)
        )
        with pytest.raises(NotImplementedError):
            obj.compute_vector(__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock())


class TestComputeEngine:
    def test_load_jar(self, container: object) -> None:
        from nutella_scraper.application.container import Container

        assert isinstance(container, Container)
        jar = container.compute_engine.load_jar("nutella_400g")
        assert jar.id == "nutella_400g"

    def test_simulate_contact_delegates_to_contact_engine(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        engine = ComputeEngine(
            model_store=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
            jar_loader=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
            contact_engine=ContactSimulationEngine(),
        )
        result = engine.simulate_contact(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
            coarse_simulation_config,
        )
        assert result.model_id == "test_scraper"
        assert result.jar_id == "test_jar"
        assert result.collision is not None
