"""Application layer tests."""

from __future__ import annotations

import pytest

from nutella_scraper.application.container import build_container
from nutella_scraper.application.dto import SimulateRequestDTO


class TestContainer:
    def test_build_container(self) -> None:
        container = build_container()
        assert container.orchestrator is not None
        assert container.compute_engine is not None
        assert container.visualization_engine is not None
        assert container.optimization_engine is not None


class TestApplicationOrchestrator:
    def test_simulate_not_implemented(self) -> None:
        container = build_container()
        with pytest.raises(NotImplementedError):
            container.orchestrator.simulate(
                SimulateRequestDTO(model_id="test", jar_id="nutella_400g")
            )
