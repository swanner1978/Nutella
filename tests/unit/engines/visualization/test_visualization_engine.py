"""Visualization engine tests."""

from __future__ import annotations

import pytest

from nutella_scraper.engines.visualization.engine import VisualizationEngine
from nutella_scraper.engines.visualization.view_projection_generator import ViewProjectionGenerator


class TestViewProjectionGenerator:
    def test_generate_not_implemented(self) -> None:
        gen = ViewProjectionGenerator()
        with pytest.raises(NotImplementedError):
            gen.generate(__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock())


class TestVisualizationEngine:
    def test_engine_instantiation(self) -> None:
        engine = VisualizationEngine()
        assert engine is not None
