"""Optimization engine tests."""

from __future__ import annotations

import pytest

from nutella_scraper.domain.models.design_space import DesignSpace, OptimizationBudget
from nutella_scraper.engines.optimization.engine import OptimizationEngine
from nutella_scraper.engines.optimization.optimizer import OptimizerRunner


class TestOptimizerRunner:
    def test_run_not_implemented(self) -> None:
        runner = OptimizerRunner()
        with pytest.raises(NotImplementedError):
            runner.run(
                evaluator=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
                design_space=DesignSpace(id="test", parameters=()),
                budget=OptimizationBudget(),
            )


class TestOptimizationEngine:
    def test_engine_instantiation(self) -> None:
        engine = OptimizationEngine()
        assert engine is not None
