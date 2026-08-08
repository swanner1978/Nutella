"""Optimization engine facade."""

from __future__ import annotations

from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.design_space import (
    DesignSpace,
    OptimizationBudget,
    OptimizationResult,
)
from nutella_scraper.domain.protocols.compute import IComputeEngine
from nutella_scraper.engines.optimization.compute_evaluator import ComputeEvaluator
from nutella_scraper.engines.optimization.design_space_sampler import DesignSpaceSampler
from nutella_scraper.engines.optimization.optimizer import OptimizerRunner
from nutella_scraper.engines.optimization.pareto import ParetoFrontManager


class OptimizationEngine:
    """
    Optimization engine — uses ComputeEngine via IEvaluator only.

    Does not depend on VisualizationEngine.
    """

    def __init__(
        self,
        optimizer: OptimizerRunner | None = None,
        pareto_manager: ParetoFrontManager | None = None,
        sampler: DesignSpaceSampler | None = None,
    ) -> None:
        self._optimizer = optimizer or OptimizerRunner()
        self._pareto_manager = pareto_manager or ParetoFrontManager()
        self._sampler = sampler or DesignSpaceSampler()

    def run(
        self,
        compute: IComputeEngine,
        design_space: DesignSpace,
        budget: OptimizationBudget,
        jar_id: str = "nutella_400g",
    ) -> OptimizationResult:
        evaluator = ComputeEvaluator(
            compute_engine=compute,
            jar_id=jar_id,
            simulation_config=ContactSimulationConfig(),
        )
        return self._optimizer.run(evaluator, design_space, budget)
