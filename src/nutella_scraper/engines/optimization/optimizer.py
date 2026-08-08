"""Optimization loop runner — Optuna/NLopt placeholder."""

from __future__ import annotations

from nutella_scraper.domain.models.design_space import (
    DesignSpace,
    OptimizationBudget,
    OptimizationResult,
)
from nutella_scraper.domain.protocols.optimization import IEvaluator


class OptimizerRunner:
    """Runs optimization trials using an evaluator."""

    def run(
        self,
        evaluator: IEvaluator,
        design_space: DesignSpace,
        budget: OptimizationBudget,
    ) -> OptimizationResult:
        raise NotImplementedError("OptimizerRunner.run not implemented")
