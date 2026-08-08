"""Optimization engine protocol."""

from __future__ import annotations

from typing import Protocol

from nutella_scraper.domain.models.contact import EvaluationResult
from nutella_scraper.domain.models.design_space import (
    DesignSpace,
    OptimizationBudget,
    OptimizationResult,
    ParameterSample,
)
from nutella_scraper.domain.protocols.compute import IComputeEngine


class IEvaluator(Protocol):
    """Evaluates a parameter sample via ComputeEngine only."""

    def evaluate(self, sample: ParameterSample) -> EvaluationResult:
        """Evaluate design — must not use VisualizationEngine."""
        ...


class IOptimizationEngine(Protocol):
    """Contract for optimization engine."""

    def run(
        self,
        compute: IComputeEngine,
        design_space: DesignSpace,
        budget: OptimizationBudget,
    ) -> OptimizationResult:
        """Run optimization loop via ComputeEngine evaluator."""
        ...
