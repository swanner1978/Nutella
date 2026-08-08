"""Compute evaluator — bridges OptimizationEngine and ComputeEngine."""

from __future__ import annotations

from nutella_scraper.domain.models.contact import ContactSimulationConfig, EvaluationResult
from nutella_scraper.domain.models.design_space import ParameterSample
from nutella_scraper.domain.protocols.compute import IComputeEngine


class ComputeEvaluator:
    """
    Evaluates parameter samples via ComputeEngine only.

    Never uses VisualizationEngine or ViewProjectionCache.
    """

    def __init__(
        self,
        compute_engine: IComputeEngine,
        jar_id: str,
        simulation_config: ContactSimulationConfig,
    ) -> None:
        self._compute = compute_engine
        self._jar_id = jar_id
        self._simulation_config = simulation_config

    def evaluate(self, sample: ParameterSample) -> EvaluationResult:
        raise NotImplementedError("ComputeEvaluator.evaluate not implemented")
