"""Objective functions for optimization — computed via ComputeEngine only."""

from __future__ import annotations

from nutella_scraper.domain.models.contact import EvaluationResult
from nutella_scraper.domain.models.design_space import ObjectiveSpec


class ObjectiveFunctions:
    """
    Aggregates objective values from evaluation results.

    Never reads ViewProjectionCache or 2D projections.
    """

    def __init__(self, specs: tuple[ObjectiveSpec, ...]) -> None:
        self._specs = specs

    @property
    def specs(self) -> tuple[ObjectiveSpec, ...]:
        return self._specs

    def compute_vector(self, evaluation: EvaluationResult) -> dict[str, float]:
        """Build objective vector from 3D evaluation metrics."""
        raise NotImplementedError("ObjectiveFunctions.compute_vector not implemented")

    def scalarize(self, evaluation: EvaluationResult) -> float:
        """Weighted scalar score for single-objective optimizers."""
        raise NotImplementedError("ObjectiveFunctions.scalarize not implemented")
