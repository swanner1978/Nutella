"""Pareto front management for multi-objective optimization."""

from __future__ import annotations

from nutella_scraper.domain.models.design_space import DesignCandidate


class ParetoFrontManager:
    """Maintains non-dominated candidate set."""

    def update(self, candidates: list[DesignCandidate]) -> list[DesignCandidate]:
        raise NotImplementedError("ParetoFrontManager.update not implemented")

    def extract_front(self, candidates: list[DesignCandidate]) -> tuple[str, ...]:
        raise NotImplementedError("ParetoFrontManager.extract_front not implemented")
