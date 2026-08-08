"""Persistence protocols."""

from __future__ import annotations

from typing import Protocol

from nutella_scraper.domain.models.contact import ContactResult
from nutella_scraper.domain.models.design_space import DesignCandidate, OptimizationRun
from nutella_scraper.domain.models.views import ViewProjectionCache


class IResultsStore(Protocol):
    """Persists optimization runs, candidates, and simulation results."""

    def save_optimization_run(self, run: OptimizationRun) -> None:
        ...

    def get_optimization_run(self, run_id: str) -> OptimizationRun | None:
        ...

    def save_candidate(self, run_id: str, candidate: DesignCandidate) -> None:
        ...

    def list_candidates(self, run_id: str) -> list[DesignCandidate]:
        ...

    def save_contact_result(self, result: ContactResult) -> str:
        ...

    def get_contact_result(self, result_id: str) -> ContactResult | None:
        ...


class IViewCacheStore(Protocol):
    """Persists visualization-only view caches."""

    def save(self, cache: ViewProjectionCache) -> str:
        ...

    def get(self, views_id: str) -> ViewProjectionCache | None:
        ...
