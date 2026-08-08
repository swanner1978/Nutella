"""SQLite persistence for runs, candidates, and contact results."""

from __future__ import annotations

from nutella_scraper.domain.models.contact import ContactResult
from nutella_scraper.domain.models.design_space import DesignCandidate, OptimizationRun
from nutella_scraper.domain.models.views import ViewProjectionCache


class ResultsStore:
    """Skeleton persistence layer — business logic not implemented."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def save_optimization_run(self, run: OptimizationRun) -> None:
        raise NotImplementedError("ResultsStore.save_optimization_run not implemented")

    def get_optimization_run(self, run_id: str) -> OptimizationRun | None:
        raise NotImplementedError("ResultsStore.get_optimization_run not implemented")

    def save_candidate(self, run_id: str, candidate: DesignCandidate) -> None:
        raise NotImplementedError("ResultsStore.save_candidate not implemented")

    def list_candidates(self, run_id: str) -> list[DesignCandidate]:
        raise NotImplementedError("ResultsStore.list_candidates not implemented")

    def save_contact_result(self, result: ContactResult) -> str:
        raise NotImplementedError("ResultsStore.save_contact_result not implemented")

    def get_contact_result(self, result_id: str) -> ContactResult | None:
        raise NotImplementedError("ResultsStore.get_contact_result not implemented")


class ViewCacheStore:
    """Stores visualization-only view projection caches."""

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir

    def save(self, cache: ViewProjectionCache) -> str:
        raise NotImplementedError("ViewCacheStore.save not implemented")

    def get(self, views_id: str) -> ViewProjectionCache | None:
        raise NotImplementedError("ViewCacheStore.get not implemented")
