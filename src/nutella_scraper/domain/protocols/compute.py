"""Compute engine protocol."""

from __future__ import annotations

from typing import Protocol

from nutella_scraper.domain.models.canonical import CanonicalModel3D, JarCanonicalModel
from nutella_scraper.domain.models.contact import (
    ContactResult,
    ContactSimulationConfig,
    EvaluationResult,
)
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose


class IComputeEngine(Protocol):
    """Contract for the compute engine (contact simulation, metrics)."""

    def load_scraper(self, model_id: str) -> CanonicalModel3D:
        """Load a persisted scraper model by ID."""
        ...

    def load_jar(self, jar_id: str) -> JarCanonicalModel:
        """Load a jar model by ID."""
        ...

    def simulate_contact(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
        config: ContactSimulationConfig,
    ) -> ContactResult:
        """Run 3D contact simulation — no 2D view input allowed."""
        ...

    def evaluate(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
        config: ContactSimulationConfig,
    ) -> EvaluationResult:
        """Full evaluation including contact and derived metrics."""
        ...
