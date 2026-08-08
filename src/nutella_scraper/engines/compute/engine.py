"""Compute engine facade — orchestrates contact simulation and analysis."""

from __future__ import annotations

from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.domain.models.canonical import CanonicalModel3D, JarCanonicalModel
from nutella_scraper.domain.models.contact import (
    ContactResult,
    ContactSimulationConfig,
    EvaluationResult,
)
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.domain.protocols.persistence import IResultsStore
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine
from nutella_scraper.engines.compute.coverage_scorer import CoverageScorer
from nutella_scraper.engines.compute.distance_field import DistanceFieldQuery
from nutella_scraper.engines.compute.fdm_printability import FDMPrintabilityChecker
from nutella_scraper.engines.compute.objective_functions import ObjectiveFunctions
from nutella_scraper.engines.compute.residual_estimator import ResidualVolumeEstimator
from nutella_scraper.io.config_loader import JarLoader


class ComputeEngine:
    """
    Main compute engine — 3D contact simulation and metrics.

    Does not depend on VisualizationEngine or ViewProjectionCache.
    """

    def __init__(
        self,
        model_store: ModelStore,
        jar_loader: JarLoader,
        contact_engine: ContactSimulationEngine | None = None,
        coverage_scorer: CoverageScorer | None = None,
        distance_query: DistanceFieldQuery | None = None,
        residual_estimator: ResidualVolumeEstimator | None = None,
        fdm_checker: FDMPrintabilityChecker | None = None,
        objective_functions: ObjectiveFunctions | None = None,
        results_store: IResultsStore | None = None,
    ) -> None:
        self._model_store = model_store
        self._jar_loader = jar_loader
        self._contact_engine = contact_engine or ContactSimulationEngine()
        self._coverage_scorer = coverage_scorer or CoverageScorer()
        self._distance_query = distance_query or DistanceFieldQuery()
        self._residual_estimator = residual_estimator or ResidualVolumeEstimator()
        self._fdm_checker = fdm_checker or FDMPrintabilityChecker()
        self._objective_functions = objective_functions
        self._results_store = results_store

    def load_scraper(self, model_id: str) -> CanonicalModel3D:
        return self._model_store.get(model_id)

    def load_jar(self, jar_id: str) -> JarCanonicalModel:
        return self._jar_loader.load(jar_id)

    def simulate_contact(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
        config: ContactSimulationConfig,
    ) -> ContactResult:
        return self._contact_engine.simulate(jar, geometry, pose, config)

    def evaluate(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
        config: ContactSimulationConfig,
    ) -> EvaluationResult:
        raise NotImplementedError("ComputeEngine.evaluate not implemented")
