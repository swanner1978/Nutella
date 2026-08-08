"""Application orchestrator — unified entry point for CLI and API."""

from __future__ import annotations

from pathlib import Path

from nutella_scraper.application.dto import (
    ImportRequestDTO,
    ImportResponseDTO,
    OptimizationRequestDTO,
    OptimizationResponseDTO,
    SimulateRequestDTO,
    SimulateResponseDTO,
    ViewOverlayResponseDTO,
)
from nutella_scraper.application.simulation_service import SimulationService
from nutella_scraper.cad_import.pipeline import ImportPipeline
from nutella_scraper.domain.models.design_space import OptimizationBudget
from nutella_scraper.engines.compute.engine import ComputeEngine
from nutella_scraper.engines.optimization.engine import OptimizationEngine
from nutella_scraper.engines.visualization.engine import VisualizationEngine
from nutella_scraper.io.persistence.results_store import ResultsStore


class ApplicationOrchestrator:
    """Coordinates import, simulation, visualization, and optimization."""

    def __init__(
        self,
        import_pipeline: ImportPipeline,
        compute_engine: ComputeEngine,
        visualization_engine: VisualizationEngine,
        optimization_engine: OptimizationEngine,
        results_store: ResultsStore,
        simulation_service: SimulationService,
    ) -> None:
        self._import_pipeline = import_pipeline
        self._compute = compute_engine
        self._visualization = visualization_engine
        self._optimization = optimization_engine
        self._results_store = results_store
        self._simulation_service = simulation_service

    def import_model(self, request: ImportRequestDTO) -> ImportResponseDTO:
        raise NotImplementedError("ApplicationOrchestrator.import_model not implemented")

    def import_sldprt(self, path: Path) -> ImportResponseDTO:
        raise NotImplementedError("ApplicationOrchestrator.import_sldprt not implemented")

    def simulate(self, request: SimulateRequestDTO) -> SimulateResponseDTO:
        return self._simulation_service.simulate(request)

    def get_view_overlay(self, model_id: str, contact_result_id: str) -> ViewOverlayResponseDTO:
        return self._simulation_service.get_overlay(model_id, contact_result_id)

    def start_optimization(self, request: OptimizationRequestDTO) -> OptimizationResponseDTO:
        raise NotImplementedError("ApplicationOrchestrator.start_optimization not implemented")

    def run_optimization_sync(
        self,
        request: OptimizationRequestDTO,
        budget: OptimizationBudget,
    ) -> OptimizationResponseDTO:
        raise NotImplementedError("ApplicationOrchestrator.run_optimization_sync not implemented")
