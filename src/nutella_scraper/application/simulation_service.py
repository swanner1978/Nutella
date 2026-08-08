"""Simulation service — coordinates ComputeEngine and VisualizationEngine."""

from __future__ import annotations

from nutella_scraper.application.dto import SimulateRequestDTO, SimulateResponseDTO, ViewOverlayResponseDTO
from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.engines.compute.engine import ComputeEngine
from nutella_scraper.engines.visualization.engine import VisualizationEngine
from nutella_scraper.io.config_loader import JarLoader, SimulationConfigLoader
from nutella_scraper.io.persistence.results_store import ViewCacheStore


class SimulationService:
    """
    Coordinates contact simulation (ComputeEngine) and view projection (VisualizationEngine).

    Compute runs first; visualization receives ContactResult read-only.
    """

    def __init__(
        self,
        compute_engine: ComputeEngine,
        visualization_engine: VisualizationEngine,
        simulation_config_loader: SimulationConfigLoader,
        jar_loader: JarLoader,
        view_cache_store: ViewCacheStore,
    ) -> None:
        self._compute = compute_engine
        self._visualization = visualization_engine
        self._simulation_config_loader = simulation_config_loader
        self._jar_loader = jar_loader
        self._view_cache_store = view_cache_store

    def simulate(self, request: SimulateRequestDTO) -> SimulateResponseDTO:
        raise NotImplementedError("SimulationService.simulate not implemented")

    def get_overlay(self, model_id: str, contact_result_id: str) -> ViewOverlayResponseDTO:
        raise NotImplementedError("SimulationService.get_overlay not implemented")

    def _load_simulation_config(self, profile: str) -> ContactSimulationConfig:
        return self._simulation_config_loader.load(profile)
