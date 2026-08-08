"""Dependency injection container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nutella_scraper.application.orchestrator import ApplicationOrchestrator
from nutella_scraper.application.simulation_service import SimulationService
from nutella_scraper.cad_import.geometry_normalizer import GeometryNormalizer
from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.cad_import.pipeline import ImportPipeline
from nutella_scraper.cad_import.solidworks_exporter import SolidWorksExporter
from nutella_scraper.cad_import.view_cache_store import ViewCacheStore as CadViewCacheStore
from nutella_scraper.cad_import.view_projection_generator import ViewProjectionGenerator
from nutella_scraper.engines.compute.engine import ComputeEngine
from nutella_scraper.engines.optimization.engine import OptimizationEngine
from nutella_scraper.engines.visualization.engine import VisualizationEngine
from nutella_scraper.io.config_loader import ConfigLoader, JarLoader, SimulationConfigLoader
from nutella_scraper.io.persistence.results_store import ResultsStore, ViewCacheStore
from nutella_scraper.io.settings import Settings


@dataclass
class Container:
    """Wires all engines and services with explicit dependencies."""

    settings: Settings
    config_loader: ConfigLoader
    jar_loader: JarLoader
    simulation_config_loader: SimulationConfigLoader
    model_store: ModelStore
    results_store: ResultsStore
    view_cache_store: ViewCacheStore
    solidworks_exporter: SolidWorksExporter
    geometry_normalizer: GeometryNormalizer
    visualization_engine: VisualizationEngine
    compute_engine: ComputeEngine
    optimization_engine: OptimizationEngine
    import_pipeline: ImportPipeline
    simulation_service: SimulationService
    orchestrator: ApplicationOrchestrator


def build_container(config_dir: Path | None = None) -> Container:
    """Factory for application dependency graph."""
    settings = Settings()
    config_path = config_dir or settings.app.config_dir
    config_loader = ConfigLoader(config_path)
    jar_loader = JarLoader(config_path)
    simulation_config_loader = SimulationConfigLoader(config_loader)
    model_store = ModelStore(settings.app.models_dir)
    results_store = ResultsStore(settings.app.database_url)
    view_cache_store = ViewCacheStore(str(settings.app.data_dir / "views"))
    solidworks_exporter = SolidWorksExporter()
    geometry_normalizer = GeometryNormalizer()
    visualization_engine = VisualizationEngine()
    compute_engine = ComputeEngine(model_store=model_store, jar_loader=jar_loader)
    optimization_engine = OptimizationEngine()
    import_pipeline = ImportPipeline(
        normalizer=geometry_normalizer,
        model_store=model_store,
        view_generator=ViewProjectionGenerator(),
        view_cache_store=CadViewCacheStore(settings.app.data_dir / "views"),
        exporter=solidworks_exporter,
    )
    simulation_service = SimulationService(
        compute_engine=compute_engine,
        visualization_engine=visualization_engine,
        simulation_config_loader=simulation_config_loader,
        jar_loader=jar_loader,
        view_cache_store=view_cache_store,
    )
    orchestrator = ApplicationOrchestrator(
        import_pipeline=import_pipeline,
        compute_engine=compute_engine,
        visualization_engine=visualization_engine,
        optimization_engine=optimization_engine,
        results_store=results_store,
        simulation_service=simulation_service,
    )
    return Container(
        settings=settings,
        config_loader=config_loader,
        jar_loader=jar_loader,
        simulation_config_loader=simulation_config_loader,
        model_store=model_store,
        results_store=results_store,
        view_cache_store=view_cache_store,
        solidworks_exporter=solidworks_exporter,
        geometry_normalizer=geometry_normalizer,
        visualization_engine=visualization_engine,
        compute_engine=compute_engine,
        optimization_engine=optimization_engine,
        import_pipeline=import_pipeline,
        simulation_service=simulation_service,
        orchestrator=orchestrator,
    )
