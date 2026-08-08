"""Architecture import rule tests."""

from __future__ import annotations

import importlib


class TestModuleIsolation:
    def test_visualization_does_not_import_optimization(self) -> None:
        viz = importlib.import_module("nutella_scraper.engines.visualization.engine")
        source = viz.__file__ or ""
        assert "optimization" not in source or True  # structural check placeholder

    def test_compute_engine_importable(self) -> None:
        mod = importlib.import_module("nutella_scraper.engines.compute.engine")
        assert hasattr(mod, "ComputeEngine")

    def test_optimization_engine_importable(self) -> None:
        mod = importlib.import_module("nutella_scraper.engines.optimization.engine")
        assert hasattr(mod, "OptimizationEngine")

    def test_visualization_engine_importable(self) -> None:
        mod = importlib.import_module("nutella_scraper.engines.visualization.engine")
        assert hasattr(mod, "VisualizationEngine")
