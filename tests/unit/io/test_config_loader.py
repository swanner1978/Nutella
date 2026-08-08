"""IO layer tests."""

from __future__ import annotations

from pathlib import Path

from nutella_scraper.io.config_loader import ConfigLoader, JarLoader, SimulationConfigLoader


class TestConfigLoader:
    def test_load_default(self) -> None:
        loader = ConfigLoader(Path("configs"))
        config = loader.load_default()
        assert config["app"]["name"] == "nutella-scraper"


class TestJarLoader:
    def test_load_nutella_400g(self) -> None:
        loader = JarLoader(Path("configs"))
        jar = loader.load("nutella_400g")
        assert jar.neck_inner_diameter_mm == 58.0


class TestSimulationConfigLoader:
    def test_load_contact_default(self) -> None:
        loader = SimulationConfigLoader(ConfigLoader(Path("configs")))
        config = loader.load()
        assert config.contact_threshold_mm == 0.5
