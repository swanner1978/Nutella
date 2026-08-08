"""Application settings from YAML and environment."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(env_prefix="NUTELLA_", env_nested_delimiter="__")

    app_name: str = "nutella-scraper"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    config_dir: Path = Path("configs")
    data_dir: Path = Path("data")
    models_dir: Path = Path("data/models")
    exports_dir: Path = Path("data/exports")
    database_url: str = "sqlite:///data/nutella_scraper.db"


class SimulationSettings(BaseSettings):
    contact_threshold_mm: float = 0.5
    trajectory_steps: int = 100


class FDMSettings(BaseSettings):
    min_wall_thickness_mm: float = 1.2
    max_overhang_angle_deg: float = 45.0
    clearance_compensation_mm: float = 0.15


class Settings(BaseSettings):
    """Aggregated settings container."""

    app: AppSettings = Field(default_factory=AppSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    fdm: FDMSettings = Field(default_factory=FDMSettings)
