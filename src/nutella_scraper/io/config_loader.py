"""YAML/JSON configuration loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from nutella_scraper.domain.models.canonical import JarCanonicalModel, JarProfilePoint
from nutella_scraper.domain.models.contact import ContactSimulationConfig, TrajectoryConfig
from nutella_scraper.domain.models.design_space import DesignSpace, ObjectiveSpec, OptimizationBudget


class ConfigLoader:
    """Loads YAML configuration files."""

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir

    def load_yaml(self, relative_path: str) -> dict[str, Any]:
        path = self._config_dir / relative_path
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_default(self) -> dict[str, Any]:
        return self.load_yaml("default.yaml")


class JarLoader:
    """Loads jar profile JSON into JarCanonicalModel."""

    def __init__(self, config_dir: Path) -> None:
        self._jars_dir = config_dir / "jars"

    def load(self, jar_id: str) -> JarCanonicalModel:
        path = self._jars_dir / f"{jar_id}.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        profile = tuple(
            JarProfilePoint(z_mm=p["z_mm"], r_mm=p["r_mm"])
            for p in data["meridian_profile"]
        )
        return JarCanonicalModel(
            id=data["id"],
            version=data["version"],
            meridian_profile=profile,
            neck_inner_diameter_mm=data["neck_inner_diameter_mm"],
            total_height_mm=data["total_height_mm"],
        )


class SimulationConfigLoader:
    """Loads contact simulation configuration."""

    def __init__(self, config_loader: ConfigLoader) -> None:
        self._config_loader = config_loader

    def load(self, name: str = "contact_default") -> ContactSimulationConfig:
        data = self._config_loader.load_yaml(f"simulation/{name}.yaml")
        traj = data.get("trajectory", {})
        contact = data.get("contact", {})
        mesh = data.get("mesh", {})
        return ContactSimulationConfig(
            trajectory=TrajectoryConfig(
                type=traj.get("type", "rotational_vertical"),
                angular_step_deg=float(traj.get("angular_step_deg", 5.0)),
                vertical_step_mm=float(traj.get("vertical_step_mm", 2.0)),
            ),
            contact_threshold_mm=float(contact.get("threshold_mm", 0.5)),
            clearance_mm=float(contact.get("clearance_mm", 0.15)),
            mesh_tolerance_mm=float(mesh.get("tolerance_mm", 0.1)),
        )


class OptimizationConfigLoader:
    """Loads optimization profile."""

    def __init__(self, config_loader: ConfigLoader) -> None:
        self._config_loader = config_loader

    def load_objectives(self, profile: str = "fdm_pla") -> tuple[ObjectiveSpec, ...]:
        data = self._config_loader.load_yaml(f"optimization/{profile}.yaml")
        objectives = data.get("objectives", {})
        return tuple(
            ObjectiveSpec(name=name, weight=float(spec["weight"]), direction=spec["direction"])
            for name, spec in objectives.items()
        )

    def load_budget(self, profile: str = "fdm_pla") -> OptimizationBudget:
        data = self._config_loader.load_yaml(f"optimization/{profile}.yaml")
        budget = data.get("budget", {})
        return OptimizationBudget(
            max_trials=int(budget.get("max_trials", 200)),
            timeout_s=int(budget.get("timeout_s", 3600)),
            seed=int(budget.get("seed", 42)),
        )

    def load_design_space(self, space_id: str = "default") -> DesignSpace:
        raise NotImplementedError("OptimizationConfigLoader.load_design_space not implemented")
