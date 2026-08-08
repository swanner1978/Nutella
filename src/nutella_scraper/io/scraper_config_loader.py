"""Load parametric scraper definitions from YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nutella_scraper.domain.models.scraper import ScraperGeometry


def load_scraper_geometry(config_path: Path) -> ScraperGeometry:
    """Build ScraperGeometry from a scraper YAML template such as racloir_v1."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid scraper config: {config_path}")

    parameters = _parameters_dict(payload)
    geometry_id = str(payload.get("id", config_path.stem))
    return ScraperGeometry(
        id=geometry_id,
        width_mm=float(parameters["width_mm"]),
        length_mm=float(parameters["length_mm"]),
        thickness_mm=float(parameters["thickness_mm"]),
        tip_radius_mm=float(parameters.get("tip_radius_mm", 1.5)),
        curvature_radius_mm=_optional_float(parameters, "curvature_radius_mm"),
        bend_angle_deg=float(parameters.get("bend_angle_deg", 0.0)),
        metadata={"config_path": str(config_path), "version": str(payload.get("version", "1"))},
    )


def default_racloir_v1(config_dir: Path) -> ScraperGeometry:
    return load_scraper_geometry(config_dir / "scrapers" / "racloir_v1.yaml")


def _parameters_dict(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("parameters")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        merged: dict[str, Any] = {}
        for entry in raw:
            if isinstance(entry, dict) and "name" in entry and "value" in entry:
                merged[str(entry["name"])] = entry["value"]
        if merged:
            return merged
    raise ValueError("Scraper config must define parameters as a mapping or name/value list")


def _optional_float(parameters: dict[str, Any], key: str) -> float | None:
    if key not in parameters or parameters[key] is None:
        return None
    return float(parameters[key])
