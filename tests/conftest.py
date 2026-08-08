"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def config_dir() -> Path:
    return Path("configs")


@pytest.fixture
def container(config_dir: Path):
    from nutella_scraper.application.container import build_container

    return build_container(config_dir=config_dir)


@pytest.fixture
def jar_step_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "Solidworks" / "jar.STEP"
    if not path.exists():
        pytest.skip("Solidworks/jar.STEP fixture missing")
    return path


@pytest.fixture
def cad_reference_geometry(jar_step_path: Path):
    pytest.importorskip("OCP", reason="cadquery-ocp required for CAD reference fixtures")
    from nutella_scraper.cad_import.cad_reference_builder import CadReferenceGeometryBuilder

    return CadReferenceGeometryBuilder().from_step(jar_step_path, model_id="jar_test")
