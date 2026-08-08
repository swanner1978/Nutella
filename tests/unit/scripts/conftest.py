"""Fixtures for viewer API integration tests."""

from __future__ import annotations

import pytest

from nutella_scraper.domain.models.canonical import JarCanonicalModel, JarProfilePoint
from nutella_scraper.domain.models.contact import ContactSimulationConfig, TrajectoryConfig
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder


@pytest.fixture
def cylindrical_jar_model() -> JarCanonicalModel:
    return JarCanonicalModel(
        id="test_jar",
        version="1",
        meridian_profile=(
            JarProfilePoint(z_mm=0.0, r_mm=50.0),
            JarProfilePoint(z_mm=100.0, r_mm=50.0),
        ),
        neck_inner_diameter_mm=60.0,
        total_height_mm=100.0,
    )


@pytest.fixture
def cylindrical_jar_canonical(cylindrical_jar_model: JarCanonicalModel):
    return JarMeshBuilder().to_canonical(cylindrical_jar_model, theta_segments=24)


@pytest.fixture
def wall_scraper_geometry() -> ScraperGeometry:
    return ScraperGeometry(
        id="test_scraper",
        width_mm=10.0,
        length_mm=90.0,
        thickness_mm=4.0,
    )


@pytest.fixture
def wall_scraper_pose() -> ScraperPose:
    return ScraperPose(position_mm=(47.0, 50.0, 0.0), yaw_deg=0.0)


@pytest.fixture
def coarse_simulation_config() -> ContactSimulationConfig:
    return ContactSimulationConfig(
        trajectory=TrajectoryConfig(
            angular_step_deg=90.0,
            vertical_step_mm=50.0,
        ),
        contact_threshold_mm=1.0,
        clearance_mm=0.15,
        mesh_tolerance_mm=0.1,
    )
