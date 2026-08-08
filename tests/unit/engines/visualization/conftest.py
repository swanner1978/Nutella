"""Shared fixtures for visualization engine tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.domain.models.canonical import JarCanonicalModel, JarProfilePoint
from nutella_scraper.domain.models.contact import ContactSimulationConfig, TrajectoryConfig
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.domain.models.views import (
    ProjectedView,
    ProjectionMetadata,
    ViewProjectionCache,
)
from nutella_scraper.engines.compute.internal_jar_profile_builder import InternalJarProfileBuilder
from nutella_scraper.engines.compute.internal_jar_surface_builder import InternalJarSurfaceBuilder
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
def internal_jar_surface(cylindrical_jar_canonical):
    return InternalJarSurfaceBuilder().from_canonical(cylindrical_jar_canonical)


@pytest.fixture
def internal_jar_profile(internal_jar_surface):
    return InternalJarProfileBuilder().from_internal(internal_jar_surface)


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


@pytest.fixture
def sample_view_cache() -> ViewProjectionCache:
    base_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="650" '
        'data-plane="XY" data-view-axis="Z">'
        '<g data-layer="contour"></g></svg>'
    )
    metadata = ProjectionMetadata(
        plane="XY",
        camera={"x": 0.0, "y": 0.0, "z": -1.0},
        scale=1.0,
        width_px=900,
        height_px=650,
    )
    return ViewProjectionCache(
        model_id="test_model",
        profile_view=ProjectedView(
            plane="XY",
            asset_path=None,
            svg_content=base_svg,
            metadata=metadata,
        ),
        top_view=ProjectedView(
            plane="XZ",
            asset_path=None,
            svg_content=base_svg.replace('data-plane="XY"', 'data-plane="XZ"').replace(
                'data-view-axis="Z"', 'data-view-axis="Y"'
            ),
            metadata=ProjectionMetadata(
                plane="XZ",
                camera={"x": 0.0, "y": -1.0, "z": 0.0},
                scale=1.0,
                width_px=900,
                height_px=650,
            ),
        ),
    )


@pytest.fixture
def viewer_dir(tmp_path: Path, cylindrical_jar_canonical: object) -> Path:
    from scripts.visualization_helpers import VIEW_CONVENTIONS, build_projection_svg

    from tests.unit.cad_import.conftest import persist_test_cad_reference

    model = cylindrical_jar_canonical
    store = ModelStore(tmp_path / "models")
    store.persist(model)

    jar_step = Path(__file__).resolve().parents[4] / "Solidworks" / "jar.STEP"
    if jar_step.exists():
        persist_test_cad_reference(store, model.id, jar_step)

    view_dir = tmp_path / "views" / model.id
    view_dir.mkdir(parents=True)

    vertices = np.array(model.mesh.vertices, dtype=np.float64)
    faces = np.array(model.mesh.faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    canonical_hash = "test_hash"
    side_svg = build_projection_svg(
        mesh,
        plane=VIEW_CONVENTIONS["side"]["plane"],
        center=model.geometry.center_mm,
        principal_axes=model.geometry.principal_axes,
        model_id=model.id,
        canonical_mesh_sha256=canonical_hash,
    )
    top_svg = build_projection_svg(
        mesh,
        plane=VIEW_CONVENTIONS["top"]["plane"],
        center=model.geometry.center_mm,
        principal_axes=model.geometry.principal_axes,
        model_id=model.id,
        canonical_mesh_sha256=canonical_hash,
    )
    (view_dir / "side_composite.svg").write_text(side_svg, encoding="utf-8")
    (view_dir / "top_composite.svg").write_text(top_svg, encoding="utf-8")
    metadata = {
        "model_id": model.id,
        "displayed_views": {
            "side": {
                "filename": "side_composite.svg",
                "plane": "XY",
                "view_axis": "Z",
                "sha256": "side",
                "canonical_mesh_sha256": canonical_hash,
            },
            "top": {
                "filename": "top_composite.svg",
                "plane": "XZ",
                "view_axis": "Y",
                "sha256": "top",
                "canonical_mesh_sha256": canonical_hash,
            },
        },
    }
    (view_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return view_dir
