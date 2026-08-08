"""Shared fixtures for CAD import tests."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import trimesh
except ImportError:
    trimesh = None  # type: ignore[assignment]

from nutella_scraper.domain.models.canonical import JarCanonicalModel, JarProfilePoint
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder


@pytest.fixture
def box_stl_path(tmp_path: Path) -> Path:
    """Watertight 10x20x30 mm box exported as STL."""
    if trimesh is None:
        pytest.skip("trimesh not installed")
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    path = tmp_path / "box.stl"
    mesh.export(str(path))
    return path


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    if trimesh is None:
        pytest.skip("trimesh not installed")
    return trimesh.creation.box(extents=(10.0, 20.0, 30.0))


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


def persist_test_cad_reference(store, model_id: str, step_path: Path):
    pytest.importorskip("OCP")
    from nutella_scraper.cad_import.cad_reference_builder import CadReferenceGeometryBuilder

    geometry = CadReferenceGeometryBuilder().from_step(step_path, model_id=model_id)
    store.persist_cad_reference(model_id, geometry, step_path=step_path)
    return geometry
