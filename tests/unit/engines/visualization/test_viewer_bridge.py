"""Viewer bridge integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from nutella_scraper.engines.visualization.viewer_bridge import (
    build_contact_visualization_response,
    view_cache_from_viewer_dir,
)


@pytest.fixture
def viewer_dir(tmp_path: Path, cylindrical_jar_canonical: object) -> Path:
    from nutella_scraper.cad_import.model_store import ModelStore
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
                "plane": VIEW_CONVENTIONS["side"]["plane"],
                "view_axis": "Y",
                "sha256": "side",
                "canonical_mesh_sha256": canonical_hash,
            },
            "top": {
                "filename": "top_composite.svg",
                "plane": VIEW_CONVENTIONS["top"]["plane"],
                "view_axis": "Z",
                "sha256": "top",
                "canonical_mesh_sha256": canonical_hash,
            },
        },
    }
    (view_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return view_dir


class TestViewerBridge:
    def test_view_cache_from_viewer_dir(self, viewer_dir: Path) -> None:
        cache = view_cache_from_viewer_dir(viewer_dir)
        assert cache.model_id == viewer_dir.name
        assert cache.profile_view.plane == "XY"
        assert cache.top_view.plane == "XZ"

    def test_build_contact_visualization_response(self, viewer_dir: Path) -> None:
        models_root = viewer_dir.parent.parent / "models"
        if not (models_root / viewer_dir.name / "cad_reference.json").exists():
            pytest.skip("cad_reference.json required for contact overlays")
        payload = build_contact_visualization_response(
            view_dir=viewer_dir,
            models_root=models_root,
        )
        assert payload["model_id"] == viewer_dir.name
        assert "contact-covered" in payload["overlays"]["side"]
