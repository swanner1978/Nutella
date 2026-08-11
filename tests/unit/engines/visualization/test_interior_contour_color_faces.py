"""Interior Contour layer uses STEP colour-selected faces (not geometric envelope)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh

pytest.importorskip("OCP")

from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.cad_import.step_face_color_diagnostics import (
    TARGET_RGB_255,
    diagnose_step_face_colors,
)
from nutella_scraper.engines.visualization.cad_reference_projector import (
    LAYER_INTERIOR_PROFILE,
)
from nutella_scraper.engines.visualization.projection_math import (
    PLANE_SIDE,
    fit_to_viewport,
    project_vertices,
)
from nutella_scraper.engines.visualization.target_face_color_projector import (
    TargetFaceColorProjector,
)
from nutella_scraper.engines.visualization.viewer_bridge import (
    build_interior_contour_response,
)
from scripts.visualization_helpers import VIEW_CONVENTIONS, build_projection_svg


@pytest.fixture
def colored_jar_step() -> Path:
    path = Path(__file__).resolve().parents[4] / "Solidworks" / "jar_color-jar.step"
    if not path.exists():
        pytest.skip("Solidworks/jar_color-jar.step missing")
    return path


def test_colored_step_still_detects_thirteen_target_faces(colored_jar_step: Path) -> None:
    diagnostic = diagnose_step_face_colors(colored_jar_step)
    assert diagnostic.matching_face_count == 13
    assert diagnostic.target_rgb_255 == TARGET_RGB_255
    assert abs(diagnostic.total_target_area_mm2 - 40467.74) < 0.1
    assert diagnostic.matching_faces[0].face_id == 18


def test_interior_contour_response_uses_color_faces_as_interior_envelope(
    tmp_path: Path,
    colored_jar_step: Path,
    cylindrical_jar_canonical: object,
) -> None:
    store = ModelStore(tmp_path / "models")
    model = replace(cylindrical_jar_canonical, id="color_interior_test")
    store.persist(model)
    model_dir = tmp_path / "models" / model.id
    (model_dir / ModelStore.REFERENCE_STEP).write_bytes(colored_jar_step.read_bytes())

    view_dir = tmp_path / "views" / model.id
    view_dir.mkdir(parents=True)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(model.mesh.vertices, dtype=np.float64),
        faces=np.asarray(model.mesh.faces, dtype=np.int64),
        process=False,
    )
    for view_name in ("side", "top", "left", "right"):
        plane = VIEW_CONVENTIONS[view_name]["plane"]
        svg = build_projection_svg(
            mesh,
            plane=plane,
            center=model.geometry.center_mm,
            principal_axes=model.geometry.principal_axes,
            model_id=model.id,
            canonical_mesh_sha256="test",
        )
        (view_dir / f"{view_name}_composite.svg").write_text(svg, encoding="utf-8")
    metadata = {
        "model_id": model.id,
        "displayed_views": {
            name: {
                "filename": f"{name}_composite.svg",
                "plane": VIEW_CONVENTIONS[name]["plane"],
                "sha256": "x",
                "canonical_mesh_sha256": "test",
            }
            for name in ("side", "top", "left", "right")
        },
    }
    (view_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    payload = build_interior_contour_response(
        view_dir=view_dir,
        models_root=tmp_path / "models",
    )

    assert payload["interior_colored_faces"] == 13
    assert payload["target_rgb_255"] == [85, 255, 255]
    assert abs(float(payload["target_area_mm2"]) - 40467.74) < 0.1
    assert payload["matching_face_ids"] == list(range(18, 31))
    assert payload["layer"] == LAYER_INTERIOR_PROFILE
    assert payload["source"] == "step_face_color_rgb_85_255_255"

    for view_name in ("side", "top", "left", "right"):
        layer_map = payload["overlays"][view_name]
        assert LAYER_INTERIOR_PROFILE in layer_map
        fragment = layer_map[LAYER_INTERIOR_PROFILE]
        assert f'data-layer="{LAYER_INTERIOR_PROFILE}"' in fragment
        assert "rgb(85,255,255)" in fragment
        assert "<path" in fragment
        assert "wireframe" not in fragment
        assert "vertices" not in fragment
        assert payload["matching_face_ids"] == list(range(18, 31))


def test_projector_uses_jar_viewport_transform_not_independent_face_fit() -> None:
    jar = np.array(
        [
            [-100.0, 0.0, -100.0],
            [100.0, 0.0, -100.0],
            [100.0, 200.0, -100.0],
            [-100.0, 200.0, -100.0],
            [-100.0, 0.0, 100.0],
            [100.0, 0.0, 100.0],
            [100.0, 200.0, 100.0],
            [-100.0, 200.0, 100.0],
        ],
        dtype=np.float64,
    )
    face_vertices = np.array(
        [[0.0, 10.0, 0.0], [1.0, 10.0, 0.0], [0.0, 11.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    jar_coords, _, _ = project_vertices(jar, PLANE_SIDE)
    scale, offset = fit_to_viewport(jar_coords)
    expected = face_vertices[:, [0, 1]] * scale + offset

    projection = TargetFaceColorProjector().project(
        face_vertices=face_vertices,
        face_triangles=faces,
        jar_vertices=jar,
        target_face_count=1,
        target_area_mm2=1.0,
        layer_type=LAYER_INTERIOR_PROFILE,
        include_labels=False,
    )
    fragment = projection.profile_layers[0].svg_fragment
    assert projection.profile_layers[0].layer_type == LAYER_INTERIOR_PROFILE
    assert f"M{expected[0, 0]:.3f},{expected[0, 1]:.3f}" in fragment
    assert "Target faces:" not in fragment
