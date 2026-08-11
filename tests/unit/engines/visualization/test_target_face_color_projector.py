"""Unit tests for STEP target-face colour SVG projector."""

from __future__ import annotations

import numpy as np

from nutella_scraper.engines.visualization.cad_reference_projector import (
    LAYER_INTERIOR_PROFILE,
)
from nutella_scraper.engines.visualization.target_face_color_projector import (
    LAYER_TARGET_FACE_COLORS,
    TargetFaceColorProjector,
)


def test_target_face_color_projector_default_is_interior_envelope() -> None:
    jar = np.array(
        [
            [-50.0, 0.0, -50.0],
            [50.0, 0.0, -50.0],
            [50.0, 100.0, -50.0],
            [-50.0, 100.0, -50.0],
            [-50.0, 0.0, 50.0],
            [50.0, 0.0, 50.0],
            [50.0, 100.0, 50.0],
            [-50.0, 100.0, 50.0],
        ],
        dtype=np.float64,
    )
    face_vertices = np.array(
        [
            [-20.0, 20.0, -20.0],
            [20.0, 20.0, -20.0],
            [0.0, 80.0, -20.0],
        ],
        dtype=np.float64,
    )
    face_triangles = np.array([[0, 1, 2]], dtype=np.int64)

    projection = TargetFaceColorProjector().project(
        face_vertices=face_vertices,
        face_triangles=face_triangles,
        jar_vertices=jar,
        target_face_count=13,
        target_area_mm2=40467.74,
        fill_rgb_255=(85, 255, 255),
    )

    assert len(projection.profile_layers) == 1
    layer = projection.profile_layers[0]
    assert layer.layer_type == LAYER_INTERIOR_PROFILE
    assert "rgb(85,255,255)" in layer.svg_fragment
    assert "Target faces:" not in layer.svg_fragment
    assert layer.svg_fragment.startswith("<path")


def test_debug_layer_can_include_labels() -> None:
    jar = np.array(
        [[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [10.0, 20.0, 10.0], [-10.0, 20.0, 10.0]],
        dtype=np.float64,
    )
    verts = np.array([[0.0, 5.0, 0.0], [2.0, 5.0, 0.0], [1.0, 8.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    projection = TargetFaceColorProjector().project(
        face_vertices=verts,
        face_triangles=faces,
        jar_vertices=jar,
        target_face_count=13,
        target_area_mm2=40467.74,
        layer_type=LAYER_TARGET_FACE_COLORS,
        include_labels=True,
    )
    assert projection.profile_layers[0].layer_type == LAYER_TARGET_FACE_COLORS
    assert "Target faces: 13" in projection.profile_layers[0].svg_fragment
