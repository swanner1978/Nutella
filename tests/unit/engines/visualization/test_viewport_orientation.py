"""Viewport orientation: jar opening toward the top of the SVG screen."""

from __future__ import annotations

import numpy as np

from nutella_scraper.engines.visualization.projection_math import fit_to_viewport, project_vertices
from scripts.visualization_helpers import VIEW_CONVENTIONS, fit_to_viewport as helpers_fit


def test_fit_to_viewport_maps_larger_world_v_toward_screen_top() -> None:
    """SVG y increases downward; larger world V must get a smaller SVG y."""
    coords = np.array([[0.0, -10.0], [0.0, 90.0]], dtype=np.float64)
    scale, offset = fit_to_viewport(coords)
    assert scale.shape == (2,)
    assert scale[0] > 0.0
    assert scale[1] < 0.0

    base_svg = coords[0] * scale + offset
    opening_svg = coords[1] * scale + offset
    assert opening_svg[1] < base_svg[1]


def test_helpers_fit_matches_projection_math_orientation() -> None:
    coords = np.array([[-5.0, 0.0], [5.0, 40.0]], dtype=np.float64)
    scale_a, offset_a = fit_to_viewport(coords)
    scale_b, offset_b = helpers_fit(coords)
    np.testing.assert_allclose(scale_a, scale_b)
    np.testing.assert_allclose(offset_a, offset_b)


def test_profile_left_right_opening_is_above_base_in_svg() -> None:
    """For vertical jar views, max world Y projects above min world Y."""
    vertices = np.array(
        [
            [-40.0, -10.0, 0.0],
            [40.0, -10.0, 0.0],
            [-30.0, 100.0, 0.0],
            [30.0, 100.0, 0.0],
            [0.0, 50.0, -40.0],
            [0.0, 50.0, 40.0],
        ],
        dtype=np.float64,
    )
    for view_name in ("side", "left", "right"):
        plane = VIEW_CONVENTIONS[view_name]["plane"]
        coords, _, _ = project_vertices(vertices, plane)
        scale, offset = fit_to_viewport(coords)
        y_world = vertices[:, 1]
        base_3d = vertices[int(y_world.argmin())]
        rim_3d = vertices[int(y_world.argmax())]
        base_2d, _, _ = project_vertices(base_3d.reshape(1, 3), plane)
        rim_2d, _, _ = project_vertices(rim_3d.reshape(1, 3), plane)
        base_svg = base_2d[0] * scale + offset
        rim_svg = rim_2d[0] * scale + offset
        assert rim_svg[1] < base_svg[1], f"{view_name}: opening must be above base"


def test_orientation_does_not_mutate_input_coordinates() -> None:
    coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    original = coords.copy()
    fit_to_viewport(coords)
    np.testing.assert_array_equal(coords, original)
