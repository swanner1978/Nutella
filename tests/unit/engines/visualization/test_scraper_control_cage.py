"""Control cage overlay — 11 search rows around scraper A centreline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.unit.engines.compute.test_scraper_parametric_v1 import (
    _profile_a,
    _reference_from_profile,
)
from tests.unit.engines.compute.test_scraper_trajectory_width import (
    _bowl,
    _station_xz_span,
)

from nutella_scraper.engines.compute.scraper_envelope_path import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    ScraperEnvelopePathBuilder,
    scraper_length_span,
)
from nutella_scraper.engines.visualization.scraper_control_cage import (
    CAGE_CENTER_ROW_INDEX,
    CAGE_ROW_OFFSETS_MM,
    build_control_cage_overlay,
)

CAGE_SRC = Path("src/nutella_scraper/engines/visualization/scraper_control_cage.py")
BRIDGE_SRC = Path("src/nutella_scraper/engines/visualization/viewer_bridge.py")
COLLISION_SRC = Path(
    "src/nutella_scraper/engines/compute/scraper_envelope_collision.py"
)
MOTION_SRC = Path("src/nutella_scraper/engines/compute/scraper_rigid_motion.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")


def _reference_path(surface, *, width_mm: float = 2.5):
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    params = _profile_a(
        width_mm=width_mm,
        length_mm=max_length,
        thickness_mm=2.5,
        clearance_mm=0.0,
    )
    return ScraperEnvelopePathBuilder().build(surface, params), params, max_length


def test_control_cage_module_is_visualization_only() -> None:
    text = CAGE_SRC.read_text(encoding="utf-8")
    assert "scraper_envelope_collision" not in text
    assert "scraper_rigid_motion" not in text
    assert "optimization" not in text
    assert "rotationCache" not in text
    assert "Play" not in text
    bridge = BRIDGE_SRC.read_text(encoding="utf-8")
    assert "build_control_cage_overlay" in bridge
    assert "control_cage" in bridge
    collision = COLLISION_SRC.read_text(encoding="utf-8")
    motion = MOTION_SRC.read_text(encoding="utf-8")
    assert "scraper_control_cage" not in collision
    assert "scraper_control_cage" not in motion


def test_control_cage_has_eleven_search_rows() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    assert cage["row_count"] == 11
    assert cage["center_row_index"] == CAGE_CENTER_ROW_INDEX
    assert cage["row_offsets_mm"] == list(CAGE_ROW_OFFSETS_MM)
    assert len(cage["polylines_mm"]) == 11
    assert len(cage["candidates"]) == 11
    assert cage["lofted"] is False
    assert "faces" not in cage
    assert "vertices" not in cage


def test_centerline_matches_trajectory_a() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    centerline = np.asarray(cage["centerline_mm"], dtype=np.float64)
    spine = np.asarray(path.wall_curve_mm, dtype=np.float64)
    assert np.allclose(centerline, spine, atol=1e-3)
    center_row = np.asarray(
        cage["polylines_mm"][CAGE_CENTER_ROW_INDEX], dtype=np.float64
    )
    assert np.allclose(center_row, spine, atol=1e-3)


def test_yellow_blade_is_thin_and_follows_centerline() -> None:
    surface = _bowl()
    path, params, _max_length = _reference_path(surface, width_mm=2.5)
    cage = build_control_cage_overlay(path, surface)
    assert params.width_mm == pytest.approx(2.5)
    assert params.thickness_mm == pytest.approx(2.5)
    spans = [_station_xz_span(station) for station in path.stations]
    assert max(spans) < 5.0
    centerline = np.asarray(cage["centerline_mm"], dtype=np.float64)
    assert np.allclose(centerline, path.wall_curve_mm, atol=1e-3)


def test_cage_points_lie_on_interior_envelope() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    points = np.asarray(cage["points_mm"], dtype=np.float64)
    assert len(points) >= len(path.stations)
    mesh = surface.to_trimesh()
    _closest, distances, _tri = mesh.nearest.on_surface(points)
    assert float(np.max(distances)) < 0.5
    assert cage["source"] == SOURCE_INTERIOR_PRODUCT_SURFACE


def test_control_cage_covers_full_reference_length() -> None:
    surface = _reference_from_profile(radius_at_y=lambda _y: 50.0, y_max=128.0)
    opening_y, lower_y, max_length = scraper_length_span(surface)
    assert max_length == pytest.approx(128.0)
    path, _params, _ = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    ys = np.asarray(cage["centerline_mm"], dtype=np.float64)[:, 1]
    assert float(np.ptp(ys)) >= 110.0
    assert float(np.max(ys)) >= opening_y - 3.0
    assert float(np.min(ys)) <= lower_y + 8.0


def test_search_rows_are_not_a_volume_plate() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    blade_span = max(_station_xz_span(station) for station in path.stations)
    assert blade_span < 5.0
    outer = np.asarray(cage["polylines_mm"][0], dtype=np.float64)
    if len(outer) > 0:
        mid_c = center[len(center) // 2]
        mid_o = outer[len(outer) // 2]
        assert float(np.linalg.norm(mid_o - mid_c)) > 5.0


def test_demo_viewer_reference_a_is_a_thin_blade() -> None:
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "width_mm: 2.5" in html
    assert "thickness_mm: 2.5" in html
    assert "length_mm: 10000" in html
    assert "center_row_index" in html
    assert "async function startRotationPlay" in html
    assert "const rotationCache" in html
    assert "rotationCache.get" in html
