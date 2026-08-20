"""Control cage overlay — 11 search rows around scraper A centreline."""

from __future__ import annotations

import math
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
    CAGE_ROW_OFFSETS_MM,
    build_control_cage_overlay,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (
    CONTACT_TOLERANCE_MM,
    DEFAULT_CANDIDATE_COUNT,
    envelope_contact_metrics,
    generate_candidate_shapes,
    lattice_from_cage,
)

CAGE_SRC = Path("src/nutella_scraper/engines/visualization/scraper_control_cage.py")
BRIDGE_SRC = Path("src/nutella_scraper/engines/visualization/viewer_bridge.py")
COLLISION_SRC = Path(
    "src/nutella_scraper/engines/compute/scraper_envelope_collision.py"
)
MOTION_SRC = Path("src/nutella_scraper/engines/compute/scraper_rigid_motion.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")


def _dense_rows(cage: dict) -> list:
    return cage.get("nominal_candidates") or cage["polylines_mm"]


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
    assert int(cage["base_row_count"]) == 11
    assert cage["row_count"] >= 11
    assert cage["row_offsets_mm"][int(cage["center_row_index"])] == 0.0
    assert list(cage["base_row_offsets_mm"]) == list(CAGE_ROW_OFFSETS_MM)
    assert len(cage["polylines_mm"]) == cage["row_count"]
    assert len(cage["candidates"]) == cage["row_count"]
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
        cage["polylines_mm"][int(cage["center_row_index"])], dtype=np.float64
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
    assert float(np.max(distances)) <= 0.5
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
    outer = np.asarray(_dense_rows(cage)[0], dtype=np.float64)
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


def test_every_row_has_one_point_per_station() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    n_stations = len(path.stations)
    for row in _dense_rows(cage):
        assert len(row) == n_stations
        assert all(point is not None for point in row)
    for row in cage["polylines_mm"]:
        assert len(row) == n_stations
        assert all(point is not None for point in row)


def test_station_points_share_one_transverse_frame() -> None:
    """All 11 samples of a station sit on the same jar parallel (constant Y)."""
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    dense = _dense_rows(cage)
    for i, origin in enumerate(center):
        ys = [
            float(np.asarray(dense[row][i], dtype=np.float64)[1])
            for row in range(len(dense))
        ]
        assert max(ys) - min(ys) < 0.25
        assert abs(float(np.mean(ys)) - float(origin[1])) < 0.25


def test_transverse_frame_does_not_flip_along_centreline() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    right = np.asarray(
        _dense_rows(cage)[int(cage["center_row_index"]) + 1], dtype=np.float64
    )
    tangents = np.vstack(
        [center[1] - center[0], center[2:] - center[:-2], center[-1] - center[-2]]
    )
    previous = None
    for i, origin in enumerate(center):
        vec = right[i] - origin
        if float(np.linalg.norm(vec)) < 3.0:
            continue
        tangent = tangents[i]
        tn = float(np.linalg.norm(tangent))
        if tn > 1e-9:
            tangent = tangent / tn
            vec = vec - tangent * float(np.dot(vec, tangent))
        direction = vec / max(float(np.linalg.norm(vec)), 1e-9)
        if previous is not None:
            assert float(np.dot(direction, previous)) > 0.7
        previous = direction


def test_row_steps_stay_continuous_including_near_the_floor() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    spine_steps = np.linalg.norm(np.diff(center, axis=0), axis=1)
    floor_count = max(6, len(center) // 5)
    for row_pts in _dense_rows(cage):
        pts = np.asarray(row_pts, dtype=np.float64)
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        allowed = np.maximum(spine_steps, 1.0) * 4.0 + 8.0
        assert float(np.max(steps)) < 20.0
        assert bool(np.all(steps <= allowed + 1e-6))
        floor_steps = steps[:floor_count]
        assert float(np.max(floor_steps)) < 12.0


def test_outer_rows_taper_toward_the_floor_without_lateral_jumps() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    outer = np.asarray(_dense_rows(cage)[0], dtype=np.float64)
    radii = np.linalg.norm(outer - center, axis=1)
    opening = int(np.argmax(center[:, 1]))
    floor = int(np.argmin(center[:, 1]))
    assert radii[opening] > 20.0
    assert radii[floor] < radii[opening] * 0.35
    delta = np.abs(np.diff(radii))
    assert float(np.max(delta)) < 12.0


def test_cage_does_not_use_independent_section_walks() -> None:
    text = CAGE_SRC.read_text(encoding="utf-8")
    assert "_offset_via_transverse_section" not in text
    assert "_offset_via_surface_walk" not in text
    assert "def _radial_constrain" not in text
    assert "nearest.on_surface" not in text
    assert "parallel" in text.lower() or "Bishop" in text or "transport" in text.lower()


def _axis_xz_from_surface(surface) -> np.ndarray:
    verts = np.asarray(surface.vertices, dtype=np.float64)
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    return 0.5 * (mins[[0, 2]] + maxs[[0, 2]])


def _row_azimuths(row: np.ndarray, axis_xz: np.ndarray) -> np.ndarray:
    dx = row[:, 0] - float(axis_xz[0])
    dz = row[:, 2] - float(axis_xz[1])
    return np.arctan2(dz, dx)


def test_side_columns_keep_stable_azimuth_in_top_view() -> None:
    """A column (constant offset) is a meridian — not a jittered polyline."""
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    axis = _axis_xz_from_surface(surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    radii = np.hypot(center[:, 0] - axis[0], center[:, 2] - axis[1])
    usable = radii > 8.0
    assert int(np.count_nonzero(usable)) >= 6
    offsets = [float(v) for v in cage["row_offsets_mm"]]
    for offset in (-50.0, -40.0, -30.0, 30.0, 40.0, 50.0):
        row_index = offsets.index(offset)
        pts = np.asarray(_dense_rows(cage)[row_index], dtype=np.float64)
        azimuth = np.unwrap(_row_azimuths(pts[usable], axis))
        assert float(np.ptp(azimuth)) < math.radians(10.0)
        if len(azimuth) >= 2:
            assert float(np.max(np.abs(np.diff(azimuth)))) < math.radians(6.0)


def test_floor_points_stay_on_centreline_halfplane() -> None:
    """No side point may jump to the opposite side of the jar."""
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    axis = _axis_xz_from_surface(surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    floor_n = max(6, len(center) // 5)
    opening = center[int(np.argmax(center[:, 1]))]
    meridian = np.array(
        [opening[0] - axis[0], opening[2] - axis[1]], dtype=np.float64
    )
    meridian = meridian / max(float(np.linalg.norm(meridian)), 1e-9)
    for row_pts in _dense_rows(cage):
        pts = np.asarray(row_pts, dtype=np.float64)[:floor_n]
        rel = pts[:, [0, 2]] - axis
        projection = rel @ meridian
        assert float(np.min(projection)) > -4.0


def test_successive_stations_do_not_flip_or_jump() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    floor_n = max(8, len(center) // 4)
    for row_pts in _dense_rows(cage):
        pts = np.asarray(row_pts, dtype=np.float64)
        delta = np.diff(pts, axis=0)
        steps = np.linalg.norm(delta, axis=1)
        headings = delta / np.maximum(steps[:, None], 1e-9)
        if len(headings) >= 2:
            turn = np.sum(headings[:-1] * headings[1:], axis=1)
            floor_turn = turn[: max(floor_n - 1, 1)]
            assert float(np.min(floor_turn)) > 0.0
        floor_steps = steps[: max(floor_n - 1, 1)]
        assert float(np.max(floor_steps)) < 10.0


_OUTER_OFFSETS = (-50.0, -40.0, -30.0, 30.0, 40.0, 50.0)


def _row_for_offset(cage: dict, offset: float) -> np.ndarray:
    offsets = [float(v) for v in cage["row_offsets_mm"]]
    return np.asarray(_dense_rows(cage)[offsets.index(offset)], dtype=np.float64)


def test_eleven_base_rows_are_complete_longitudinal_trajectories() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    assert int(cage["base_row_count"]) == 11
    n_stations = len(path.stations)
    for offset in CAGE_ROW_OFFSETS_MM:
        pts = _row_for_offset(cage, offset)
        assert pts.shape == (n_stations, 3)
        assert np.all(np.isfinite(pts))
        ys = pts[:, 1]
        diffs = np.diff(ys)
        assert bool(np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6))
    assert np.allclose(
        lattice.points_mm[lattice.center_row_index],
        path.wall_curve_mm,
        atol=1e-3,
    )


def test_outer_rows_do_not_lose_stations_including_at_the_floor() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    floor = int(np.argmin(center[:, 1]))
    opening = int(np.argmax(center[:, 1]))
    for offset in _OUTER_OFFSETS:
        pts = _row_for_offset(cage, offset)
        assert np.all(np.isfinite(pts))
        assert pts.shape[0] == len(center)
        signed, unsigned = envelope_contact_metrics(surface, pts)
        assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM
        assert float(np.max(unsigned)) <= CONTACT_TOLERANCE_MM + 0.05
        assert np.linalg.norm(pts[opening] - center[opening]) > 8.0
        assert np.all(np.isfinite(pts[floor]))
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        spine_steps = np.linalg.norm(np.diff(center, axis=0), axis=1)
        assert float(np.max(steps)) < float(np.max(spine_steps)) * 4.0 + 10.0


def test_cage_points_stay_on_interior_envelope_not_in_glass() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    points = np.asarray(cage["points_mm"], dtype=np.float64)
    signed, unsigned = envelope_contact_metrics(surface, points)
    assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM
    assert float(np.max(unsigned)) <= CONTACT_TOLERANCE_MM + 0.05


def test_a0_and_candidates_survive_outer_row_repair() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    catalog = generate_candidate_shapes(lattice, count=8)
    assert catalog[0].candidate_id == "A0"
    assert np.allclose(
        np.asarray(catalog[0].control_points_mm, dtype=np.float64),
        path.wall_curve_mm,
        atol=1e-3,
    )
    assert len(catalog) == 8
    assert all(item.valid for item in catalog)
    default_catalog = generate_candidate_shapes(lattice)
    assert len(default_catalog) == DEFAULT_CANDIDATE_COUNT
    assert default_catalog[0].candidate_id == "A0"
    collision = COLLISION_SRC.read_text(encoding="utf-8")
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "scraper_control_cage" not in collision
    assert "const rotationCache = new Map()" in html
    assert "CANDIDATE_CATALOG_SIZE = 100" in html


def test_base_row_spacing_stays_ordered_and_regular() -> None:
    """Offsets of A stay a regular family; they may bunch at the floor, not vanish."""
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    axis = _axis_xz_from_surface(surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    opening = int(np.argmax(center[:, 1]))
    floor = int(np.argmin(center[:, 1]))

    def _azimuths(station: int) -> np.ndarray:
        values = [
            float(
                np.arctan2(
                    _row_for_offset(cage, offset)[station, 2] - axis[1],
                    _row_for_offset(cage, offset)[station, 0] - axis[0],
                )
            )
            for offset in CAGE_ROW_OFFSETS_MM
        ]
        return np.unwrap(np.asarray(values, dtype=np.float64))

    opening_az = _azimuths(opening)
    opening_step = np.diff(opening_az)
    assert bool(np.all(opening_step > 0.0) or np.all(opening_step < 0.0))
    assert (
        float(np.max(np.abs(opening_step)))
        / max(float(np.min(np.abs(opening_step))), 1e-9)
        < 3.0
    )
    # At the floor the centreline is near the axis, so its atan2 is unstable.
    # The five rows on each side must keep the same cyclic order as at the opening.
    floor_az = _azimuths(floor)
    sign = float(np.sign(opening_step[0]))
    left_step = np.diff(floor_az[:5]) * sign
    right_step = np.diff(floor_az[6:]) * sign
    assert bool(np.all(left_step >= -1e-6))
    assert bool(np.all(right_step >= -1e-6))
    for offset in _OUTER_OFFSETS:
        assert np.all(np.isfinite(_row_for_offset(cage, offset)[floor]))


def _axis_radius(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.hypot(points[:, 0] - axis[0], points[:, 2] - axis[1])


def test_outer_rows_follow_a_across_the_floor_not_the_far_wall() -> None:
    """On a nearly flat floor, outer rows stay beside A instead of jumping to the wall."""
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    axis = _axis_xz_from_surface(surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    r_a = _axis_radius(center, axis)
    floor_stations = np.flatnonzero(r_a < 20.0)
    assert len(floor_stations) >= 3
    for offset in _OUTER_OFFSETS:
        pts = _row_for_offset(cage, offset)
        r_row = _axis_radius(pts, axis)
        for station in floor_stations:
            assert r_row[station] < r_a[station] + 12.0


def test_cached_jar_outer_rows_cover_the_floor_beside_a() -> None:
    models = Path("output/models")
    caches = sorted(
        models.glob("*/interior_product_surface.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not caches:
        pytest.skip("no interior cache")
    from nutella_scraper.engines.compute.interior_surface_reference import (
        load_interior_surface_reference,
    )

    surface = load_interior_surface_reference(
        models_root=models,
        model_id=caches[0].parent.name,
        use_cache=True,
    )
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    axis = _axis_xz_from_surface(surface)
    center = np.asarray(cage["centerline_mm"], dtype=np.float64)
    r_a = _axis_radius(center, axis)
    floor_stations = np.flatnonzero(r_a < 20.0)
    if len(floor_stations) < 3:
        pytest.skip("cached jar has no floor stations near the axis")
    for offset in _OUTER_OFFSETS:
        pts = _row_for_offset(cage, offset)
        assert pts.shape[0] == len(center)
        assert np.all(np.isfinite(pts))
        r_row = _axis_radius(pts, axis)
        for station in floor_stations:
            # Must not remain on the far wall (~35 mm) while A is on the floor.
            assert r_row[station] < r_a[station] + 12.0
        signed, unsigned = envelope_contact_metrics(surface, pts)
        assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM
        assert float(np.max(unsigned)) <= CONTACT_TOLERANCE_MM + 0.05
