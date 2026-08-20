"""ContactConstraintLattice keeps only envelope-admissible cage samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.unit.engines.compute.test_scraper_parametric_v1 import _ellipse
from tests.unit.engines.compute.test_scraper_trajectory_width import _bowl
from tests.unit.engines.visualization.test_scraper_control_cage import _reference_path

from nutella_scraper.engines.visualization.scraper_control_cage import (
    CAGE_ROW_OFFSETS_MM,
    build_control_cage_overlay,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (
    CONTACT_TOLERANCE_MM,
    cage_segment_crosses_exterior,
    envelope_contact_metrics,
    filter_control_cage,
    form_a_row_indices,
    generate_candidate_shapes,
    lattice_from_cage,
)

CAGE_SRC = Path("src/nutella_scraper/engines/visualization/scraper_control_cage.py")
SHAPE_SRC = Path("src/nutella_scraper/engines/visualization/scraper_shape_space.py")
COLLISION_SRC = Path(
    "src/nutella_scraper/engines/compute/scraper_envelope_collision.py"
)
HTML_SRC = Path("scripts/templates/demo_viewer.html")


def test_admissible_points_are_not_outside_the_envelope() -> None:
    surface = _ellipse()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    points = np.asarray(cage["points_mm"], dtype=np.float64)
    assert len(points) == lattice.admissible_count
    signed, _unsigned = envelope_contact_metrics(surface, points)
    assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM
    assert np.all(np.isfinite(lattice.points_mm[lattice.admissible]))


def test_admissible_points_do_not_cross_the_glass() -> None:
    surface = _ellipse()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    assert cage_segment_crosses_exterior(lattice, surface) is False
    signed = lattice.signed_mm[lattice.admissible]
    assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM


def test_center_row_a0_is_unchanged() -> None:
    surface = _ellipse()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    centerline = np.asarray(cage["centerline_mm"], dtype=np.float64)
    assert np.allclose(centerline, path.wall_curve_mm, atol=1e-3)
    assert np.all(lattice.admissible[lattice.center_row_index])
    stored = np.asarray(
        lattice.points_mm[lattice.center_row_index], dtype=np.float64
    )
    assert np.allclose(stored, path.wall_curve_mm, atol=1e-3)
    catalog = generate_candidate_shapes(lattice, count=1)
    assert np.allclose(
        np.asarray(catalog[0].control_points_mm, dtype=np.float64),
        path.wall_curve_mm,
        atol=1e-3,
    )
    assert catalog[0].row_indices == form_a_row_indices(lattice)


def test_lateral_rows_stay_complete_when_the_envelope_narrows() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    assert int(cage["base_row_count"]) == 11
    assert lattice.admissible_count == lattice.nominal_count
    assert int(cage["removed_point_count"]) == 0
    assert np.all(lattice.admissible)
    assert np.all(np.asarray(cage["admissible"][0], dtype=bool))
    assert np.all(np.asarray(cage["admissible"][int(cage["center_row_index"])], dtype=bool))
    signed, unsigned = envelope_contact_metrics(
        surface, np.asarray(lattice.points_mm[0], dtype=np.float64)
    )
    assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM
    assert float(np.max(unsigned)) <= CONTACT_TOLERANCE_MM + 0.05


def test_no_cage_segment_crosses_a_deleted_zone() -> None:
    surface = _ellipse()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    assert cage_segment_crosses_exterior(lattice, surface) is False
    for row_index, row in enumerate(cage["polylines_mm"]):
        assert all(point is not None for point in row)
        for station, _point in enumerate(row[:-1]):
            nxt = row[station + 1]
            assert cage["admissible"][row_index][station] is True
            assert cage["admissible"][row_index][station + 1] is True
            assert nxt is not None


def test_filtered_count_never_exceeds_nominal() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    assert lattice.admissible_count == lattice.nominal_count
    assert int(cage["point_count"]) == int(cage["nominal_point_count"])
    assert int(cage["point_count"]) == lattice.admissible_count
    assert int(cage["removed_point_count"]) == 0


def test_regenerated_cage_has_stable_admissible_fingerprint() -> None:
    surface = _ellipse()
    path, _params, _max_length = _reference_path(surface)
    first = build_control_cage_overlay(path, surface)
    second = build_control_cage_overlay(path, surface)
    lattice_a = lattice_from_cage(first, surface)
    lattice_b = lattice_from_cage(second, surface)
    assert first["admissible_fingerprint"] == second["admissible_fingerprint"]
    assert lattice_a.fingerprint == lattice_b.fingerprint
    assert first["admissible"] == second["admissible"]
    assert first["points_mm"] == second["points_mm"]


def test_filter_does_not_touch_collision_play_or_candidate_families() -> None:
    collision = COLLISION_SRC.read_text(encoding="utf-8")
    shape = SHAPE_SRC.read_text(encoding="utf-8")
    cage = CAGE_SRC.read_text(encoding="utf-8")
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "scraper_control_cage" not in collision
    assert "filter_control_cage" not in collision
    assert "rotationCache" not in cage
    assert "Play" not in cage
    assert "generate_candidate_shapes" in shape
    overlay = html[
        html.index("function drawControlCageOverlay") : html.index(
            "function pchipSlopes"
        )
    ]
    assert "pt.length < 3" in overlay
    assert "started = false" in overlay
    assert "API.buildScraper" not in overlay
    assert "loadShapeCandidateCatalog" not in overlay


def _radial_scale(point: np.ndarray, axis_xz: np.ndarray, factor: float) -> list[float]:
    out = np.asarray(point, dtype=np.float64).copy()
    out[0] = float(axis_xz[0]) + factor * (out[0] - float(axis_xz[0]))
    out[2] = float(axis_xz[1]) + factor * (out[2] - float(axis_xz[1]))
    return [float(out[0]), float(out[1]), float(out[2])]


def test_original_grid_is_kept_and_only_outside_points_are_dropped() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    raw = build_control_cage_overlay(path, None)
    filtered = build_control_cage_overlay(path, surface)
    assert raw["row_count"] == 11
    assert int(filtered["base_row_count"]) == 11
    assert filtered["row_count"] >= 11
    assert len(raw["nominal_candidates"]) == 11
    assert len(filtered["polylines_mm"]) == filtered["row_count"]
    center = np.asarray(filtered["centerline_mm"], dtype=np.float64)
    assert np.allclose(center, path.wall_curve_mm, atol=1e-3)
    assert np.allclose(center, np.asarray(raw["centerline_mm"], dtype=np.float64), atol=1e-9)
    for row in filtered["polylines_mm"]:
        assert all(pt is not None for pt in row)
        ys = [float(pt[1]) for pt in row]
        if len(ys) >= 2:
            diffs = np.diff(ys)
            assert bool(np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6))


def test_outside_point_is_reprojected_on_and_inside_points_are_kept() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    raw = build_control_cage_overlay(path, None)
    verts = np.asarray(surface.vertices, dtype=np.float64)
    axis = 0.5 * (np.min(verts, axis=0)[[0, 2]] + np.max(verts, axis=0)[[0, 2]])
    station = len(path.stations) // 2
    original = list(raw["candidates"][0][station])
    outside = _radial_scale(original, axis, 1.35)
    on_wall = list(raw["centerline_mm"][station])
    inside = _radial_scale(on_wall, axis, 0.85)
    signed_out, _u_out = envelope_contact_metrics(surface, np.asarray([outside]))
    signed_on, unsigned_on = envelope_contact_metrics(surface, np.asarray([on_wall]))
    signed_in, unsigned_in = envelope_contact_metrics(surface, np.asarray([inside]))
    assert float(signed_out[0]) < -CONTACT_TOLERANCE_MM
    assert float(unsigned_on[0]) <= CONTACT_TOLERANCE_MM + 1e-6
    assert float(signed_on[0]) >= -CONTACT_TOLERANCE_MM
    assert float(signed_in[0]) > 0.0
    assert float(unsigned_in[0]) > CONTACT_TOLERANCE_MM

    mutated = dict(raw)
    rows = [list(row) for row in raw["candidates"]]
    rows[0] = list(rows[0])
    rows[2] = list(rows[2])
    rows[0][station] = outside
    rows[2][station] = inside
    mutated["candidates"] = rows
    mutated["nominal_candidates"] = rows
    mutated["polylines_mm"] = rows
    filtered = filter_control_cage(mutated, surface)
    snapped = np.asarray(filtered["polylines_mm"][0][station], dtype=np.float64)
    interior_snapped = np.asarray(filtered["polylines_mm"][2][station], dtype=np.float64)
    signed_snap, unsigned_snap = envelope_contact_metrics(surface, snapped.reshape(1, 3))
    signed_in_snap, unsigned_in_snap = envelope_contact_metrics(
        surface, interior_snapped.reshape(1, 3)
    )
    assert filtered["polylines_mm"][0][station] is not None
    assert filtered["polylines_mm"][2][station] is not None
    assert filtered["polylines_mm"][0][station - 1] is not None
    assert float(unsigned_snap[0]) <= CONTACT_TOLERANCE_MM + 0.05
    assert float(signed_snap[0]) >= -CONTACT_TOLERANCE_MM
    assert float(unsigned_in_snap[0]) <= CONTACT_TOLERANCE_MM + 0.05
    assert filtered["admissible"][0][station] is True
    assert filtered["admissible"][int(filtered["center_row_index"])][station] is True
    lattice = lattice_from_cage(mutated, surface)
    assert bool(lattice.admissible[0, station]) is True
    assert bool(np.all(lattice.admissible[0]))


def test_one_invalid_station_does_not_delete_the_row() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    raw = build_control_cage_overlay(path, None)
    verts = np.asarray(surface.vertices, dtype=np.float64)
    axis = 0.5 * (np.min(verts, axis=0)[[0, 2]] + np.max(verts, axis=0)[[0, 2]])
    station = 4
    rows = [list(row) for row in raw["candidates"]]
    rows[1] = list(rows[1])
    rows[1][station] = _radial_scale(rows[1][station], axis, 1.4)
    payload = dict(raw)
    payload["candidates"] = rows
    payload["nominal_candidates"] = rows
    payload["polylines_mm"] = rows
    filtered = filter_control_cage(payload, surface)
    kept = [pt is not None for pt in filtered["polylines_mm"][1]]
    assert all(kept)
    snapped = np.asarray(filtered["polylines_mm"][1][station], dtype=np.float64)
    _signed, unsigned = envelope_contact_metrics(surface, snapped.reshape(1, 3))
    assert float(unsigned[0]) <= CONTACT_TOLERANCE_MM + 0.05
    assert int(filtered["base_row_count"]) == 11
    assert filtered["row_count"] >= 11


def test_a0_and_candidate_pipeline_are_untouched() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    assert np.allclose(
        np.asarray(cage["centerline_mm"], dtype=np.float64),
        path.wall_curve_mm,
        atol=1e-3,
    )
    assert int(np.count_nonzero(lattice.admissible[lattice.center_row_index])) == (
        lattice.station_count
    )
    catalog = generate_candidate_shapes(lattice, count=1)
    assert catalog[0].shape.vertices is None
    assert catalog[0].shape.thickness_mm == 2.5
    collision = COLLISION_SRC.read_text(encoding="utf-8")
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "scraper_control_cage" not in collision
    assert "const rotationCache = new Map()" in html
    assert "function generate_candidate_shapes" not in html


def test_circular_bowl_keeps_the_full_constructed_lattice() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    raw = build_control_cage_overlay(path, None)
    filtered = build_control_cage_overlay(path, surface)
    expected = len(CAGE_ROW_OFFSETS_MM) * len(path.stations)
    assert int(raw["nominal_point_count"]) == expected
    assert int(filtered["base_row_count"]) == 11
    assert int(filtered["point_count"]) >= expected
    assert int(filtered["removed_point_count"]) == 0
    assert all(all(row) for row in filtered["admissible"])
    assert CONTACT_TOLERANCE_MM == 0.5


def test_unsigned_band_is_not_a_deletion_rule() -> None:
    """Interior samples are snapped onto the envelope, not deleted."""
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    raw = build_control_cage_overlay(path, None)
    verts = np.asarray(surface.vertices, dtype=np.float64)
    axis = 0.5 * (np.min(verts, axis=0)[[0, 2]] + np.max(verts, axis=0)[[0, 2]])
    station = len(path.stations) // 2
    interior = _radial_scale(raw["centerline_mm"][station], axis, 0.7)
    signed, unsigned = envelope_contact_metrics(surface, np.asarray([interior]))
    assert float(signed[0]) >= -CONTACT_TOLERANCE_MM
    assert float(unsigned[0]) > CONTACT_TOLERANCE_MM
    rows = [list(row) for row in raw["candidates"]]
    rows[3] = list(rows[3])
    rows[3][station] = interior
    payload = dict(raw)
    payload["candidates"] = rows
    payload["nominal_candidates"] = rows
    payload["polylines_mm"] = rows
    filtered = filter_control_cage(payload, surface)
    assert filtered["polylines_mm"][3][station] is not None
    assert filtered["admissible"][3][station] is True
    assert filtered["admissible"][3][station - 1] is True
    assert filtered["admissible"][3][station + 1] is True
    snapped = np.asarray(filtered["polylines_mm"][3][station], dtype=np.float64)
    _signed, unsigned_snap = envelope_contact_metrics(surface, snapped.reshape(1, 3))
    assert float(unsigned_snap[0]) <= CONTACT_TOLERANCE_MM + 0.05


def test_lattice_has_continuous_longitudinal_rows() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    assert lattice.row_count >= 11
    assert lattice.station_count == len(path.stations)
    assert np.all(lattice.admissible)
    for row in range(lattice.row_count):
        pts = np.asarray(lattice.points_mm[row], dtype=np.float64)
        assert pts.shape == (lattice.station_count, 3)
        assert np.all(np.isfinite(pts))
        ys = pts[:, 1]
        diffs = np.diff(ys)
        assert bool(np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6))
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert float(np.max(steps)) < 25.0
        overlay_row = cage["polylines_mm"][row]
        assert all(point is not None for point in overlay_row)


def test_lateral_rows_are_available_for_candidate_generation() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    lattice = lattice_from_cage(cage, surface)
    catalog = generate_candidate_shapes(lattice, count=12)
    assert catalog[0].candidate_id == "A0"
    assert np.allclose(
        np.asarray(catalog[0].control_points_mm, dtype=np.float64),
        path.wall_curve_mm,
        atol=1e-3,
    )
    others = [c for c in catalog if c.candidate_id != "A0" and c.valid]
    assert len(others) >= 3
    fingerprints = {c.shape_fingerprint for c in others}
    assert catalog[0].shape_fingerprint not in fingerprints
    assert len(fingerprints) >= 3
    families = {c.family for c in others}
    assert "parallel" in families
