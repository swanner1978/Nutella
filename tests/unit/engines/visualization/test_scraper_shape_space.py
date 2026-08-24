"""Admissible scraper-shape space — lattice curves, not meshes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from tests.unit.engines.compute.test_scraper_parametric_v1 import _profile_a
from tests.unit.engines.compute.test_scraper_trajectory_width import _bowl

from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.engines.compute.scraper_envelope_path import (
    ScraperEnvelopePathBuilder,
    scraper_length_span,
)
from nutella_scraper.engines.visualization.scraper_control_cage import (
    CAGE_ROW_OFFSETS_MM,
    build_control_cage_overlay,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (
    BLADE_THICKNESS_MM,
    BLADE_WIDTH_MM,
    CONTACT_TOLERANCE_MM,
    REFERENCE_CANDIDATE_ID,
    build_candidate,
    curve_behind_or_through_glass,
    envelope_contact_metrics,
    form_a_row_indices,
    generate_candidate_shapes,
    lattice_from_cage,
    posed_control_points,
    segments_self_intersect,
    validate_row_sequence,
)

SHAPE_SRC = Path("src/nutella_scraper/engines/visualization/scraper_shape_space.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")
COLLISION_SRC = Path(
    "src/nutella_scraper/engines/compute/scraper_envelope_collision.py"
)


def _lattice_from_bowl():
    surface = _bowl()
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    params = _profile_a(
        width_mm=2.5,
        length_mm=max_length,
        thickness_mm=2.5,
        clearance_mm=0.0,
    )
    path = ScraperEnvelopePathBuilder().build(surface, params)
    cage = build_control_cage_overlay(path, surface)
    return lattice_from_cage(cage, surface), path, surface, cage


def test_shape_space_does_not_import_collision_or_optimization() -> None:
    text = SHAPE_SRC.read_text(encoding="utf-8")
    assert "scraper_envelope_collision" not in text
    assert "optimization" not in text
    assert "rotationCache" not in text
    collision = COLLISION_SRC.read_text(encoding="utf-8")
    assert "scraper_shape_space" not in collision


def test_control_points_lie_on_interior_surface() -> None:
    lattice, path, surface, _cage = _lattice_from_bowl()
    candidates = generate_candidate_shapes(lattice, count=12)
    mesh = surface.to_trimesh()
    for candidate in candidates:
        assert candidate.valid
        points = np.asarray(candidate.control_points_mm, dtype=np.float64)
        _closest, distances, _tri = mesh.nearest.on_surface(points)
        assert float(np.max(distances)) <= CONTACT_TOLERANCE_MM + 1e-6
        assert len(points) == len(path.stations)


def test_row_jump_is_rejected() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    rows = list(form_a_row_indices(lattice))
    rows[3] = min(lattice.row_count - 1, rows[2] + 3)
    ok, reason = validate_row_sequence(lattice, tuple(rows))
    assert ok is False
    assert reason is not None
    assert "step" in reason


def test_zigzag_sequence_is_rejected() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    center = lattice.center_row_index
    other = min(center + 1, lattice.row_count - 1)
    rows = tuple(center if i % 2 == 0 else other for i in range(lattice.station_count))
    ok, reason = validate_row_sequence(lattice, rows)
    assert ok is False
    assert reason in {"zigzag oscillation", "curvature second difference exceeds 1"}


def test_self_intersection_detector() -> None:
    cross = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert segments_self_intersect(cross) is True
    line = np.column_stack([np.zeros(8), np.linspace(0, 20, 8), np.zeros(8)])
    assert segments_self_intersect(line) is False


def test_candidate_is_a_thin_curve_not_a_plate() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    candidate = generate_candidate_shapes(lattice, count=1)[0]
    assert len(candidate.control_points_mm) == lattice.station_count
    assert len(candidate.control_points_mm) < lattice.row_count * lattice.station_count
    assert candidate.shape.width_mm == pytest.approx(BLADE_WIDTH_MM)
    assert candidate.shape.thickness_mm == pytest.approx(BLADE_THICKNESS_MM)
    assert abs(CAGE_ROW_OFFSETS_MM[-1] - CAGE_ROW_OFFSETS_MM[0]) == pytest.approx(100.0)
    assert candidate.shape.width_mm < 10.0


def test_switching_candidate_does_not_mutate_previous() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    catalog = generate_candidate_shapes(lattice, count=8)
    first = catalog[0]
    snapshot = tuple(first.control_points_mm)
    fingerprint = first.shape_fingerprint
    second = catalog[1]
    assert second.candidate_id != first.candidate_id
    assert first.control_points_mm == snapshot
    assert first.shape_fingerprint == fingerprint
    with pytest.raises(FrozenInstanceError):
        first.row_indices = second.row_indices  # type: ignore[misc]


def test_pose_is_se3_and_does_not_reshape() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    shape = generate_candidate_shapes(lattice, count=1)[0].shape
    original = tuple(shape.control_points_mm)
    posed = posed_control_points(
        shape,
        ScraperPose(position_mm=(5.0, -3.0, 2.0), yaw_deg=15.0),
    )
    assert len(posed) == len(original)
    assert tuple(shape.control_points_mm) == original
    assert not np.allclose(posed, np.asarray(original), atol=0.5)
    assert shape.vertices is None
    assert shape.faces is None


def test_candidate_zero_is_form_a_centerline() -> None:
    lattice, path, _surface, cage = _lattice_from_bowl()
    catalog = generate_candidate_shapes(lattice, count=5)
    reference = catalog[0]
    assert reference.candidate_id == REFERENCE_CANDIDATE_ID
    assert reference.index == 0
    assert reference.valid is True
    assert reference.family == "A0"
    assert reference.row_indices == form_a_row_indices(lattice)
    assert reference.row_indices == (lattice.center_row_index,) * lattice.station_count
    centerline = np.asarray(cage["centerline_mm"], dtype=np.float64)
    points = np.asarray(reference.control_points_mm, dtype=np.float64)
    assert np.allclose(points, centerline, atol=1e-3)
    assert np.allclose(points, path.wall_curve_mm, atol=1e-3)
    wall_length = float(
        np.sum(np.linalg.norm(np.diff(path.wall_curve_mm, axis=0), axis=1))
    )
    assert reference.curve_length_mm == pytest.approx(wall_length, abs=1e-3)
    assert reference.shape.thickness_mm == pytest.approx(BLADE_THICKNESS_MM)
    assert reference.shape.width_mm == pytest.approx(BLADE_WIDTH_MM)
    assert len(reference.shape_fingerprint) >= 16
    rebuilt = generate_candidate_shapes(lattice, count=1)[0]
    assert rebuilt.shape_fingerprint == reference.shape_fingerprint
    assert rebuilt.as_rigid_shape().fingerprint == reference.shape_fingerprint


def test_no_point_is_behind_the_envelope() -> None:
    lattice, path, surface, _cage = _lattice_from_bowl()
    catalog = generate_candidate_shapes(lattice, count=16)
    mesh_points = np.asarray(path.wall_curve_mm, dtype=np.float64)
    signed, unsigned = envelope_contact_metrics(surface, mesh_points)
    assert float(np.max(unsigned)) <= CONTACT_TOLERANCE_MM + 1e-6
    assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM
    assert np.all(
        lattice.signed_mm[lattice.admissible] >= -CONTACT_TOLERANCE_MM - 1e-9
    )
    for candidate in catalog:
        points = np.asarray(candidate.control_points_mm, dtype=np.float64)
        signed_c, unsigned_c = envelope_contact_metrics(surface, points)
        assert float(np.max(unsigned_c)) <= CONTACT_TOLERANCE_MM + 1e-6
        assert float(np.min(signed_c)) >= -CONTACT_TOLERANCE_MM
        assert curve_behind_or_through_glass(surface, points) is False


def test_outward_curve_is_through_glass() -> None:
    _lattice, path, surface, _cage = _lattice_from_bowl()
    pts = np.asarray(path.wall_curve_mm, dtype=np.float64)
    outward = pts.copy()
    outward[:, 0] *= 1.25
    outward[:, 2] *= 1.25
    signed, _unsigned = envelope_contact_metrics(surface, outward)
    assert float(np.min(signed)) < -CONTACT_TOLERANCE_MM
    assert curve_behind_or_through_glass(surface, outward) is True


def test_two_poses_share_the_same_shape_fingerprint() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    candidate = generate_candidate_shapes(lattice, count=3)[2]
    fingerprint = candidate.shape_fingerprint
    posed_a = posed_control_points(
        candidate.shape,
        ScraperPose(position_mm=(4.0, -2.0, 1.0), yaw_deg=12.0),
    )
    posed_b = posed_control_points(
        candidate.shape,
        ScraperPose(position_mm=(-3.0, 1.5, 0.5), yaw_deg=-8.0, pitch_deg=3.0),
    )
    assert candidate.shape_fingerprint == fingerprint
    assert candidate.as_rigid_shape().fingerprint == fingerprint
    assert not np.allclose(posed_a, posed_b, atol=0.1)
    assert tuple(candidate.shape.control_points_mm) == candidate.control_points_mm


def test_invalid_candidate_cannot_be_promoted() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    rows = list(form_a_row_indices(lattice))
    rows[0] = 99
    built = build_candidate(lattice, tuple(rows), index=9)
    assert built.valid is False
    with pytest.raises(ValueError, match="cannot be promoted"):
        built.as_rigid_shape()
    valid = generate_candidate_shapes(lattice, count=1)[0]
    rigid = valid.as_rigid_shape()
    assert rigid.fingerprint == valid.shape_fingerprint
    assert rigid.vertices is None


def test_catalog_covers_printable_families() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    catalog = generate_candidate_shapes(lattice, count=40)
    families = {item.family for item in catalog}
    assert "A0" in families
    assert "parallel" in families
    assert "inclined" in families
    assert all(item.shape.vertices is None for item in catalog)
    assert all(len(item.row_indices) == lattice.station_count for item in catalog)


def test_generate_candidate_shapes_all_valid_and_diverse() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    catalog = generate_candidate_shapes(lattice, count=40)
    assert len(catalog) >= 11
    assert all(item.valid for item in catalog)
    fingerprints = {item.shape_fingerprint for item in catalog}
    assert len(fingerprints) == len(catalog)
    assert catalog[0].start_row == lattice.center_row_index


def test_build_candidate_rejects_off_lattice_row() -> None:
    lattice, _path, _surface, _cage = _lattice_from_bowl()
    rows = list(form_a_row_indices(lattice))
    rows[0] = 99
    built = build_candidate(lattice, tuple(rows), index=9)
    assert built.valid is False


def test_demo_viewer_has_candidate_navigator_and_keeps_play_cache() -> None:
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "const candidateCache = new Map()" in html
    assert "let candidateCatalogLoaded = false" in html
    assert "const CANDIDATE_CATALOG_SIZE = 100" in html
    assert "CANDIDATE_CATALOG_SIZE = 1000" not in html
    assert 'id="candidate-navigator"' in html
    assert "loadShapeCandidateCatalog" in html
    assert "applyCachedCandidate" in html
    assert "API.scraperShapeCandidates" in html
    assert "const rotationCache = new Map()" in html
    enter = html[
        html.index("async function enterScraperSoloView") : html.index(
            "function cacheReferenceCandidate"
        )
    ]
    assert "loadShapeCandidateCatalog({ resetToBest: true })" in enter
    assert "ensureVisualA0" in enter
    start = html.index("async function stepShapeCandidate")
    chunk = html[start : start + 500]
    assert "rotationCache.clear" not in chunk
    assert "clearRotationCache" not in chunk
    replay = html.index("function rotationReplayTick")
    replay_chunk = html[replay : replay + 900]
    assert "API.buildScraper" not in replay_chunk
    assert "scraperShapeCandidates" not in replay_chunk
