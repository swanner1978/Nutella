"""Length along InteriorSurfaceReference — no degenerate loft, explicit bound."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.unit.engines.compute.test_scraper_parametric_v1 import (
    _build,
    _cylinder,
    _ellipse,
    _profile_a,
    _reference_from_profile,
)

from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.scraper_envelope_collision import (
    evaluate_envelope_collision,
)
from nutella_scraper.engines.compute.scraper_envelope_path import (
    LONGITUDINAL_ANCHOR,
    ScraperEnvelopePathBuilder,
    apply_effective_length,
    jar_longitudinal_limits,
    scraper_length_span,
)
from nutella_scraper.engines.compute.scraper_geometry_generator import (
    ScraperGeometryGenerator,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    apply_rigid_transform,
    build_rigid_scraper_artifact,
)


def _min_consecutive_vertex_gap(mesh) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    gaps: list[float] = []
    for face in np.asarray(mesh.faces, dtype=np.int64):
        pts = vertices[face]
        edges = np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1)
        gaps.append(float(np.min(edges)))
    return min(gaps) if gaps else 0.0


def _assert_valid_scraper(
    params: ScraperParameters,
    surface,
    *,
    require_pose: bool = True,
) -> object:
    if require_pose:
        path, _pose, posed = _build(params, surface)
    else:
        artifact = build_rigid_scraper_artifact(surface, params)
        path = artifact.design_path
        posed = artifact.mesh
    areas = np.asarray(posed.area_faces, dtype=np.float64)
    assert float(np.min(areas)) >= ScraperGeometryGenerator._MIN_FACE_AREA_MM2
    assert _min_consecutive_vertex_gap(posed) > 1e-9
    mesh = surface.to_trimesh()
    _median_y, opening_y = jar_longitudinal_limits(surface)
    wall = np.vstack(
        [
            station.wall_points_mm
            for station in path.stations
            if float(station.y_mm) <= opening_y + 1e-6
        ]
    )
    dists = mesh.nearest.on_surface(wall)[1]
    assert float(np.mean(dists)) < 0.5
    mids = [station.wall_points_mm[len(station.wall_points_mm) // 2] for station in path.stations]
    for prev, curr in zip(mids, mids[1:], strict=False):
        assert float(np.linalg.norm(curr - prev)) > 1e-3
    return path


@pytest.mark.parametrize("length_mm", [20.0, 50.0, 80.0])
def test_length_short_medium_long_build(length_mm: float) -> None:
    surface = _cylinder(50.0)
    params = _profile_a(position_z_mm=50.0, width_mm=15.0, length_mm=length_mm)
    _assert_valid_scraper(params, surface)


@pytest.mark.parametrize(
    ("length_mm", "bevel", "helix"),
    [
        (20.0, 0.0, 0.0),
        (20.0, 30.0, 0.0),
        (50.0, 30.0, 2.0),
        (80.0, 30.0, 2.0),
    ],
)
def test_length_with_bevel_and_helix(length_mm: float, bevel: float, helix: float) -> None:
    surface = _ellipse(a=53.0, b=32.0)
    params = ScraperParameters.default().with_updates(
        position_z_mm=50.0,
        width_mm=15.0,
        length_mm=length_mm,
        bevel_angle_deg=bevel,
        relief_angle_deg=10.0,
        helix_rate_deg_per_mm=helix,
    )
    # Vrille can block the free pose on a tight ellipse; manufacturing loft must still succeed.
    _assert_valid_scraper(params, surface, require_pose=helix == 0.0)


def test_length_continuity_extends_envelope_path() -> None:
    surface = _cylinder(50.0)
    _median_y, opening_y = jar_longitudinal_limits(surface)
    lengths = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    spans: list[float] = []
    arcs: list[float] = []
    mids: list[np.ndarray] = []
    prev_wall = None
    for length_mm in lengths:
        params = _profile_a(position_z_mm=50.0, width_mm=15.0, length_mm=length_mm)
        path = _assert_valid_scraper(params, surface)
        wall = path.wall_curve_mm
        in_band = wall[wall[:, 1] <= opening_y + 1e-6]
        spans.append(float(np.ptp(wall[:, 1])))
        seg = np.linalg.norm(np.diff(wall, axis=0), axis=1)
        arcs.append(float(np.sum(seg)))
        mids.append(np.mean(in_band, axis=0))
        if prev_wall is not None:
            # Longer path extends the shorter one: previous Y-range is inside.
            assert float(np.min(wall[:, 1])) <= float(np.min(prev_wall[:, 1])) + 0.6
            assert float(np.max(wall[:, 1])) >= float(np.max(prev_wall[:, 1])) - 0.6
        prev_wall = wall
    for index in range(1, len(arcs)):
        assert arcs[index] >= arcs[index - 1] - 0.5
        assert spans[index] >= spans[index - 1] - 0.5
        assert float(np.linalg.norm(mids[index] - mids[index - 1])) < 12.0


def _tall_cylinder(*, radius: float = 50.0, y_max: float = 200.0):
    return _reference_from_profile(radius_at_y=lambda _y: radius, y_max=y_max)


def test_jar_longitudinal_limits_from_interior_aabb() -> None:
    surface = _tall_cylinder(y_max=200.0)
    median_y, opening_y = jar_longitudinal_limits(surface)
    assert median_y == pytest.approx(100.0)
    assert opening_y == pytest.approx(200.0)
    assert opening_y == pytest.approx(float(surface.y_max_mm))
    assert median_y == pytest.approx(0.5 * (surface.y_min_mm + surface.y_max_mm))


def test_length_beyond_useful_zone_does_not_grow_above_opening() -> None:
    surface = _cylinder(50.0)
    params = _profile_a(position_z_mm=50.0, width_mm=15.0, length_mm=250.0)
    assert params.length_mm == pytest.approx(250.0)
    path = ScraperEnvelopePathBuilder().build(surface, params)
    artifact = build_rigid_scraper_artifact(surface, params)
    opening_y, lower_y, max_length = scraper_length_span(surface)
    wall = path.wall_curve_mm
    assert float(np.min(wall[:, 1])) >= lower_y - 0.3
    assert float(np.max(wall[:, 1])) <= opening_y + 0.5
    report = evaluate_envelope_collision(artifact.mesh, surface, params)
    assert not report.has_collision
    assert report.admissible
    arc = float(np.sum(np.linalg.norm(np.diff(wall, axis=0), axis=1)))
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    assert arc == pytest.approx(max_length, abs=4.0)
    assert arc < 250.0 - 10.0


@pytest.mark.parametrize("length_mm", [190.0, 200.0, 210.0, 220.0, 230.0])
def test_long_physical_length_stays_admissible(length_mm: float) -> None:
    surface = _tall_cylinder(y_max=200.0)
    opening_y, lower_y, max_length = scraper_length_span(surface)
    params = _profile_a(
        position_z_mm=float(0.5 * (lower_y + opening_y)),
        width_mm=15.0,
        length_mm=length_mm,
    )
    assert params.length_mm == pytest.approx(length_mm)
    path, _pose, posed = _build(params, surface)
    wall = path.wall_curve_mm
    assert float(np.min(wall[:, 1])) >= lower_y - 0.3
    assert float(np.max(wall[:, 1])) >= opening_y - 2.0
    assert float(np.max(wall[:, 1])) <= opening_y + 0.5
    report = evaluate_envelope_collision(posed, surface, params)
    assert not report.has_collision
    assert report.admissible
    arc = float(np.sum(np.linalg.norm(np.diff(wall, axis=0), axis=1)))
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    assert arc == pytest.approx(min(length_mm, max_length), abs=4.0)


@pytest.mark.parametrize("progress_deg", [0.0, 45.0, 90.0, 180.0, 270.0])
def test_long_scraper_rotation_is_not_false_collision(progress_deg: float) -> None:
    surface = _tall_cylinder(y_max=200.0)
    median_y, _opening_y = jar_longitudinal_limits(surface)
    params = _profile_a(
        position_z_mm=float(median_y),
        width_mm=15.0,
        length_mm=220.0,
        surface_progress_deg=progress_deg,
    )
    _path, pose, posed = _build(params, surface)
    assert pose.yaw_deg == pytest.approx(progress_deg)
    report = evaluate_envelope_collision(posed, surface, params)
    assert not report.has_collision
    assert report.admissible


def test_interior_penetration_still_blocks_long_scraper() -> None:
    surface = _tall_cylinder(y_max=200.0)
    median_y, _opening_y = jar_longitudinal_limits(surface)
    params = _profile_a(position_z_mm=float(median_y), width_mm=20.0, length_mm=220.0)
    artifact = build_rigid_scraper_artifact(surface, params)
    design_ok = evaluate_envelope_collision(artifact.mesh, surface, params)
    assert design_ok.admissible
    shoved = apply_rigid_transform(
        artifact.mesh,
        np.array(
            [
                [1.0, 0.0, 0.0, 8.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    report = evaluate_envelope_collision(shoved, surface, params)
    assert report.has_collision
    assert not report.admissible


def _cylinder_with_rim() -> object:
    """Sidewall r=50 then a radial flange at the opening (real-jar rim analog)."""
    return _reference_from_profile(radius_at_y=lambda y: 80.0 if y >= 98.0 else 50.0)


@pytest.mark.parametrize("length_mm", [180.0, 200.0])
def test_long_length_on_flanged_opening_lofts(length_mm: float) -> None:
    surface = _cylinder_with_rim()
    median_y, opening_y = jar_longitudinal_limits(surface)
    params = ScraperParameters.default().with_updates(
        position_z_mm=float(median_y),
        width_mm=15.0,
        length_mm=length_mm,
    )
    path = _assert_valid_scraper(params, surface, require_pose=False)
    wall = path.wall_curve_mm
    opening_y, lower_y, _max_length = scraper_length_span(surface)
    assert float(np.max(wall[:, 1])) <= opening_y + 0.5
    assert float(np.max(wall[:, 1])) > 90.0
    assert float(np.min(wall[:, 1])) >= lower_y - 0.3


@pytest.mark.parametrize("length_mm", [180.0, 200.0])
def test_cached_jar_long_length_lofts(length_mm: float) -> None:
    models = Path("output/models")
    if not models.exists():
        pytest.skip("no cached models")
    candidates = list(models.glob("*/interior_product_surface.npz"))
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        pytest.skip("no interior cache")
    from nutella_scraper.engines.compute.interior_surface_reference import (
        load_interior_surface_reference,
    )

    model_id = candidates[0].parent.name
    surface = load_interior_surface_reference(
        models_root=models,
        model_id=model_id,
        use_cache=True,
    )
    opening_y, lower_y, _max_length = scraper_length_span(surface)
    params = ScraperParameters.default().with_updates(
        position_z_mm=float(0.5 * (lower_y + opening_y)),
        width_mm=15.0,
        length_mm=length_mm,
    )
    artifact = build_rigid_scraper_artifact(surface, params)
    areas = np.asarray(artifact.mesh.area_faces, dtype=np.float64)
    assert float(np.min(areas)) >= ScraperGeometryGenerator._MIN_FACE_AREA_MM2
    wall = artifact.design_path.wall_curve_mm
    assert float(np.max(wall[:, 1])) <= float(opening_y) + 0.5
    assert float(np.max(wall[:, 1])) >= float(opening_y) - 3.0
    assert float(np.min(wall[:, 1])) >= lower_y - 0.5


@pytest.mark.parametrize("progress_deg", [0.0, 20.0, 45.0, 90.0])
def test_rotation_preserves_rigid_topology(progress_deg: float) -> None:
    surface = _ellipse(a=53.0, b=32.0)
    base = _profile_a(position_z_mm=50.0, width_mm=15.0, length_mm=50.0)
    artifact = build_rigid_scraper_artifact(surface, base)
    rotated = base.with_updates(surface_progress_deg=progress_deg)
    _, pose, posed = _build(rotated, surface)
    assert pose.yaw_deg == pytest.approx(progress_deg)
    assert posed.faces.shape == artifact.mesh.faces.shape
    assert posed.vertices.shape == artifact.mesh.vertices.shape
    e0 = artifact.mesh.edges_unique
    e1 = posed.edges_unique
    len0 = np.linalg.norm(
        artifact.mesh.vertices[e0[:, 0]] - artifact.mesh.vertices[e0[:, 1]],
        axis=1,
    )
    len1 = np.linalg.norm(posed.vertices[e1[:, 0]] - posed.vertices[e1[:, 1]], axis=1)
    assert np.allclose(np.sort(len0), np.sort(len1), atol=1e-6)


def test_top_anchor_stays_fixed_while_bottom_descends() -> None:
    surface = _tall_cylinder(y_max=200.0)
    opening_y, lower_y, max_length = scraper_length_span(surface)
    tops: list[float] = []
    bottoms: list[float] = []
    for length_mm in (30.0, 100.0, 150.0, 190.0):
        params = _profile_a(width_mm=15.0, length_mm=length_mm)
        wall = ScraperEnvelopePathBuilder().build(surface, params).wall_curve_mm
        tops.append(float(np.max(wall[:, 1])))
        bottoms.append(float(np.min(wall[:, 1])))
        assert tops[-1] >= opening_y - 2.0
        assert tops[-1] <= opening_y + 0.5
        assert bottoms[-1] == pytest.approx(opening_y - length_mm, abs=3.0)
    assert max(tops) - min(tops) < 1.0
    for index in range(1, len(bottoms)):
        assert bottoms[index] < bottoms[index - 1] - 20.0
    assert bottoms[-1] > lower_y - 0.3
    assert max_length == pytest.approx(200.0)


def test_apply_effective_length_clamps_and_warns() -> None:
    surface = _tall_cylinder(y_max=200.0)
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    requested = ScraperParameters.default().with_updates(length_mm=800.0)
    clamped, info = apply_effective_length(requested, surface)
    assert info["clamped"] is True
    assert clamped.length_mm == pytest.approx(max_length)
    assert info["effective_length_mm"] == pytest.approx(max_length)
    assert info["warning"] == f"Attention : longueur maximale = {max_length:.0f} mm"
    short = ScraperParameters.default().with_updates(length_mm=80.0)
    same, short_info = apply_effective_length(short, surface)
    assert short_info["clamped"] is False
    assert same.length_mm == pytest.approx(80.0)
    assert short_info["warning"] is None


def test_anchor_is_upper_opening_not_red_axis() -> None:
    surface = _cylinder(50.0)
    assert LONGITUDINAL_ANCHOR == "upper_opening"
    median_y, opening_y = jar_longitudinal_limits(surface)
    params = _profile_a(position_z_mm=50.0, width_mm=15.0, length_mm=30.0)
    wall = ScraperEnvelopePathBuilder().build(surface, params).wall_curve_mm
    assert float(np.max(wall[:, 1])) >= opening_y - 2.0
    assert float(np.max(wall[:, 1])) <= opening_y + 0.5
    assert float(np.min(wall[:, 1])) > median_y + 10.0
    # Independent of position_z (viewer red-axis / AABB centre height).
    other = _profile_a(position_z_mm=70.0, width_mm=15.0, length_mm=30.0)
    wall_b = ScraperEnvelopePathBuilder().build(surface, other).wall_curve_mm
    assert np.allclose(wall, wall_b, atol=0.2)


def test_length_grows_downward_from_opening() -> None:
    surface = _cylinder(50.0)
    _median_y, opening_y = jar_longitudinal_limits(surface)
    prev_min = opening_y
    prev_max = -1e9
    for length_mm in (30.0, 50.0, 100.0, 150.0):
        params = _profile_a(width_mm=15.0, length_mm=length_mm)
        wall = ScraperEnvelopePathBuilder().build(surface, params).wall_curve_mm
        assert float(np.max(wall[:, 1])) >= opening_y - 2.0
        assert float(np.max(wall[:, 1])) <= opening_y + 0.5
        assert float(np.max(wall[:, 1])) >= prev_max - 0.6
        assert float(np.min(wall[:, 1])) <= prev_min + 0.6
        prev_min = float(np.min(wall[:, 1]))
        prev_max = float(np.max(wall[:, 1]))


@pytest.mark.parametrize("length_mm", [230.0, 300.0, 400.0, 600.0])
def test_extreme_length_caps_useful_course_without_exception(length_mm: float) -> None:
    surface = _tall_cylinder(y_max=200.0)
    opening_y, lower_y, max_length = scraper_length_span(surface)
    params = ScraperParameters.default().with_updates(
        width_mm=15.0,
        length_mm=length_mm,
    )
    assert params.length_mm == pytest.approx(length_mm)
    artifact = build_rigid_scraper_artifact(surface, params)
    wall = artifact.design_path.wall_curve_mm
    assert float(np.min(wall[:, 1])) >= lower_y - 0.3
    assert float(np.max(wall[:, 1])) <= opening_y + 0.5
    arc = float(np.sum(np.linalg.norm(np.diff(wall, axis=0), axis=1)))
    assert arc == pytest.approx(max_length, abs=4.0)
    areas = np.asarray(artifact.mesh.area_faces, dtype=np.float64)
    assert float(np.min(areas)) >= ScraperGeometryGenerator._MIN_FACE_AREA_MM2
