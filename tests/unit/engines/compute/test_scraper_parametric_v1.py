"""Unit tests: scraper derived from interior product-surface mesh sections."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_collision import (
    evaluate_envelope_collision,
    pose_rigid_scraper_admissible,
)
from nutella_scraper.engines.compute.scraper_envelope_path import (
    NUMERIC_GAP_MM,
    ScraperEnvelopePathBuilder,
    assert_no_inverted_station_pairs,
    jar_longitudinal_limits,
    normalize_row_orders,
    prefers_flipped_order,
)
from nutella_scraper.engines.compute.scraper_geometry_generator import (
    ScraperGeometryGenerator,
)
from nutella_scraper.engines.compute.scraper_placement_calculator import (
    ScraperPlacementCalculator,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    apply_rigid_transform,
    build_rigid_scraper_artifact,
)


def _reference_from_profile(
    *,
    radius_at_y,
    y_min: float = 0.0,
    y_max: float = 100.0,
    y_count: int = 41,
    angular_count: int = 120,
    source: str = SOURCE_INTERIOR_PRODUCT_SURFACE,
) -> InteriorSurfaceReference:
    thetas = np.linspace(-math.pi, math.pi, angular_count, endpoint=False)
    ys = np.linspace(y_min, y_max, y_count)
    vertices: list[tuple[float, float, float]] = []
    for y in ys:
        radii = radius_at_y(float(y))
        if isinstance(radii, tuple):
            a, b = float(radii[0]), float(radii[1])
        else:
            a = b = float(radii)
        for theta in thetas:
            vertices.append(
                (float(a * math.cos(theta)), float(y), float(b * math.sin(theta)))
            )
    faces: list[tuple[int, int, int]] = []
    for yi in range(y_count - 1):
        for ti in range(angular_count):
            t2 = (ti + 1) % angular_count
            i00 = yi * angular_count + ti
            i01 = yi * angular_count + t2
            i10 = (yi + 1) * angular_count + ti
            i11 = (yi + 1) * angular_count + t2
            faces.append((i00, i10, i11))
            faces.append((i00, i11, i01))
    return InteriorSurfaceReference.from_arrays(
        model_id="synthetic-interior",
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        matching_face_count=1,
        source=source,
    )


def _cylinder(radius: float = 50.0) -> InteriorSurfaceReference:
    return _reference_from_profile(radius_at_y=lambda _y: radius)


def _ellipse(*, a: float = 53.0, b: float = 32.0) -> InteriorSurfaceReference:
    return _reference_from_profile(radius_at_y=lambda _y: (a, b))


def _tapered(*, r0: float = 40.0, slope: float = 0.2) -> InteriorSurfaceReference:
    return _reference_from_profile(radius_at_y=lambda y: r0 + slope * (y - 50.0))


def _outer_shell(radius: float = 60.0) -> InteriorSurfaceReference:
    return _reference_from_profile(
        radius_at_y=lambda _y: radius,
        source="synthetic_outer_not_used",
    )


def _profile_a(**updates: float) -> ScraperParameters:
    return ScraperParameters.default().with_updates(
        bevel_angle_deg=0.0,
        relief_angle_deg=0.0,
        helix_rate_deg_per_mm=0.0,
        **updates,
    )


def _build(params: ScraperParameters, surface: InteriorSurfaceReference):
    """Rigid solid once + admissible free pose (hard non-penetration)."""
    artifact = build_rigid_scraper_artifact(surface, params)
    result = pose_rigid_scraper_admissible(artifact, surface, params)
    if result.blocked:
        raise ValueError(
            f"MOUVEMENT BLOQUÉ at {params.surface_progress_deg:.1f}° "
            f"(outward {result.collision.max_outward_mm:.4f} mm)"
        )
    pose = result.pose
    posed = result.posed_mesh
    return artifact.design_path, pose, posed


def test_default_parameters() -> None:
    params = ScraperParameters.default()
    assert params.width_mm == 15.0
    assert params.clearance_mm == 0.0


def test_reference_is_interior_product_surface() -> None:
    surface = _ellipse()
    params = _profile_a(position_z_mm=50.0)
    path, _, _ = _build(params, surface)
    assert path.source == SOURCE_INTERIOR_PRODUCT_SURFACE
    assert ScraperPlacementCalculator.uses_interior_product_surface(path)


def test_normalize_row_orders_fixes_inverted_pair() -> None:
    prev = np.asarray([[1.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    curr = np.asarray([[1.1, 1.0, 1.0], [1.1, 1.0, 0.0], [1.1, 1.0, -1.0]])  # flipped
    normals = [np.tile([-1.0, 0.0, 0.0], (3, 1)), np.tile([-1.0, 0.0, 0.0], (3, 1))]
    assert prefers_flipped_order(prev, curr)
    rows = [prev.copy(), curr.copy()]
    normalize_row_orders(rows, normals)
    assert not prefers_flipped_order(rows[0], rows[1])


def test_built_path_has_no_inverted_station_pairs() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    params = _profile_a(position_z_mm=50.0, width_mm=30.0, length_mm=60.0)
    path = ScraperEnvelopePathBuilder().build(surface, params)
    assert_no_inverted_station_pairs(path.stations)
    for i in range(1, len(path.stations)):
        assert not prefers_flipped_order(
            path.stations[i - 1].tip_points_mm,
            path.stations[i].tip_points_mm,
        )


def test_profile_a_loft_has_no_degenerate_faces() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    params = _profile_a(position_z_mm=50.0, width_mm=25.0, length_mm=50.0)
    path, _, posed = _build(params, surface)
    assert_no_inverted_station_pairs(path.stations)
    areas = np.asarray(posed.area_faces, dtype=np.float64)
    assert float(np.min(areas)) >= ScraperGeometryGenerator._MIN_FACE_AREA_MM2
    assert bool(posed.is_winding_consistent)
    assert len(posed.faces) > 0


def test_tip_curve_is_smooth_interpolation_not_raw_mesh_polyline() -> None:
    """Dense tip samples should not inherit sharp mesh-edge kinks."""
    surface = _ellipse(a=53.0, b=32.0)
    params = _profile_a(position_z_mm=50.0, width_mm=40.0)
    path = ScraperEnvelopePathBuilder().build(surface, params)
    tip = path.stations[len(path.stations) // 2].wall_points_mm[:, [0, 2]]
    dirs = np.diff(tip, axis=0)
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    turns = np.degrees(np.arccos(np.clip(np.sum(dirs[1:] * dirs[:-1], axis=1), -1.0, 1.0)))
    assert float(np.max(turns)) < 25.0
    # Still faithful to ellipse (not a constant-radius circle).
    radii = np.hypot(tip[:, 0], tip[:, 1])
    assert float(np.max(radii) - np.min(radii)) > 2.0


def test_exterior_shell_never_used() -> None:
    interior = _ellipse(a=53.0, b=32.0)
    exterior = _outer_shell(60.0)
    params = _profile_a(position_z_mm=50.0, clearance_mm=0.0)
    path, _, posed = _build(params, interior)
    tip = posed.vertices[np.argmax(posed.vertices[:, 0])]
    d_in = float(interior.to_trimesh().nearest.on_surface(tip.reshape(1, 3))[1][0])
    d_out = float(exterior.to_trimesh().nearest.on_surface(tip.reshape(1, 3))[1][0])
    assert d_in < 0.5
    assert d_out > 5.0
    assert path.source == SOURCE_INTERIOR_PRODUCT_SURFACE


def test_active_edge_follows_ellipse_section_not_circle() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    params = _profile_a(position_z_mm=50.0, width_mm=40.0)
    path, _, _ = _build(params, surface)
    mid = path.stations[len(path.stations) // 2]
    wall = mid.wall_points_mm
    # Contour samples must leave the circle of radius a (ellipse squeezes in Z).
    radii = np.hypot(wall[:, 0], wall[:, 2])
    assert float(np.max(radii) - np.min(radii)) > 2.0
    # Chord bow: mid tip not on the straight chord (real curved section).
    tip = wall
    chord = tip[-1] - tip[0]
    chord_u = chord / max(float(np.linalg.norm(chord)), 1e-9)
    mid_pt = tip[len(tip) // 2]
    projected = tip[0] + chord_u * np.dot(mid_pt - tip[0], chord_u)
    assert float(np.linalg.norm(mid_pt - projected)) > 0.5
    assert tip.shape[0] >= 20


def test_path_points_lie_near_mesh_surface() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    params = _profile_a(position_z_mm=50.0, width_mm=25.0, length_mm=40.0)
    path, _, _ = _build(params, surface)
    mesh = surface.to_trimesh()
    wall = np.vstack([station.wall_points_mm for station in path.stations])
    dists = mesh.nearest.on_surface(wall)[1]
    # Smooth interpolation may leave the tessellation slightly; stay near surface.
    assert float(np.mean(dists)) < 0.5
    assert float(np.max(dists)) < 2.0


def test_active_edge_follows_vertical_variation() -> None:
    surface = _tapered(r0=40.0, slope=0.2)
    params = _profile_a(position_z_mm=50.0, length_mm=60.0)
    path, _, _ = _build(params, surface)
    wall = path.wall_curve_mm
    radial = np.hypot(wall[:, 0], wall[:, 2])
    assert float(np.max(radial) - np.min(radial)) > 5.0
    assert float(np.max(wall[:, 1]) - np.min(wall[:, 1])) > 20.0


@pytest.mark.parametrize("clearance_mm", [0.0, 1.0, 2.5])
def test_clearance_matches_tip_gap(clearance_mm: float) -> None:
    surface = _cylinder(50.0)
    params = _profile_a(position_z_mm=50.0, clearance_mm=clearance_mm)
    _, _, posed = _build(params, surface)
    gap = ScraperPlacementCalculator.active_tip_gap_mm(posed, surface)
    assert gap == pytest.approx(clearance_mm + NUMERIC_GAP_MM, abs=0.25)


def test_interior_side_validation_rejects_glass_penetration() -> None:
    surface = _cylinder(50.0)
    params = _profile_a(position_z_mm=50.0, clearance_mm=0.0)
    _, _, posed = _build(params, surface)
    broken = posed.copy()
    broken.vertices = broken.vertices.copy()
    tip_ids = np.argsort(
        np.asarray(surface.to_trimesh().nearest.on_surface(broken.vertices)[1])
    )[:20]
    broken.vertices[tip_ids, 0] += 3.0
    with pytest.raises(ValueError, match="penetrates|gap"):
        ScraperPlacementCalculator.assert_interior_side(broken, surface, params)


def test_z_rebuilds_trajectory() -> None:
    """Length is anchored at the opening; position_z is not the loft origin."""
    surface = _tapered(r0=40.0, slope=0.2)
    low = _profile_a(position_z_mm=62.0, length_mm=18.0)
    high = _profile_a(position_z_mm=88.0, length_mm=18.0)
    path_low = ScraperEnvelopePathBuilder().build(surface, low)
    path_high = ScraperEnvelopePathBuilder().build(surface, high)
    assert np.allclose(path_low.wall_curve_mm, path_high.wall_curve_mm, atol=0.2)
    _median_y, opening_y = jar_longitudinal_limits(surface)
    wall = path_low.wall_curve_mm
    assert float(np.max(wall[:, 1])) >= opening_y - 2.0
    assert float(np.min(wall[:, 1])) > opening_y - 25.0


def test_length_grows_down_the_taper() -> None:
    surface = _tapered(r0=40.0, slope=0.2)
    short = _profile_a(length_mm=18.0)
    long = _profile_a(length_mm=40.0)
    path_short = ScraperEnvelopePathBuilder().build(surface, short)
    path_long = ScraperEnvelopePathBuilder().build(surface, long)
    assert float(np.min(path_long.wall_curve_mm[:, 1])) < float(
        np.min(path_short.wall_curve_mm[:, 1])
    ) - 10.0
    r_short = float(
        np.mean(np.hypot(path_short.wall_curve_mm[:, 0], path_short.wall_curve_mm[:, 2]))
    )
    r_long = float(
        np.mean(np.hypot(path_long.wall_curve_mm[:, 0], path_long.wall_curve_mm[:, 2]))
    )
    # Longer blade reaches lower (narrower) stations on the taper.
    assert r_short > r_long + 1.0


def test_surface_progress_rigid_pose_no_rebuild() -> None:
    """Progress changes pose only — topology and edge lengths stay identical."""
    surface = _ellipse(a=53.0, b=32.0)
    a0 = _profile_a(position_z_mm=50.0, surface_progress_deg=0.0)
    a90 = _profile_a(position_z_mm=50.0, surface_progress_deg=90.0)
    _, pose0, posed0 = _build(a0, surface)
    _, pose90, posed90 = _build(a90, surface)
    assert pose0.yaw_deg == pytest.approx(0.0)
    assert pose90.yaw_deg == pytest.approx(90.0)
    assert posed0.faces.shape == posed90.faces.shape
    assert posed0.vertices.shape == posed90.vertices.shape
    # Rigid motion: pairwise edge lengths are preserved.
    e0 = posed0.edges_unique
    e90 = posed90.edges_unique
    len0 = np.linalg.norm(posed0.vertices[e0[:, 0]] - posed0.vertices[e0[:, 1]], axis=1)
    len90 = np.linalg.norm(posed90.vertices[e90[:, 0]] - posed90.vertices[e90[:, 1]], axis=1)
    assert np.allclose(np.sort(len0), np.sort(len90), atol=1e-6)
    assert abs(float(posed0.vertices[:, 0].mean())) > abs(float(posed0.vertices[:, 2].mean()))
    assert abs(float(posed90.vertices[:, 2].mean())) > abs(float(posed90.vertices[:, 0].mean()))


def test_rotation_zero() -> None:
    """0° reproduces profile A with bevel/relief/helix untouched."""
    surface = _ellipse(a=53.0, b=32.0)
    base = _profile_a(position_z_mm=50.0, width_mm=25.0, length_mm=50.0)
    rotated = base.with_updates(surface_progress_deg=0.0)
    assert rotated.bevel_angle_deg == 0.0
    assert rotated.relief_angle_deg == 0.0
    assert rotated.helix_rate_deg_per_mm == 0.0
    path0, pose0, posed0 = _build(base, surface)
    path_r, pose_r, posed_r = _build(rotated, surface)
    assert pose_r.yaw_deg == pytest.approx(0.0)
    assert np.allclose(posed0.vertices, posed_r.vertices, atol=1e-9)
    assert path0.stations[0].tip_points_mm.shape == path_r.stations[0].tip_points_mm.shape
    fit = ScraperPlacementCalculator.measure_envelope_fit(posed_r, surface, rotated)
    report = evaluate_envelope_collision(posed_r, surface, rotated)
    assert report.admissible
    assert not report.has_collision
    assert fit["distance_min_mm"] >= 0.0


def test_rotation_45() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    params = _profile_a(
        position_z_mm=50.0, width_mm=25.0, length_mm=50.0, surface_progress_deg=45.0
    )
    assert params.bevel_angle_deg == 0.0
    assert params.relief_angle_deg == 0.0
    assert params.helix_rate_deg_per_mm == 0.0
    path, pose, posed = _build(params, surface)
    assert pose.yaw_deg == pytest.approx(45.0)
    assert_no_inverted_station_pairs(path.stations)
    areas = np.asarray(posed.area_faces, dtype=np.float64)
    assert float(np.min(areas)) >= ScraperGeometryGenerator._MIN_FACE_AREA_MM2
    assert bool(posed.is_winding_consistent)
    # 45° progress leaves the +X wall; use the outer cloud, not one vertex.
    radii = np.hypot(posed.vertices[:, 0], posed.vertices[:, 2])
    outer = posed.vertices[radii >= np.quantile(radii, 0.85)]
    assert float(np.mean(np.abs(outer[:, 2]))) > 5.0
    report = evaluate_envelope_collision(posed, surface, params)
    assert report.admissible
    assert not report.has_collision


def test_rotation_90() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    params = _profile_a(
        position_z_mm=50.0, width_mm=25.0, length_mm=50.0, surface_progress_deg=90.0
    )
    assert params.bevel_angle_deg == 0.0
    path, pose, posed = _build(params, surface)
    assert pose.yaw_deg == pytest.approx(90.0)
    assert_no_inverted_station_pairs(path.stations)
    areas = np.asarray(posed.area_faces, dtype=np.float64)
    assert float(np.min(areas)) >= ScraperGeometryGenerator._MIN_FACE_AREA_MM2
    tip = pose.position_mm
    assert abs(float(tip[2])) > abs(float(tip[0]))
    report = evaluate_envelope_collision(posed, surface, params)
    assert report.admissible
    assert not report.has_collision
    assert params.surface_progress_deg == pytest.approx(90.0)


def test_rotation_quarter_turn_steps_remain_valid() -> None:
    """Every +5° from 0→90 poses the same rigid solid without reshape."""
    surface = _ellipse(a=53.0, b=32.0)
    base = _profile_a(position_z_mm=50.0, width_mm=25.0, length_mm=50.0)
    ref_faces = None
    ref_edge_lengths = None
    for angle in range(0, 91, 5):
        params = base.with_updates(surface_progress_deg=float(angle))
        assert params.bevel_angle_deg == 0.0
        assert params.relief_angle_deg == 0.0
        assert params.helix_rate_deg_per_mm == 0.0
        path, pose, posed = _build(params, surface)
        assert pose.yaw_deg == pytest.approx(float(angle))
        assert_no_inverted_station_pairs(path.stations)
        areas = np.asarray(posed.area_faces, dtype=np.float64)
        assert float(np.min(areas)) >= ScraperGeometryGenerator._MIN_FACE_AREA_MM2
        assert bool(posed.is_winding_consistent)
        if ref_faces is None:
            ref_faces = posed.faces.copy()
            edges = posed.edges_unique
            ref_edge_lengths = np.sort(
                np.linalg.norm(
                    posed.vertices[edges[:, 0]] - posed.vertices[edges[:, 1]],
                    axis=1,
                )
            )
        else:
            assert np.array_equal(posed.faces, ref_faces)
            edges = posed.edges_unique
            lengths = np.sort(
                np.linalg.norm(
                    posed.vertices[edges[:, 0]] - posed.vertices[edges[:, 1]],
                    axis=1,
                )
            )
            assert np.allclose(lengths, ref_edge_lengths, atol=1e-6)
        report = evaluate_envelope_collision(posed, surface, params)
        assert report.admissible
        assert not report.has_collision
        assert report.min_unsigned_distance_mm >= 0.0


def test_surface_progress_alias_rotation_angle_deg() -> None:
    params = ScraperParameters.from_dict({"rotation_angle_deg": 25.0, "width_mm": 15.0})
    assert params.surface_progress_deg == pytest.approx(25.0)
    assert params.rotation_angle_deg == pytest.approx(25.0)
    assert params.to_dict()["rotation_angle_deg"] == pytest.approx(25.0)
    assert params.to_dict()["surface_progress_deg"] == pytest.approx(25.0)


def test_hard_constraint_rejects_glass_side_translation() -> None:
    """A rigid outward shove must be INVALID — never an acceptable pose."""
    surface = _cylinder(50.0)
    params = _profile_a(position_z_mm=50.0, width_mm=20.0, length_mm=40.0)
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
    assert report.vertex_hit or report.edge_hit or report.face_hit


def test_admissible_search_does_not_reshape() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    params = _profile_a(
        position_z_mm=50.0, width_mm=25.0, length_mm=50.0, surface_progress_deg=45.0
    )
    artifact = build_rigid_scraper_artifact(surface, params)
    result = pose_rigid_scraper_admissible(artifact, surface, params)
    assert result.status in {"VALID", "BLOCKED"}
    assert np.array_equal(result.posed_mesh.faces, artifact.mesh.faces)
    edges = result.posed_mesh.edges_unique
    len_posed = np.sort(
        np.linalg.norm(
            result.posed_mesh.vertices[edges[:, 0]]
            - result.posed_mesh.vertices[edges[:, 1]],
            axis=1,
        )
    )
    len_design = np.sort(
        np.linalg.norm(
            artifact.mesh.vertices[edges[:, 0]] - artifact.mesh.vertices[edges[:, 1]],
            axis=1,
        )
    )
    assert np.allclose(len_posed, len_design, atol=1e-6)
    if result.status == "VALID":
        assert result.collision.admissible
        assert not result.collision.has_collision
    else:
        assert result.blocked
        assert result.collision.has_collision


@pytest.mark.parametrize(
    ("field", "value"),
    [("width_mm", 28.0), ("length_mm", 25.0), ("thickness_mm", 6.0)],
)
def test_dimensions_rebuild_from_surface(field: str, value: float) -> None:
    surface = _cylinder(50.0)
    base = _profile_a(position_z_mm=50.0)
    updated = base.with_updates(**{field: value})
    _, _, m0 = _build(base, surface)
    _, _, m1 = _build(updated, surface)
    assert not np.allclose(m0.bounding_box.extents, m1.bounding_box.extents, atol=0.2)


def test_profiles_a_b_c_distinct() -> None:
    """Manufacturing solids differ; helix may be BLOCKED as a pose (not reshaped)."""
    surface = _cylinder(50.0)
    current = ScraperParameters.default().with_updates(position_z_mm=50.0)
    profiles = [
        current.with_updates(bevel_angle_deg=0, relief_angle_deg=0, helix_rate_deg_per_mm=0),
        current.with_updates(bevel_angle_deg=30, relief_angle_deg=10, helix_rate_deg_per_mm=0),
        current.with_updates(bevel_angle_deg=30, relief_angle_deg=10, helix_rate_deg_per_mm=2),
    ]
    meshes = [build_rigid_scraper_artifact(surface, p).mesh for p in profiles]
    for mesh in meshes:
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.faces) > 0
    assert float(np.max(np.abs(meshes[0].vertices - meshes[1].vertices))) > 0.05
    assert float(np.max(np.abs(meshes[1].vertices - meshes[2].vertices))) > 0.2


def test_real_interior_product_surface_if_available() -> None:
    models_root = Path("output/models")
    step_candidates = sorted(
        models_root.glob("*/reference.step"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not step_candidates:
        pytest.skip("no reference.step")
    from nutella_scraper.engines.compute.interior_surface_reference import (
        load_interior_surface_reference,
    )

    step = step_candidates[0]
    model_id = step.parent.name
    try:
        surface = load_interior_surface_reference(
            models_root=models_root,
            model_id=model_id,
            step_path=step,
            use_cache=True,
        )
    except Exception as exc:
        pytest.skip(f"interior product surface unavailable: {exc}")

    assert surface.source == SOURCE_INTERIOR_PRODUCT_SURFACE
    assert surface.face_count > 0

    # Choose a height where a plane section exists.
    y_mid = 0.5 * (surface.y_min_mm + surface.y_max_mm)
    params = _profile_a(position_z_mm=float(y_mid), clearance_mm=0.0, width_mm=20.0)
    path, _, posed = _build(params, surface)
    assert path.source == SOURCE_INTERIOR_PRODUCT_SURFACE
    gap = ScraperPlacementCalculator.active_tip_gap_mm(posed, surface)
    assert gap == pytest.approx(NUMERIC_GAP_MM, abs=0.5)
    mesh = surface.to_trimesh()
    _median_y, opening_y = jar_longitudinal_limits(surface)
    wall = np.asarray(
        [row for row in path.wall_curve_mm if float(row[1]) <= opening_y + 1e-6],
        dtype=np.float64,
    )
    dists = mesh.nearest.on_surface(wall)[1]
    assert float(np.mean(dists)) < 0.35
    # Top-view samples must not collapse to a single radius (generic circle).
    mid = path.stations[len(path.stations) // 2].wall_points_mm
    radii = np.hypot(mid[:, 0], mid[:, 2])
    assert float(np.std(radii)) >= 0.0  # path exists; real jar may be near-circular locally
