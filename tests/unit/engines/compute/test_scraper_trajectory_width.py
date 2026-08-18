"""3D meridian strip: constant width, continuous fillet, no envelope penetration."""

from __future__ import annotations

import math
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
    ScraperEnvelopePathBuilder,
    scraper_length_span,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    build_rigid_scraper_artifact,
)


def _bowl(*, radius: float = 50.0, y_max: float = 200.0, fillet: float = 30.0):
    """Surface of revolution with a smooth bowl into the axis at y=0."""

    def radius_at_y(y: float) -> float:
        if y >= fillet:
            return radius
        dy = fillet - y
        return math.sqrt(max(fillet * fillet - dy * dy, 0.0)) * (radius / fillet)

    return _reference_from_profile(
        radius_at_y=radius_at_y,
        y_min=0.0,
        y_max=y_max,
        y_count=81,
    )


def _station_arc_width(station) -> float:
    wall = np.asarray(station.wall_points_mm, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(wall, axis=0), axis=1)))


def _station_xz_span(station) -> float:
    xz = np.asarray(station.wall_points_mm, dtype=np.float64)[:, [0, 2]]
    delta = xz[:, None, :] - xz[None, :, :]
    return float(np.sqrt(np.max(np.sum(delta * delta, axis=2))))


def _midpoint_steps(path) -> np.ndarray:
    mids = path.wall_curve_mm
    return np.linalg.norm(np.diff(mids, axis=0), axis=1)


def _max_turn_deg(path) -> float:
    mids = path.wall_curve_mm
    tangents = np.diff(mids, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-9)
    if len(tangents) < 2:
        return 0.0
    cos = np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0)
    return float(np.degrees(np.max(np.arccos(cos))))


def test_short_length_stays_anchored_at_opening() -> None:
    surface = _bowl()
    opening_y, _lower_y, _max_length = scraper_length_span(surface)
    params = _profile_a(width_mm=10.0, length_mm=30.0)
    wall = ScraperEnvelopePathBuilder().build(surface, params).wall_curve_mm
    assert float(np.max(wall[:, 1])) >= opening_y - 2.0
    assert float(np.max(wall[:, 1])) <= opening_y + 0.5
    assert float(np.min(wall[:, 1])) == pytest.approx(opening_y - 30.0, abs=4.0)


def test_medium_length_keeps_width() -> None:
    surface = _bowl()
    params = _profile_a(width_mm=10.0, length_mm=100.0)
    path = ScraperEnvelopePathBuilder().build(surface, params)
    widths = [_station_arc_width(station) for station in path.stations]
    assert min(widths) == pytest.approx(10.0, abs=1.5)
    assert max(widths) == pytest.approx(10.0, abs=1.5)


def test_max_length_reaches_low_zone() -> None:
    surface = _bowl()
    opening_y, lower_y, max_length = scraper_length_span(surface)
    params = _profile_a(width_mm=10.0, length_mm=max_length)
    wall = ScraperEnvelopePathBuilder().build(surface, params).wall_curve_mm
    assert float(np.max(wall[:, 1])) >= opening_y - 2.0
    assert float(np.min(wall[:, 1])) <= lower_y + 8.0
    # Top view: the strip continues toward the axis, not a wall ring.
    bottom = wall[np.argmin(wall[:, 1])]
    assert float(np.hypot(bottom[0], bottom[2])) < 20.0


def test_width_is_constant_along_length() -> None:
    surface = _bowl()
    opening_y, lower_y, max_length = scraper_length_span(surface)
    params = _profile_a(width_mm=10.0, length_mm=max_length)
    path = ScraperEnvelopePathBuilder().build(surface, params)
    ys = np.asarray([station.y_mm for station in path.stations], dtype=np.float64)
    widths = np.asarray(
        [_station_arc_width(station) for station in path.stations], dtype=np.float64
    )
    spans = np.asarray(
        [_station_xz_span(station) for station in path.stations], dtype=np.float64
    )
    high = widths[ys >= opening_y - 20.0]
    mid = widths[(ys > 80.0) & (ys < 120.0)]
    turn = widths[(ys > lower_y + 5.0) & (ys < 40.0)]
    low = widths[ys <= lower_y + 12.0]
    for group in (high, mid, turn, low):
        if len(group) == 0:
            continue
        assert float(np.mean(group)) == pytest.approx(10.0, abs=1.5)
    assert float(np.max(spans)) < 18.0


def test_bottom_does_not_fill_the_floor_disk() -> None:
    surface = _bowl()
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    params = _profile_a(width_mm=10.0, length_mm=max_length)
    path = ScraperEnvelopePathBuilder().build(surface, params)
    for station in path.stations:
        assert _station_xz_span(station) < 18.0
        assert _station_arc_width(station) == pytest.approx(10.0, abs=1.8)


def test_fillet_has_no_geometric_kink() -> None:
    surface = _bowl()
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    params = _profile_a(width_mm=10.0, length_mm=max_length)
    path = ScraperEnvelopePathBuilder().build(surface, params)
    steps = _midpoint_steps(path)
    assert float(np.max(steps)) < 8.0
    assert _max_turn_deg(path) < 50.0


def test_max_length_does_not_penetrate_envelope() -> None:
    surface = _bowl()
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    params = _profile_a(width_mm=10.0, length_mm=max_length, clearance_mm=0.0)
    artifact = build_rigid_scraper_artifact(surface, params)
    report = evaluate_envelope_collision(artifact.mesh, surface, params)
    assert not report.has_collision
    assert report.admissible
    assert report.max_outward_mm <= 0.2


@pytest.mark.parametrize("progress_deg", [0.0, 20.0, 45.0, 90.0])
def test_rotation_keeps_design_vertices(progress_deg: float) -> None:
    surface = _ellipse(a=53.0, b=32.0)
    base = _profile_a(width_mm=10.0, length_mm=50.0)
    artifact = build_rigid_scraper_artifact(surface, base)
    rotated = base.with_updates(surface_progress_deg=progress_deg)
    _, pose, posed = _build(rotated, surface)
    assert pose.yaw_deg == pytest.approx(progress_deg)
    assert posed.vertices.shape == artifact.mesh.vertices.shape
    e0 = artifact.mesh.edges_unique
    e1 = posed.edges_unique
    len0 = np.linalg.norm(
        artifact.mesh.vertices[e0[:, 0]] - artifact.mesh.vertices[e0[:, 1]],
        axis=1,
    )
    len1 = np.linalg.norm(posed.vertices[e1[:, 0]] - posed.vertices[e1[:, 1]], axis=1)
    assert np.allclose(np.sort(len0), np.sort(len1), atol=1e-6)


def test_cached_jar_full_length_is_a_narrow_non_penetrating_strip() -> None:
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

    surface = load_interior_surface_reference(
        models_root=models,
        model_id=candidates[0].parent.name,
        use_cache=True,
    )
    opening_y, lower_y, max_length = scraper_length_span(surface)
    params = ScraperParameters.default().with_updates(
        width_mm=10.0,
        length_mm=max_length,
        clearance_mm=0.0,
        bevel_angle_deg=0.0,
        relief_angle_deg=0.0,
        helix_rate_deg_per_mm=0.0,
    )
    artifact = build_rigid_scraper_artifact(surface, params)
    path = artifact.design_path
    wall = path.wall_curve_mm
    assert float(np.max(wall[:, 1])) >= opening_y - 3.0
    assert float(np.min(wall[:, 1])) <= lower_y + 8.0
    assert float(np.max(_midpoint_steps(path))) < 10.0
    assert _max_turn_deg(path) < 55.0
    for station in path.stations:
        assert _station_xz_span(station) < 18.0
        assert _station_arc_width(station) == pytest.approx(10.0, abs=2.0)
    report = evaluate_envelope_collision(artifact.mesh, surface, params)
    assert not report.has_collision
    assert report.admissible
    assert report.max_outward_mm <= 0.35


def test_cylinder_short_path_still_builds() -> None:
    surface = _cylinder(50.0)
    params = _profile_a(width_mm=10.0, length_mm=30.0)
    path = ScraperEnvelopePathBuilder().build(surface, params)
    widths = [_station_arc_width(station) for station in path.stations]
    assert min(widths) == pytest.approx(10.0, abs=1.2)
    assert max(widths) == pytest.approx(10.0, abs=1.2)
