"""Constructability gates for sagittal profiles."""

from __future__ import annotations

import numpy as np
from tests.unit.engines.compute.test_shape_families import _frame

from nutella_scraper.engines.compute.shape_constraints import (
    MIN_CURVATURE_RADIUS_MM,
    GeometryValidity,
    sagittal_self_intersects,
    validate_profile,
)
from nutella_scraper.engines.compute.shape_families import (
    FAMILY_BY_ID,
    SagittalFrame,
    sample_profile,
)


def test_default_straight_on_cylinder_is_valid() -> None:
    frame = SagittalFrame(
        y_top_mm=80.0,
        y_bot_mm=0.0,
        meridian_y_mm=np.linspace(80.0, 0.0, 9),
        meridian_r_mm=np.full(9, 50.0),
        meridian_xyz_mm=np.column_stack(
            (np.full(9, 50.0), np.linspace(80.0, 0.0, 9), np.zeros(9))
        ),
        r_max_mm=50.0,
        useful_height_mm=80.0,
    )
    family = FAMILY_BY_ID["straight"]
    profile = sample_profile(family, (50.0, 50.0), frame, length_mm=40.0)
    report = validate_profile(profile, frame)
    assert report.valid
    assert report.thickness_mm == 2.0
    assert report.width_mm == 2.0
    assert report.monotonic_down
    assert not report.self_intersects


def test_radius_beyond_wall_is_rejected() -> None:
    frame = _frame()
    family = FAMILY_BY_ID["straight"]
    profile = sample_profile(family, (80.0, 80.0), frame, length_mm=40.0)
    report = validate_profile(profile, frame)
    assert report.valid is False
    assert "outside_interior_envelope" in report.reasons


def test_self_intersecting_polyline_is_detected() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert sagittal_self_intersects(points) is True


def test_geometry_validity_keeps_reasons_separate() -> None:
    report = GeometryValidity(
        valid=False,
        reasons=("curvature_radius_too_small",),
        length_mm=10.0,
        min_curvature_radius_mm=0.1,
        max_turn_deg=90.0,
        self_intersects=False,
        monotonic_down=True,
        inside_envelope=True,
    )
    assert MIN_CURVATURE_RADIUS_MM == 2.5
    from nutella_scraper.engines.compute.shape_constraints import MAX_CURVATURE_MM_INV

    assert abs(MAX_CURVATURE_MM_INV - 0.032) < 1e-9
    assert report.reasons == ("curvature_radius_too_small",)
