"""Parametric scraper families share one sagittal frame. No CoverageSimulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nutella_scraper.engines.compute.shape_families import (
    FAMILY_BY_ID,
    PRELIMINARY_FAMILY_IDS,
    SHAPE_FAMILIES,
    SagittalFrame,
    sample_profile,
)

SRC = Path("src/nutella_scraper/engines/compute/shape_families.py")


def _frame() -> SagittalFrame:
    y = np.linspace(80.0, 0.0, 17, dtype=np.float64)
    r = np.full(len(y), 50.0, dtype=np.float64)
    r = np.where(y < 20.0, 50.0 * (y / 20.0), r)
    xyz = np.column_stack((r, y, np.zeros(len(y))))
    return SagittalFrame(
        y_top_mm=80.0,
        y_bot_mm=0.0,
        meridian_y_mm=y,
        meridian_r_mm=r,
        meridian_xyz_mm=xyz,
        r_max_mm=50.0,
        useful_height_mm=80.0,
    )


def test_registered_families_include_simple_blades() -> None:
    ids = [item.family_id for item in SHAPE_FAMILIES]
    assert ids[:5] == [
        "straight",
        "concave",
        "convex",
        "circular_arc",
        "poly_2",
    ]
    assert "bezier_4" in ids
    assert PRELIMINARY_FAMILY_IDS == (
        "straight",
        "concave",
        "convex",
        "circular_arc",
        "bezier_4",
    )
    assert "poly_2" not in PRELIMINARY_FAMILY_IDS
    assert "fourier_5" in ids
    for family in SHAPE_FAMILIES:
        assert "width" not in family.family_id


def test_every_family_is_monotonic_down_in_the_same_frame() -> None:
    frame = _frame()
    window = frame.window_for_length(40.0)
    for family in SHAPE_FAMILIES:
        profile = sample_profile(
            family,
            family.default_params(window),
            frame,
            sample_count=24,
            length_mm=40.0,
        )
        assert profile.family_id == family.family_id
        assert len(profile.parameters) == family.n_parameters
        assert np.all(np.diff(profile.y_mm) <= 1e-9)
        assert float(profile.y_mm[0]) > float(profile.y_mm[-1])
        assert profile.points_mm.shape == (24, 3)
        assert abs(float(profile.length_mm) - 40.0) < 1.0
        y_span = abs(float(profile.y_mm[0]) - float(profile.y_mm[-1]))
        assert y_span < 55.0


def test_short_blade_does_not_span_the_jar() -> None:
    from nutella_scraper.engines.compute.shape_families import FAMILY_BY_ID

    frame = _frame()
    family = FAMILY_BY_ID["straight"]
    window = frame.window_for_length(20.0)
    profile = sample_profile(
        family, family.default_params(window), frame, sample_count=12, length_mm=20.0
    )
    y_span = abs(float(profile.y_mm[0]) - float(profile.y_mm[-1]))
    assert y_span < 25.0
    assert float(profile.length_mm) == 20.0


def test_concave_and_convex_bend_opposite_ways() -> None:
    frame = _frame()
    concave = sample_profile(
        FAMILY_BY_ID["concave"],
        FAMILY_BY_ID["concave"].default_params(frame.window_for_length(40.0)),
        frame,
        sample_count=24,
        length_mm=40.0,
    )
    convex = sample_profile(
        FAMILY_BY_ID["convex"],
        FAMILY_BY_ID["convex"].default_params(frame.window_for_length(40.0)),
        frame,
        sample_count=24,
        length_mm=40.0,
    )
    mid = len(concave.r_mm) // 2
    assert float(concave.r_mm[mid]) != float(convex.r_mm[mid])


def test_module_does_not_import_simulator_or_viewer() -> None:
    text = SRC.read_text(encoding="utf-8")
    assert "coverage_simulator" not in text
    assert "engines.visualization" not in text
    assert "engines.optimization" not in text
