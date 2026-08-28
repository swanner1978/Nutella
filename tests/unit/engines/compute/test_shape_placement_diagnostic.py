"""Placement diagnostic: A0 vs short 2 mm blade. No campaign."""

from __future__ import annotations

from tests.unit.engines.compute.test_coverage_simulator import _fast_surface

from nutella_scraper.engines.compute.shape_families import (
    DEFAULT_SCRAPER_LENGTH_MM,
    SCRAPER_THICKNESS_MM,
    SCRAPER_WIDTH_MM,
)
from nutella_scraper.engines.compute.shape_materialize import materialize_a0
from nutella_scraper.engines.compute.shape_placement_diagnostic import (
    compare_a0_and_straight40,
    local_section_extents_mm,
)


def test_straight_40_is_a_short_thin_blade_not_a_wall() -> None:
    surface = _fast_surface()
    report = compare_a0_and_straight40(surface)
    assert report["same_pose"] is True
    assert abs(float(report["a0"]["y_mm"]) - float(report["straight_40_2mm"]["y_mm"])) < 1e-9
    assert report["a0"]["azimuth_deg"] == report["straight_40_2mm"]["azimuth_deg"]
    blade = report["straight_40_2mm"]["local_extents_mm"]
    assert abs(float(blade["thickness_mm"]) - SCRAPER_THICKNESS_MM) < 0.8
    assert abs(float(blade["width_mm"]) - SCRAPER_WIDTH_MM) < 0.8
    assert float(blade["length_mm"]) < 55.0
    assert float(blade["length_mm"]) > 25.0
    a0 = report["a0"]["local_extents_mm"]
    assert abs(float(a0["length_mm"]) - DEFAULT_SCRAPER_LENGTH_MM) < 12.0
    assert float(report["origin_delta_mm"]) < 5.0


def test_a0_local_extents_match_historical_solid() -> None:
    surface = _fast_surface()
    extents = local_section_extents_mm(materialize_a0(surface))
    assert abs(extents["thickness_mm"] - 2.5) < 0.8
    assert abs(extents["width_mm"] - 2.5) < 0.8
    assert 28.0 < extents["length_mm"] < 55.0
