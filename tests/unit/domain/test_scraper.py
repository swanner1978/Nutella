"""Scraper geometry and pose domain model tests."""

from __future__ import annotations

import pytest

from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose


class TestScraperGeometry:
    def test_accepts_extended_parametric_fields(self) -> None:
        geometry = ScraperGeometry(
            width_mm=18.0,
            length_mm=95.0,
            thickness_mm=2.5,
            tip_radius_mm=1.5,
            curvature_radius_mm=35.0,
            bend_angle_deg=12.0,
            metadata={"variant": "v1"},
        )

        assert geometry.width_mm == 18.0
        assert geometry.tip_radius_mm == 1.5
        assert geometry.metadata["variant"] == "v1"

    def test_rejects_invalid_curvature_radius(self) -> None:
        with pytest.raises(ValueError):
            ScraperGeometry(
                width_mm=10.0,
                length_mm=10.0,
                thickness_mm=2.0,
                curvature_radius_mm=-1.0,
            )


class TestScraperPose:
    def test_stores_placement_only(self) -> None:
        pose = ScraperPose(
            position_mm=(46.0, 40.0, 0.0),
            yaw_deg=15.0,
            pitch_deg=3.0,
            roll_deg=-2.0,
        )

        assert pose.position_mm == (46.0, 40.0, 0.0)
        assert pose.yaw_deg == 15.0
