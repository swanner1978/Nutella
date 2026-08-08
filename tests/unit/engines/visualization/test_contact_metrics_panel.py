"""Contact metrics panel tests."""

from __future__ import annotations

import numpy as np
import pytest

from nutella_scraper.domain.models.contact import (
    CollisionResult,
    ContactOverlayData,
    ContactPoint3D,
    ContactResult,
)
from nutella_scraper.engines.visualization.contact_metrics_panel import ContactMetricsPanel


def _sample_contact(*, diagnostics: dict | None = None) -> ContactResult:
    distances = np.array([0.2, 1.5, np.inf], dtype=np.float64)
    overlay = ContactOverlayData(
        contact_points=(
            ContactPoint3D(position_mm=(1.0, 2.0, 3.0), jar_face_id=0, distance_mm=0.2),
        ),
        face_coverage=(True, False, False),
        min_distance_per_face_mm=(0.2, 1.5, float("inf")),
        scraper_pose_count=4,
    )
    return ContactResult(
        model_id="scraper_a",
        jar_id="jar_a",
        coverage_score=0.42,
        touched_face_ids=frozenset({0}),
        untouched_face_ids=frozenset({1, 2}),
        contact_distance_map=distances,
        trajectory_pose_count=4,
        overlay=overlay,
        collision=CollisionResult(
            has_collision=True,
            penetration_depth_mm=0.35,
            collision_points=(),
            colliding_face_ids=frozenset({2}),
        ),
        diagnostics=diagnostics or {"contact_point_count": 1},
    )


class TestContactMetricsPanel:
    def test_extracts_core_metrics_from_contact_result(self) -> None:
        panel = ContactMetricsPanel.from_contact_result(_sample_contact())

        assert panel.coverage_score_percent == pytest.approx(42.0)
        assert panel.covered_face_count == 1
        assert panel.total_face_count == 3
        assert panel.contact_point_count == 1
        assert panel.mean_distance_mm == pytest.approx(0.85)
        assert panel.max_distance_mm == pytest.approx(1.5)
        assert panel.has_collision is True
        assert panel.max_penetration_depth_mm == pytest.approx(0.35)
        assert panel.covered_surface_mm2 is None

    def test_surface_areas_from_diagnostics(self) -> None:
        panel = ContactMetricsPanel.from_contact_result(
            _sample_contact(
                diagnostics={
                    "total_inner_surface_mm2": 1000.0,
                    "simulation_duration_ms": 12.5,
                }
            )
        )

        assert panel.covered_surface_mm2 == pytest.approx(420.0)
        assert panel.uncovered_surface_mm2 == pytest.approx(580.0)
        assert panel.simulation_duration_ms == pytest.approx(12.5)

    def test_to_dict_is_json_ready(self) -> None:
        payload = ContactMetricsPanel.from_contact_result(_sample_contact()).to_dict()
        assert payload["coverage_score_percent"] == 42.0
        assert payload["has_collision"] is True
        assert payload["covered_surface_mm2"] is None
