"""Trajectory projection tests."""

from __future__ import annotations

from nutella_scraper.engines.visualization.trajectory_projector import (
    LAYER_TRAJECTORY,
    TrajectoryProjector,
)


def test_trajectory_projector_emits_polyline(internal_jar_surface) -> None:
    positions = (
        (40.0, 10.0, 0.0),
        (42.0, 35.0, 0.0),
        (44.0, 60.0, 0.0),
    )
    projection = TrajectoryProjector().project(positions, internal_jar_surface)

    assert projection.profile_layers
    assert projection.top_layers
    assert projection.profile_layers[0].layer_type == LAYER_TRAJECTORY
    assert "scraper-trajectory" in projection.profile_layers[0].svg_fragment
    assert "#fbbf24" in projection.profile_layers[0].svg_fragment
