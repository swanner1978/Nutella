"""Visual contact-curve smoothing must not move control points."""

from __future__ import annotations

import numpy as np

from nutella_scraper.engines.visualization.scraper_curve_smoothing import (
    smooth_contact_polyline,
)


def test_smoothing_does_not_mutate_control_points() -> None:
    points = np.array(
        [
            [10.0, 80.0, 0.0],
            [12.0, 60.0, 1.0],
            [11.0, 40.0, 0.5],
            [13.0, 20.0, -1.0],
        ],
        dtype=np.float64,
    )
    snapshot = points.copy()
    smoothed = smooth_contact_polyline(points)
    assert np.array_equal(points, snapshot)
    assert not np.shares_memory(smoothed, points)


def test_smoothed_curve_interpolates_and_is_y_monotone() -> None:
    points = np.array(
        [
            [0.0, 30.0, 10.0],
            [4.0, 20.0, 8.0],
            [5.0, 10.0, 9.0],
            [1.0, 0.0, 7.0],
        ],
        dtype=np.float64,
    )
    smoothed = smooth_contact_polyline(points, samples_per_segment=8)
    for point in points:
        distances = np.linalg.norm(smoothed - point, axis=1)
        assert float(np.min(distances)) < 1e-9
    assert np.all(np.diff(smoothed[:, 1]) <= 1e-12)


def test_smoothing_has_no_coordinate_overshoot() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 1.0],
            [3.0, 4.0, 1.0],
        ],
        dtype=np.float64,
    )
    smoothed = smooth_contact_polyline(points)
    for axis in range(3):
        assert float(np.min(smoothed[:, axis])) >= float(np.min(points[:, axis])) - 1e-9
        assert float(np.max(smoothed[:, axis])) <= float(np.max(points[:, axis])) + 1e-9
