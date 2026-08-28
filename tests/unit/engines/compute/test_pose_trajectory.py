"""Pose-graph trajectory model — no cloud waypoints, no campaign."""

from __future__ import annotations

from pathlib import Path

from tests.unit.engines.compute.test_coverage_simulator import _fast_surface

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
    MATRIX_SPACING_MM,
)
from nutella_scraper.engines.compute.pose_contact_cache import (
    PoseContactEntry,
    mask_indices,
    pose_cache_from_entries,
)
from nutella_scraper.engines.compute.pose_space import (
    DIAGNOSTIC_LABEL,
    TARGET_MATRIX,
    TRAJECTORY_MODEL,
    PoseMotionLimits,
    PoseSampleSpec,
    PoseSamplingConfig,
    motion_limits_from_surface,
    sample_pose_specs,
    transition_allowed,
)
from nutella_scraper.engines.compute.pose_trajectory import (
    OPTIMIZATION_LABEL,
    assert_path_physical,
    beam_search_pose_trajectories,
    build_pose_edges,
    opening_pose_ids,
)

SPACE_SRC = Path("src/nutella_scraper/engines/compute/pose_space.py")
CACHE_SRC = Path("src/nutella_scraper/engines/compute/pose_contact_cache.py")
TRAJ_SRC = Path("src/nutella_scraper/engines/compute/pose_trajectory.py")


def _entry(
    pose_id: int,
    y_mm: float,
    azimuth_deg: float,
    mask: int,
    *,
    admissible: bool = True,
    origin: tuple[float, float, float] | None = None,
) -> PoseContactEntry:
    rad = 50.0
    yaw = float(azimuth_deg)
    ox = rad if origin is None else origin[0]
    oz = origin[2] if origin is not None else 0.0
    if origin is None:
        import math

        ox = rad * math.cos(math.radians(yaw))
        oz = -rad * math.sin(math.radians(yaw))
    return PoseContactEntry(
        pose_id=pose_id,
        y_mm=float(y_mm),
        azimuth_deg=yaw,
        origin_mm=(float(ox), float(y_mm), float(oz)),
        yaw_deg=yaw,
        length_axis=(0.0, -1.0, 0.0),
        admissible=admissible,
        neighborhood_used=False,
        covered_mask=int(mask) if admissible else 0,
        covered_count=int(mask).bit_count() if admissible else 0,
        physics_queries=0,
    )


def _limits(
    *,
    opening_y: float = 10.0,
    min_y: float = 0.0,
    vertical: float = 5.0,
    lateral: float = 40.0,
    rotation: float = 20.0,
    band: float = 1.0,
) -> PoseMotionLimits:
    return PoseMotionLimits(
        max_vertical_step_mm=vertical,
        max_lateral_step_mm=lateral,
        max_rotation_step_deg=rotation,
        opening_y_mm=opening_y,
        min_useful_y_mm=min_y,
        opening_band_mm=band,
        finish_band_mm=band,
    )


def test_modules_stay_in_compute_and_ignore_cloud_rows() -> None:
    for src in (SPACE_SRC, CACHE_SRC, TRAJ_SRC):
        text = src.read_text(encoding="utf-8")
        assert "evaluate_candidate(" not in text
        assert "from nutella_scraper.engines.compute.coverage_simulator" not in text
        assert "engines.visualization" not in text
        assert "engines.optimization" not in text
        assert "dst.row" not in text
        assert "state.cell.row" not in text
        assert "MAX_DOWNWARD_STEP" not in text
    assert DIAGNOSTIC_LABEL == "TRAJECTORY_MODEL_V2_A0_ONLY"
    assert TRAJECTORY_MODEL == "POSE_GRAPH"
    assert TARGET_MATRIX == COVERAGE_TARGET_REGION
    assert OPTIMIZATION_LABEL == "HEURISTIC"


def test_pose_sampling_is_independent_of_cloud_cardinality() -> None:
    surface = _fast_surface()
    specs = sample_pose_specs(
        surface,
        PoseSamplingConfig(height_step_mm=20.0, azimuth_step_deg=30.0),
    )
    assert specs
    assert len(specs) != 608
    ids = [item.pose_id for item in specs]
    assert ids == list(range(len(specs)))
    assert all(isinstance(item, PoseSampleSpec) for item in specs)
    ys = {round(item.y_mm, 6) for item in specs}
    az = {round(item.azimuth_deg, 6) for item in specs}
    assert min(ys) <= 1.0
    assert max(ys) >= 79.0
    assert min(az) == 0.0
    assert max(az) == 90.0


def test_motion_limits_are_physical_units() -> None:
    surface = _fast_surface()
    limits = motion_limits_from_surface(surface, scraper_length_mm=40.0)
    assert limits.max_vertical_step_mm == 40.0
    assert limits.max_lateral_step_mm == 2 * MATRIX_SPACING_MM
    assert limits.opening_y_mm > limits.min_useful_y_mm
    assert limits.is_opening_pose(limits.opening_y_mm)
    assert not limits.is_opening_pose(limits.min_useful_y_mm)
    assert limits.reached_useful_depth(limits.min_useful_y_mm)


def test_transition_uses_height_not_row_and_forbids_climb() -> None:
    limits = _limits(vertical=5.0, rotation=15.0, lateral=20.0)
    high = (50.0, 10.0, 0.0)
    mid = (50.0, 6.0, 0.0)
    low = (50.0, 0.0, 0.0)
    side = (48.0, 10.0, 8.0)
    assert transition_allowed(10.0, 0.0, high, 6.0, 0.0, mid, limits)
    assert not transition_allowed(6.0, 0.0, mid, 10.0, 0.0, high, limits)
    assert not transition_allowed(10.0, 0.0, high, 0.0, 0.0, low, limits)
    assert transition_allowed(10.0, 0.0, high, 10.0, 10.0, side, limits)


def test_start_is_opening_height_not_cloud_row_zero() -> None:
    limits = _limits(opening_y=10.0, min_y=0.0, band=1.0)
    entries = (
        _entry(0, 10.0, 0.0, 1 << 0),
        _entry(1, 5.0, 0.0, 1 << 1),
        _entry(2, 0.0, 0.0, 1 << 2),
    )
    cache = pose_cache_from_entries(entries, n_points=3)
    starts = opening_pose_ids(cache, limits)
    assert starts == (0,)
    assert 1 not in starts


def test_beam_does_not_require_last_cloud_point() -> None:
    limits = _limits(opening_y=10.0, min_y=-10.0, vertical=6.0, band=1.0)
    entries = (
        _entry(0, 10.0, 0.0, 1 << 0),
        _entry(1, 5.0, 0.0, 1 << 1),
        _entry(2, 1.0, 0.0, 1 << 2),
    )
    cache = pose_cache_from_entries(entries, n_points=8)
    ranked = beam_search_pose_trajectories(cache, limits, beam_width=8, top_k=3)
    assert ranked
    best = ranked[0]
    assert best.optimization_label == "HEURISTIC"
    assert best.trajectory_model == "POSE_GRAPH"
    assert best.covered_points == 3
    assert best.coverage_percent == 100.0 * 3 / 8
    assert best.path[-1].y_mm == 1.0
    assert_path_physical(best.path, limits)


def test_union_coverage_does_not_sum_overlaps() -> None:
    limits = _limits(opening_y=10.0, min_y=0.0, vertical=6.0, band=1.0)
    first = (1 << 0) | (1 << 1) | (1 << 2)
    second = (1 << 2) | (1 << 3)
    entries = (
        _entry(0, 10.0, 0.0, first),
        _entry(1, 5.0, 0.0, second),
    )
    cache = pose_cache_from_entries(entries, n_points=5)
    ranked = beam_search_pose_trajectories(cache, limits, beam_width=4, top_k=1)
    assert ranked[0].covered_points == 4
    assert ranked[0].covered_points != 3 + 2
    assert mask_indices(ranked[0].covered_mask, 5) == (0, 1, 2, 3)


def test_skipped_heights_still_yield_positive_union() -> None:
    """Old MAX_DOWNWARD_STEP=1 died on empty rows. Pose graph may skip."""
    limits = _limits(opening_y=12.0, min_y=0.0, vertical=8.0, band=1.0)
    entries = (
        _entry(0, 12.0, 0.0, 1 << 0),
        _entry(1, 5.0, 0.0, 1 << 1),
        _entry(2, 0.0, 0.0, 1 << 2),
    )
    cache = pose_cache_from_entries(entries, n_points=3)
    edges = build_pose_edges(cache, limits)
    assert (0, 1) in edges
    assert (1, 2) in edges
    ranked = beam_search_pose_trajectories(cache, limits, edges=edges, beam_width=4, top_k=1)
    assert ranked[0].covered_points == 3
    assert ranked[0].coverage_percent > 0.0
