"""Export of best shape-search candidates. No STEP during search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nutella_scraper.engines.compute.shape_export import (
    export_best_candidates,
    write_solidworks_xyz,
)
from nutella_scraper.engines.compute.shape_result import ShapeCandidate


def _item() -> ShapeCandidate:
    return ShapeCandidate(
        candidate_id="straight-001",
        family_id="straight",
        parameters=(50.0, 40.0),
        n_parameters=2,
        coverage_percent=50.0,
        covered_points=2,
        total_points=4,
        covered_point_indices=(0, 1),
        untouched_point_indices=(2, 3),
        mean_geometric_error_mm=1.0,
        max_geometric_error_mm=2.0,
        scraper_length_mm=80.0,
        min_curvature_radius_mm=10.0,
        trajectory_steps=4,
        trajectory_length_mm=12.0,
        lateral_changes=1,
        direction_changes=1,
        geometric_valid=True,
        physical_valid=True,
        geometric_reasons=(),
        profile_points_mm=np.array([[50.0, 80.0, 0.0], [40.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_export_writes_json_and_xyz_not_step(tmp_path: Path) -> None:
    root = export_best_candidates((_item(),), tmp_path, top_k=1)
    assert (root / "best_candidates.json").is_file()
    xyz = root / "straight-001_straight_sw_curve.txt"
    assert xyz.is_file()
    assert "STEP" not in xyz.read_text(encoding="utf-8")
    write_solidworks_xyz(np.array([[1.0, 2.0, 3.0]]), tmp_path / "one.txt")
    text = (tmp_path / "one.txt").read_text(encoding="utf-8")
    assert text.startswith("1.0000000000,2.0000000000,3.0000000000")
