"""Export best shape-search candidates. No STEP during the search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nutella_scraper.engines.compute.shape_result import ShapeCandidate

DEFAULT_EXPORT_DIR = Path("output/coverage/shape_search")


def candidate_payload(item: ShapeCandidate) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate_id,
        "family": item.family_id,
        "parameters": list(item.parameters),
        "n_parameters": item.n_parameters,
        "coverage_percent": item.coverage_percent,
        "covered_points": item.covered_points,
        "total_points": item.total_points,
        "untouched_point_indices": list(item.untouched_point_indices),
        "mean_geometric_error_mm": item.mean_geometric_error_mm,
        "max_geometric_error_mm": item.max_geometric_error_mm,
        "scraper_length_mm": item.scraper_length_mm,
        "thickness_mm": item.thickness_mm,
        "width_mm": item.width_mm,
        "min_curvature_radius_mm": item.min_curvature_radius_mm,
        "trajectory_steps": item.trajectory_steps,
        "trajectory_length_mm": item.trajectory_length_mm,
        "lateral_changes": item.lateral_changes,
        "direction_changes": item.direction_changes,
        "n_pose_candidates": item.n_pose_candidates,
        "n_admissible_poses": item.n_admissible_poses,
        "n_contacting_poses": item.n_contacting_poses,
        "n_reachable_poses": item.n_reachable_poses,
        "trajectory_found": item.trajectory_found,
        "opening_start_available": item.opening_start_available,
        "floor_reached": item.floor_reached,
        "termination_reason": item.termination_reason,
        "max_depth_reached_mm": item.max_depth_reached_mm,
        "geometric_valid": item.geometric_valid,
        "physical_valid": item.physical_valid,
        "optimization_label": item.optimization_label,
        "optimization_method": item.optimization_method,
        "profile_points_mm": np.round(item.profile_points_mm, 8).tolist(),
    }


def write_solidworks_xyz(points_mm: np.ndarray, path: Path) -> None:
    """XYZ millimetre list for SolidWorks 'Curve Through XYZ Points'."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for x, y, z in np.asarray(points_mm, dtype=np.float64):
        rows.append(f"{x:.10f},{y:.10f},{z:.10f}")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def export_best_candidates(
    candidates: tuple[ShapeCandidate, ...],
    output_dir: Path | None = None,
    *,
    top_k: int = 5,
) -> Path:
    root = Path(output_dir) if output_dir is not None else DEFAULT_EXPORT_DIR
    if root.name.startswith("candidate_coverage_100"):
        raise ValueError("Refus d'écraser les résultats candidate_coverage_100")
    root.mkdir(parents=True, exist_ok=True)
    chosen = [item for item in candidates if item.geometric_valid][: max(1, int(top_k))]
    payload = {
        "optimization_label": "HEURISTIC",
        "disclaimer": (
            "Meilleures formes trouvées. Aucune garantie d'optimum mathématique."
        ),
        "candidates": [candidate_payload(item) for item in chosen],
    }
    (root / "best_candidates.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    for item in chosen:
        stem = f"{item.candidate_id}_{item.family_id}"
        (root / f"{stem}_parameters.json").write_text(
            json.dumps(candidate_payload(item), indent=2),
            encoding="utf-8",
        )
        write_solidworks_xyz(item.profile_points_mm, root / f"{stem}_sw_curve.txt")
    return root
