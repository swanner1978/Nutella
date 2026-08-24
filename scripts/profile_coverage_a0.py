"""One-shot A0 coverage profiler. Does not change simulator math."""

from __future__ import annotations

import json
import statistics
import time

import numpy as np
from tests.unit.engines.compute.test_scraper_parametric_v1 import (
    _profile_a,
    _reference_from_profile,
)

from nutella_scraper.engines.compute import coverage_simulator as cov_mod
from nutella_scraper.engines.compute import scraper_envelope_collision as col_mod
from nutella_scraper.engines.compute import scraper_rigid_motion as rig_mod
from nutella_scraper.engines.compute.coverage_simulator import (
    CoverageSimulator,
    unique_edge_lengths_mm,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    build_rigid_scraper_artifact,
)


def _fast_surface():
    return _reference_from_profile(
        radius_at_y=lambda _y: 50.0,
        y_min=0.0,
        y_max=80.0,
        y_count=21,
        angular_count=48,
    )


def _a0_parameters(surface):
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    return _profile_a(
        width_mm=2.5,
        thickness_mm=2.5,
        length_mm=min(40.0, max_length),
        clearance_mm=0.0,
        position_z_mm=float(0.5 * (surface.y_min_mm + surface.y_max_mm)),
    )

EXPECTED = {
    "coverage_percent": 63.3333,
    "covered_area_mm2": 1988.2551,
    "target_area_mm2": 3139.3502,
}

# Uninstrumented A0 after optimisation #1 (P2/P3). This cycle starts from here.
BEFORE = {
    "evaluate_s": 71.852,
    "n_proximity": 768,
    "n_nearest": 791,
    "n_mesh_copy": 0,
    "n_to_trimesh": 0,
    "n_poses": 408,
    "n_points": 2453616,
}


class CallClock:
    def __init__(self) -> None:
        self.total_s = 0.0
        self.count = 0
        self.min_s = float("inf")
        self.max_s = 0.0

    def add(self, elapsed: float) -> None:
        self.total_s += elapsed
        self.count += 1
        self.min_s = min(self.min_s, elapsed)
        self.max_s = max(self.max_s, elapsed)


def wrap(clocks: dict[str, CallClock], name: str, fn):
    clock = clocks.setdefault(name, CallClock())

    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        clock.add(time.perf_counter() - started)
        return result

    return wrapped


def main() -> None:
    clocks: dict[str, CallClock] = {}
    per_angle: list[dict[str, float]] = []
    current: dict[str, float] | None = None

    orig_best = CoverageSimulator._best_pose_for_angle
    orig_evaluate = CoverageSimulator._evaluate
    orig_to_trimesh = InteriorSurfaceReference.to_trimesh
    orig_copy = __import__("trimesh").Trimesh.copy
    orig_on_surface = None

    def snapshot() -> dict[str, tuple[float, int]]:
        return {name: (clk.total_s, clk.count) for name, clk in clocks.items()}

    def delta(before: dict[str, tuple[float, int]], name: str) -> tuple[float, int]:
        after_t, after_n = clocks.get(name, CallClock()).total_s, clocks.get(
            name, CallClock()
        ).count
        if name in clocks:
            after_t, after_n = clocks[name].total_s, clocks[name].count
        else:
            return 0.0, 0
        prev_t, prev_n = before.get(name, (0.0, 0))
        return after_t - prev_t, after_n - prev_n

    def timed_best(self, artifact, angle_deg, **kwargs):
        nonlocal current
        before = snapshot()
        t0 = time.perf_counter()
        result = orig_best(self, artifact, angle_deg, **kwargs)
        elapsed = time.perf_counter() - t0
        pose_prep_s, _n = delta(before, "envelope_contact_frame")
        neigh_s, n_frames = delta(before, "rigid_pose_neighborhood")
        xform_s, n_xform = delta(before, "transform_points")
        copy_s, n_copy = delta(before, "apply_rigid_transform")
        tf_s, _n = delta(before, "rigid_transform_between_frames")
        coll_s, n_coll = delta(before, "evaluate_envelope_collision")
        prox_s, n_prox = delta(before, "_proximity")
        mesh_copy_s, n_mesh_copy = delta(before, "trimesh.copy")
        nearest_s, n_nearest = delta(before, "nearest.on_surface")
        to_tri_s, n_to_tri = delta(before, "to_trimesh")
        pose_s, _n = delta(before, "pose_from_matrix")
        current = {
            "angle": float(angle_deg),
            "total_s": elapsed,
            "pose_prep_s": pose_prep_s + neigh_s,
            "se3_search_s": xform_s + tf_s + copy_s,
            "n_candidates": float(n_xform or n_coll or n_frames),
            "collision_s": coll_s,
            "proximity_s": prox_s,
            "face_ids_in_collision_s": max(0.0, coll_s - prox_s),
            "area_and_union_s": 0.0,
            "nearest_s": nearest_s,
            "n_nearest": float(n_nearest),
            "n_proximity": float(n_prox),
            "n_collision": float(n_coll),
            "n_mesh_copy": float(n_mesh_copy),
            "n_to_trimesh": float(n_to_tri),
            "pose_from_matrix_s": pose_s,
        }
        per_angle.append(current)
        return result

    def timed_evaluate(self, candidate_id, artifact):
        covered: set[int] = set()
        orig_update = covered.update

        def counting_evaluate():
            return orig_evaluate(self, candidate_id, artifact)

        t0 = time.perf_counter()
        result = counting_evaluate()
        result_s = time.perf_counter() - t0
        del orig_update
        return result, result_s

    CoverageSimulator._best_pose_for_angle = timed_best  # type: ignore[method-assign]
    col_mod.evaluate_envelope_collision = wrap(
        clocks, "evaluate_envelope_collision", col_mod.evaluate_envelope_collision
    )
    col_mod._proximity = wrap(clocks, "_proximity", col_mod._proximity)
    col_mod.closest_on_envelope_surface = wrap(
        clocks, "closest_on_envelope", col_mod.closest_on_envelope_surface
    )
    col_mod.rigid_pose_neighborhood = wrap(
        clocks, "rigid_pose_neighborhood", col_mod.rigid_pose_neighborhood
    )
    cov_mod.evaluate_envelope_collision = col_mod.evaluate_envelope_collision
    cov_mod.rigid_pose_neighborhood = col_mod.rigid_pose_neighborhood
    rig_mod.envelope_contact_frame = wrap(
        clocks, "envelope_contact_frame", rig_mod.envelope_contact_frame
    )
    rig_mod.apply_rigid_transform = wrap(
        clocks, "apply_rigid_transform", rig_mod.apply_rigid_transform
    )
    rig_mod.transform_points = wrap(clocks, "transform_points", rig_mod.transform_points)
    rig_mod.rigid_transform_between_frames = wrap(
        clocks, "rigid_transform_between_frames", rig_mod.rigid_transform_between_frames
    )
    cov_mod.envelope_contact_frame = rig_mod.envelope_contact_frame
    cov_mod.transform_points = rig_mod.transform_points
    cov_mod.rigid_transform_between_frames = rig_mod.rigid_transform_between_frames
    cov_mod.pose_from_matrix = wrap(clocks, "pose_from_matrix", cov_mod.pose_from_matrix)
    cov_mod.unique_edge_lengths_mm = wrap(
        clocks, "unique_edge_lengths_mm", cov_mod.unique_edge_lengths_mm
    )

    InteriorSurfaceReference.to_trimesh = wrap(  # type: ignore[method-assign]
        clocks, "to_trimesh", orig_to_trimesh
    )
    __import__("trimesh").Trimesh.copy = wrap(clocks, "trimesh.copy", orig_copy)

    from trimesh.proximity import ProximityQuery

    orig_on_surface = ProximityQuery.on_surface
    ProximityQuery.on_surface = wrap(clocks, "nearest.on_surface", orig_on_surface)

    from nutella_scraper.engines.compute.coverage_scorer import CoverageScorer

    CoverageScorer.score = wrap(clocks, "coverage_score", CoverageScorer.score)

    t_surf0 = time.perf_counter()
    surface = _fast_surface()
    surf_build_s = time.perf_counter() - t_surf0

    t_scrap0 = time.perf_counter()
    params = _a0_parameters(surface)
    artifact = build_rigid_scraper_artifact(surface, params)
    scrap_s = time.perf_counter() - t_scrap0
    edges_before = unique_edge_lengths_mm(artifact.mesh)
    verts_before = np.asarray(artifact.mesh.vertices).copy()

    t_init0 = time.perf_counter()
    simulator = CoverageSimulator(surface, parameters=params)
    init_s = time.perf_counter() - t_init0

    from nutella_scraper.engines.compute.envelope_surface_proximity import (
        PROXIMITY_STATS,
        reset_proximity_stats,
    )

    simulator.register("A0", artifact)

    reset_proximity_stats()
    pre_eval = {name: (clk.total_s, clk.count) for name, clk in clocks.items()}
    t_eval0 = time.perf_counter()
    result = simulator.evaluate_candidate("A0")
    eval_s = time.perf_counter() - t_eval0

    # Attribute leftover evaluate time (union/area/score) after per-angle sums.
    angle_sum = sum(row["total_s"] for row in per_angle)
    leftover_s = max(0.0, eval_s - angle_sum)
    if per_angle:
        per_angle[-1]["area_and_union_s"] += leftover_s / max(len(per_angle), 1)
        share = leftover_s / max(len(per_angle), 1)
        for row in per_angle:
            row["area_and_union_s"] = share

    edges_after = unique_edge_lengths_mm(artifact.mesh)
    n_valid = sum(1 for _a, pose in result.best_pose_by_angle if pose is not None)

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(statistics.fmean(values)) if values else 0.0,
            "median": float(statistics.median(values)) if values else 0.0,
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
            "total": float(sum(values)) if values else 0.0,
        }

    n_poses = int(sum(row["n_candidates"] for row in per_angle))
    collision_total = clocks["evaluate_envelope_collision"].total_s
    report = {
        "result": {
            "coverage_percent": result.coverage_percent,
            "covered_area_mm2": result.covered_area_mm2,
            "target_area_mm2": result.target_area_mm2,
            "n_covered_faces": len(result.covered_face_ids),
            "n_valid_poses": n_valid,
            "n_angles": len(result.evaluated_angles),
            "shape_unchanged": bool(
                np.allclose(edges_before, edges_after, atol=1e-6)
                and np.allclose(verts_before, np.asarray(artifact.mesh.vertices))
            ),
            "matches_baseline": (
                abs(result.coverage_percent - EXPECTED["coverage_percent"]) < 1e-3
                and abs(result.covered_area_mm2 - EXPECTED["covered_area_mm2"]) < 1e-2
                and abs(result.target_area_mm2 - EXPECTED["target_area_mm2"]) < 1e-2
            ),
        },
        "sizes": {
            "scraper_vertices": int(len(artifact.mesh.vertices)),
            "scraper_faces": int(len(artifact.mesh.faces)),
            "interior_vertices": int(len(surface.vertices)),
            "interior_faces": int(len(surface.faces)),
            "neighborhood_frames": 17,
        },
        "global_s": {
            "A_scraper_prep": scrap_s,
            "B_surface_prep": surf_build_s + init_s,
            "B1_synthetic_surface": surf_build_s,
            "B2_simulator_init": init_s,
            "C_pose_search": clocks.get("envelope_contact_frame", CallClock()).total_s
            + clocks.get("rigid_pose_neighborhood", CallClock()).total_s
            + clocks.get("apply_rigid_transform", CallClock()).total_s
            + clocks.get("transform_points", CallClock()).total_s
            + clocks.get("rigid_transform_between_frames", CallClock()).total_s,
            "D_collision": collision_total,
            "E_proximity_inside_collision": clocks.get("_proximity", CallClock()).total_s,
            "E2_nearest_on_surface": clocks.get("nearest.on_surface", CallClock()).total_s,
            "F_union_and_area": leftover_s,
            "G_coverage_percent": clocks.get("coverage_score", CallClock()).total_s,
            "evaluate_candidate": eval_s,
            "wall_clock_eval_ms": result.evaluation_ms,
        },
        "calls": {
            name: {"count": clk.count, "total_s": clk.total_s}
            for name, clk in clocks.items()
        },
        "per_angle": per_angle,
        "per_angle_stats": {
            "total_s": stats([row["total_s"] for row in per_angle]),
            "pose_prep_s": stats([row["pose_prep_s"] for row in per_angle]),
            "se3_search_s": stats([row["se3_search_s"] for row in per_angle]),
            "collision_s": stats([row["collision_s"] for row in per_angle]),
            "proximity_s": stats([row["proximity_s"] for row in per_angle]),
            "n_candidates": stats([row["n_candidates"] for row in per_angle]),
            "n_nearest": stats([row["n_nearest"] for row in per_angle]),
            "n_mesh_copy": stats([row["n_mesh_copy"] for row in per_angle]),
            "n_to_trimesh": stats([row["n_to_trimesh"] for row in per_angle]),
        },
        "n_poses_evaluated": n_poses,
        "eval_phase_calls": {},
        "comparison": {},
    }

    def eval_delta(name: str) -> tuple[float, int]:
        after_t, after_n = clocks.get(name, CallClock()).total_s, clocks.get(
            name, CallClock()
        ).count
        if name in clocks:
            after_t, after_n = clocks[name].total_s, clocks[name].count
        prev_t, prev_n = pre_eval.get(name, (0.0, 0))
        return after_t - prev_t, after_n - prev_n

    eval_phase = {}
    for name in (
        "closest_on_envelope",
        "_proximity",
        "nearest.on_surface",
        "trimesh.copy",
        "to_trimesh",
        "evaluate_envelope_collision",
        "transform_points",
        "apply_rigid_transform",
        "envelope_contact_frame",
    ):
        t_s, n = eval_delta(name)
        eval_phase[name] = {"count": n, "total_s": t_s}
    report["eval_phase_calls"] = eval_phase

    after_metrics = {
        "evaluate_s": eval_s,
        "n_proximity": eval_phase["_proximity"]["count"],
        "n_nearest": eval_phase["nearest.on_surface"]["count"],
        "n_mesh_copy": eval_phase["trimesh.copy"]["count"],
        "n_to_trimesh": eval_phase["to_trimesh"]["count"],
        "n_poses": n_poses,
        "n_points": int(PROXIMITY_STATS["points"]),
    }
    report["proximity_stats"] = dict(PROXIMITY_STATS)
    comparison = {}
    for key, before_v in BEFORE.items():
        after_v = after_metrics[key]
        gain = None
        if isinstance(before_v, float) and before_v > 0:
            gain = 100.0 * (before_v - after_v) / before_v
        elif isinstance(before_v, int) and before_v > 0:
            gain = 100.0 * (before_v - after_v) / before_v
        comparison[key] = {"before": before_v, "after": after_v, "gain_pct": gain}
    report["comparison"] = comparison
    print(json.dumps(report, indent=2))
    print("\nA0 AVANT / APRÈS")
    print(f"{'metric':<18} {'before':>12} {'after':>12} {'gain %':>10}")
    for key, row in comparison.items():
        gain = "" if row["gain_pct"] is None else f"{row['gain_pct']:.1f}"
        print(f"{key:<18} {row['before']:>12} {row['after']:>12} {gain:>10}")


if __name__ == "__main__":
    main()
