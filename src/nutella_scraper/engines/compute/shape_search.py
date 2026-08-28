"""Heuristic search of scraper *shapes* on interior_matrix_a0_0_90.

Pipeline (shape and trajectory stay distinct):

    SHAPE PARAMETERS (longitudinal blade profile)
            ↓
    SCRAPER 3D (existing loft, 2 mm thickness)
            ↓
    POSE GRAPH V2 (Y, azimuth) — not cloud waypoints
            ↓
    CONTACT UNION (existing collision masks)
            ↓
    COVERAGE = unique_touched / N

Does not import CoverageSimulator. Does not use the historical A0 point grid.
Trajectory model: POSE_GRAPH. Label: HEURISTIC.
Meilleure trajectoire trouvee, not a proven optimum.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
    LEGACY_A0_QUADRANT_REGION,
    CoverageReferenceMatrix,
    build_coverage_reference_matrix,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.pose_contact_cache import (
    PoseContactCache,
    build_pose_contact_cache,
)
from nutella_scraper.engines.compute.pose_space import (
    TRAJECTORY_MODEL,
    PoseMotionLimits,
    PoseSampleSpec,
    PoseSamplingConfig,
    motion_limits_from_surface,
    sample_pose_specs,
)
from nutella_scraper.engines.compute.pose_trajectory import (
    BEAM_WIDTH,
    assert_path_physical,
    beam_search_pose_trajectories,
    build_pose_edges,
    opening_pose_ids,
    reachable_pose_ids,
)
from nutella_scraper.engines.compute.shape_constraints import (
    GeometryValidity,
    validate_profile,
)
from nutella_scraper.engines.compute.shape_families import (
    BLADE_THICKNESS_MM,
    BLADE_WIDTH_MM,
    DEFAULT_SCRAPER_LENGTH_MM,
    FAMILY_BY_ID,
    LENGTH_PRETEST_SPECS,
    SCRAPER_LENGTHS_MM,
    SEARCH_FAMILY_IDS,
    VALIDATION_FAMILY_IDS,
    SagittalFrame,
    SampledProfile,
    ShapeFamily,
    build_sagittal_frame,
    clip_params_to_bounds,
    sample_profile,
)
from nutella_scraper.engines.compute.shape_fitter import (
    fit_family_geometrically,
    geometric_errors,
    profile_length_mm,
)
from nutella_scraper.engines.compute.shape_materialize import (
    materialize_a0,
    materialize_profile,
    profile_fingerprint,
)
from nutella_scraper.engines.compute.shape_result import (
    DISCLAIMER,
    OPTIMIZATION_LABEL,
    OPTIMIZATION_METHOD,
    SearchStats,
    ShapeCandidate,
)
from nutella_scraper.engines.compute.trajectory_search import (
    TrajectoryGrid,
    index_reference_matrix,
)

MAX_SHAPE_EVALUATIONS = 100
A0_REFERENCE_ID = "A0"
PhysicsBuilder = Callable[..., PoseContactCache]


@dataclass(frozen=True)
class ShapeSearchConfig:
    max_shape_evaluations: int = MAX_SHAPE_EVALUATIONS
    family_ids: tuple[str, ...] = SEARCH_FAMILY_IDS
    scraper_lengths_mm: tuple[float, ...] = (DEFAULT_SCRAPER_LENGTH_MM,)
    shape_specs: tuple[tuple[str, float], ...] | None = None
    beam_width: int = BEAM_WIDTH
    top_k_per_family: int = 3
    sample_count: int = 32
    rng_seed: int = 20260825
    run_a0_reference: bool = True
    de_popsize: int = 4
    verbose: bool = False
    pose_sampling: PoseSamplingConfig = PoseSamplingConfig()
    stage_label: str = "PRELIMINARY"

    def iter_shape_specs(self) -> tuple[tuple[str, float], ...]:
        if self.shape_specs is not None:
            return tuple(self.shape_specs)
        return tuple(
            (str(family_id), float(length_mm))
            for family_id in self.family_ids
            for length_mm in self.scraper_lengths_mm
        )


@dataclass(frozen=True)
class ShapeSearchReport:
    grid: TrajectoryGrid
    frame: SagittalFrame
    config: ShapeSearchConfig
    candidates: tuple[ShapeCandidate, ...]
    best_per_family: tuple[ShapeCandidate, ...]
    a0_reference: ShapeCandidate | None
    stats: SearchStats
    optimization_label: str = OPTIMIZATION_LABEL
    optimization_method: str = OPTIMIZATION_METHOD
    disclaimer: str = DISCLAIMER
    target_definition: str = COVERAGE_TARGET_REGION
    cache_shape_fingerprints: tuple[str, ...] = ()


class _ContactCacheStore:
    """One (shape, matrix) → per-pose contact masks. Never recomputed."""

    def __init__(self) -> None:
        self._store: dict[str, PoseContactCache] = {}
        self.hits = 0
        self.misses = 0
        self.generated = 0

    def get(self, fingerprint: str) -> PoseContactCache | None:
        item = self._store.get(fingerprint)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        return item

    def put(self, fingerprint: str, cache: PoseContactCache) -> PoseContactCache:
        self._store[fingerprint] = cache
        return cache


def _require_validated_matrix(matrix: CoverageReferenceMatrix, grid: TrajectoryGrid) -> None:
    if matrix.uses_legacy_a0_point_matrix or grid.uses_legacy_a0_point_matrix:
        raise ValueError("Legacy A0 point matrix cannot be used as the search grid")
    if str(matrix.coverage_target_region) != COVERAGE_TARGET_REGION:
        raise ValueError(
            f"Shape search target must be {COVERAGE_TARGET_REGION}, "
            f"got {matrix.coverage_target_region}"
        )
    if str(matrix.coverage_target_region) == LEGACY_A0_QUADRANT_REGION:
        raise ValueError("Legacy A0 quadrant cannot be used as the search grid")
    if grid.target_definition != COVERAGE_TARGET_REGION:
        raise ValueError("Trajectory grid is not the validated interior matrix")


def _indices_from_mask(mask: int, n_points: int) -> tuple[int, ...]:
    return tuple(index for index in range(int(n_points)) if mask & (1 << index))


def _empty_candidate(
    *,
    candidate_id: str,
    family_id: str,
    profile: SampledProfile,
    validity: GeometryValidity,
    total_points: int,
    mean_err: float,
    max_err: float,
    fingerprint: str,
) -> ShapeCandidate:
    n_points = int(total_points)
    return ShapeCandidate(
        candidate_id=candidate_id,
        family_id=family_id,
        parameters=profile.parameters,
        n_parameters=len(profile.parameters),
        coverage_percent=0.0,
        covered_points=0,
        total_points=n_points,
        covered_point_indices=(),
        untouched_point_indices=tuple(range(n_points)),
        mean_geometric_error_mm=float(mean_err),
        max_geometric_error_mm=float(max_err),
        scraper_length_mm=float(profile.length_mm or validity.length_mm),
        min_curvature_radius_mm=float(validity.min_curvature_radius_mm),
        trajectory_steps=0,
        trajectory_length_mm=0.0,
        lateral_changes=0,
        direction_changes=0,
        geometric_valid=bool(validity.valid),
        physical_valid=False,
        geometric_reasons=validity.reasons,
        profile_points_mm=np.asarray(profile.points_mm, dtype=np.float64),
        shape_fingerprint=fingerprint,
        trajectory_model=TRAJECTORY_MODEL,
        thickness_mm=float(validity.thickness_mm),
        width_mm=float(validity.width_mm),
        termination_reason="geometry_rejected" if not validity.valid else "no_trajectory",
    )


def _limits_for_length(
    surface: InteriorSurfaceReference,
    length_mm: float,
    sampling: PoseSamplingConfig,
) -> PoseMotionLimits:
    return motion_limits_from_surface(
        surface,
        scraper_length_mm=max(float(length_mm), 1.0),
        sampling=sampling,
    )


def _pose_instrumentation(
    cache: PoseContactCache,
    limits: PoseMotionLimits,
) -> dict[str, object]:
    n_adm = sum(1 for entry in cache.entries if entry.admissible)
    n_touch = sum(1 for entry in cache.entries if entry.covered_count > 0)
    starts = opening_pose_ids(cache, limits)
    edges = build_pose_edges(cache, limits)
    reachable = reachable_pose_ids(cache, limits, edges=edges)
    return {
        "n_pose_candidates": int(len(cache.entries)),
        "n_admissible_poses": int(n_adm),
        "n_contacting_poses": int(n_touch),
        "n_reachable_poses": int(len(reachable)),
        "opening_start_available": bool(starts),
        "edges": edges,
    }


def _candidate_from_pose_cache(
    *,
    candidate_id: str,
    profile: SampledProfile,
    validity: GeometryValidity,
    cache: PoseContactCache,
    limits: PoseMotionLimits,
    mean_err: float,
    max_err: float,
    fingerprint: str,
    beam_width: int,
    thickness_mm: float,
) -> ShapeCandidate:
    ranked = beam_search_pose_trajectories(
        cache, limits, beam_width=int(beam_width), top_k=1
    )
    n_points = int(cache.n_points)
    stats = _pose_instrumentation(cache, limits)
    n_adm = int(stats["n_admissible_poses"])
    n_touch = int(stats["n_contacting_poses"])
    if n_points >= 20:
        print(
            f"      poses admissibles={n_adm}/{len(cache.entries)}  "
            f"avec_contact={n_touch}  trajets_beam={len(ranked)}",
            flush=True,
        )
    extra = {
        "n_pose_candidates": int(stats["n_pose_candidates"]),
        "n_admissible_poses": n_adm,
        "n_contacting_poses": n_touch,
        "n_reachable_poses": int(stats["n_reachable_poses"]),
        "opening_start_available": bool(stats["opening_start_available"]),
        "width_mm": BLADE_WIDTH_MM,
        "thickness_mm": float(thickness_mm),
        "scraper_length_mm": float(profile.length_mm or validity.length_mm),
        "poses_evaluated": int(cache.physics_queries),
        "scraper_fingerprint": str(cache.scraper_fingerprint),
    }
    if not ranked:
        empty = _empty_candidate(
            candidate_id=candidate_id,
            family_id=profile.family_id,
            profile=profile,
            validity=validity,
            total_points=n_points,
            mean_err=mean_err,
            max_err=max_err,
            fingerprint=fingerprint,
        )
        reason = (
            "no_opening_start"
            if not stats["opening_start_available"]
            else "no_trajectory"
        )
        return replace(
            empty,
            termination_reason=reason,
            **extra,
        )
    best = ranked[0]
    assert_path_physical(best.path, limits)
    covered = tuple(int(i) for i in best.covered_point_indices)
    untouched = tuple(i for i in range(n_points) if i not in set(covered))
    poses = tuple((float(item.y_mm), float(item.azimuth_deg)) for item in best.path)
    origins = tuple(tuple(float(v) for v in item.origin_mm) for item in best.path)
    floor = bool(limits.reached_useful_depth(best.max_depth_reached_mm))
    return ShapeCandidate(
        candidate_id=candidate_id,
        family_id=profile.family_id,
        parameters=profile.parameters,
        n_parameters=len(profile.parameters),
        coverage_percent=float(best.coverage_percent),
        covered_points=int(best.covered_points),
        total_points=n_points,
        covered_point_indices=covered,
        untouched_point_indices=untouched,
        mean_geometric_error_mm=float(mean_err),
        max_geometric_error_mm=float(max_err),
        scraper_length_mm=float(profile.length_mm or validity.length_mm),
        min_curvature_radius_mm=float(validity.min_curvature_radius_mm),
        trajectory_steps=int(best.position_count),
        trajectory_length_mm=float(best.path_length_mm),
        lateral_changes=int(best.lateral_moves),
        direction_changes=int(best.rotation_changes),
        geometric_valid=True,
        physical_valid=True,
        geometric_reasons=(),
        profile_points_mm=np.asarray(profile.points_mm, dtype=np.float64),
        trajectory_id=str(best.trajectory_id),
        shape_fingerprint=fingerprint,
        scraper_fingerprint=str(cache.scraper_fingerprint),
        poses_evaluated=int(cache.physics_queries),
        beam_trajectories_explored=int(best.beam_trajectories_explored),
        trajectory_poses=poses,
        trajectory_origins=origins,
        trajectory_model=TRAJECTORY_MODEL,
        thickness_mm=float(thickness_mm),
        width_mm=BLADE_WIDTH_MM,
        max_depth_reached_mm=float(best.max_depth_reached_mm),
        n_pose_candidates=int(stats["n_pose_candidates"]),
        n_admissible_poses=n_adm,
        n_contacting_poses=n_touch,
        n_reachable_poses=int(stats["n_reachable_poses"]),
        trajectory_found=True,
        opening_start_available=bool(stats["opening_start_available"]),
        floor_reached=floor,
        termination_reason=str(best.termination_reason or "trajectory_found"),
    )


def evaluate_profile_coverage(
    profile: SampledProfile,
    *,
    surface: InteriorSurfaceReference,
    matrix: CoverageReferenceMatrix,
    specs: tuple[PoseSampleSpec, ...],
    sampling: PoseSamplingConfig,
    frame: SagittalFrame,
    store: _ContactCacheStore,
    candidate_id: str,
    beam_width: int = BEAM_WIDTH,
    physics_builder: PhysicsBuilder = build_pose_contact_cache,
) -> ShapeCandidate:
    """Shape → 3D 2 mm blade → pose cache → V2 beam → UNION coverage."""
    started = time.perf_counter()
    hits0, misses0 = store.hits, store.misses
    store.generated += 1
    validity = validate_profile(profile, frame, requested_length_mm=profile.length_mm)
    mean_err, max_err = geometric_errors(profile, frame)
    fingerprint = profile_fingerprint(profile)
    length_mm = float(profile.length_mm) or max(
        profile_length_mm(profile), float(validity.length_mm), 1.0
    )
    limits = _limits_for_length(surface, length_mm, sampling)
    if not validity.valid:
        return replace(
            _empty_candidate(
                candidate_id=candidate_id,
                family_id=profile.family_id,
                profile=profile,
                validity=validity,
                total_points=int(matrix.point_count),
                mean_err=mean_err,
                max_err=max_err,
                fingerprint=fingerprint,
            ),
            elapsed_seconds=time.perf_counter() - started,
            cache_hits=store.hits - hits0,
            cache_misses=store.misses - misses0,
        )
    cached = store.get(fingerprint)
    if cached is None:
        try:
            artifact = materialize_profile(
                profile,
                surface,
                length_mm=length_mm,
            )
            cached = physics_builder(
                surface,
                matrix,
                specs,
                artifact=artifact,
                parameters=None,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failed = GeometryValidity(
                valid=False,
                reasons=validity.reasons + ("loft_or_physics_failed",),
                length_mm=validity.length_mm,
                min_curvature_radius_mm=validity.min_curvature_radius_mm,
                max_turn_deg=validity.max_turn_deg,
                self_intersects=validity.self_intersects,
                monotonic_down=validity.monotonic_down,
                inside_envelope=validity.inside_envelope,
            )
            return replace(
                _empty_candidate(
                    candidate_id=candidate_id,
                    family_id=profile.family_id,
                    profile=profile,
                    validity=failed,
                    total_points=int(matrix.point_count),
                    mean_err=mean_err,
                    max_err=max_err,
                    fingerprint=fingerprint,
                ),
                elapsed_seconds=time.perf_counter() - started,
                cache_hits=store.hits - hits0,
                cache_misses=store.misses - misses0,
            )
        store.put(fingerprint, cached)
    return replace(
        _candidate_from_pose_cache(
            candidate_id=candidate_id,
            profile=profile,
            validity=validity,
            cache=cached,
            limits=limits,
            mean_err=mean_err,
            max_err=max_err,
            fingerprint=fingerprint,
            beam_width=beam_width,
            thickness_mm=BLADE_THICKNESS_MM,
        ),
        elapsed_seconds=time.perf_counter() - started,
        cache_hits=store.hits - hits0,
        cache_misses=store.misses - misses0,
    )


def evaluate_a0_reference(
    *,
    surface: InteriorSurfaceReference,
    matrix: CoverageReferenceMatrix,
    specs: tuple[PoseSampleSpec, ...],
    sampling: PoseSamplingConfig,
    frame: SagittalFrame,
    store: _ContactCacheStore,
    beam_width: int = BEAM_WIDTH,
    physics_builder: PhysicsBuilder = build_pose_contact_cache,
) -> ShapeCandidate:
    """A0 manufacturing solid as a historical baseline. Not a 2 mm search blade."""
    started = time.perf_counter()
    hits0, misses0 = store.hits, store.misses
    store.generated += 1
    from nutella_scraper.engines.compute.trajectory_contact_cache import (
        reference_scraper_parameters,
    )

    a0_length = float(reference_scraper_parameters(surface).length_mm)
    window = frame.window_for_length(a0_length)
    wall = np.asarray(window.meridian_xyz_mm, dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(wall), dtype=np.float64)
    profile = SampledProfile(
        family_id=A0_REFERENCE_ID,
        parameters=(),
        t=t,
        y_mm=np.asarray(wall[:, 1], dtype=np.float64),
        r_mm=np.hypot(wall[:, 0], wall[:, 2]),
        points_mm=wall,
        length_mm=float(a0_length),
    )
    validity = validate_profile(profile, frame, requested_length_mm=a0_length)
    mean_err, max_err = geometric_errors(profile, frame)
    fingerprint = "A0-manufacturing-solid"
    limits = _limits_for_length(surface, a0_length, sampling)
    cached = store.get(fingerprint)
    if cached is None:
        artifact = materialize_a0(surface)
        cached = physics_builder(
            surface,
            matrix,
            specs,
            artifact=artifact,
            parameters=None,
        )
        store.put(fingerprint, cached)
    ranked = _candidate_from_pose_cache(
        candidate_id=A0_REFERENCE_ID,
        profile=profile,
        validity=validity,
        cache=cached,
        limits=limits,
        mean_err=mean_err,
        max_err=max_err,
        fingerprint=fingerprint,
        beam_width=beam_width,
        thickness_mm=2.5,
    )
    return replace(
        ranked,
        family_id=A0_REFERENCE_ID,
        candidate_id=A0_REFERENCE_ID,
        parameters=(),
        n_parameters=0,
        geometric_valid=True,
        optimization_method="a0_reference",
        elapsed_seconds=time.perf_counter() - started,
        cache_hits=store.hits - hits0,
        cache_misses=store.misses - misses0,
        thickness_mm=2.5,
        width_mm=2.5,
        scraper_length_mm=float(a0_length),
    )


def _initial_param_sets(
    family: ShapeFamily,
    frame: SagittalFrame,
    rng: np.random.Generator,
    budget: int,
) -> list[NDArray[np.float64]]:
    seeds = [clip_params_to_bounds(family, family.default_params(frame), frame)]
    try:
        fitted = fit_family_geometrically(family, frame)
        seeds.append(clip_params_to_bounds(family, fitted, frame))
    except (ValueError, np.linalg.LinAlgError):
        pass
    bounds = family.bounds(frame)
    while len(seeds) < min(max(int(budget), 1), 6):
        draw = bounds[:, 0] + rng.random(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        seeds.append(clip_params_to_bounds(family, draw, frame))
    unique: list[NDArray[np.float64]] = []
    seen: set[tuple[float, ...]] = set()
    for item in seeds:
        key = tuple(round(float(v), 6) for v in item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[: max(int(budget), 1)]


def _optimize_family(
    family: ShapeFamily,
    *,
    surface: InteriorSurfaceReference,
    matrix: CoverageReferenceMatrix,
    specs: tuple[PoseSampleSpec, ...],
    sampling: PoseSamplingConfig,
    frame: SagittalFrame,
    store: _ContactCacheStore,
    config: ShapeSearchConfig,
    rng: np.random.Generator,
    physics_builder: PhysicsBuilder,
    id_prefix: str,
    length_mm: float,
) -> list[ShapeCandidate]:
    budget = max(1, int(config.max_shape_evaluations))
    evaluated: list[ShapeCandidate] = []
    counter = 0
    window = frame.window_for_length(float(length_mm))

    def _eval(vector: NDArray[np.float64]) -> ShapeCandidate:
        nonlocal counter
        counter += 1
        params = clip_params_to_bounds(family, vector, window)
        profile = sample_profile(
            family,
            params,
            frame,
            sample_count=config.sample_count,
            length_mm=float(length_mm),
        )
        candidate_id = f"{id_prefix}{counter:03d}"
        return evaluate_profile_coverage(
            profile,
            surface=surface,
            matrix=matrix,
            specs=specs,
            sampling=sampling,
            frame=frame,
            store=store,
            candidate_id=candidate_id,
            beam_width=config.beam_width,
            physics_builder=physics_builder,
        )

    seeds: list[NDArray[np.float64]] = []
    fallback: list[NDArray[np.float64]] = []
    for seed in _initial_param_sets(family, window, rng, budget):
        params = clip_params_to_bounds(family, seed, window)
        profile = sample_profile(
            family,
            params,
            frame,
            sample_count=config.sample_count,
            length_mm=float(length_mm),
        )
        if validate_profile(profile, frame, requested_length_mm=float(length_mm)).valid:
            seeds.append(params)
        else:
            fallback.append(params)
    for seed in seeds + fallback:
        if len(evaluated) >= budget:
            break
        evaluated.append(_eval(seed))

    remaining = budget - len(evaluated)
    if remaining >= max(8, family.n_parameters * config.de_popsize):
        bounds = family.bounds(window)

        def _objective(vector: NDArray[np.float64]) -> float:
            if len(evaluated) >= budget:
                return 0.0
            item = _eval(vector)
            evaluated.append(item)
            return -float(item.covered_points)

        differential_evolution(
            _objective,
            bounds=list(zip(bounds[:, 0], bounds[:, 1], strict=True)),
            maxiter=max(1, remaining // max(family.n_parameters, 1)),
            popsize=int(config.de_popsize),
            seed=int(config.rng_seed),
            polish=False,
            updating="immediate",
        )
    evaluated.sort(key=lambda item: item.rank_tuple())
    return evaluated[: max(1, int(config.top_k_per_family))]


def rank_shape_candidates(
    candidates: Sequence[ShapeCandidate],
) -> tuple[ShapeCandidate, ...]:
    ordered = sorted(candidates, key=lambda item: item.rank_tuple())
    return tuple(ordered)


def comparison_rows(
    ranked: Sequence[ShapeCandidate],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for rank, item in enumerate(ranked, start=1):
        rows.append(
            {
                "Rank": rank,
                "Family": item.family_id,
                "Candidate": item.candidate_id,
                "Coverage %": round(float(item.coverage_percent), 4),
                "Covered points": int(item.covered_points),
                "Parameters": int(item.n_parameters),
                "Mean geometric error": round(float(item.mean_geometric_error_mm), 4),
                "Max geometric error": round(float(item.max_geometric_error_mm), 4),
                "Trajectory length": round(float(item.trajectory_length_mm), 4),
                "Direction changes": int(item.direction_changes),
                "Min curvature radius": round(float(item.min_curvature_radius_mm), 4),
                "Length mm": round(float(item.scraper_length_mm), 4),
                "Thickness mm": round(float(item.thickness_mm), 4),
                "Width mm": round(float(item.width_mm), 4),
                "Admissible poses": int(item.n_admissible_poses),
                "Trajectory found": bool(item.trajectory_found),
                "Opening start": bool(item.opening_start_available),
                "Termination": item.termination_reason,
                "Valid": bool(item.geometric_valid and item.physical_valid),
            }
        )
    return tuple(rows)


def a0_gain_summary(
    a0: ShapeCandidate | None,
    best: ShapeCandidate | None,
) -> dict[str, Any]:
    if a0 is None or best is None:
        return {}
    gain = int(best.covered_points) - int(a0.covered_points)
    return {
        "a0_historique_coverage_percent": float(a0.coverage_percent),
        "a0_historique_covered_points": int(a0.covered_points),
        "meilleure_forme_trouvee_coverage_percent": float(best.coverage_percent),
        "meilleure_forme_trouvee_covered_points": int(best.covered_points),
        "gain_points": gain,
        "optimization_label": OPTIMIZATION_LABEL,
    }


def search_scraper_shapes(
    surface: InteriorSurfaceReference,
    *,
    matrix: CoverageReferenceMatrix | None = None,
    config: ShapeSearchConfig | None = None,
    physics_builder: PhysicsBuilder = build_pose_contact_cache,
) -> ShapeSearchReport:
    settings = config or ShapeSearchConfig()
    if matrix is None:
        matrix = build_coverage_reference_matrix(surface)
    grid = index_reference_matrix(matrix, surface=surface)
    _require_validated_matrix(matrix, grid)
    frame = build_sagittal_frame(surface)
    specs = sample_pose_specs(surface, settings.pose_sampling)
    store = _ContactCacheStore()
    rng = np.random.default_rng(int(settings.rng_seed))
    stats = SearchStats()
    started = time.perf_counter()
    found: list[ShapeCandidate] = []
    a0_ref: ShapeCandidate | None = None
    if settings.verbose:
        print(
            f"  pose lattice n={len(specs)}  families={list(settings.family_ids)}  "
            f"stage={settings.stage_label}",
            flush=True,
        )
    if settings.run_a0_reference:
        if settings.verbose:
            print("  A0 historique (référence V2, pas une lame 2 mm) …", flush=True)
        t0 = time.perf_counter()
        a0_ref = evaluate_a0_reference(
            surface=surface,
            matrix=matrix,
            specs=specs,
            sampling=settings.pose_sampling,
            frame=frame,
            store=store,
            beam_width=settings.beam_width,
            physics_builder=physics_builder,
        )
        stats.time_per_family_s[A0_REFERENCE_ID] = time.perf_counter() - t0
        if settings.verbose:
            print(
                f"      couverture={a0_ref.coverage_percent:.3f}%  "
                f"t={stats.time_per_family_s[A0_REFERENCE_ID]:.1f}s",
                flush=True,
            )
    shape_specs = settings.iter_shape_specs()
    family_total = len(shape_specs)
    for offset, (family_id, length_mm) in enumerate(shape_specs, start=2):
        if settings.verbose:
            print(
                f"  [{offset}/{family_total + 1}] {family_id} L={length_mm:.0f} mm …",
                flush=True,
            )
        family = FAMILY_BY_ID[str(family_id)]
        t0 = time.perf_counter()
        prefix = f"{family.family_id}-L{length_mm:.0f}-"
        items = _optimize_family(
            family,
            surface=surface,
            matrix=matrix,
            specs=specs,
            sampling=settings.pose_sampling,
            frame=frame,
            store=store,
            config=settings,
            rng=rng,
            physics_builder=physics_builder,
            id_prefix=prefix,
            length_mm=float(length_mm),
        )
        key = f"{family.family_id}:{length_mm:.0f}"
        stats.time_per_family_s[key] = time.perf_counter() - t0
        found.extend(items)
        if settings.verbose and items:
            print(
                f"      couverture={items[0].coverage_percent:.3f}%  "
                f"t={stats.time_per_family_s[key]:.1f}s",
                flush=True,
            )
    stats.shapes_generated = int(store.generated)
    stats.cache_hits = int(store.hits)
    stats.cache_misses = int(store.misses)
    stats.physics_simulations = int(store.misses)
    stats.total_time_s = time.perf_counter() - started
    ranked = rank_shape_candidates(found)
    best_per_family = []
    seen_family: set[str] = set()
    for item in ranked:
        if item.family_id in seen_family:
            continue
        seen_family.add(item.family_id)
        best_per_family.append(item)
    return ShapeSearchReport(
        grid=grid,
        frame=frame,
        config=settings,
        candidates=ranked,
        best_per_family=tuple(best_per_family),
        a0_reference=a0_ref,
        stats=stats,
        cache_shape_fingerprints=tuple(store._store.keys()),
    )


def validation_config() -> ShapeSearchConfig:
    """Tiny budget: straight, simple Bézier, plus A0. Not a campaign."""
    return ShapeSearchConfig(
        max_shape_evaluations=1,
        family_ids=VALIDATION_FAMILY_IDS,
        scraper_lengths_mm=(DEFAULT_SCRAPER_LENGTH_MM,),
        beam_width=8,
        top_k_per_family=1,
        sample_count=16,
        run_a0_reference=True,
        pose_sampling=PoseSamplingConfig(height_step_mm=40.0, azimuth_step_deg=45.0),
        stage_label="PRELIMINARY",
    )


def preliminary_config() -> ShapeSearchConfig:
    """Five simple families at 40 mm. Not a 35-form length sweep."""
    return ShapeSearchConfig(
        max_shape_evaluations=1,
        family_ids=SEARCH_FAMILY_IDS,
        scraper_lengths_mm=(DEFAULT_SCRAPER_LENGTH_MM,),
        beam_width=BEAM_WIDTH,
        top_k_per_family=1,
        sample_count=32,
        run_a0_reference=True,
        verbose=True,
        stage_label="PRELIMINARY",
    )


def length_pretest_config() -> ShapeSearchConfig:
    """A0 + 6 short 2 mm blades. STOP gate before any 5x7 campaign."""
    return ShapeSearchConfig(
        max_shape_evaluations=1,
        family_ids=("straight", "concave", "convex"),
        shape_specs=LENGTH_PRETEST_SPECS,
        beam_width=BEAM_WIDTH,
        top_k_per_family=1,
        sample_count=32,
        run_a0_reference=True,
        verbose=True,
        stage_label="LENGTH_PRETEST",
    )


def length_sweep_config() -> ShapeSearchConfig:
    """5 families x 7 lengths. Do not call unless pretest gate passes."""
    return ShapeSearchConfig(
        max_shape_evaluations=1,
        family_ids=SEARCH_FAMILY_IDS,
        scraper_lengths_mm=SCRAPER_LENGTHS_MM,
        beam_width=BEAM_WIDTH,
        top_k_per_family=1,
        sample_count=32,
        run_a0_reference=True,
        verbose=True,
        stage_label="LENGTH_SWEEP",
    )


def descending_contact_trajectory(item: ShapeCandidate) -> bool:
    """True iff a 2 mm candidate actually scrapes downward with UNION > 0."""
    if str(item.family_id) == A0_REFERENCE_ID:
        return False
    ys = [float(y) for y, _az in item.trajectory_poses]
    if len(ys) < 2:
        return False
    if int(item.covered_points) < 1:
        return False
    return float(ys[-1]) < float(ys[0]) - 1e-6


def real_matrix_validation_config() -> ShapeSearchConfig:
    """Alias of the short V2 preliminary (no 14-family campaign)."""
    return preliminary_config()


def report_to_payload(report: ShapeSearchReport) -> dict[str, Any]:
    ranked = list(report.candidates)
    best = ranked[0] if ranked else None
    return {
        "target_definition": report.target_definition,
        "uses_legacy_a0_point_matrix": report.grid.uses_legacy_a0_point_matrix,
        "optimization_label": report.optimization_label,
        "stage_label": report.config.stage_label,
        "trajectory_model": TRAJECTORY_MODEL,
        "optimization_method": report.optimization_method,
        "disclaimer": report.disclaimer,
        "grid": {
            "rows": report.grid.n_rows,
            "cols": report.grid.n_cols,
            "points": len(report.grid.cells),
            "target_definition": report.grid.target_definition,
        },
        "stats": report.stats.to_payload(),
        "table": list(comparison_rows(ranked)),
        "best_per_family": list(comparison_rows(report.best_per_family)),
        "a0_comparison": a0_gain_summary(report.a0_reference, best),
        "best_candidate_found": (
            {
                "candidate_id": best.candidate_id,
                "family": best.family_id,
                "coverage_percent": best.coverage_percent,
                "covered_points": best.covered_points,
            }
            if best is not None
            else None
        ),
    }
