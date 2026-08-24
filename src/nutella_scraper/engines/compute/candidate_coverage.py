"""Rigid catalog coverage batch — ranking only, no viewer, no reshape.

Each candidate is lofted once, then evaluated independently by
``CoverageSimulator``. SE(3) poses may translate/rotate the frozen solid;
relative vertices, length, curvature, thickness and width stay unchanged.

Does not import visualization. Does not change CoverageSimulator math.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.coverage_simulator import (
    REFERENCE_CANDIDATE_ID,
    CoverageResult,
    CoverageSimulator,
    unique_edge_lengths_mm,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    RigidScraperArtifact,
    build_rigid_scraper_artifact_from_path,
    remap_envelope_path_to_wall_curve,
)

COVERAGE_BATCH_SIZE = 10
PREFERRED_FAMILIES: tuple[str, ...] = (
    "parallel",
    "inclined",
    "asymmetric",
    "progressive",
    "s_curve",
    "combined",
)

A0_BASELINE_COVERAGE_PERCENT = 63.33333333333344
A0_BASELINE_COVERED_AREA_MM2 = 1988.2551285963507
A0_BASELINE_TARGET_AREA_MM2 = 3139.3502030468644
A0_BASELINE_TOUCHED_FACE_COUNT = 152
A0_BASELINE_VALID_POSE_COUNT = 24
A0_BASELINE_TOTAL_POSE_COUNT = 24
A0_BASELINE_FINGERPRINT = (
    "synthetic-interior|w=2.5|l=40|t=2.5|z=40|bevel=0|relief=0|helix=0|clear=0"
)

# Sequential 10-shape ranking validated before the batch refactor.
KNOWN_TEN_RANKING: tuple[tuple[str, float], ...] = (
    ("A0", 63.3333),
    ("S0002", 63.3333),
    ("S0001", 60.0000),
    ("S0003", 60.0000),
    ("S0022", 58.7500),
    ("S0455", 58.3333),
    ("S0023", 55.8333),
    ("S0024", 37.5000),
    ("S0025", 32.9167),
    ("S0454", 25.8333),
)


class A0BaselineRegressionError(RuntimeError):
    """A0 drifted from the validated baseline. Other candidates must not be ranked."""


class CatalogCandidate(Protocol):
    """Minimal catalog row. Visualization CandidateShape satisfies this."""

    candidate_id: str
    family: str
    valid: bool
    shape_fingerprint: str
    control_points_mm: Sequence[Sequence[float]] | tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class CandidateCoverageResult:
    """Comparable coverage of one frozen catalog candidate."""

    candidate_id: str
    family: str
    coverage_percent: float
    covered_area_mm2: float
    useful_area_mm2: float
    touched_face_count: int
    valid_pose_count: int
    total_pose_count: int
    elapsed_seconds: float
    shape_fingerprint: str


@dataclass(frozen=True)
class RankedCoverageBatch:
    """Sequential batch: evaluation order, then ranked table."""

    evaluated: tuple[CandidateCoverageResult, ...]
    ranked: tuple[CandidateCoverageResult, ...]
    total_elapsed_seconds: float
    fingerprints_unchanged: bool


@dataclass(frozen=True)
class _GeometrySnapshot:
    fingerprint: str
    vertices: NDArray[np.float64]
    edge_lengths: NDArray[np.float64]


def select_coverage_catalog(
    catalog: Sequence[CatalogCandidate],
    *,
    count: int = COVERAGE_BATCH_SIZE,
) -> tuple[CatalogCandidate, ...]:
    """A0 first, then distinct valid representatives of several families."""
    if not catalog:
        raise ValueError("candidate catalog is empty")
    first = catalog[0]
    if str(first.candidate_id) != REFERENCE_CANDIDATE_ID:
        raise ValueError(
            f"catalog[0] must be {REFERENCE_CANDIDATE_ID!r}, got {first.candidate_id!r}"
        )
    if not first.valid:
        raise ValueError("catalog A0 is not valid")
    selected: list[CatalogCandidate] = [first]
    seen = {str(first.shape_fingerprint)}
    n = int(count)

    def _try_add(item: CatalogCandidate) -> bool:
        if not item.valid:
            return False
        fingerprint = str(item.shape_fingerprint)
        if fingerprint in seen:
            return False
        selected.append(item)
        seen.add(fingerprint)
        return True

    for _round in range(3):
        for family in PREFERRED_FAMILIES:
            if len(selected) >= n:
                break
            for item in catalog[1:]:
                if str(item.family) != family:
                    continue
                if _try_add(item):
                    break
        if len(selected) >= n:
            break
    for item in catalog[1:]:
        if len(selected) >= n:
            break
        _try_add(item)
    if len(selected) != n:
        raise ValueError(
            f"need {n} distinct valid candidates, selected {len(selected)}"
        )
    ids = [str(item.candidate_id) for item in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("selected candidate ids are not unique")
    if len(seen) != n:
        raise ValueError("selected candidate fingerprints are not unique")
    return tuple(selected)


def rank_candidate_coverage(
    results: Sequence[CandidateCoverageResult],
) -> tuple[CandidateCoverageResult, ...]:
    """Coverage desc, then covered area, valid poses, candidate_id."""
    return tuple(
        sorted(
            results,
            key=lambda item: (
                -float(item.coverage_percent),
                -float(item.covered_area_mm2),
                -int(item.valid_pose_count),
                str(item.candidate_id),
            ),
        )
    )


def a0_matches_baseline(result: CoverageResult | CandidateCoverageResult) -> bool:
    """True if A0 reproduces the validated 0–45° golden numbers."""
    if str(result.candidate_id) != REFERENCE_CANDIDATE_ID:
        return False
    valid = int(getattr(result, "valid_pose_count", -1))
    if valid < 0:
        valid = sum(
            1
            for _angle, pose in result.best_pose_by_angle  # type: ignore[union-attr]
            if pose is not None
        )
    faces = int(getattr(result, "touched_face_count", -1))
    if faces < 0:
        faces = len(result.covered_face_ids)  # type: ignore[union-attr]
    useful = float(
        getattr(result, "useful_area_mm2", getattr(result, "target_area_mm2", 0.0))
    )
    fingerprint = str(result.shape_fingerprint)
    return (
        abs(float(result.coverage_percent) - A0_BASELINE_COVERAGE_PERCENT) <= 1e-6
        and abs(float(result.covered_area_mm2) - A0_BASELINE_COVERED_AREA_MM2) <= 1e-2
        and abs(useful - A0_BASELINE_TARGET_AREA_MM2) <= 1e-2
        and faces == A0_BASELINE_TOUCHED_FACE_COUNT
        and valid == A0_BASELINE_VALID_POSE_COUNT
        and fingerprint == A0_BASELINE_FINGERPRINT
    )


def assert_a0_matches_baseline(
    result: CoverageResult | CandidateCoverageResult,
) -> None:
    if a0_matches_baseline(result):
        return
    raise A0BaselineRegressionError(
        "A0 coverage drifted from the validated baseline; ranking stopped. "
        f"got coverage={result.coverage_percent!r} "
        f"area={result.covered_area_mm2!r} "
        f"fingerprint={result.shape_fingerprint!r}"
    )


def materialize_catalog_candidate(
    candidate: CatalogCandidate,
    *,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
    reference: RigidScraperArtifact,
) -> RigidScraperArtifact:
    """Loft a catalog curve once. A0 reuses the validated manufacturing solid."""
    if str(candidate.candidate_id) == REFERENCE_CANDIDATE_ID:
        return reference
    wall = np.asarray(candidate.control_points_mm, dtype=np.float64)
    path = remap_envelope_path_to_wall_curve(
        reference.design_path, wall, surface
    )
    return build_rigid_scraper_artifact_from_path(
        surface,
        parameters,
        path,
        shape_fingerprint=str(candidate.shape_fingerprint),
    )


def select_and_materialize_catalog(
    catalog: Sequence[CatalogCandidate],
    *,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
    reference: RigidScraperArtifact,
    count: int = COVERAGE_BATCH_SIZE,
) -> tuple[tuple[CatalogCandidate, ...], tuple[tuple[str, str, RigidScraperArtifact], ...]]:
    """A0 first, then distinct loftable catalog curves. Failed lofts are skipped."""
    if not catalog:
        raise ValueError("candidate catalog is empty")
    first = catalog[0]
    if str(first.candidate_id) != REFERENCE_CANDIDATE_ID:
        raise ValueError(
            f"catalog[0] must be {REFERENCE_CANDIDATE_ID!r}, got {first.candidate_id!r}"
        )
    if not first.valid:
        raise ValueError("catalog A0 is not valid")
    selected: list[CatalogCandidate] = [first]
    entries: list[tuple[str, str, RigidScraperArtifact]] = [
        (
            str(first.candidate_id),
            str(first.family),
            materialize_catalog_candidate(
                first,
                surface=surface,
                parameters=parameters,
                reference=reference,
            ),
        )
    ]
    seen = {str(first.shape_fingerprint)}
    n = int(count)

    def _try_add(item: CatalogCandidate) -> bool:
        if not item.valid:
            return False
        fingerprint = str(item.shape_fingerprint)
        if fingerprint in seen:
            return False
        try:
            artifact = materialize_catalog_candidate(
                item,
                surface=surface,
                parameters=parameters,
                reference=reference,
            )
        except (ValueError, AssertionError):
            return False
        selected.append(item)
        entries.append((str(item.candidate_id), str(item.family), artifact))
        seen.add(fingerprint)
        return True

    for _round in range(3):
        for family in PREFERRED_FAMILIES:
            if len(selected) >= n:
                break
            for item in catalog[1:]:
                if str(item.family) != family:
                    continue
                if _try_add(item):
                    break
        if len(selected) >= n:
            break
    for item in catalog[1:]:
        if len(selected) >= n:
            break
        _try_add(item)
    if len(selected) != n:
        raise ValueError(
            f"need {n} distinct loftable candidates, selected {len(selected)}"
        )
    return tuple(selected), tuple(entries)


def snapshot_artifact_geometry(artifact: RigidScraperArtifact) -> _GeometrySnapshot:
    return _GeometrySnapshot(
        fingerprint=str(artifact.shape_fingerprint),
        vertices=np.asarray(artifact.mesh.vertices, dtype=np.float64).copy(),
        edge_lengths=unique_edge_lengths_mm(artifact.mesh).copy(),
    )


def geometry_unchanged(
    artifact: RigidScraperArtifact,
    snapshot: _GeometrySnapshot,
) -> bool:
    if str(artifact.shape_fingerprint) != snapshot.fingerprint:
        return False
    vertices = np.asarray(artifact.mesh.vertices, dtype=np.float64)
    if vertices.shape != snapshot.vertices.shape:
        return False
    if not np.allclose(vertices, snapshot.vertices, atol=1e-9):
        return False
    return bool(
        np.allclose(
            unique_edge_lengths_mm(artifact.mesh),
            snapshot.edge_lengths,
            atol=1e-6,
        )
    )


def candidate_result_from_coverage(
    coverage: CoverageResult,
    *,
    family: str,
    elapsed_seconds: float,
) -> CandidateCoverageResult:
    valid = sum(1 for _angle, pose in coverage.best_pose_by_angle if pose is not None)
    return CandidateCoverageResult(
        candidate_id=str(coverage.candidate_id),
        family=str(family),
        coverage_percent=float(coverage.coverage_percent),
        covered_area_mm2=float(coverage.covered_area_mm2),
        useful_area_mm2=float(coverage.target_area_mm2),
        touched_face_count=len(coverage.covered_face_ids),
        valid_pose_count=int(valid),
        total_pose_count=len(coverage.evaluated_angles),
        elapsed_seconds=float(elapsed_seconds),
        shape_fingerprint=str(coverage.shape_fingerprint),
    )


def evaluate_rigid_candidate_batch(
    simulator: CoverageSimulator,
    entries: Sequence[tuple[str, str, RigidScraperArtifact]],
    *,
    a0_id: str = REFERENCE_CANDIDATE_ID,
    use_batch_invariants: bool = False,
) -> RankedCoverageBatch:
    """Evaluate each frozen artifact sequentially. Abort if A0 drifts.

    ``use_batch_invariants=True`` prepares jar-level envelope frames once
    via ``CoverageSimulator.evaluate_candidates_batch``. Physics stay the
    same as ``evaluate_candidate``.
    """
    if not entries:
        raise ValueError("coverage batch is empty")
    first_id, _first_family, _first_artifact = entries[0]
    if str(first_id) != str(a0_id):
        raise ValueError(f"batch[0] must be {a0_id!r}, got {first_id!r}")

    snapshots: dict[str, _GeometrySnapshot] = {}
    for candidate_id, _family, artifact in entries:
        key = str(candidate_id)
        simulator.register(key, artifact)
        snapshots[key] = snapshot_artifact_geometry(artifact)

    evaluated: list[CandidateCoverageResult] = []
    started = time.perf_counter()
    if use_batch_invariants:
        ranked_cov = simulator.evaluate_candidates_batch(
            [str(candidate_id) for candidate_id, _family, _art in entries]
        )
        by_id = {str(item.candidate_id): item for item in ranked_cov}
        for index, (candidate_id, family, artifact) in enumerate(entries):
            key = str(candidate_id)
            coverage = by_id[key]
            snapshot = snapshots[key]
            if not geometry_unchanged(artifact, snapshot):
                raise ValueError(f"candidate {key!r} deformed during coverage evaluation")
            if str(coverage.shape_fingerprint) != snapshot.fingerprint:
                raise ValueError(
                    f"candidate {key!r} fingerprint changed "
                    f"{snapshot.fingerprint!r} → {coverage.shape_fingerprint!r}"
                )
            item = candidate_result_from_coverage(
                coverage,
                family=family,
                elapsed_seconds=float(coverage.evaluation_ms) / 1000.0,
            )
            if index == 0:
                assert_a0_matches_baseline(item)
            evaluated.append(item)
    else:
        for index, (candidate_id, family, artifact) in enumerate(entries):
            key = str(candidate_id)
            snapshot = snapshots[key]
            t0 = time.perf_counter()
            coverage = simulator.evaluate_candidate(key)
            elapsed = time.perf_counter() - t0
            if not geometry_unchanged(artifact, snapshot):
                raise ValueError(f"candidate {key!r} deformed during coverage evaluation")
            if str(coverage.shape_fingerprint) != snapshot.fingerprint:
                raise ValueError(
                    f"candidate {key!r} fingerprint changed "
                    f"{snapshot.fingerprint!r} → {coverage.shape_fingerprint!r}"
                )
            item = candidate_result_from_coverage(
                coverage, family=family, elapsed_seconds=elapsed
            )
            if index == 0:
                assert_a0_matches_baseline(item)
            evaluated.append(item)

    fingerprints_unchanged = all(
        geometry_unchanged(artifact, snapshots[str(candidate_id)])
        for candidate_id, _family, artifact in entries
    )
    ranked = rank_candidate_coverage(evaluated)
    return RankedCoverageBatch(
        evaluated=tuple(evaluated),
        ranked=ranked,
        total_elapsed_seconds=float(time.perf_counter() - started),
        fingerprints_unchanged=bool(fingerprints_unchanged),
    )


def format_coverage_rank_report(
    batch: RankedCoverageBatch,
) -> str:
    lines = [
        "Rank | Candidate | Family | Coverage | Faces | Valid poses | Time",
        "",
    ]
    for rank, item in enumerate(batch.ranked, start=1):
        lines.append(
            f"{rank} | {item.candidate_id} | {item.family} | "
            f"{item.coverage_percent:.4f} % | {item.touched_face_count} | "
            f"{item.valid_pose_count}/{item.total_pose_count} | "
            f"{item.elapsed_seconds:.2f} s"
        )
    return "\n".join(lines)
