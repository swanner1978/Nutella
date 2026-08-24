"""Rigid-shape coverage over the interior reference matrix (A0 → +45°).

The scraper solid is lofted once. Each sample angle searches a small SE(3)
neighbourhood and keeps the admissible pose that maximises contact with the
faces of that angle. Coverage is the union of touched faces in the matrix
zone, never a sum across frames and never a ×4 / ×8 shortcut.

Target faces follow coverage_reference_matrix (same zone as the viewer
white point cloud). The yellow A0 scraper is visual-only.

Does not import visualization. Does not read the viewer visual mesh.
The legacy 90° A0-centroid region is not used as the simulation target.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.coverage_reference_matrix import (
    A0_MERIDIAN_AZIMUTH_DEG,
    COVERAGE_TARGET_REGION,
    COVERAGE_TARGET_SURFACE,
    REFERENCE_ZONE_SPAN_DEG,
    build_coverage_reference_matrix,
)
from nutella_scraper.engines.compute.coverage_scorer import CoverageScorer
from nutella_scraper.engines.compute.envelope_surface_proximity import (
    bind_envelope_proximity,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.mesh_utils import face_areas
from nutella_scraper.engines.compute.scraper_envelope_collision import (
    evaluate_envelope_collision,
    rigid_pose_neighborhood,
)
from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    EnvelopeContactFrame,
    RigidScraperArtifact,
    envelope_contact_frame,
    rigid_transform_between_frames,
    transform_points,
)
from nutella_scraper.engines.compute.scraper_transform import pose_from_matrix

ANGLE_START_DEG = 0.0
ANGLE_END_DEG = 45.0
ANGLE_STEP_DEG = 2.0
REFERENCE_CANDIDATE_ID = "A0"


def coverage_angle_samples_deg(
    *,
    start_deg: float = ANGLE_START_DEG,
    end_deg: float = ANGLE_END_DEG,
    step_deg: float = ANGLE_STEP_DEG,
) -> tuple[float, ...]:
    """0, 2, …, 44, 45. The end angle is always included."""
    if float(step_deg) <= 0.0:
        raise ValueError("angle step must be positive")
    values = [
        round(float(start_deg) + float(step_deg) * i, 10)
        for i in range(int(np.floor((float(end_deg) - float(start_deg)) / float(step_deg))) + 1)
        if float(start_deg) + float(step_deg) * i < float(end_deg) - 1e-9
    ]
    if not values or abs(values[-1] - float(end_deg)) > 1e-9:
        values.append(float(end_deg))
    return tuple(float(v) for v in values)


def unique_edge_lengths_mm(mesh: trimesh.Trimesh) -> NDArray[np.float64]:
    """Sorted unique-edge lengths — intrinsic shape fingerprint under SE(3)."""
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(edges) == 0:
        return np.asarray([], dtype=np.float64)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return np.sort(lengths)


@dataclass(frozen=True)
class CoverageResult:
    """Coverage of one rigid candidate over the 90° interior-wall quadrant."""

    candidate_id: str
    coverage_percent: float
    covered_area_mm2: float
    target_area_mm2: float
    angle_start_deg: float
    angle_end_deg: float
    angle_step_deg: float
    evaluated_angles: tuple[float, ...]
    covered_face_ids: frozenset[int]
    best_pose_by_angle: tuple[tuple[float, ScraperPose | None], ...]
    touched_face_ids_by_angle: tuple[tuple[float, tuple[int, ...]], ...]
    shape_fingerprint: str
    evaluation_ms: float = 0.0
    quadrant_areas_mm2: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    symmetry_multiplier_applied: bool = False
    coverage_target_surface: str = COVERAGE_TARGET_SURFACE
    coverage_target_region: str = COVERAGE_TARGET_REGION
    coverage_target_azimuth_range: tuple[float, float] = (
        A0_MERIDIAN_AZIMUTH_DEG,
        float(np.mod(A0_MERIDIAN_AZIMUTH_DEG + REFERENCE_ZONE_SPAN_DEG, 360.0)),
    )

    @property
    def uses_visual_stl(self) -> bool:
        return False


@dataclass(frozen=True)
class CoverageBatchInvariants:
    """Jar-level structures shared by every candidate in a coverage batch.

    A. Batch-invariant (built once): interior mesh, KD-trees, face areas,
       A0→+45° interior-matrix mask, azimuths, angle samples, envelope frames for φ≠0,
       SE(3) neighbourhoods of those frames, collision corridor limits.
    B. Candidate-specific: lofted mesh, design frame, fingerprint.
    C. Pose-specific: SE(3) vertex transform, collision/contact query.
    D. Angle-specific (jar): target faces of φ; (candidate): best pose of φ.
    """

    parameters_by_angle: dict[float, ScraperParameters]
    envelope_frame_by_angle: dict[float, EnvelopeContactFrame]
    neighborhood_by_angle: dict[float, tuple[EnvelopeContactFrame, ...]]
    neighborhood_matrices_by_angle: dict[float, tuple[NDArray[np.float64], ...]]


def rank_coverage_results(
    results: Sequence[CoverageResult],
) -> tuple[CoverageResult, ...]:
    """Coverage desc, then covered area, valid poses, candidate_id."""
    return tuple(
        sorted(
            results,
            key=lambda item: (
                -float(item.coverage_percent),
                -float(item.covered_area_mm2),
                -_valid_pose_count(item),
                str(item.candidate_id),
            ),
        )
    )


def _valid_pose_count(result: CoverageResult) -> int:
    return sum(1 for _angle, pose in result.best_pose_by_angle if pose is not None)


class CoverageSimulator:
    """Evaluate rigid scraper coverage on InteriorSurfaceReference only."""

    def __init__(
        self,
        surface: InteriorSurfaceReference,
        *,
        parameters: ScraperParameters | None = None,
        catalog: Mapping[str, RigidScraperArtifact] | None = None,
    ) -> None:
        self._surface = surface
        self._parameters = parameters if parameters is not None else ScraperParameters.default()
        self._catalog: dict[str, RigidScraperArtifact] = dict(catalog or {})
        self._results: dict[tuple[str, str], CoverageResult] = {}
        self._scorer = CoverageScorer()
        self._surface_mesh = surface.to_trimesh()
        # Build vertex/centroid KD-trees once; collision/contact reuse them.
        bind_envelope_proximity(self._surface_mesh)
        self._areas = face_areas(self._surface_mesh)
        opening_y, lower_y, _span = scraper_length_span(surface)
        self._opening_y = opening_y
        self._lower_y = lower_y
        self._azimuths = _face_azimuths_deg(self._surface_mesh, surface)
        centroids = np.asarray(self._surface_mesh.triangles_center, dtype=np.float64)
        useful = (centroids[:, 1] >= lower_y - 1e-3) & (centroids[:, 1] <= opening_y)
        self._useful_mask = useful
        self._quadrant_areas = _quadrant_useful_areas_mm2(
            self._areas, self._azimuths, useful
        )
        self._reference_matrix = build_coverage_reference_matrix(surface)
        in_sector = np.zeros(len(self._areas), dtype=np.bool_)
        in_sector[list(self._reference_matrix.target_face_ids)] = True
        self._target_face_ids = frozenset(self._reference_matrix.target_face_ids)
        self._target_area_mm2 = float(self._reference_matrix.target_area_mm2)
        self._angles = coverage_angle_samples_deg()
        self._faces_by_angle = _assign_faces_to_angles(
            self._azimuths, in_sector, self._angles
        )
        self._batch_invariants: CoverageBatchInvariants | None = None

    def register(self, candidate_id: str, artifact: RigidScraperArtifact) -> None:
        self._catalog[str(candidate_id)] = artifact

    def _prepare_batch_invariants(self) -> CoverageBatchInvariants:
        """Build jar-level angle frames and SE(3) neighbourhoods once."""
        cached = self._batch_invariants
        if cached is not None:
            return cached
        parameters_by_angle: dict[float, ScraperParameters] = {}
        envelope_frame_by_angle: dict[float, EnvelopeContactFrame] = {}
        neighborhood_by_angle: dict[float, tuple[EnvelopeContactFrame, ...]] = {}
        neighborhood_matrices_by_angle: dict[
            float, tuple[NDArray[np.float64], ...]
        ] = {}
        for angle in self._angles:
            key = float(angle)
            params = self._parameters.with_updates(surface_progress_deg=key)
            parameters_by_angle[key] = params
            if abs(key) <= 1e-9:
                continue
            frame = envelope_contact_frame(
                self._surface,
                params,
                surface_progress_deg=key,
                surface_mesh=self._surface_mesh,
            )
            neighborhood = rigid_pose_neighborhood(frame)
            envelope_frame_by_angle[key] = frame
            neighborhood_by_angle[key] = neighborhood
            neighborhood_matrices_by_angle[key] = tuple(
                item.matrix() for item in neighborhood
            )
        invariants = CoverageBatchInvariants(
            parameters_by_angle=parameters_by_angle,
            envelope_frame_by_angle=envelope_frame_by_angle,
            neighborhood_by_angle=neighborhood_by_angle,
            neighborhood_matrices_by_angle=neighborhood_matrices_by_angle,
        )
        self._batch_invariants = invariants
        return invariants

    def evaluate_candidate(self, candidate_id: str) -> CoverageResult:
        key = str(candidate_id)
        if key not in self._catalog:
            raise KeyError(f"unknown candidate {key!r}")
        artifact = self._catalog[key]
        cache_key = (key, str(artifact.shape_fingerprint))
        cached = self._results.get(cache_key)
        if cached is not None:
            return cached
        result = self._evaluate(key, artifact, use_batch_invariants=False)
        self._results[cache_key] = result
        return result

    def evaluate_candidates(
        self,
        candidate_ids: Sequence[str] | None = None,
    ) -> tuple[CoverageResult, ...]:
        ids = (
            [str(v) for v in candidate_ids]
            if candidate_ids is not None
            else list(self._catalog)
        )
        results = [self.evaluate_candidate(item) for item in ids]
        return tuple(
            sorted(results, key=lambda item: (-item.coverage_percent, item.candidate_id))
        )

    def evaluate_candidates_batch(
        self,
        candidate_ids: Sequence[str] | None = None,
    ) -> tuple[CoverageResult, ...]:
        """Evaluate several rigid shapes, preparing jar invariants once.

        Physics match ``evaluate_candidate``: same angles, same 17 SE(3)
        poses, same collision and coverage. Envelope frames for φ≠0 and
        their neighbourhoods are computed once for the batch.
        """
        self._prepare_batch_invariants()
        ids = (
            [str(v) for v in candidate_ids]
            if candidate_ids is not None
            else list(self._catalog)
        )
        results: list[CoverageResult] = []
        for item in ids:
            key = str(item)
            if key not in self._catalog:
                raise KeyError(f"unknown candidate {key!r}")
            artifact = self._catalog[key]
            cache_key = (key, str(artifact.shape_fingerprint))
            cached = self._results.get(cache_key)
            if cached is not None:
                results.append(cached)
                continue
            result = self._evaluate(key, artifact, use_batch_invariants=True)
            self._results[cache_key] = result
            results.append(result)
        return rank_coverage_results(results)

    def _evaluate(
        self,
        candidate_id: str,
        artifact: RigidScraperArtifact,
        *,
        use_batch_invariants: bool = False,
    ) -> CoverageResult:
        started = time.perf_counter()
        src_vertices = np.asarray(artifact.mesh.vertices, dtype=np.float64)
        src_faces = np.asarray(artifact.mesh.faces, dtype=np.int64)
        src_edges = np.asarray(artifact.mesh.edges_unique, dtype=np.int64)
        design_edges = unique_edge_lengths_mm(artifact.mesh)
        design_inv = (
            np.linalg.inv(artifact.design_frame.matrix())
            if use_batch_invariants
            else None
        )
        covered: set[int] = set()
        poses: list[tuple[float, ScraperPose | None]] = []
        touched_by_angle: list[tuple[float, tuple[int, ...]]] = []
        last_vertices: NDArray[np.float64] | None = None
        for angle in self._angles:
            pose, face_ids, posed_vertices = self._best_pose_for_angle(
                artifact,
                angle,
                src_vertices=src_vertices,
                src_faces=src_faces,
                src_edges=src_edges,
                use_batch_invariants=use_batch_invariants,
                design_inv=design_inv,
            )
            if posed_vertices is not None:
                last_vertices = posed_vertices
            in_target = tuple(
                sorted(face_id for face_id in face_ids if face_id in self._target_face_ids)
            )
            covered.update(in_target)
            poses.append((float(angle), pose))
            touched_by_angle.append((float(angle), in_target))
        if last_vertices is not None and len(src_edges) > 0:
            posed_edges = np.sort(
                np.linalg.norm(
                    last_vertices[src_edges[:, 0]] - last_vertices[src_edges[:, 1]],
                    axis=1,
                )
            )
            if not np.allclose(design_edges, posed_edges, atol=1e-6):
                raise ValueError("SE(3) pose changed intrinsic edge lengths")
        covered_ids = frozenset(covered)
        covered_area = float(sum(self._areas[face_id] for face_id in covered_ids))
        if self._target_area_mm2 <= 0.0:
            percent = 0.0
        else:
            percent = 100.0 * self._scorer.score(
                covered_ids,
                self._target_face_ids - covered_ids,
                self._surface_mesh,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return CoverageResult(
            candidate_id=str(candidate_id),
            coverage_percent=float(max(0.0, min(100.0, percent))),
            covered_area_mm2=covered_area,
            target_area_mm2=self._target_area_mm2,
            angle_start_deg=ANGLE_START_DEG,
            angle_end_deg=ANGLE_END_DEG,
            angle_step_deg=ANGLE_STEP_DEG,
            evaluated_angles=self._angles,
            covered_face_ids=covered_ids,
            best_pose_by_angle=tuple(poses),
            touched_face_ids_by_angle=tuple(touched_by_angle),
            shape_fingerprint=str(artifact.shape_fingerprint),
            evaluation_ms=float(elapsed_ms),
            quadrant_areas_mm2=self._quadrant_areas,
            symmetry_multiplier_applied=False,
            coverage_target_surface=self._reference_matrix.coverage_target_surface,
            coverage_target_region=self._reference_matrix.coverage_target_region,
            coverage_target_azimuth_range=self._reference_matrix.coverage_target_azimuth_range,
        )

    def _best_pose_for_angle(
        self,
        artifact: RigidScraperArtifact,
        angle_deg: float,
        src_vertices: NDArray[np.float64] | None = None,
        src_faces: NDArray[np.int64] | None = None,
        src_edges: NDArray[np.int64] | None = None,
        *,
        use_batch_invariants: bool = False,
        design_inv: NDArray[np.float64] | None = None,
    ) -> tuple[ScraperPose | None, frozenset[int], NDArray[np.float64] | None]:
        vertices0 = (
            np.asarray(src_vertices, dtype=np.float64)
            if src_vertices is not None
            else np.asarray(artifact.mesh.vertices, dtype=np.float64)
        )
        faces0 = (
            np.asarray(src_faces, dtype=np.int64)
            if src_faces is not None
            else np.asarray(artifact.mesh.faces, dtype=np.int64)
        )
        edges0 = (
            np.asarray(src_edges, dtype=np.int64)
            if src_edges is not None
            else np.asarray(artifact.mesh.edges_unique, dtype=np.int64)
        )
        angle_key = float(angle_deg)
        invariants = self._batch_invariants if use_batch_invariants else None
        if invariants is not None:
            parameters = invariants.parameters_by_angle[angle_key]
        else:
            parameters = self._parameters.with_updates(surface_progress_deg=angle_key)
        cached_matrices: tuple[NDArray[np.float64], ...] | None = None
        if abs(angle_key) <= 1e-9:
            frames = rigid_pose_neighborhood(artifact.design_frame)
        elif invariants is not None:
            frames = invariants.neighborhood_by_angle[angle_key]
            cached_matrices = invariants.neighborhood_matrices_by_angle[angle_key]
        else:
            nominal = envelope_contact_frame(
                self._surface,
                parameters,
                surface_progress_deg=angle_key,
                surface_mesh=self._surface_mesh,
            )
            frames = rigid_pose_neighborhood(nominal)
        target_faces = self._faces_by_angle[angle_key]
        best_area = -1.0
        best_pose: ScraperPose | None = None
        best_faces: frozenset[int] = frozenset()
        best_vertices: NDArray[np.float64] | None = None
        inv = design_inv
        for index, frame in enumerate(frames):
            try:
                if cached_matrices is not None:
                    if inv is None:
                        inv = np.linalg.inv(artifact.design_frame.matrix())
                    transform = cached_matrices[index] @ inv
                else:
                    transform = rigid_transform_between_frames(
                        artifact.design_frame, frame
                    )
            except np.linalg.LinAlgError:
                continue
            posed_vertices = transform_points(vertices0, transform)
            report = evaluate_envelope_collision(
                artifact.mesh,
                self._surface,
                parameters,
                surface_mesh=self._surface_mesh,
                vertices=posed_vertices,
                faces=faces0,
                edges_unique=edges0,
            )
            if not report.admissible:
                continue
            face_ids = report.contact_face_ids
            scored = face_ids & target_faces
            area = float(sum(self._areas[face_id] for face_id in scored))
            if area > best_area:
                best_area = area
                best_pose = pose_from_matrix(transform)
                best_faces = face_ids
                best_vertices = posed_vertices
        if best_vertices is None:
            return None, frozenset(), None
        return best_pose, best_faces, best_vertices


def _surface_axis_xz(surface: InteriorSurfaceReference) -> NDArray[np.float64]:
    verts = np.asarray(surface.vertices, dtype=np.float64)
    if len(verts) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    return 0.5 * (mins[[0, 2]] + maxs[[0, 2]])


def _face_azimuths_deg(
    mesh: trimesh.Trimesh,
    surface: InteriorSurfaceReference,
) -> NDArray[np.float64]:
    """Progress convention: 0° = +X, 90° = −Z (cos θ, −sin θ)."""
    axis = _surface_axis_xz(surface)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    dx = centroids[:, 0] - float(axis[0])
    dz = centroids[:, 2] - float(axis[1])
    raw = np.rad2deg(np.arctan2(-dz, dx))
    return np.mod(raw, 360.0)


def _quadrant_useful_areas_mm2(
    areas: NDArray[np.float64],
    azimuths: NDArray[np.float64],
    useful: NDArray[np.bool_],
) -> tuple[float, float, float, float]:
    """Areas of useful faces in 0–45, 45–90, 90–135, 135–180. Never used as ×4."""
    bands = ((0.0, 45.0), (45.0, 90.0), (90.0, 135.0), (135.0, 180.0))
    totals: list[float] = []
    for lo, hi in bands:
        if lo == 0.0:
            mask = useful & (azimuths >= lo) & (azimuths <= hi)
        else:
            mask = useful & (azimuths > lo) & (azimuths <= hi)
        totals.append(float(np.sum(areas[mask])))
    return (totals[0], totals[1], totals[2], totals[3])


def _assign_faces_to_angles(
    azimuths: NDArray[np.float64],
    in_sector: NDArray[np.bool_],
    angles: tuple[float, ...],
) -> dict[float, frozenset[int]]:
    angle_arr = np.asarray(angles, dtype=np.float64)
    assigned: dict[float, set[int]] = {float(a): set() for a in angles}
    for face_id in np.flatnonzero(in_sector):
        az = float(azimuths[int(face_id)])
        nearest = float(angle_arr[int(np.argmin(np.abs(angle_arr - az)))])
        assigned[nearest].add(int(face_id))
    return {key: frozenset(value) for key, value in assigned.items()}
