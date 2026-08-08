"""Contact simulation engine — 3D only, no 2D view input."""

from __future__ import annotations

import time
from collections.abc import Callable, MutableMapping

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import (
    CollisionResult,
    ContactOverlayData,
    ContactResult,
    ContactSimulationConfig,
)
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.collision_analyzer import analyze_collision, merge_collisions
from nutella_scraper.engines.compute.constrained_trajectory_sampler import (
    generate_validated_poses,
)
from nutella_scraper.engines.compute.contact_analyzer import merge_face_distances
from nutella_scraper.engines.compute.coverage_scorer import CoverageScorer
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.engines.compute.internal_jar_surface_builder import (
    InternalJarSurfaceBuilder,
    internal_mesh_to_trimesh,
    resolve_internal_jar_surface,
)
from nutella_scraper.engines.compute.interior_contact_analyzer import analyze_interior_contact
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder
from nutella_scraper.engines.compute.pose_constraint_engine import PoseConstraintEngine
from nutella_scraper.engines.compute.scraper_builder import ScraperBuilder
from nutella_scraper.engines.compute.scraper_geometry import ScraperGeometryBuilder
from nutella_scraper.engines.compute.scraper_transform import pose_matrix

PoseResultCallback = Callable[
    [
        int,
        int,
        trimesh.Trimesh,
        np.ndarray,
        np.ndarray,
        tuple,
        CollisionResult,
    ],
    None,
]


class ContactSimulationEngine:
    """
    Simulates contact between a parametric scraper solid and jar inner walls.

    Uses only 3D canonical geometry — never ViewProjectionCache or 2D overlays.
    """

    def __init__(
        self,
        *,
        jar_mesh_builder: JarMeshBuilder | None = None,
        scraper_geometry: ScraperGeometryBuilder | None = None,
        scraper_builder: ScraperBuilder | None = None,
        pose_constraint_engine: PoseConstraintEngine | None = None,
        interior_surface_builder: InternalJarSurfaceBuilder | None = None,
        coverage_scorer: CoverageScorer | None = None,
    ) -> None:
        self._jar_mesh_builder = jar_mesh_builder or JarMeshBuilder()
        self._scraper_builder = scraper_builder or ScraperBuilder()
        self._scraper_geometry = scraper_geometry or ScraperGeometryBuilder(
            builder=self._scraper_builder
        )
        self._pose_constraint_engine = pose_constraint_engine or PoseConstraintEngine(
            jar_mesh_builder=self._jar_mesh_builder,
            scraper_builder=self._scraper_builder,
        )
        self._internal_builder = interior_surface_builder or InternalJarSurfaceBuilder(
            jar_mesh_builder=self._jar_mesh_builder
        )
        self._coverage_scorer = coverage_scorer or CoverageScorer()

    def simulate(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
        config: ContactSimulationConfig,
        *,
        internal: InternalJarSurface | None = None,
        progress_callback: Callable[[str, str, float | None], None] | None = None,
        profile_ms: MutableMapping[str, float] | None = None,
        pose_result_callback: PoseResultCallback | None = None,
    ) -> ContactResult:
        timings = profile_ms if profile_ms is not None else {}

        def add_timing(phase: str, duration_ms: float) -> None:
            timings[phase] = timings.get(phase, 0.0) + duration_ms

        generation_started = time.perf_counter()
        if progress_callback is not None:
            progress_callback(
                "scraper_generation",
                "Construction du maillage 3D du racloir",
                5.0,
            )
        jar_mesh = self._resolve_jar_mesh(jar, internal)
        interior_started = time.perf_counter()
        if progress_callback is not None:
            progress_callback(
                "interior_surface_calculation",
                "Construction de la surface intérieure accessible",
                2.0,
            )
        internal_surface = resolve_internal_jar_surface(
            jar,
            cached=internal,
            builder=self._internal_builder,
        )
        add_timing(
            "interior_surface_calculation",
            (time.perf_counter() - interior_started) * 1000.0,
        )
        base_scraper_mesh = self._scraper_builder.build(geometry)
        base_pose = pose_matrix(pose)

        envelope_started = time.perf_counter()
        if progress_callback is not None:
            progress_callback(
                "envelope_calculation",
                "Calcul de l'enveloppe intérieure utilisable",
                3.0,
            )
        envelope = self._pose_constraint_engine.build_envelope(
            jar,
            config,
            internal=internal_surface,
        )
        add_timing(
            "envelope_calculation",
            (time.perf_counter() - envelope_started) * 1000.0,
        )

        if progress_callback is not None:
            progress_callback(
                "pose_generation",
                "Génération et validation des poses réalistes",
                6.0,
            )
        pose_generation, pose_diagnostics = generate_validated_poses(
            jar,
            jar_mesh,
            geometry,
            pose,
            config,
            constraint_engine=self._pose_constraint_engine,
            envelope=envelope,
            scraper_builder=self._scraper_builder,
            profile_ms=timings,
        )
        poses = list(pose_generation.accepted_transforms)
        add_timing(
            "scraper_generation",
            (time.perf_counter() - generation_started) * 1000.0,
        )

        face_count = len(jar_mesh.faces)
        min_distances = np.full(face_count, np.inf, dtype=np.float64)
        contact_points: list = []
        collision = CollisionResult(
            has_collision=False,
            penetration_depth_mm=0.0,
            collision_points=(),
            colliding_face_ids=frozenset(),
        )

        pose_count = len(poses)
        for pose_index, trajectory_pose in enumerate(poses, start=1):
            posed_scraper = base_scraper_mesh.copy()
            scraper_transform = trajectory_pose @ base_pose
            posed_scraper.apply_transform(scraper_transform)
            if progress_callback is not None:
                progress_callback(
                    "distance_calculation",
                    f"Recherche de proximité — pose {pose_index}/{pose_count}",
                    10.0 + (pose_index - 1) / max(pose_count, 1) * 65.0,
                )
            pose_distances, pose_contacts = analyze_interior_contact(
                internal_surface,
                posed_scraper,
                contact_threshold_mm=config.contact_threshold_mm,
                timing_callback=add_timing,
            )
            contact_started = time.perf_counter()
            if progress_callback is not None:
                progress_callback(
                    "contact_calculation",
                    f"Consolidation des contacts — pose {pose_index}/{pose_count}",
                    10.0 + (pose_index - 0.5) / max(pose_count, 1) * 65.0,
                )
            min_distances = merge_face_distances(min_distances, pose_distances)
            contact_points.extend(pose_contacts)
            pose_collision = analyze_collision(
                jar_mesh,
                posed_scraper,
                mesh_tolerance_mm=config.mesh_tolerance_mm,
            )
            collision = merge_collisions(collision, pose_collision)
            add_timing(
                "contact_calculation",
                (time.perf_counter() - contact_started) * 1000.0,
            )
            if pose_result_callback is not None:
                capture_started = time.perf_counter()
                pose_result_callback(
                    pose_index - 1,
                    pose_count,
                    posed_scraper,
                    scraper_transform,
                    pose_distances,
                    pose_contacts,
                    pose_collision,
                )
                add_timing(
                    "pose_capture",
                    (time.perf_counter() - capture_started) * 1000.0,
                )

        metrics_started = time.perf_counter()
        if progress_callback is not None:
            progress_callback(
                "metrics_calculation",
                "Calcul de la couverture et des métriques 3D",
                78.0,
            )
        touched, untouched = self._partition_faces(min_distances, config.contact_threshold_mm)
        coverage_score = self._coverage_scorer.score(touched, untouched, jar_mesh)
        overlay = self._build_overlay(
            jar_mesh=jar_mesh,
            min_distances=min_distances,
            touched=touched,
            contact_points=tuple(contact_points),
            pose_count=len(poses),
            threshold_mm=config.contact_threshold_mm,
        )
        add_timing(
            "metrics_calculation",
            (time.perf_counter() - metrics_started) * 1000.0,
        )

        return ContactResult(
            model_id=geometry.id,
            jar_id=jar.id,
            coverage_score=coverage_score,
            touched_face_ids=touched,
            untouched_face_ids=untouched,
            contact_distance_map=min_distances,
            trajectory_pose_count=len(poses),
            overlay=overlay,
            collision=collision,
            diagnostics={
                "scraper_id": geometry.id,
                "contact_point_count": len(contact_points),
                "contact_threshold_mm": config.contact_threshold_mm,
                "clearance_mm": config.clearance_mm,
                "has_collision": collision.has_collision,
                "penetration_depth_mm": collision.penetration_depth_mm,
                "envelope_duration_ms": timings.get("envelope_calculation", 0.0),
                "pose_generation_duration_ms": pose_diagnostics.pose_generation_duration_ms,
                "candidate_pose_count": pose_diagnostics.candidate_pose_count,
                "accepted_pose_count": pose_diagnostics.accepted_pose_count,
                "rejected_pose_count": pose_diagnostics.rejected_pose_count,
                "simulated_pose_count": len(poses),
                "rejections_by_reason": dict(pose_diagnostics.rejections_by_reason),
                "pose_rejections": [
                    {
                        "pose_index": rejection.pose_index,
                        "reason": rejection.reason.value,
                        "detail": rejection.detail,
                    }
                    for rejection in pose_diagnostics.rejections
                ],
                "envelope": {
                    "jar_id": envelope.jar_id,
                    "y_min_mm": envelope.y_min_mm,
                    "y_max_mm": envelope.y_max_mm,
                    "neck_radius_mm": envelope.neck_radius_mm,
                    "clearance_mm": envelope.clearance_mm,
                    "slice_count": len(envelope.slices),
                    "slices": [
                        {"y_mm": slice_.y_mm, "max_radial_mm": slice_.max_radial_mm}
                        for slice_ in envelope.slices
                    ],
                },
                "compute_geometry": {
                    "geometry_source": "InternalJarSurface",
                    "jar_mesh_faces": internal_surface.face_count,
                    "interior_surface_samples": internal_surface.sample_count,
                    "interior_surface_slices": len(internal_surface.slices),
                    "sample_reduction_ratio": round(
                        internal_surface.sample_count
                        / max(internal_surface.face_count, 1),
                        4,
                    ),
                    "contact_source": "InternalJarSurface.sample_points_mm",
                    "visualization_source": "InternalJarSurface.mesh",
                    "canonical_source_face_count": internal_surface.source_face_count,
                },
            },
        )

    def _resolve_jar_mesh(
        self,
        jar: CanonicalModel3D,
        internal: InternalJarSurface | None = None,
    ) -> trimesh.Trimesh:
        surface = resolve_internal_jar_surface(
            jar,
            cached=internal,
            builder=self._internal_builder,
        )
        return internal_mesh_to_trimesh(surface)

    @staticmethod
    def _partition_faces(
        min_distances: np.ndarray,
        threshold_mm: float,
    ) -> tuple[frozenset[int], frozenset[int]]:
        touched: set[int] = set()
        untouched: set[int] = set()
        for face_id, distance in enumerate(min_distances):
            if np.isfinite(distance) and distance <= threshold_mm:
                touched.add(face_id)
            else:
                untouched.add(face_id)
        return frozenset(touched), frozenset(untouched)

    @staticmethod
    def _build_overlay(
        *,
        jar_mesh: trimesh.Trimesh,
        min_distances: np.ndarray,
        touched: frozenset[int],
        contact_points: tuple,
        pose_count: int,
        threshold_mm: float,
    ) -> ContactOverlayData:
        face_coverage = tuple(
            face_id in touched and float(min_distances[face_id]) <= threshold_mm
            for face_id in range(len(jar_mesh.faces))
        )
        return ContactOverlayData(
            contact_points=contact_points,
            face_coverage=face_coverage,
            min_distance_per_face_mm=tuple(float(v) for v in min_distances),
            scraper_pose_count=pose_count,
        )


def jar_model_to_canonical(jar_model: object) -> CanonicalModel3D:
    """Helper for callers that still hold a JarCanonicalModel instance."""
    if isinstance(jar_model, CanonicalModel3D):
        return jar_model
    from nutella_scraper.domain.models.canonical import JarCanonicalModel

    if isinstance(jar_model, JarCanonicalModel):
        return JarMeshBuilder().to_canonical(jar_model)
    raise TypeError(f"Unsupported jar model type: {type(jar_model)!r}")
