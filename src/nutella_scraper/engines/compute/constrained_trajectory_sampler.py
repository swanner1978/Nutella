"""Constrained trajectory pose generation with pre-simulation rejection."""

from __future__ import annotations

import time
from collections.abc import MutableMapping

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.envelope import (
    InteriorEnvelope,
    PoseConstraintDiagnostics,
    PoseGenerationResult,
    PoseRejection,
)
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.pose_constraint_engine import PoseConstraintEngine
from nutella_scraper.engines.compute.scraper_builder import ScraperBuilder
from nutella_scraper.engines.compute.scraper_transform import pose_matrix
from nutella_scraper.engines.compute.trajectory_sampler import sample_trajectory_poses


def generate_validated_poses(
    jar: CanonicalModel3D,
    jar_mesh: trimesh.Trimesh,
    geometry: ScraperGeometry,
    base_pose: ScraperPose,
    config: ContactSimulationConfig,
    *,
    constraint_engine: PoseConstraintEngine,
    envelope: InteriorEnvelope,
    scraper_builder: ScraperBuilder | None = None,
    profile_ms: MutableMapping[str, float] | None = None,
) -> tuple[PoseGenerationResult, PoseConstraintDiagnostics]:
    """
    Generate trajectory poses and reject physically implausible ones upfront.

    Invalid poses never reach ContactSimulationEngine.
    """
    started = time.perf_counter()
    builder = scraper_builder or ScraperBuilder()
    base_scraper_mesh = builder.build(geometry)
    base_pose_matrix = pose_matrix(base_pose)
    candidates = sample_trajectory_poses(jar_mesh, config.trajectory)

    accepted: list[np.ndarray] = []
    rejections: list[PoseRejection] = []
    rejections_by_reason: dict[str, int] = {}

    for pose_index, trajectory_transform in enumerate(candidates):
        validation = constraint_engine.validate_transform(
            base_scraper_mesh=base_scraper_mesh,
            base_pose_matrix=base_pose_matrix,
            trajectory_transform=trajectory_transform,
            jar_mesh=jar_mesh,
            envelope=envelope,
            geometry=geometry,
            config=config,
        )
        if validation.is_valid:
            accepted.append(trajectory_transform)
            continue
        reason = validation.reason
        assert reason is not None
        rejections.append(
            PoseRejection(
                pose_index=pose_index,
                reason=reason,
                detail=validation.detail,
            )
        )
        key = reason.value
        rejections_by_reason[key] = rejections_by_reason.get(key, 0) + 1

    duration_ms = (time.perf_counter() - started) * 1000.0
    if profile_ms is not None:
        profile_ms["pose_generation"] = profile_ms.get("pose_generation", 0.0) + duration_ms

    generation = PoseGenerationResult(
        accepted_transforms=tuple(accepted),
        rejections=tuple(rejections),
        candidate_count=len(candidates),
    )
    diagnostics = PoseConstraintDiagnostics(
        pose_generation_duration_ms=duration_ms,
        candidate_pose_count=len(candidates),
        accepted_pose_count=generation.accepted_count,
        rejected_pose_count=generation.rejected_count,
        rejections_by_reason=rejections_by_reason,
        rejections=tuple(rejections),
    )
    return generation, diagnostics
