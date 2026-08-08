"""Envelope and pose constraint engine tests."""

from __future__ import annotations

import numpy as np

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.envelope import PoseRejectionReason
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.constrained_trajectory_sampler import (
    generate_validated_poses,
)
from nutella_scraper.engines.compute.envelope_builder import EnvelopeBuilder
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder
from nutella_scraper.engines.compute.pose_constraint_engine import PoseConstraintEngine


class TestEnvelopeBuilder:
    def test_envelope_is_inside_jar_wall(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        jar_mesh = JarMeshBuilder().from_canonical(cylindrical_jar_canonical)
        envelope = EnvelopeBuilder().from_canonical(
            cylindrical_jar_canonical,
            clearance_mm=coarse_simulation_config.clearance_mm,
        )
        jar_radial = np.sqrt(jar_mesh.vertices[:, 0] ** 2 + jar_mesh.vertices[:, 2] ** 2)

        assert envelope.y_min_mm <= float(jar_mesh.vertices[:, 1].min())
        assert envelope.y_max_mm >= float(jar_mesh.vertices[:, 1].max())
        assert len(envelope.slices) > 0
        for slice_ in envelope.slices:
            near_y = np.isclose(jar_mesh.vertices[:, 1], slice_.y_mm, atol=5.0)
            wall_radius = float(np.median(jar_radial[near_y]))
            if np.isfinite(wall_radius):
                assert slice_.max_radial_mm <= wall_radius


class TestPoseConstraintEngine:
    def test_accepts_realistic_wall_pose(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        engine = PoseConstraintEngine()
        result = engine.validate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
            coarse_simulation_config,
        )

        assert result.is_valid

    def test_rejects_penetrating_pose_before_simulation(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        engine = PoseConstraintEngine()
        result = engine.validate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            ScraperPose(position_mm=(55.0, 50.0, 0.0)),
            coarse_simulation_config,
        )

        assert not result.is_valid
        assert result.reason in {
            PoseRejectionReason.INITIAL_COLLISION,
            PoseRejectionReason.OUT_OF_ENVELOPE,
            PoseRejectionReason.CROSSES_WALL,
            PoseRejectionReason.ENTIRELY_OUTSIDE,
        }

    def test_rejects_invalid_orientation(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        engine = PoseConstraintEngine(max_tilt_deg=5.0)
        result = engine.validate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            ScraperPose(position_mm=(47.0, 50.0, 0.0), pitch_deg=30.0),
            coarse_simulation_config,
        )

        assert not result.is_valid
        assert result.reason == PoseRejectionReason.INVALID_ORIENTATION


class TestConstrainedTrajectorySampler:
    def test_rejects_unrealistic_poses(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        jar_mesh = JarMeshBuilder().from_canonical(cylindrical_jar_canonical)
        engine = PoseConstraintEngine()
        envelope = engine.build_envelope(cylindrical_jar_canonical, coarse_simulation_config)

        generation, diagnostics = generate_validated_poses(
            cylindrical_jar_canonical,
            jar_mesh,
            wall_scraper_geometry,
            ScraperPose(position_mm=(55.0, 50.0, 0.0)),
            coarse_simulation_config,
            constraint_engine=engine,
            envelope=envelope,
        )

        assert generation.candidate_count > 0
        assert generation.accepted_count == 0
        assert diagnostics.rejected_pose_count == generation.candidate_count

    def test_keeps_realistic_poses(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        jar_mesh = JarMeshBuilder().from_canonical(cylindrical_jar_canonical)
        engine = PoseConstraintEngine()
        envelope = engine.build_envelope(cylindrical_jar_canonical, coarse_simulation_config)

        generation, diagnostics = generate_validated_poses(
            cylindrical_jar_canonical,
            jar_mesh,
            wall_scraper_geometry,
            wall_scraper_pose,
            coarse_simulation_config,
            constraint_engine=engine,
            envelope=envelope,
        )

        assert generation.accepted_count > 0
        assert diagnostics.accepted_pose_count == generation.accepted_count
        assert diagnostics.rejected_pose_count == generation.rejected_count
