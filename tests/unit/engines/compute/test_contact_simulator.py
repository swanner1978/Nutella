"""Contact simulation engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import ContactSimulationConfig, TrajectoryConfig
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.collision_analyzer import analyze_collision
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine
from nutella_scraper.engines.compute.coverage_scorer import CoverageScorer
from nutella_scraper.engines.compute.distance_field import DistanceFieldQuery
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder
from nutella_scraper.engines.compute.scraper_builder import ScraperBuilder
from nutella_scraper.engines.compute.scraper_geometry import ScraperGeometryBuilder
from nutella_scraper.engines.compute.trajectory_sampler import sample_trajectory_poses


class TestScraperGeometryBuilder:
    def test_builds_solid_volume_not_degenerate(self) -> None:
        geometry = ScraperGeometry(
            width_mm=20.0,
            length_mm=80.0,
            thickness_mm=3.0,
            curvature_radius_mm=40.0,
            bend_angle_deg=20.0,
        )
        mesh = ScraperGeometryBuilder().build(geometry)

        assert len(mesh.vertices) >= 8
        assert len(mesh.faces) >= 12
        assert mesh.volume > 0.0

    def test_pose_is_applied_separately_from_geometry(self) -> None:
        geometry = ScraperGeometry(width_mm=10.0, length_mm=10.0, thickness_mm=2.0)
        builder = ScraperGeometryBuilder()
        local = builder.build(geometry)
        posed = builder.build_posed(geometry, ScraperPose(position_mm=(5.0, 0.0, 0.0)))

        assert not np.allclose(local.centroid, posed.centroid)


class TestJarMeshBuilder:
    def test_profile_revolve_produces_inner_wall_mesh(
        self,
        cylindrical_jar_model: object,
    ) -> None:
        from nutella_scraper.domain.models.canonical import JarCanonicalModel

        assert isinstance(cylindrical_jar_model, JarCanonicalModel)
        mesh = JarMeshBuilder().from_profile(cylindrical_jar_model, theta_segments=16)

        assert len(mesh.faces) > 0
        assert mesh.bounds[1, 1] - mesh.bounds[0, 1] == pytest.approx(100.0, rel=0.05)


class TestContactSimulationEngine:
    def test_simulate_returns_coverage_overlay_and_collision(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        engine = ContactSimulationEngine()
        result = engine.simulate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
            coarse_simulation_config,
        )

        assert result.provenance == "computed_metric"
        assert 0.0 <= result.coverage_score <= 1.0
        assert result.coverage_score > 0.03
        assert result.trajectory_pose_count > 0
        assert result.overlay is not None
        assert result.collision is not None
        assert result.collision.has_collision is False
        assert len(result.overlay.face_coverage) == len(result.contact_distance_map)
        assert result.touched_face_ids | result.untouched_face_ids
        assert result.diagnostics["scraper_id"] == "test_scraper"
        assert result.diagnostics["candidate_pose_count"] > 0
        assert result.diagnostics["accepted_pose_count"] == result.trajectory_pose_count
        assert "envelope" in result.diagnostics

    def test_scraper_is_positioned_and_produces_contact_points(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        result = ContactSimulationEngine().simulate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
            coarse_simulation_config,
        )

        assert result.overlay is not None
        assert len(result.overlay.contact_points) > 0
        assert all(
            point.distance_mm <= coarse_simulation_config.contact_threshold_mm
            for point in result.overlay.contact_points
        )

    def test_detects_collision_for_penetrating_pose(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        penetrating_pose = ScraperPose(position_mm=(55.0, 50.0, 0.0))
        result = ContactSimulationEngine().simulate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            penetrating_pose,
            coarse_simulation_config,
        )

        assert result.diagnostics["rejected_pose_count"] > 0
        assert result.diagnostics["simulated_pose_count"] == 0
        assert result.trajectory_pose_count == 0
        assert result.collision.has_collision is False

    def test_pose_callback_receives_exact_mesh_used_for_contact(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
        coarse_simulation_config: ContactSimulationConfig,
    ) -> None:
        captured: list[tuple] = []

        ContactSimulationEngine().simulate(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
            coarse_simulation_config,
            pose_result_callback=lambda *values: captured.append(values),
        )

        assert captured
        index, total, posed_mesh, transform, distances, _contacts, _collision = captured[0]
        expected = ScraperBuilder().build(wall_scraper_geometry)
        expected.apply_transform(transform)
        assert index == 0
        assert total == len(captured)
        assert np.array_equal(posed_mesh.faces, expected.faces)
        assert np.allclose(posed_mesh.vertices, expected.vertices)
        assert len(distances) == len(cylindrical_jar_canonical.mesh.faces)


class TestCollisionAnalyzer:
    def test_distinguishes_contact_from_collision(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
    ) -> None:
        jar_mesh = JarMeshBuilder().from_canonical(cylindrical_jar_canonical)
        geometry = ScraperGeometry(width_mm=10.0, length_mm=90.0, thickness_mm=4.0)
        builder = ScraperGeometryBuilder()

        near_wall = builder.build_posed(geometry, ScraperPose(position_mm=(47.0, 50.0, 0.0)))
        penetrating = builder.build_posed(geometry, ScraperPose(position_mm=(55.0, 50.0, 0.0)))

        near_result = analyze_collision(jar_mesh, near_wall, mesh_tolerance_mm=0.1)
        penetrating_result = analyze_collision(jar_mesh, penetrating, mesh_tolerance_mm=0.1)

        assert near_result.has_collision is False
        assert penetrating_result.has_collision
        assert penetrating_result.penetration_depth_mm > near_result.penetration_depth_mm


class TestCoverageScorer:
    def test_score_is_area_weighted(self, cylindrical_jar_canonical: CanonicalModel3D) -> None:
        mesh = JarMeshBuilder().from_canonical(cylindrical_jar_canonical)
        areas = np.asarray(mesh.area_faces)
        largest_face = int(np.argmax(areas))

        scorer = CoverageScorer()
        untouched = frozenset(set(range(len(areas))) - {largest_face})
        score = scorer.score(frozenset({largest_face}), untouched, mesh)

        expected = float(areas[largest_face] / areas.sum())
        assert score == pytest.approx(expected, rel=1e-6)


class TestDistanceFieldQuery:
    def test_min_distance_is_finite_for_close_scraper(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
    ) -> None:
        distance = DistanceFieldQuery().min_distance(
            cylindrical_jar_canonical,
            wall_scraper_geometry,
            wall_scraper_pose,
        )

        assert np.isfinite(distance)
        assert distance < 5.0


class TestTrajectorySampler:
    def test_pose_count_matches_config(self, cylindrical_jar_canonical: CanonicalModel3D) -> None:
        jar_mesh = JarMeshBuilder().from_canonical(cylindrical_jar_canonical)
        config = TrajectoryConfig(angular_step_deg=90.0, vertical_step_mm=50.0)
        poses = sample_trajectory_poses(jar_mesh, config)

        assert len(poses) == 12
        assert all(pose.shape == (4, 4) for pose in poses)


class TestScraperGeometryValidation:
    def test_rejects_non_positive_dimensions(self) -> None:
        with pytest.raises(ValueError):
            ScraperGeometry(width_mm=0.0, length_mm=10.0, thickness_mm=2.0)
