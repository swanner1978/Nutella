"""Geometric pose constraints derived from CanonicalModel3D."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.contact import ContactSimulationConfig
from nutella_scraper.domain.models.envelope import (
    InteriorEnvelope,
    PoseRejectionReason,
    PoseValidationResult,
)
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.domain.protocols.manual_pose import IManualPoseValidator
from nutella_scraper.engines.compute.collision_analyzer import analyze_collision
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.engines.compute.envelope_builder import EnvelopeBuilder
from nutella_scraper.engines.compute.internal_jar_surface_builder import (
    internal_mesh_to_trimesh,
    resolve_internal_jar_surface,
)
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder
from nutella_scraper.engines.compute.scraper_builder import ScraperBuilder
from nutella_scraper.engines.compute.scraper_transform import pose_matrix


class PoseConstraintEngine(IManualPoseValidator):
    """
    Computes jar constraints and validates scraper poses before simulation.

    Uses CanonicalModel3D only — never view projections or overlays.
    """

    def __init__(
        self,
        *,
        jar_mesh_builder: JarMeshBuilder | None = None,
        envelope_builder: EnvelopeBuilder | None = None,
        scraper_builder: ScraperBuilder | None = None,
        max_tilt_deg: float = 25.0,
    ) -> None:
        self._jar_mesh_builder = jar_mesh_builder or JarMeshBuilder()
        self._envelope_builder = envelope_builder or EnvelopeBuilder()
        self._scraper_builder = scraper_builder or ScraperBuilder()
        self._max_tilt_deg = max_tilt_deg

    def build_envelope(
        self,
        jar: CanonicalModel3D,
        config: ContactSimulationConfig,
        *,
        internal: InternalJarSurface | None = None,
    ) -> InteriorEnvelope:
        surface = resolve_internal_jar_surface(jar, cached=internal)
        return self._envelope_builder.from_internal(
            surface,
            clearance_mm=config.clearance_mm,
        )

    def validate(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
        config: ContactSimulationConfig,
        *,
        internal: InternalJarSurface | None = None,
    ) -> PoseValidationResult:
        surface = resolve_internal_jar_surface(jar, cached=internal)
        jar_mesh = internal_mesh_to_trimesh(surface)
        envelope = self.build_envelope(jar, config, internal=surface)
        orientation = self._validate_orientation(pose_matrix(pose))
        if not orientation.is_valid:
            return orientation
        posed_mesh = self._scraper_builder.build_posed(geometry, pose)
        return self.validate_posed_mesh(
            posed_mesh,
            jar_mesh=jar_mesh,
            envelope=envelope,
            geometry=geometry,
            config=config,
        )

    def validate_transform(
        self,
        *,
        base_scraper_mesh: trimesh.Trimesh,
        base_pose_matrix: np.ndarray,
        trajectory_transform: np.ndarray,
        jar_mesh: trimesh.Trimesh,
        envelope: InteriorEnvelope,
        geometry: ScraperGeometry,
        config: ContactSimulationConfig,
    ) -> PoseValidationResult:
        posed_mesh = base_scraper_mesh.copy()
        composed = trajectory_transform @ base_pose_matrix
        posed_mesh.apply_transform(composed)
        return self.validate_posed_mesh(
            posed_mesh,
            jar_mesh=jar_mesh,
            envelope=envelope,
            geometry=geometry,
            config=config,
        )

    def validate_posed_mesh(
        self,
        posed_mesh: trimesh.Trimesh,
        *,
        jar_mesh: trimesh.Trimesh,
        envelope: InteriorEnvelope,
        geometry: ScraperGeometry,
        config: ContactSimulationConfig,
    ) -> PoseValidationResult:
        tolerance = config.mesh_tolerance_mm
        vertices = np.asarray(posed_mesh.vertices, dtype=np.float64)
        y_values = vertices[:, 1]
        radial = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 2] ** 2)

        if all(
            radial_value > envelope.max_radial_at(float(y_value)) + tolerance
            for y_value, radial_value in zip(y_values, radial, strict=True)
        ):
            return PoseValidationResult(
                is_valid=False,
                reason=PoseRejectionReason.ENTIRELY_OUTSIDE,
                detail="Scraper volume lies completely outside the interior envelope",
            )

        if np.any(y_values < envelope.y_min_mm - tolerance) or np.any(
            y_values > envelope.y_max_mm + tolerance
        ):
            return PoseValidationResult(
                is_valid=False,
                reason=PoseRejectionReason.OUTSIDE_JAR,
                detail="Scraper extends beyond jar vertical bounds",
            )

        local_mesh = self._scraper_builder.build(geometry)
        local_extent = float(
            np.sqrt(local_mesh.vertices[:, 0] ** 2 + local_mesh.vertices[:, 2] ** 2).max()
        )
        if local_extent > envelope.neck_radius_mm + tolerance:
            return PoseValidationResult(
                is_valid=False,
                reason=PoseRejectionReason.CANNOT_INSERT,
                detail="Scraper cross-section exceeds neck opening",
            )

        for y_value, radial_value in zip(y_values, radial, strict=True):
            allowed = envelope.max_radial_at(float(y_value))
            if radial_value > allowed + tolerance:
                return PoseValidationResult(
                    is_valid=False,
                    reason=PoseRejectionReason.OUT_OF_ENVELOPE,
                    detail=(
                        f"Radial extent {radial_value:.2f} mm exceeds envelope "
                        f"{allowed:.2f} mm at Y={y_value:.2f} mm"
                    ),
                )

        centroid = posed_mesh.centroid
        centroid_radial = float(np.sqrt(centroid[0] ** 2 + centroid[2] ** 2))
        centroid_allowed = envelope.max_radial_at(float(centroid[1]))
        if centroid_radial > centroid_allowed + tolerance:
            return PoseValidationResult(
                is_valid=False,
                reason=PoseRejectionReason.CROSSES_WALL,
                detail="Scraper centroid starts beyond the interior wall",
            )

        collision = analyze_collision(
            jar_mesh,
            posed_mesh,
            mesh_tolerance_mm=config.mesh_tolerance_mm,
        )
        if collision.has_collision:
            return PoseValidationResult(
                is_valid=False,
                reason=PoseRejectionReason.INITIAL_COLLISION,
                detail=(
                    f"Initial penetration depth {collision.penetration_depth_mm:.3f} mm"
                ),
            )

        return PoseValidationResult(is_valid=True)

    def _validate_orientation(self, transform: np.ndarray) -> PoseValidationResult:
        rotation = transform[:3, :3]
        pitch_deg, roll_deg = _extract_pitch_roll_deg(rotation)
        if abs(pitch_deg) > self._max_tilt_deg or abs(roll_deg) > self._max_tilt_deg:
            return PoseValidationResult(
                is_valid=False,
                reason=PoseRejectionReason.INVALID_ORIENTATION,
                detail=(
                    f"Orientation pitch={pitch_deg:.1f}° roll={roll_deg:.1f}° "
                    f"exceeds ±{self._max_tilt_deg:.1f}°"
                ),
            )
        return PoseValidationResult(is_valid=True)


def compose_scraper_pose(base_pose: ScraperPose, trajectory_transform: np.ndarray) -> np.ndarray:
    return trajectory_transform @ pose_matrix(base_pose)


def _extract_pitch_roll_deg(rotation: np.ndarray) -> tuple[float, float]:
    sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    if sy >= 1e-6:
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    else:
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
    return np.rad2deg(pitch), np.rad2deg(roll)
