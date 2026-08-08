"""Trajectory pose generation for scraping motion."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.domain.models.contact import TrajectoryConfig


def sample_trajectory_poses(
    jar_mesh: trimesh.Trimesh,
    config: TrajectoryConfig,
) -> list[np.ndarray]:
    """
    Generate 4x4 rigid transforms for a rotational + vertical scrape.

    Each pose rotates the scraper around the jar vertical axis (Y) and shifts
    it vertically inside the jar bounding box.
    """
    y_min = float(jar_mesh.bounds[0, 1])
    y_max = float(jar_mesh.bounds[1, 1])
    y_steps = np.arange(y_min, y_max + 1e-6, config.vertical_step_mm)
    if len(y_steps) == 0:
        y_steps = np.array([(y_min + y_max) / 2.0])

    angles = np.arange(0.0, 360.0, config.angular_step_deg)
    if len(angles) == 0:
        angles = np.array([0.0])

    center_y = (y_min + y_max) / 2.0
    poses: list[np.ndarray] = []
    for y_offset in y_steps - center_y:
        for angle_deg in angles:
            poses.append(_pose_y_rotation_with_translation(angle_deg, y_offset))
    return poses


def _pose_y_rotation_with_translation(angle_deg: float, y_offset: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    matrix[1, 3] = y_offset
    return matrix
