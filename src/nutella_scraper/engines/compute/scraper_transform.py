"""Shared scraper pose and mesh deformation helpers."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.domain.models.scraper import ScraperPose


def pose_from_matrix(matrix: np.ndarray) -> ScraperPose:
    """Recover a scraper pose from a 4×4 homogeneous transform (Y-X-Z rotation order)."""
    transform = np.asarray(matrix, dtype=np.float64)
    position_mm = (
        float(transform[0, 3]),
        float(transform[1, 3]),
        float(transform[2, 3]),
    )
    rotation = transform[:3, :3]
    pitch = float(np.arcsin(np.clip(-rotation[1, 2], -1.0, 1.0)))
    cos_pitch = float(np.cos(pitch))
    if abs(cos_pitch) > 1e-6:
        yaw = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        roll = float(np.arctan2(rotation[0, 1], rotation[0, 0]))
    else:
        yaw = float(np.arctan2(-rotation[2, 0], rotation[1, 1]))
        roll = 0.0
    return ScraperPose(
        position_mm=position_mm,
        yaw_deg=float(np.rad2deg(yaw)),
        pitch_deg=float(np.rad2deg(pitch)),
        roll_deg=float(np.rad2deg(roll)),
    )


def pose_to_dict(pose: ScraperPose) -> dict[str, float | tuple[float, float, float]]:
    """Serialize pose fields for the viewer API."""
    return {
        "position_mm": pose.position_mm,
        "height_mm": pose.position_mm[1],
        "yaw_deg": pose.yaw_deg,
        "pitch_deg": pose.pitch_deg,
        "roll_deg": pose.roll_deg,
    }


def pose_matrix(pose: ScraperPose) -> np.ndarray:
    yaw = np.deg2rad(pose.yaw_deg)
    pitch = np.deg2rad(pose.pitch_deg)
    roll = np.deg2rad(pose.roll_deg)

    rot_y = _rotation_y(yaw)
    rot_x = _rotation_x(pitch)
    rot_z = _rotation_z(roll)
    rotation = rot_y @ rot_x @ rot_z

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(pose.position_mm, dtype=np.float64)
    return matrix


def bend_solid_around_y(
    mesh: trimesh.Trimesh,
    *,
    radius_mm: float,
    bend_angle_deg: float,
) -> trimesh.Trimesh:
    """Bend a straight blade solid along a circular arc in the X/Y plane."""
    bent = mesh.copy()
    vertices = bent.vertices.copy()
    y_values = vertices[:, 1]
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    span = max(y_max - y_min, 1e-6)
    max_angle = np.deg2rad(bend_angle_deg)

    for index, (x, y, z) in enumerate(vertices):
        t = (y - y_min) / span
        angle = max_angle * t
        arc_x = radius_mm * (1.0 - np.cos(angle))
        arc_y = y_min + radius_mm * np.sin(angle)
        local_x = x - float(np.mean(mesh.vertices[:, 0]))
        vertices[index] = (local_x + arc_x, arc_y, z)

    bent.vertices = vertices
    return bent


def apply_tip_radius(mesh: trimesh.Trimesh, *, tip_radius_mm: float) -> trimesh.Trimesh:
    """Round the scraping tip (max-Y end) with a spherical cap."""
    rounded = mesh.copy()
    vertices = rounded.vertices.copy()
    y_max = float(vertices[:, 1].max())
    y_min = float(vertices[:, 1].min())
    tip_plane_y = y_max - tip_radius_mm
    if tip_plane_y <= y_min:
        return rounded

    x_center = float(np.mean(vertices[:, 0]))
    z_center = float(np.mean(vertices[:, 2]))
    cap_center = np.array([x_center, y_max - tip_radius_mm, z_center], dtype=np.float64)

    for index, vertex in enumerate(vertices):
        if vertex[1] < tip_plane_y:
            continue
        offset = vertex - cap_center
        distance = float(np.linalg.norm(offset))
        if distance <= tip_radius_mm or distance <= 1e-9:
            continue
        vertices[index] = cap_center + offset * (tip_radius_mm / distance)

    rounded.vertices = vertices
    return rounded


def _rotation_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
