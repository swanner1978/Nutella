"""3D viewer cameras — visualization only, never used by compute/collision.

One scene, six lookAt cameras. The pot (visual.stl) and scraper are never
rebuilt when the view changes.

Jar / CanonicalModel3D / CAD orientation (from the real model AABB, not Z-up):

  +Y  vertical, toward the cavity OPENING  (bbox face at max Y)
  −Y  toward the BASE (bbox face at min Y)
  X,Z horizontal plane

There is no Three.js scene graph. The demo viewer projects with these lookAt
presets. Top and Bottom sit on opposite sides of the opening axis and look
toward the center — opposite look directions, not a 2D image flip.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Opening axis of the jar: CAD +Y = bbox face at maximum Y. Not Z.
JAR_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
JAR_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float64)
JAR_FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float64)
# Screen-up for Top/Bottom so lookAt is not degenerate (up ⟂ look).
TOP_BOTTOM_SCREEN_UP = np.array([0.0, 0.0, -1.0], dtype=np.float64)

JAR_FRAME_CONVENTION: dict[str, Any] = {
    "frame": "Y-up jar",
    "vertical_axis": "Y",
    "opening": "+Y",
    "base": "-Y",
    "horizontal_plane": "XZ",
    "viewer": "canvas lookAt (no Three.js)",
    "note": (
        "Opening is the AABB face at max Y of visual.stl (CAD orientation). "
        "Vertical is not Z. Top/Bottom are opposite cameras around the center. "
        "Profil/Gauche/Droite/Iso are other cameras on the same scene."
    ),
}


def _normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def look_at_camera(
    *,
    eye: NDArray[np.float64],
    target: NDArray[np.float64],
    up: NDArray[np.float64],
    projection: str,
) -> dict[str, Any]:
    """Right-handed lookAt camera. ``look_direction`` points from eye to target."""
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = _normalize(np.asarray(up, dtype=np.float64))
    look = _normalize(target_v - eye_v)
    back = -look
    right = np.cross(up_v, back)
    if float(np.linalg.norm(right)) <= 1e-9:
        right = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), back)
    right = _normalize(right)
    screen_up = _normalize(np.cross(back, right))
    return {
        "eye": eye_v.tolist(),
        "target": target_v.tolist(),
        "up": up_v.tolist(),
        "look_direction": look.tolist(),
        "projection": projection,
        "basis": {
            "x": right.tolist(),
            "y": screen_up.tolist(),
            "z": back.tolist(),
        },
    }


def opening_direction_from_bounds(
    mins_mm: NDArray[np.float64],
    maxs_mm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unit vector from AABB center toward the real opening (max-Y face).

    The CAD frame stores the cavity opening at maximum Y. Z is never treated
    as vertical just because it is the third coordinate.
    """
    mins = np.asarray(mins_mm, dtype=np.float64)
    maxs = np.asarray(maxs_mm, dtype=np.float64)
    center = 0.5 * (mins + maxs)
    toward_opening = np.array(
        [0.0, float(maxs[1] - center[1]), 0.0],
        dtype=np.float64,
    )
    if float(np.linalg.norm(toward_opening)) <= 1e-12:
        return JAR_UP.copy()
    return _normalize(toward_opening)


def cameras_from_center_and_distance(
    center_mm: NDArray[np.float64],
    distance_mm: float,
    *,
    opening: NDArray[np.float64] | None = None,
) -> dict[str, dict[str, Any]]:
    """Six cameras sharing the same look-at center and the same scene."""
    center = np.asarray(center_mm, dtype=np.float64)
    distance = float(distance_mm)
    up_axis = JAR_UP if opening is None else _normalize(np.asarray(opening, dtype=np.float64))
    # Opening side of the AABB. Opposite side is the base.
    top_eye = center + up_axis * distance
    bottom_eye = center - up_axis * distance
    # Profil: look along −Z (XY on screen, opening toward the top).
    side_eye = center + JAR_FORWARD * distance
    # Gauche / Droite: opposite cameras on ±X.
    left_eye = center + JAR_RIGHT * distance
    right_eye = center - JAR_RIGHT * distance
    # Iso: above the opening, offset laterally so depth and interior show.
    iso_dir = _normalize(up_axis * 1.25 + JAR_RIGHT * 0.95 + JAR_FORWARD * 0.70)
    iso_eye = center + iso_dir * distance
    return {
        "top": look_at_camera(
            eye=top_eye,
            target=center,
            up=TOP_BOTTOM_SCREEN_UP,
            projection="orthographic",
        ),
        "bottom": look_at_camera(
            eye=bottom_eye,
            target=center,
            up=TOP_BOTTOM_SCREEN_UP,
            projection="orthographic",
        ),
        "side": look_at_camera(
            eye=side_eye,
            target=center,
            up=up_axis,
            projection="orthographic",
        ),
        "left": look_at_camera(
            eye=left_eye,
            target=center,
            up=up_axis,
            projection="orthographic",
        ),
        "right": look_at_camera(
            eye=right_eye,
            target=center,
            up=up_axis,
            projection="orthographic",
        ),
        "iso": look_at_camera(
            eye=iso_eye,
            target=center,
            up=up_axis,
            projection="perspective",
        ),
    }


def cameras_from_bounds(
    mins_mm: NDArray[np.float64],
    maxs_mm: NDArray[np.float64],
) -> tuple[dict[str, dict[str, Any]], NDArray[np.float64], float]:
    mins = np.asarray(mins_mm, dtype=np.float64)
    maxs = np.asarray(maxs_mm, dtype=np.float64)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    distance = max(span * 2.2, 1.0)
    opening = opening_direction_from_bounds(mins, maxs)
    return cameras_from_center_and_distance(center, distance, opening=opening), center, distance
