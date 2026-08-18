"""3D viewer cameras — visualization only, never used by compute/collision.

One scene (visual.stl + rigid scraper). A view is only a different lookAt camera.
Meshes are never rebuilt when the view changes.

CAD / STEP convention confirmed on the imported jar (identity frame):

  +Y  opening / top of the pot
  −Y  base / bottom
  X,Z horizontal plane  —  Z is not vertical

lookAt target is the AABB centre of visual.stl:

  center = (bbox.min + bbox.max) / 2
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Explicit jar axes. Do not treat Z as up.
JAR_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
JAR_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float64)
JAR_FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float64)
# Shared screen-up for Top/Bottom (look is ±Y, so up cannot be Y).
TOP_BOTTOM_SCREEN_UP = np.array([0.0, 0.0, -1.0], dtype=np.float64)

# Iso sits farther than the ortho cameras so the whole pot stays in frame.
ISO_DISTANCE_SCALE = 1.25
# Above the opening, offset in X and Z (not a second Top view).
ISO_OFFSET = np.array([1.05, 1.55, 0.80], dtype=np.float64)

JAR_FRAME_CONVENTION: dict[str, Any] = {
    "frame": "Y-up jar",
    "vertical_axis": "Y",
    "opening": "+Y",
    "base": "-Y",
    "horizontal_plane": "XZ",
    "viewer": "canvas lookAt (no Three.js)",
    "look_at": "AABB centre of visual.stl",
    "top_eye": "center + (0, +d, 0)",
    "bottom_eye": "center + (0, -d, 0)",
    "note": (
        "Opening is +Y. Z is not vertical. Top and Bottom are opposite cameras "
        "around the AABB centre. Profil/Gauche/Droite keep their existing axes. "
        "Iso is perspective from above with a lateral offset."
    ),
}


def _normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def aabb_center(
    mins_mm: NDArray[np.float64],
    maxs_mm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """lookAt / orbit centre = AABB midpoint of the visual mesh."""
    mins = np.asarray(mins_mm, dtype=np.float64)
    maxs = np.asarray(maxs_mm, dtype=np.float64)
    return 0.5 * (mins + maxs)


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
    """Opening axis for this CAD frame: always +Y (max-Y AABB face)."""
    del mins_mm, maxs_mm
    return JAR_UP.copy()


def cameras_from_center_and_distance(
    center_mm: NDArray[np.float64],
    distance_mm: float,
    *,
    opening: NDArray[np.float64] | None = None,
) -> dict[str, dict[str, Any]]:
    """Six cameras sharing the same AABB look-at centre and the same scene."""
    del opening  # CAD opening is +Y; kept in the signature for call-site stability.
    center = np.asarray(center_mm, dtype=np.float64)
    distance = float(distance_mm)

    # Dessus / Dessous: opposite cameras on ±Y, both looking at the centre.
    top_eye = center + np.array([0.0, +distance, 0.0], dtype=np.float64)
    bottom_eye = center + np.array([0.0, -distance, 0.0], dtype=np.float64)
    # Profil: existing convention (XY on screen, opening toward the top).
    side_eye = center + JAR_FORWARD * distance
    # Gauche / Droite: existing opposite cameras on ±X.
    left_eye = center + JAR_RIGHT * distance
    right_eye = center - JAR_RIGHT * distance
    # Iso: above the opening, offset laterally, farther so the whole pot is visible.
    iso_eye = center + _normalize(ISO_OFFSET) * (distance * ISO_DISTANCE_SCALE)

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
            up=JAR_UP,
            projection="orthographic",
        ),
        "left": look_at_camera(
            eye=left_eye,
            target=center,
            up=JAR_UP,
            projection="orthographic",
        ),
        "right": look_at_camera(
            eye=right_eye,
            target=center,
            up=JAR_UP,
            projection="orthographic",
        ),
        "iso": look_at_camera(
            eye=iso_eye,
            target=center,
            up=JAR_UP,
            projection="perspective",
        ),
    }


def cameras_from_bounds(
    mins_mm: NDArray[np.float64],
    maxs_mm: NDArray[np.float64],
) -> tuple[dict[str, dict[str, Any]], NDArray[np.float64], float]:
    mins = np.asarray(mins_mm, dtype=np.float64)
    maxs = np.asarray(maxs_mm, dtype=np.float64)
    center = aabb_center(mins, maxs)
    span = float(np.max(maxs - mins))
    distance = max(span * 2.2, 1.0)
    return cameras_from_center_and_distance(center, distance), center, distance
