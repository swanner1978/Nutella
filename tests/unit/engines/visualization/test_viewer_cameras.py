"""Top / Bottom / Iso cameras must be real opposite lookAt views."""

from __future__ import annotations

import numpy as np

from nutella_scraper.engines.visualization.viewer_cameras import (
    ISO_DISTANCE_SCALE,
    JAR_FRAME_CONVENTION,
    aabb_center,
    cameras_from_bounds,
    cameras_from_center_and_distance,
    opening_direction_from_bounds,
)


def test_jar_frame_is_y_up_opening() -> None:
    assert JAR_FRAME_CONVENTION["vertical_axis"] == "Y"
    assert JAR_FRAME_CONVENTION["opening"] == "+Y"
    assert JAR_FRAME_CONVENTION["base"] == "-Y"
    assert JAR_FRAME_CONVENTION["horizontal_plane"] == "XZ"
    assert JAR_FRAME_CONVENTION["top_eye"] == "center + (0, +d, 0)"
    assert JAR_FRAME_CONVENTION["bottom_eye"] == "center + (0, -d, 0)"


def test_lookat_center_is_aabb_midpoint() -> None:
    mins = np.array([-55.0, -18.0, -41.5])
    maxs = np.array([55.0, 114.0, 41.5])
    cameras, center, _distance = cameras_from_bounds(mins, maxs)
    np.testing.assert_allclose(center, 0.5 * (mins + maxs), atol=1e-12)
    np.testing.assert_allclose(center, aabb_center(mins, maxs), atol=1e-12)
    for name, camera in cameras.items():
        np.testing.assert_allclose(
            np.asarray(camera["target"], dtype=np.float64),
            center,
            atol=1e-9,
            err_msg=name,
        )


def test_top_eye_is_plus_y_bottom_eye_is_minus_y() -> None:
    cameras, center, distance = cameras_from_bounds(
        np.array([-40.0, -18.0, -40.0]),
        np.array([40.0, 114.0, 40.0]),
    )
    top_eye = np.asarray(cameras["top"]["eye"], dtype=np.float64)
    bottom_eye = np.asarray(cameras["bottom"]["eye"], dtype=np.float64)
    np.testing.assert_allclose(
        top_eye,
        center + np.array([0.0, +distance, 0.0]),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        bottom_eye,
        center + np.array([0.0, -distance, 0.0]),
        atol=1e-9,
    )
    np.testing.assert_allclose(bottom_eye - center, -(top_eye - center), atol=1e-9)


def test_top_look_is_opposite_bottom() -> None:
    cameras, center, _distance = cameras_from_bounds(
        np.array([-40.0, 0.0, -40.0]),
        np.array([40.0, 80.0, 40.0]),
    )
    top = np.asarray(cameras["top"]["look_direction"], dtype=np.float64)
    bottom = np.asarray(cameras["bottom"]["look_direction"], dtype=np.float64)

    np.testing.assert_allclose(top, -bottom, atol=1e-9)
    assert not np.allclose(top, bottom)
    assert cameras["top"]["eye"] != cameras["bottom"]["eye"]
    assert cameras["top"]["eye"][1] > center[1]
    assert cameras["bottom"]["eye"][1] < center[1]
    assert top[1] < 0.0
    assert bottom[1] > 0.0
    np.testing.assert_allclose(top, np.array([0.0, -1.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(bottom, np.array([0.0, 1.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(cameras["top"]["target"], dtype=np.float64),
        center,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(cameras["bottom"]["target"], dtype=np.float64),
        center,
        atol=1e-9,
    )


def test_opening_is_explicit_plus_y_not_z() -> None:
    opening = opening_direction_from_bounds(
        np.array([-40.0, -18.0, -40.0]),
        np.array([40.0, 114.0, 40.0]),
    )
    np.testing.assert_allclose(opening, np.array([0.0, 1.0, 0.0]), atol=1e-9)
    assert not np.allclose(opening, np.array([0.0, 0.0, 1.0]))


def test_all_cameras_share_the_same_target() -> None:
    cameras, center, _distance = cameras_from_bounds(
        np.array([-40.0, 0.0, -40.0]),
        np.array([40.0, 80.0, 40.0]),
    )
    assert set(cameras) == {"top", "bottom", "side", "left", "right", "iso"}
    for name, camera in cameras.items():
        np.testing.assert_allclose(
            np.asarray(camera["target"], dtype=np.float64),
            center,
            atol=1e-9,
            err_msg=name,
        )


def test_left_look_is_opposite_right() -> None:
    cameras, center, distance = cameras_from_bounds(
        np.array([-40.0, 0.0, -40.0]),
        np.array([40.0, 80.0, 40.0]),
    )
    left = np.asarray(cameras["left"]["look_direction"], dtype=np.float64)
    right = np.asarray(cameras["right"]["look_direction"], dtype=np.float64)
    left_eye = np.asarray(cameras["left"]["eye"], dtype=np.float64)
    right_eye = np.asarray(cameras["right"]["eye"], dtype=np.float64)
    np.testing.assert_allclose(left, -right, atol=1e-9)
    np.testing.assert_allclose(right_eye - center, -(left_eye - center), atol=1e-9)
    np.testing.assert_allclose(
        left_eye,
        center + np.array([+distance, 0.0, 0.0]),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        right_eye,
        center + np.array([-distance, 0.0, 0.0]),
        atol=1e-9,
    )
    assert cameras["left"]["projection"] == "orthographic"
    assert cameras["right"]["projection"] == "orthographic"
    assert cameras["side"]["projection"] == "orthographic"


def test_iso_is_above_laterally_offset_and_farther() -> None:
    center = np.array([0.0, 48.0, 0.0])
    distance = 200.0
    cameras = cameras_from_center_and_distance(center, distance)
    iso_eye = np.asarray(cameras["iso"]["eye"], dtype=np.float64)
    iso_look = np.asarray(cameras["iso"]["look_direction"], dtype=np.float64)
    offset = iso_eye - center
    assert iso_eye[1] > center[1]
    assert iso_look[1] < 0.0
    assert abs(float(offset[0])) > 1.0
    assert abs(float(offset[2])) > 1.0
    np.testing.assert_allclose(
        float(np.linalg.norm(offset)),
        distance * ISO_DISTANCE_SCALE,
        atol=1e-9,
    )
    assert cameras["iso"]["projection"] == "perspective"
    assert cameras["top"]["projection"] == "orthographic"
    assert cameras["bottom"]["projection"] == "orthographic"
    np.testing.assert_allclose(
        np.asarray(cameras["iso"]["target"], dtype=np.float64),
        center,
        atol=1e-9,
    )
