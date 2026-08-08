"""Exact per-pose simulation snapshot persistence tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nutella_scraper.domain.models.contact import (
    CollisionResult,
    ContactPoint3D,
)
from nutella_scraper.engines.visualization.pose_snapshot_store import PoseSnapshotStore


def test_pose_snapshot_round_trip_preserves_exact_geometry(tmp_path: Path) -> None:
    store = PoseSnapshotStore(tmp_path / "poses")
    vertices = np.asarray(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    transform = np.asarray(
        [
            [0.0, 0.0, 1.0, 12.0],
            [0.0, 1.0, 0.0, 34.0],
            [-1.0, 0.0, 0.0, 56.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distances = np.asarray([0.2, 1.3, np.inf], dtype=np.float64)
    coverage = np.asarray([True, False, False], dtype=np.bool_)
    points = (
        ContactPoint3D(
            position_mm=(1.0, 2.0, 3.0),
            jar_face_id=0,
            distance_mm=0.2,
        ),
    )
    collision = CollisionResult(
        has_collision=False,
        penetration_depth_mm=0.0,
        collision_points=(),
        colliding_face_ids=frozenset(),
    )

    store.persist(
        index=0,
        total=2,
        scraper_vertices=vertices,
        scraper_faces=faces,
        scraper_transform=transform,
        face_distances=distances,
        face_coverage=coverage,
        contact_points=points,
        collision=collision,
    )
    store.finalize(
        model_id="jar",
        pose_count=1,
        view_dir_name="jar",
    )
    manifest = store.manifest()
    assert manifest["pose_count"] == 1
    assert manifest["view_dir_name"] == "jar"
    restored = store.load(0)

    assert np.array_equal(restored.scraper_vertices, vertices)
    assert np.array_equal(restored.scraper_faces, faces)
    assert np.array_equal(restored.scraper_transform, transform)
    assert np.array_equal(restored.face_distances, distances)
    assert np.array_equal(restored.face_coverage, coverage)
    assert restored.contact_points == points
