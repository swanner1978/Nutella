"""Persistence for exact per-pose 3D simulation outputs used by the viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.domain.models.contact import (
    CollisionPoint3D,
    CollisionResult,
    ContactPoint3D,
)
from nutella_scraper.engines.compute.scraper_transform import pose_from_matrix, pose_to_dict


@dataclass(frozen=True)
class PoseSnapshot:
    """Exact geometry and contact output captured for one trajectory pose."""

    index: int
    total: int
    scraper_vertices: NDArray[np.float64]
    scraper_faces: NDArray[np.int64]
    scraper_transform: NDArray[np.float64]
    face_distances: NDArray[np.float64]
    face_coverage: NDArray[np.bool_]
    contact_points: tuple[ContactPoint3D, ...]
    collision: CollisionResult


class PoseSnapshotStore:
    """Write and load immutable per-pose snapshots without recomputation."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def persist(
        self,
        *,
        index: int,
        total: int,
        scraper_vertices: NDArray[np.float64],
        scraper_faces: NDArray[np.int64],
        scraper_transform: NDArray[np.float64],
        face_distances: NDArray[np.float64],
        face_coverage: NDArray[np.bool_],
        contact_points: tuple[ContactPoint3D, ...],
        collision: CollisionResult,
    ) -> None:
        contact_positions = np.asarray(
            [point.position_mm for point in contact_points],
            dtype=np.float64,
        ).reshape((-1, 3))
        contact_face_ids = np.asarray(
            [point.jar_face_id for point in contact_points],
            dtype=np.int64,
        )
        contact_distances = np.asarray(
            [point.distance_mm for point in contact_points],
            dtype=np.float64,
        )
        collision_positions = np.asarray(
            [point.position_mm for point in collision.collision_points],
            dtype=np.float64,
        ).reshape((-1, 3))
        collision_face_ids = np.asarray(
            [point.jar_face_id for point in collision.collision_points],
            dtype=np.int64,
        )
        collision_depths = np.asarray(
            [point.penetration_depth_mm for point in collision.collision_points],
            dtype=np.float64,
        )
        np.savez_compressed(
            self._pose_path(index),
            index=np.asarray(index, dtype=np.int64),
            total=np.asarray(total, dtype=np.int64),
            scraper_vertices=np.asarray(scraper_vertices, dtype=np.float64),
            scraper_faces=np.asarray(scraper_faces, dtype=np.int64),
            scraper_transform=np.asarray(scraper_transform, dtype=np.float64),
            face_distances=np.asarray(face_distances, dtype=np.float64),
            face_coverage=np.asarray(face_coverage, dtype=np.bool_),
            contact_positions=contact_positions,
            contact_face_ids=contact_face_ids,
            contact_distances=contact_distances,
            collision_positions=collision_positions,
            collision_face_ids=collision_face_ids,
            collision_depths=collision_depths,
            collision_has=np.asarray(collision.has_collision, dtype=np.bool_),
            collision_max_depth=np.asarray(
                collision.penetration_depth_mm,
                dtype=np.float64,
            ),
            colliding_face_ids=np.asarray(
                sorted(collision.colliding_face_ids),
                dtype=np.int64,
            ),
        )

    def finalize(
        self,
        *,
        model_id: str,
        pose_count: int,
        view_dir_name: str,
        scraper_pipeline: dict[str, Any] | None = None,
        reference_pose: bool = False,
    ) -> None:
        manifest = {
            "model_id": model_id,
            "view_dir_name": view_dir_name,
            "pose_count": pose_count,
            "reference_pose": reference_pose,
            "scraper_pipeline": scraper_pipeline or {},
            "poses": self._pose_manifest_entries(pose_count),
        }
        temporary = self.directory / "manifest.tmp"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.directory / "manifest.json")

    def manifest(self) -> dict:
        return json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))

    def load(self, index: int) -> PoseSnapshot:
        path = self._pose_path(index)
        if not path.exists():
            raise FileNotFoundError(f"Pose de simulation introuvable : {index}")
        with np.load(path, allow_pickle=False) as data:
            contact_points = tuple(
                ContactPoint3D(
                    position_mm=tuple(float(value) for value in position),
                    jar_face_id=int(face_id),
                    distance_mm=float(distance),
                )
                for position, face_id, distance in zip(
                    data["contact_positions"],
                    data["contact_face_ids"],
                    data["contact_distances"],
                    strict=True,
                )
            )
            collision_points = tuple(
                CollisionPoint3D(
                    position_mm=tuple(float(value) for value in position),
                    jar_face_id=int(face_id),
                    penetration_depth_mm=float(depth),
                )
                for position, face_id, depth in zip(
                    data["collision_positions"],
                    data["collision_face_ids"],
                    data["collision_depths"],
                    strict=True,
                )
            )
            collision = CollisionResult(
                has_collision=bool(data["collision_has"]),
                penetration_depth_mm=float(data["collision_max_depth"]),
                collision_points=collision_points,
                colliding_face_ids=frozenset(
                    int(value) for value in data["colliding_face_ids"]
                ),
            )
            return PoseSnapshot(
                index=int(data["index"]),
                total=int(data["total"]),
                scraper_vertices=np.asarray(data["scraper_vertices"], dtype=np.float64),
                scraper_faces=np.asarray(data["scraper_faces"], dtype=np.int64),
                scraper_transform=np.asarray(data["scraper_transform"], dtype=np.float64),
                face_distances=np.asarray(data["face_distances"], dtype=np.float64),
                face_coverage=np.asarray(data["face_coverage"], dtype=np.bool_),
                contact_points=contact_points,
                collision=collision,
            )

    def _pose_manifest_entries(self, pose_count: int) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for index in range(pose_count):
            path = self._pose_path(index)
            entry: dict[str, Any] = {
                "index": index,
                "snapshot": path.name,
            }
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    transform = np.asarray(data["scraper_transform"], dtype=np.float64)
                pose = pose_from_matrix(transform)
                entry["pose"] = pose_to_dict(pose)
            entries.append(entry)
        return entries

    def _pose_path(self, index: int) -> Path:
        if index < 0:
            raise ValueError("L'index de pose doit être positif")
        return self.directory / f"pose-{index:04d}.npz"
