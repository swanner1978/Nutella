"""Contact masks for sampled scraper poses. Target = interior_matrix_a0_0_90.

The 608 cloud points are coverage targets only. Each pose is (y_mm, azimuth)
in scraper space. Collision uses the existing envelope engine unchanged.

Cache key: shape_fingerprint + pose (y, azimuth).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
    LEGACY_A0_QUADRANT_REGION,
    CoverageReferenceMatrix,
)
from nutella_scraper.engines.compute.envelope_surface_proximity import (
    bind_envelope_proximity,
    closest_on_envelope_surface,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.pose_space import TARGET_MATRIX, PoseSampleSpec
from nutella_scraper.engines.compute.scraper_envelope_collision import (
    evaluate_envelope_collision,
    rigid_pose_neighborhood,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    RigidScraperArtifact,
    envelope_contact_frame,
    rigid_transform_between_frames,
    transform_points,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import (
    reference_scraper_parameters,
)

MOTION_DIRECTION_AFFECTS_CONTACT = False


def pose_cache_key(shape_fingerprint: str, y_mm: float, azimuth_deg: float) -> str:
    return f"{shape_fingerprint}|y={float(y_mm):.6f}|az={float(azimuth_deg):.6f}"


@dataclass(frozen=True)
class PoseContactEntry:
    """Cached contact of one sampled scraper pose against the 608 targets."""

    pose_id: int
    y_mm: float
    azimuth_deg: float
    origin_mm: tuple[float, float, float]
    yaw_deg: float
    length_axis: tuple[float, float, float]
    admissible: bool
    neighborhood_used: bool
    covered_mask: int
    covered_count: int
    physics_queries: int

    @property
    def cache_key_pose(self) -> str:
        return f"y={self.y_mm:.6f}|az={self.azimuth_deg:.6f}"


@dataclass(frozen=True)
class PoseContactCache:
    """Reusable contact masks for sampled poses. Target = matrix points only."""

    target_definition: str
    n_points: int
    point_face_ids: tuple[int, ...]
    entries: tuple[PoseContactEntry, ...]
    scraper_fingerprint: str
    physics_queries: int
    uses_legacy_a0_point_matrix: bool
    angle_window_deg: tuple[float, float]
    symmetry_multiplier_applied: bool = False

    def entry_at(self, pose_id: int) -> PoseContactEntry | None:
        if pose_id < 0 or pose_id >= len(self.entries):
            return None
        return self.entries[int(pose_id)]

    def admissible_entries(self) -> tuple[PoseContactEntry, ...]:
        return tuple(item for item in self.entries if item.admissible)


def pose_cache_from_entries(
    entries: tuple[PoseContactEntry, ...],
    *,
    n_points: int,
    fingerprint: str = "test",
    angle_window_deg: tuple[float, float] = (0.0, 90.0),
) -> PoseContactCache:
    """Test helper: inject pose masks without calling the collision engine."""
    return PoseContactCache(
        target_definition=COVERAGE_TARGET_REGION,
        n_points=int(n_points),
        point_face_ids=tuple(0 for _ in range(int(n_points))),
        entries=entries,
        scraper_fingerprint=str(fingerprint),
        physics_queries=sum(int(item.physics_queries) for item in entries),
        uses_legacy_a0_point_matrix=False,
        angle_window_deg=tuple(angle_window_deg),
        symmetry_multiplier_applied=False,
    )


def mask_indices(mask: int, n_points: int) -> tuple[int, ...]:
    return tuple(index for index in range(int(n_points)) if mask & (1 << index))


def _mask_from_faces(
    point_face_ids: NDArray[np.int64],
    contact_face_ids: frozenset[int],
) -> int:
    mask = 0
    if not contact_face_ids:
        return 0
    for index, face_id in enumerate(point_face_ids):
        if int(face_id) in contact_face_ids:
            mask |= 1 << index
    return int(mask)


_PROCESS_STATE: dict[str, object] = {}


def _contact_worker_count(n_poses: int) -> int:
    if n_poses < 32:
        return 1
    try:
        requested = int(os.environ.get("NUTELLA_CONTACT_WORKERS", "4"))
    except ValueError:
        requested = 4
    cpu = int(os.cpu_count() or 1)
    return max(1, min(requested, cpu, n_poses))


def _init_pose_process(
    surface: InteriorSurfaceReference,
    artifact: RigidScraperArtifact,
    base_params: ScraperParameters,
    src_vertices: NDArray[np.float64],
    src_faces: NDArray[np.int64],
    src_edges: NDArray[np.int64],
    point_face_ids: NDArray[np.int64],
) -> None:
    mesh = surface.to_trimesh()
    bind_envelope_proximity(mesh)
    _PROCESS_STATE["surface"] = surface
    _PROCESS_STATE["artifact"] = artifact
    _PROCESS_STATE["params"] = base_params
    _PROCESS_STATE["surface_mesh"] = mesh
    _PROCESS_STATE["src_vertices"] = src_vertices
    _PROCESS_STATE["src_faces"] = src_faces
    _PROCESS_STATE["src_edges"] = src_edges
    _PROCESS_STATE["point_face_ids"] = point_face_ids


def _parameters_for_spec(
    base: ScraperParameters,
    spec: PoseSampleSpec,
) -> ScraperParameters:
    return base.with_updates(
        surface_progress_deg=float(spec.azimuth_deg),
        position_z_mm=float(spec.y_mm),
    )


def _process_pose_spec(spec: PoseSampleSpec) -> PoseContactEntry:
    return evaluate_sampled_pose(
        spec=spec,
        artifact=_PROCESS_STATE["artifact"],  # type: ignore[arg-type]
        surface=_PROCESS_STATE["surface"],  # type: ignore[arg-type]
        parameters=_parameters_for_spec(
            _PROCESS_STATE["params"],  # type: ignore[arg-type]
            spec,
        ),
        surface_mesh=_PROCESS_STATE["surface_mesh"],  # type: ignore[arg-type]
        src_vertices=_PROCESS_STATE["src_vertices"],  # type: ignore[arg-type]
        src_faces=_PROCESS_STATE["src_faces"],  # type: ignore[arg-type]
        src_edges=_PROCESS_STATE["src_edges"],  # type: ignore[arg-type]
        point_face_ids=_PROCESS_STATE["point_face_ids"],  # type: ignore[arg-type]
    )


def evaluate_sampled_pose(
    *,
    spec: PoseSampleSpec,
    artifact: RigidScraperArtifact,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
    surface_mesh: trimesh.Trimesh,
    src_vertices: NDArray[np.float64],
    src_faces: NDArray[np.int64],
    src_edges: NDArray[np.int64],
    point_face_ids: NDArray[np.int64],
) -> PoseContactEntry:
    queries = 0
    nominal = envelope_contact_frame(
        surface,
        parameters,
        surface_progress_deg=float(spec.azimuth_deg),
        surface_mesh=surface_mesh,
    )
    frames = rigid_pose_neighborhood(nominal)
    chosen = None
    neighborhood_used = False
    covered_mask = 0
    for index, frame in enumerate(frames):
        try:
            transform = rigid_transform_between_frames(artifact.design_frame, frame)
        except np.linalg.LinAlgError:
            continue
        vertices = transform_points(src_vertices, transform)
        try:
            report = evaluate_envelope_collision(
                artifact.mesh,
                surface,
                parameters,
                surface_mesh=surface_mesh,
                vertices=vertices,
                faces=src_faces,
                edges_unique=src_edges,
            )
        except MemoryError:
            # Proximity expansion can OOM on isolated poses. Do not invent contact.
            queries += 1
            continue
        queries += 1
        if not report.admissible:
            continue
        chosen = frame
        neighborhood_used = index > 0
        covered_mask = _mask_from_faces(point_face_ids, report.contact_face_ids)
        break
    if chosen is None:
        origin = (
            float(nominal.origin_mm[0]),
            float(nominal.origin_mm[1]),
            float(nominal.origin_mm[2]),
        )
        axis = (
            float(nominal.rotation[0, 1]),
            float(nominal.rotation[1, 1]),
            float(nominal.rotation[2, 1]),
        )
        yaw = float(spec.azimuth_deg)
    else:
        origin = (
            float(chosen.origin_mm[0]),
            float(chosen.origin_mm[1]),
            float(chosen.origin_mm[2]),
        )
        axis = (
            float(chosen.rotation[0, 1]),
            float(chosen.rotation[1, 1]),
            float(chosen.rotation[2, 1]),
        )
        yaw = float(chosen.surface_progress_deg)
    return PoseContactEntry(
        pose_id=int(spec.pose_id),
        y_mm=float(spec.y_mm),
        azimuth_deg=float(spec.azimuth_deg),
        origin_mm=origin,
        yaw_deg=yaw,
        length_axis=axis,
        admissible=chosen is not None,
        neighborhood_used=neighborhood_used,
        covered_mask=int(covered_mask),
        covered_count=int(covered_mask).bit_count(),
        physics_queries=queries,
    )


def build_pose_contact_cache(
    surface: InteriorSurfaceReference,
    matrix: CoverageReferenceMatrix,
    specs: tuple[PoseSampleSpec, ...],
    *,
    artifact: RigidScraperArtifact | None = None,
    parameters: ScraperParameters | None = None,
) -> PoseContactCache:
    """Run the existing collision engine once per sampled pose."""
    if matrix.uses_legacy_a0_point_matrix:
        raise ValueError("Legacy A0 point matrix cannot be used as the contact target")
    if str(matrix.coverage_target_region) != COVERAGE_TARGET_REGION:
        raise ValueError(
            f"Contact target must be {COVERAGE_TARGET_REGION}, "
            f"got {matrix.coverage_target_region}"
        )
    if str(matrix.coverage_target_region) == LEGACY_A0_QUADRANT_REGION:
        raise ValueError("Legacy A0 quadrant cannot be used as the contact target")
    if str(matrix.coverage_target_region) != TARGET_MATRIX:
        raise ValueError(f"Pose cache target must be {TARGET_MATRIX}")

    params = parameters if parameters is not None else reference_scraper_parameters(surface)
    rigid = artifact if artifact is not None else build_rigid_from_params(surface, params)
    surface_mesh = surface.to_trimesh()
    bind_envelope_proximity(surface_mesh)
    points = np.asarray(matrix.points_mm, dtype=np.float64)
    _closest, _dist, face_ids = closest_on_envelope_surface(surface_mesh, points)
    point_face_ids = np.asarray(face_ids, dtype=np.int64)
    src_vertices = np.asarray(rigid.mesh.vertices, dtype=np.float64)
    src_faces = np.asarray(rigid.mesh.faces, dtype=np.int64)
    src_edges = np.asarray(rigid.mesh.edges_unique, dtype=np.int64)
    n_poses = len(specs)
    workers = _contact_worker_count(n_poses)
    entries: list[PoseContactEntry] = []
    queries = 0

    def _consume(iterator: object) -> None:
        nonlocal queries
        for index, entry in enumerate(iterator, start=1):  # type: ignore[arg-type]
            queries += entry.physics_queries
            entries.append(entry)
            if n_poses >= 20 and (index == 1 or index == n_poses or index % 25 == 0):
                print(
                    f"    pose {index}/{n_poses}  queries={queries}  "
                    f"admissible={sum(1 for item in entries if item.admissible)}",
                    flush=True,
                )

    if workers == 1:
        _consume(
            evaluate_sampled_pose(
                spec=spec,
                artifact=rigid,
                surface=surface,
                parameters=_parameters_for_spec(params, spec),
                surface_mesh=surface_mesh,
                src_vertices=src_vertices,
                src_faces=src_faces,
                src_edges=src_edges,
                point_face_ids=point_face_ids,
            )
            for spec in specs
        )
    else:
        print(f"    pose workers={workers}", flush=True)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_pose_process,
            initargs=(
                surface,
                rigid,
                params,
                src_vertices,
                src_faces,
                src_edges,
                point_face_ids,
            ),
        ) as pool:
            _consume(pool.map(_process_pose_spec, specs, chunksize=4))
    entries.sort(key=lambda item: int(item.pose_id))
    return PoseContactCache(
        target_definition=str(matrix.coverage_target_region),
        n_points=int(matrix.point_count),
        point_face_ids=tuple(int(v) for v in point_face_ids),
        entries=tuple(entries),
        scraper_fingerprint=str(rigid.shape_fingerprint),
        physics_queries=int(queries),
        uses_legacy_a0_point_matrix=False,
        angle_window_deg=tuple(matrix.coverage_target_azimuth_range),
        symmetry_multiplier_applied=False,
    )


def build_rigid_from_params(
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
) -> RigidScraperArtifact:
    from nutella_scraper.engines.compute.scraper_rigid_motion import (
        build_rigid_scraper_artifact,
    )

    return build_rigid_scraper_artifact(surface, parameters)


def cache_entries_payload(cache: PoseContactCache) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in cache.entries:
        rows.append(
            {
                "pose_id": int(item.pose_id),
                "y_mm": float(item.y_mm),
                "azimuth_deg": float(item.azimuth_deg),
                "origin_mm": [float(v) for v in item.origin_mm],
                "yaw_deg": float(item.yaw_deg),
                "length_axis": [float(v) for v in item.length_axis],
                "admissible": bool(item.admissible),
                "neighborhood_used": bool(item.neighborhood_used),
                "covered_mask": int(item.covered_mask),
                "covered_count": int(item.covered_count),
                "physics_queries": int(item.physics_queries),
            }
        )
    return rows


def cache_from_payload(
    rows: list[dict[str, object]],
    *,
    n_points: int,
    fingerprint: str,
    angle_window_deg: tuple[float, float],
) -> PoseContactCache:
    entries = []
    for item in rows:
        origin = item["origin_mm"]
        axis = item["length_axis"]
        mask = int(item["covered_mask"])
        ok = bool(item["admissible"])
        entries.append(
            PoseContactEntry(
                pose_id=int(item["pose_id"]),
                y_mm=float(item["y_mm"]),
                azimuth_deg=float(item["azimuth_deg"]),
                origin_mm=(float(origin[0]), float(origin[1]), float(origin[2])),
                yaw_deg=float(item["yaw_deg"]),
                length_axis=(float(axis[0]), float(axis[1]), float(axis[2])),
                admissible=ok,
                neighborhood_used=bool(item["neighborhood_used"]),
                covered_mask=mask if ok else 0,
                covered_count=int(item["covered_count"]) if ok else 0,
                physics_queries=int(item["physics_queries"]),
            )
        )
    entries.sort(key=lambda entry: entry.pose_id)
    return pose_cache_from_entries(
        tuple(entries),
        n_points=n_points,
        fingerprint=fingerprint,
        angle_window_deg=angle_window_deg,
    )
