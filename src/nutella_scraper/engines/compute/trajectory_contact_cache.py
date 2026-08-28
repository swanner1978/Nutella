"""Per-cell contact cache on the validated 0–90° point cloud.

Pose mapping (documented before optimization)
--------------------------------------------
A trajectory waypoint is a ``TrajectoryCell`` of ``interior_matrix_a0_0_90``.

The existing rigid engine places a scraper with two independent pose
variables, plus an optional SE(3) recovery neighbourhood:

1. ``surface_progress_deg`` = cell azimuth (yaw / wall longitude).
   Same convention as ``envelope_contact_frame``: 0° = +X, 90° = −Z.
2. ``position_z_mm`` = cell Y (height of the tip contour). This is used
   only to choose the envelope frame; the manufacturing solid is lofted
   once and moved by SE(3).
3. Motion direction is **not** a contact variable. Contact is quasi-static
   at the pose. Direction is recorded on the trajectory for kinematics
   (length, direction changes), never mixed into the contact mask.

One cell → one **nominal** envelope pose. The 17-sample ``rigid_pose_neighborhood``
is used only when the nominal pose is inadmissible (same recovery as
``pose_rigid_scraper_admissible``), not as extra beam dimensions.

This module does not import CoverageSimulator, does not sweep 0–45°,
does not apply ×4/×8 symmetry, and does not use A0 as a point grid.
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
from nutella_scraper.engines.compute.scraper_envelope_collision import (
    evaluate_envelope_collision,
    rigid_pose_neighborhood,
)
from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    RigidScraperArtifact,
    build_rigid_scraper_artifact,
    envelope_contact_frame,
    rigid_transform_between_frames,
    transform_points,
)
from nutella_scraper.engines.compute.trajectory_search import TrajectoryCell, TrajectoryGrid

POSE_VARIABLES = (
    "surface_progress_deg",  # cell.azimuth_deg
    "position_z_mm",  # cell.y_mm
    "se3_neighborhood_if_nominal_blocked",
)
MOTION_DIRECTION_AFFECTS_CONTACT = False


def contact_cache_key(shape_fingerprint: str, row: int, col: int) -> str:
    """Minimum cache identity: shape (incl. parameters) + grid cell."""
    return f"{shape_fingerprint}|r{int(row)}|c{int(col)}"


def reference_scraper_parameters(surface: InteriorSurfaceReference) -> ScraperParameters:
    """A0 manufacturing solid (shape only). Not the A0 point grid."""
    _opening, _lower, max_length = scraper_length_span(surface)
    return ScraperParameters.default().with_updates(
        bevel_angle_deg=0.0,
        relief_angle_deg=0.0,
        helix_rate_deg_per_mm=0.0,
        width_mm=2.5,
        thickness_mm=2.5,
        length_mm=min(40.0, float(max_length)),
        clearance_mm=0.0,
        position_z_mm=float(0.5 * (surface.y_min_mm + surface.y_max_mm)),
        surface_progress_deg=0.0,
    )


@dataclass(frozen=True)
class CellContactEntry:
    """Cached contact of one admissible (or blocked) cell pose."""

    point_index: int
    row: int
    col: int
    yaw_deg: float
    origin_mm: tuple[float, float, float]
    admissible: bool
    neighborhood_used: bool
    covered_mask: int
    covered_count: int
    physics_queries: int


@dataclass(frozen=True)
class TrajectoryContactCache:
    """Reusable contact masks for every grid cell. Target = matrix points only."""

    target_definition: str
    n_points: int
    point_face_ids: tuple[int, ...]
    entries: tuple[CellContactEntry, ...]
    scraper_fingerprint: str
    physics_queries: int
    uses_legacy_a0_point_matrix: bool
    angle_window_deg: tuple[float, float]
    symmetry_multiplier_applied: bool = False

    def entry_at(self, row: int, col: int) -> CellContactEntry | None:
        for item in self.entries:
            if item.row == int(row) and item.col == int(col):
                return item
        return None

    def mask_for(self, cell: TrajectoryCell) -> int:
        item = self.entry_at(cell.row, cell.col)
        if item is None or not item.admissible:
            return 0
        return int(item.covered_mask)


def contact_cache_from_masks(
    grid: TrajectoryGrid,
    masks: dict[tuple[int, int], int],
    *,
    n_points: int | None = None,
    admissible: dict[tuple[int, int], bool] | None = None,
) -> TrajectoryContactCache:
    """Test helper: inject per-cell masks without calling the collision engine."""
    total = int(n_points if n_points is not None else len(grid.cells))
    flags = admissible or {}
    entries = []
    for cell in grid.cells:
        key = (cell.row, cell.col)
        mask = int(masks.get(key, 0))
        ok = bool(flags.get(key, True))
        entries.append(
            CellContactEntry(
                point_index=int(cell.index),
                row=int(cell.row),
                col=int(cell.col),
                yaw_deg=float(cell.azimuth_deg),
                origin_mm=(float(cell.x_mm), float(cell.y_mm), float(cell.z_mm)),
                admissible=ok,
                neighborhood_used=False,
                covered_mask=mask if ok else 0,
                covered_count=mask.bit_count() if ok else 0,
                physics_queries=0,
            )
        )
    return TrajectoryContactCache(
        target_definition=str(grid.target_definition),
        n_points=total,
        point_face_ids=tuple(0 for _ in range(total)),
        entries=tuple(entries),
        scraper_fingerprint="test",
        physics_queries=0,
        uses_legacy_a0_point_matrix=False,
        angle_window_deg=tuple(grid.angle_range_deg),
        symmetry_multiplier_applied=False,
    )


_PROCESS_STATE: dict[str, object] = {}


def _contact_worker_count(n_cells: int) -> int:
    if n_cells < 32:
        return 1
    try:
        requested = int(os.environ.get("NUTELLA_CONTACT_WORKERS", "4"))
    except ValueError:
        requested = 4
    cpu = int(os.cpu_count() or 1)
    return max(1, min(requested, cpu, n_cells))


def _init_contact_process(
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


def _process_contact_cell(cell: TrajectoryCell) -> CellContactEntry:
    return _evaluate_cell_pose(
        cell=cell,
        artifact=_PROCESS_STATE["artifact"],  # type: ignore[arg-type]
        surface=_PROCESS_STATE["surface"],  # type: ignore[arg-type]
        parameters=_parameters_for_cell(
            _PROCESS_STATE["params"],  # type: ignore[arg-type]
            cell,
        ),
        surface_mesh=_PROCESS_STATE["surface_mesh"],  # type: ignore[arg-type]
        src_vertices=_PROCESS_STATE["src_vertices"],  # type: ignore[arg-type]
        src_faces=_PROCESS_STATE["src_faces"],  # type: ignore[arg-type]
        src_edges=_PROCESS_STATE["src_edges"],  # type: ignore[arg-type]
        point_face_ids=_PROCESS_STATE["point_face_ids"],  # type: ignore[arg-type]
    )


def _parameters_for_cell(
    base: ScraperParameters,
    cell: TrajectoryCell,
) -> ScraperParameters:
    return base.with_updates(
        surface_progress_deg=float(cell.azimuth_deg),
        position_z_mm=float(cell.y_mm),
    )


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


def _evaluate_cell_pose(
    *,
    cell: TrajectoryCell,
    artifact: RigidScraperArtifact,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
    surface_mesh: trimesh.Trimesh,
    src_vertices: NDArray[np.float64],
    src_faces: NDArray[np.int64],
    src_edges: NDArray[np.int64],
    point_face_ids: NDArray[np.int64],
) -> CellContactEntry:
    queries = 0
    nominal = envelope_contact_frame(
        surface,
        parameters,
        surface_progress_deg=float(cell.azimuth_deg),
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
        report = evaluate_envelope_collision(
            artifact.mesh,
            surface,
            parameters,
            surface_mesh=surface_mesh,
            vertices=vertices,
            faces=src_faces,
            edges_unique=src_edges,
        )
        queries += 1
        if not report.admissible:
            continue
        chosen = frame
        neighborhood_used = index > 0
        covered_mask = _mask_from_faces(point_face_ids, report.contact_face_ids)
        break
    origin = (
        (float(chosen.origin_mm[0]), float(chosen.origin_mm[1]), float(chosen.origin_mm[2]))
        if chosen is not None
        else (float(cell.x_mm), float(cell.y_mm), float(cell.z_mm))
    )
    return CellContactEntry(
        point_index=int(cell.index),
        row=int(cell.row),
        col=int(cell.col),
        yaw_deg=float(cell.azimuth_deg),
        origin_mm=origin,
        admissible=chosen is not None,
        neighborhood_used=neighborhood_used,
        covered_mask=int(covered_mask),
        covered_count=int(covered_mask).bit_count(),
        physics_queries=queries,
    )


def build_contact_cache(
    surface: InteriorSurfaceReference,
    matrix: CoverageReferenceMatrix,
    grid: TrajectoryGrid,
    *,
    artifact: RigidScraperArtifact | None = None,
    parameters: ScraperParameters | None = None,
) -> TrajectoryContactCache:
    """Run the existing collision engine once per grid cell. Not a trajectory enum."""
    if matrix.uses_legacy_a0_point_matrix:
        raise ValueError("Legacy A0 point matrix cannot be used as the contact target")
    if str(matrix.coverage_target_region) != COVERAGE_TARGET_REGION:
        raise ValueError(
            f"Contact target must be {COVERAGE_TARGET_REGION}, "
            f"got {matrix.coverage_target_region}"
        )
    if str(matrix.coverage_target_region) == LEGACY_A0_QUADRANT_REGION:
        raise ValueError("Legacy A0 quadrant cannot be used as the contact target")
    if grid.target_definition != COVERAGE_TARGET_REGION:
        raise ValueError("Trajectory grid is not the validated interior matrix")
    if grid.uses_legacy_a0_point_matrix:
        raise ValueError("Trajectory grid still flags the legacy A0 matrix")
    if len(grid.cells) != int(matrix.point_count):
        raise ValueError("Grid cells must be exactly the matrix points")

    params = parameters if parameters is not None else reference_scraper_parameters(surface)
    rigid = artifact if artifact is not None else build_rigid_scraper_artifact(surface, params)
    surface_mesh = surface.to_trimesh()
    bind_envelope_proximity(surface_mesh)
    points = np.asarray(matrix.points_mm, dtype=np.float64)
    _closest, _dist, face_ids = closest_on_envelope_surface(surface_mesh, points)
    point_face_ids = np.asarray(face_ids, dtype=np.int64)
    src_vertices = np.asarray(rigid.mesh.vertices, dtype=np.float64)
    src_faces = np.asarray(rigid.mesh.faces, dtype=np.int64)
    src_edges = np.asarray(rigid.mesh.edges_unique, dtype=np.int64)
    n_cells = len(grid.cells)
    workers = _contact_worker_count(n_cells)
    entries: list[CellContactEntry] = []
    queries = 0

    def _consume(iterator: object) -> None:
        nonlocal queries
        for index, entry in enumerate(iterator, start=1):  # type: ignore[arg-type]
            queries += entry.physics_queries
            entries.append(entry)
            if n_cells >= 100 and (index == 1 or index == n_cells or index % 50 == 0):
                print(
                    f"    contact {index}/{n_cells}  queries={queries}  "
                    f"admissible={sum(1 for item in entries if item.admissible)}",
                    flush=True,
                )

    if workers == 1:
        _consume(
            _evaluate_cell_pose(
                cell=cell,
                artifact=rigid,
                surface=surface,
                parameters=_parameters_for_cell(params, cell),
                surface_mesh=surface_mesh,
                src_vertices=src_vertices,
                src_faces=src_faces,
                src_edges=src_edges,
                point_face_ids=point_face_ids,
            )
            for cell in grid.cells
        )
    else:
        print(f"    contact workers={workers}", flush=True)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_contact_process,
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
            _consume(pool.map(_process_contact_cell, grid.cells, chunksize=4))
    return TrajectoryContactCache(
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
