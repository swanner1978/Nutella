"""Bridge helpers for the local demo viewer — visualization only."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.domain.models.cad_reference_geometry import CadReferenceGeometry
from nutella_scraper.domain.models.contact import (
    CollisionResult,
    ContactSimulationConfig,
    TrajectoryConfig,
)
from nutella_scraper.domain.models.envelope import EnvelopeSlice, InteriorEnvelope
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.domain.models.scraper import ScraperPose
from nutella_scraper.domain.models.views import (
    ProjectedView,
    ProjectionMetadata,
    ViewOverlayPayload,
    ViewProjectionCache,
)
from nutella_scraper.engines.compute.contact_simulator import ContactSimulationEngine
from nutella_scraper.engines.compute.scraper_builder import ScraperBuilder
from nutella_scraper.engines.compute.scraper_transform import pose_matrix, pose_to_dict
from nutella_scraper.engines.visualization.contact_metrics_panel import ContactMetricsPanel
from nutella_scraper.engines.visualization.engine import VisualizationEngine
from nutella_scraper.engines.visualization.envelope_projector import EnvelopeProjector
from nutella_scraper.engines.visualization.overlay_renderer import OverlayRenderer
from nutella_scraper.engines.visualization.pose_snapshot_store import PoseSnapshotStore
from nutella_scraper.engines.visualization.scraper_result_projector import ScraperResultProjector
from nutella_scraper.engines.visualization.target_face_color_projector import (
    LAYER_TARGET_FACE_COLORS,
    TargetFaceColorProjector,
)
from nutella_scraper.engines.visualization.trajectory_projector import TrajectoryProjector
from nutella_scraper.engines.visualization.viewer_cameras import (
    JAR_FRAME_CONVENTION,
    cameras_from_bounds,
)
from nutella_scraper.io.scraper_config_loader import default_racloir_v1

_CONFIG_ROOT = Path(__file__).resolve().parents[4] / "configs"

DEFAULT_SCRAPER_GEOMETRY = default_racloir_v1(_CONFIG_ROOT)
DEFAULT_SCRAPER_POSE = ScraperPose(position_mm=(47.0, 50.0, 0.0), yaw_deg=0.0)
DEFAULT_SIMULATION_CONFIG = ContactSimulationConfig(
    trajectory=TrajectoryConfig(angular_step_deg=45.0, vertical_step_mm=25.0),
    contact_threshold_mm=1.0,
    clearance_mm=0.15,
    mesh_tolerance_mm=0.1,
)

# Manufacturing solid cache: shape fingerprint → RigidScraperArtifact.
# Progress / pose changes must reuse the same mesh (rigid SE(3) only).
_RIGID_SCRAPER_CACHE: dict[str, Any] = {}

SimulationProgressCallback = Callable[
    [str, str, float | None, dict[str, float]],
    None,
]


def view_cache_from_viewer_dir(view_dir: Path) -> ViewProjectionCache:
    """Build a ViewProjectionCache from immutable viewer SVG assets."""
    metadata = json.loads((view_dir / "metadata.json").read_text(encoding="utf-8"))
    model_id = str(metadata["model_id"])
    displayed = metadata["displayed_views"]
    side_entry = displayed.get("side") or displayed["profile"]
    top_entry = displayed["top"]

    side_svg = (view_dir / side_entry["filename"]).read_text(encoding="utf-8")
    top_svg = (view_dir / top_entry["filename"]).read_text(encoding="utf-8")

    return ViewProjectionCache(
        model_id=model_id,
        profile_view=ProjectedView(
            plane=str(side_entry.get("plane", "XY")),
            asset_path=view_dir / side_entry["filename"],
            svg_content=side_svg,
            metadata=ProjectionMetadata(
                plane=str(side_entry.get("plane", "XY")),
                camera={"x": 0.0, "y": -1.0, "z": 0.0},
                scale=1.0,
                width_px=900,
                height_px=650,
            ),
        ),
        top_view=ProjectedView(
            plane=str(top_entry.get("plane", "XZ")),
            asset_path=view_dir / top_entry["filename"],
            svg_content=top_svg,
            metadata=ProjectionMetadata(
                plane=str(top_entry.get("plane", "XZ")),
                camera={"x": 0.0, "y": 0.0, "z": -1.0},
                scale=1.0,
                width_px=900,
                height_px=650,
            ),
        ),
    )


def build_contact_visualization_response(
    *,
    view_dir: Path,
    models_root: Path,
    progress_callback: SimulationProgressCallback | None = None,
    pose_snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run contact simulation and project overlays for the active viewer model.

    Uses CanonicalModel3D from ModelStore — does not reload STEP or SVG assets.
    """
    total_started = time.perf_counter()
    profile_ms: dict[str, float] = {}

    def report(phase: str, detail: str, percent: float | None) -> None:
        if progress_callback is not None:
            progress_callback(phase, detail, percent, dict(profile_ms))

    report("preparation", "Lecture du manifeste de visualisation", 1.0)
    started = time.perf_counter()
    view_cache = view_cache_from_viewer_dir(view_dir)
    profile_ms["preparation"] = (time.perf_counter() - started) * 1000.0

    report("model_loading", "Chargement du CanonicalModel3D", 3.0)
    started = time.perf_counter()
    store = ModelStore(models_root)
    jar = store.get(view_cache.model_id)
    internal = store.get_internal(view_cache.model_id)
    cad_reference = store.get_cad_reference(view_cache.model_id)
    profile_ms["model_loading"] = (time.perf_counter() - started) * 1000.0

    scraper_builder = ScraperBuilder()
    scraper_pipeline: dict[str, Any] = {
        "loss_stage": None,
        "stages": {
            "config_loaded": {
                "ok": True,
                "scraper_id": DEFAULT_SCRAPER_GEOMETRY.id,
                "source": str(DEFAULT_SCRAPER_GEOMETRY.metadata.get("config_path", "inline")),
            },
        },
    }

    report("scraper_generation", "Construction du Scraper3D paramétrique V1", 4.0)
    started = time.perf_counter()
    try:
        base_scraper_mesh = scraper_builder.build(DEFAULT_SCRAPER_GEOMETRY)
        scraper_pipeline["stages"]["mesh_built"] = {
            "ok": bool(base_scraper_mesh.volume > 0.0),
            "vertex_count": len(base_scraper_mesh.vertices),
            "face_count": len(base_scraper_mesh.faces),
            "volume_mm3": float(base_scraper_mesh.volume),
        }
        if base_scraper_mesh.volume <= 0.0:
            scraper_pipeline["loss_stage"] = "mesh_built"
    except Exception as exc:
        scraper_pipeline["stages"]["mesh_built"] = {
            "ok": False,
            "error": str(exc),
        }
        scraper_pipeline["loss_stage"] = "mesh_built"
        raise
    profile_ms["scraper_generation"] = (time.perf_counter() - started) * 1000.0

    simulation_started = time.perf_counter()
    pose_store = PoseSnapshotStore(pose_snapshot_dir) if pose_snapshot_dir else None

    def capture_pose(
        index: int,
        total: int,
        posed_scraper: Any,
        scraper_transform: Any,
        face_distances: Any,
        contact_points: tuple,
        collision: Any,
    ) -> None:
        if pose_store is None:
            return
        pose_store.persist(
            index=index,
            total=total,
            scraper_vertices=posed_scraper.vertices,
            scraper_faces=posed_scraper.faces,
            scraper_transform=scraper_transform,
            face_distances=face_distances,
            face_coverage=(
                np.isfinite(face_distances)
                & (face_distances <= DEFAULT_SIMULATION_CONFIG.contact_threshold_mm)
            ),
            contact_points=contact_points,
            collision=collision,
        )

    def report_compute(phase: str, detail: str, percent: float | None) -> None:
        report(phase, detail, percent)

    contact = ContactSimulationEngine(scraper_builder=scraper_builder).simulate(
        jar,
        DEFAULT_SCRAPER_GEOMETRY,
        DEFAULT_SCRAPER_POSE,
        DEFAULT_SIMULATION_CONFIG,
        internal=internal,
        progress_callback=report_compute,
        profile_ms=profile_ms,
        pose_result_callback=capture_pose if pose_store is not None else None,
    )

    captured_pose_count = contact.trajectory_pose_count
    manifest_pose_count = captured_pose_count
    reference_pose = False
    if pose_store is not None and captured_pose_count == 0:
        reference_pose = True
        manifest_pose_count = 1
        posed_reference = scraper_builder.build_posed(
            DEFAULT_SCRAPER_GEOMETRY,
            DEFAULT_SCRAPER_POSE,
        )
        reference_transform = pose_matrix(DEFAULT_SCRAPER_POSE)
        face_count = internal.face_count
        empty_distances = np.full(face_count, np.inf, dtype=np.float64)
        empty_collision = CollisionResult(
            has_collision=False,
            penetration_depth_mm=0.0,
            collision_points=(),
            colliding_face_ids=frozenset(),
        )
        pose_store.persist(
            index=0,
            total=0,
            scraper_vertices=posed_reference.vertices,
            scraper_faces=posed_reference.faces,
            scraper_transform=reference_transform,
            face_distances=empty_distances,
            face_coverage=np.zeros(face_count, dtype=np.bool_),
            contact_points=(),
            collision=empty_collision,
        )

    scraper_pipeline["stages"]["simulation"] = {
        "ok": bool(scraper_pipeline["loss_stage"] is None),
        "simulated_pose_count": captured_pose_count,
        "candidate_pose_count": contact.diagnostics.get("candidate_pose_count"),
        "rejected_pose_count": contact.diagnostics.get("rejected_pose_count"),
    }
    scraper_pipeline["stages"]["pose_capture"] = {
        "ok": bool(captured_pose_count > 0 or reference_pose),
        "mode": "reference_base_pose" if reference_pose else "trajectory",
        "captured_count": captured_pose_count,
        "manifest_pose_count": manifest_pose_count,
    }
    scraper_pipeline["stages"]["visualization"] = {
        "ok": bool(pose_store is not None and manifest_pose_count > 0),
        "mode": "per_pose_on_demand" if pose_store is not None else "disabled",
    }
    if pose_store is not None and manifest_pose_count == 0:
        scraper_pipeline["loss_stage"] = "pose_capture"

    if pose_store is not None:
        pose_store.finalize(
            model_id=view_cache.model_id,
            pose_count=manifest_pose_count,
            view_dir_name=view_dir.name,
            scraper_pipeline=scraper_pipeline,
            reference_pose=reference_pose,
        )
    duration_ms = (time.perf_counter() - simulation_started) * 1000.0

    report("overlay_generation", "Projection des résultats 3D dans les vues", 85.0)
    started = time.perf_counter()
    overlay_profile: dict[str, Any] = {}
    if pose_store is not None:
        contact_for_panel = replace(
            contact,
            diagnostics={
                **contact.diagnostics,
                "simulation_duration_ms": duration_ms,
            },
        )
        metrics = ContactMetricsPanel.from_contact_result(contact_for_panel)
        fragments: dict[str, dict[str, str]] = {"profile": {}, "top": {}}
        overlay_profile.update(
            {
                "mode": "per_pose_on_demand",
                "face_count": len(contact.contact_distance_map),
                "pose_count": manifest_pose_count,
                "simulated_pose_count": captured_pose_count,
                "reference_pose": reference_pose,
                "graphic_element_count": 0,
                "payload_bytes": 0,
            }
        )
    else:
        engine = VisualizationEngine()
        _overlay, metrics, fragments = engine.build_contact_visualization(
            contact,
            view_cache,
            jar,
            internal=internal,
            simulation_duration_ms=duration_ms,
            overlay_profile=overlay_profile,
        )
    profile_ms["overlay_generation"] = (time.perf_counter() - started) * 1000.0
    profile_ms["ui_refresh"] = 0.0
    profile_ms["total"] = (time.perf_counter() - total_started) * 1000.0
    report("overlay_generation", "Overlays prêts pour l'interface", 95.0)

    jar_vertices = np.asarray(jar.mesh.vertices, dtype=np.float64)
    envelope_payload = _envelope_overlay_payload(
        cad_reference,
        contact.diagnostics.get("envelope"),
        jar_vertices=jar_vertices,
    )
    trajectory_payload = _trajectory_overlay_payload(
        internal,
        pose_store,
        manifest_pose_count,
    )

    return {
        "model_id": view_cache.model_id,
        "coverage_score_display": contact.coverage_score,
        "metrics": metrics.to_dict(),
        "overlays": {
            "side": {
                **fragments["profile"],
                **envelope_payload.get("side", {}),
                **trajectory_payload.get("side", {}),
            },
            "top": {
                **fragments["top"],
                **envelope_payload.get("top", {}),
                **trajectory_payload.get("top", {}),
            },
            "left": {
                **fragments.get("left", {}),
                **envelope_payload.get("left", {}),
                **trajectory_payload.get("left", {}),
            },
            "right": {
                **fragments.get("right", {}),
                **envelope_payload.get("right", {}),
                **trajectory_payload.get("right", {}),
            },
        },
        "pose_constraints": {
            "candidate_pose_count": contact.diagnostics.get("candidate_pose_count"),
            "accepted_pose_count": contact.diagnostics.get("accepted_pose_count"),
            "rejected_pose_count": contact.diagnostics.get("rejected_pose_count"),
            "simulated_pose_count": contact.diagnostics.get("simulated_pose_count"),
            "rejections_by_reason": contact.diagnostics.get("rejections_by_reason", {}),
            "envelope_duration_ms": contact.diagnostics.get("envelope_duration_ms"),
            "pose_generation_duration_ms": contact.diagnostics.get(
                "pose_generation_duration_ms"
            ),
        },
        "performance_profile_ms": {
            phase: round(duration, 1) for phase, duration in profile_ms.items()
        },
        "overlay_profile": overlay_profile,
        "trajectory": {
            "pose_count": manifest_pose_count,
            "simulated_pose_count": captured_pose_count,
            "reference_pose": reference_pose,
            "selected_pose_index": 0,
        },
        "scraper_pipeline": scraper_pipeline,
    }


def build_scraper_visualization_response(
    *,
    view_dir: Path,
    models_root: Path,
    parameters: object | None = None,
) -> dict[str, Any]:
    """
    Build parametric Scraper V1 from the cyan STEP interior (RGB 85,255,255).

    Same surface as Contour intérieur — not InternalJarSurface approximations.
    """
    from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
    from nutella_scraper.engines.compute.interior_surface_reference import (
        load_interior_surface_reference,
    )
    from nutella_scraper.engines.compute.scraper_envelope_collision import (
        collision_payload,
        pose_rigid_scraper_admissible,
    )
    from nutella_scraper.engines.compute.scraper_rigid_motion import (
        build_rigid_scraper_artifact,
        manufacturing_fingerprint,
    )

    view_cache = view_cache_from_viewer_dir(view_dir)
    store = ModelStore(models_root)
    store.get(view_cache.model_id)
    internal = store.get_internal(view_cache.model_id)
    model_id = view_cache.model_id
    interior_surface = load_interior_surface_reference(
        models_root=models_root,
        model_id=model_id,
        step_path=models_root / model_id / ModelStore.REFERENCE_STEP,
    )

    if isinstance(parameters, ScraperParameters):
        params = parameters
    elif isinstance(parameters, dict):
        params = ScraperParameters.from_dict(parameters)
    else:
        mid_height = 0.5 * (
            float(interior_surface.y_min_mm) + float(interior_surface.y_max_mm)
        )
        params = ScraperParameters.default().with_updates(position_z_mm=mid_height)

    scraper_pipeline: dict[str, Any] = {
        "loss_stage": None,
        "stages": {
            "config_loaded": {
                "ok": True,
                "scraper_id": "parametric_v1_rigid_pose",
                "source": interior_surface.source,
                "matching_face_count": interior_surface.matching_face_count,
                "parameters": params.to_dict(),
            },
        },
    }
    try:
        shape_key = manufacturing_fingerprint(params, model_id=model_id)
        artifact = _RIGID_SCRAPER_CACHE.get(shape_key)
        rebuilt = False
        if artifact is None:
            artifact = build_rigid_scraper_artifact(interior_surface, params)
            _RIGID_SCRAPER_CACHE[shape_key] = artifact
            rebuilt = True
        admissible = pose_rigid_scraper_admissible(
            artifact,
            interior_surface,
            params,
        )
        posed_mesh = admissible.posed_mesh
        pose = admissible.pose
        transform_se3 = admissible.transform
        envelope_validation = collision_payload(admissible, params)
        active_edge_mm = np.asarray(admissible.wall_edge_mm, dtype=np.float64)
        scraper_pipeline["stages"]["envelope_path"] = {
            "ok": not admissible.blocked,
            "source": interior_surface.source,
            "station_count": len(artifact.design_path.stations),
            "sampling": "rigid_pose_along_envelope",
            "surface_progress_deg": float(params.surface_progress_deg),
            "geometry_rebuilt": rebuilt,
            "active_edge_point_count": int(len(active_edge_mm)),
            "pose_status": admissible.status,
            "blocked": bool(admissible.blocked),
            "alternative_used": bool(admissible.alternative_used),
            "has_collision": bool(
                admissible.collision.has_collision and admissible.status != "VALID"
            ),
        }
        scraper_pipeline["stages"]["mesh_built"] = {
            "ok": bool(posed_mesh.volume > 0.0) or len(posed_mesh.faces) > 0,
            "vertex_count": len(posed_mesh.vertices),
            "face_count": len(posed_mesh.faces),
            "volume_mm3": float(getattr(posed_mesh, "volume", 0.0) or 0.0),
            "rigid_geometry": True,
            "shape_fingerprint": shape_key,
        }
        if len(posed_mesh.faces) == 0:
            scraper_pipeline["loss_stage"] = "mesh_built"
        if admissible.blocked:
            scraper_pipeline["loss_stage"] = "envelope_collision"
            scraper_pipeline["stages"]["envelope_collision"] = {
                "ok": False,
                "status": "BLOCKED",
                "message": "MOUVEMENT BLOQUÉ — collision avec l'enveloppe intérieure",
                "surface_progress_deg": float(params.surface_progress_deg),
            }
        else:
            scraper_pipeline["stages"]["envelope_collision"] = {
                "ok": True,
                "status": admissible.status,
                "alternative_used": bool(admissible.alternative_used),
                "has_collision": False,
            }
    except Exception as exc:
        scraper_pipeline["stages"]["mesh_built"] = {"ok": False, "error": str(exc)}
        scraper_pipeline["loss_stage"] = "mesh_built"
        raise

    transform = pose_matrix(pose)
    scraper_projection = ScraperResultProjector().project(
        scraper_vertices=posed_mesh.vertices,
        scraper_faces=posed_mesh.faces,
        internal=internal,
    )
    tip_projection = TrajectoryProjector().project(
        positions_mm=tuple(
            (float(p[0]), float(p[1]), float(p[2])) for p in active_edge_mm
        ),
        internal=internal,
    )
    overlay = ViewOverlayPayload(
        model_id=view_cache.model_id,
        profile_layers=scraper_projection.profile_layers + tip_projection.profile_layers,
        top_layers=scraper_projection.top_layers + tip_projection.top_layers,
        left_layers=scraper_projection.left_layers + tip_projection.left_layers,
        right_layers=scraper_projection.right_layers + tip_projection.right_layers,
        bottom_layers=scraper_projection.bottom_layers + tip_projection.bottom_layers,
        coverage_score_display=0.0,
    )
    fragments = OverlayRenderer().layer_fragments(overlay)
    return {
        "model_id": view_cache.model_id,
        "parameters": params.to_dict(),
        "pose": pose_to_dict(pose),
        "scraper_transform": transform_se3.tolist(),
        "scraper_geometry": {
            "vertices": np.asarray(artifact.mesh.vertices, dtype=np.float64).tolist(),
            "faces": np.asarray(artifact.mesh.faces, dtype=np.int64).tolist(),
        },
        "scraper_pipeline": scraper_pipeline,
        "scraper": {
            "vertex_count": scraper_projection.vertex_count,
            "face_count": scraper_projection.face_count,
            "provenance": "parametric_v1_rigid_pose",
            "interior_source": interior_surface.source,
            "rigid_geometry": True,
        },
        "active_edge": {
            "point_count": int(len(active_edge_mm)),
            "points_mm": active_edge_mm.tolist(),
            "source": "rigid_pose_tip_edge",
        },
        "validation": envelope_validation,
        "collision": envelope_validation,
        "overlays": {
            "side": fragments["profile"],
            "top": fragments["top"],
            "left": fragments["left"],
            "right": fragments["right"],
            "bottom": fragments["bottom"],
        },
    }


def _compact_mesh_payload(
    vertices: np.ndarray,
    *,
    faces: np.ndarray | None = None,
    edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Serialize a mesh for the canvas viewer (same vertices, rounded for JSON)."""
    payload: dict[str, Any] = {
        "vertices": np.round(np.asarray(vertices, dtype=np.float64), 3).tolist(),
    }
    if faces is not None:
        payload["faces"] = np.asarray(faces, dtype=np.int32).tolist()
    if edges is not None:
        payload["edges"] = np.asarray(edges, dtype=np.int32).tolist()
    return payload


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    faces_i = np.asarray(faces, dtype=np.int64)
    if faces_i.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    raw = np.concatenate(
        [faces_i[:, [0, 1]], faces_i[:, [1, 2]], faces_i[:, [2, 0]]],
        axis=0,
    )
    raw.sort(axis=1)
    structured = np.ascontiguousarray(raw).view(
        np.dtype([("a", np.int64), ("b", np.int64)])
    )
    unique = np.unique(structured)
    return unique.view(np.int64).reshape(-1, 2).astype(np.int32)


def build_viewer_scene_response(
    *,
    view_dir: Path,
    models_root: Path,
) -> dict[str, Any]:
    """
    Single 3D camera scene for all viewer views — visualization only.

    The displayed jar comes from ``visual.stl``. Compute still uses
    CanonicalModel3D.mesh and InteriorSurfaceReference elsewhere.
    Changing the view only swaps the lookAt camera; meshes are not rebuilt.
    """
    from nutella_scraper.engines.compute.interior_surface_reference import (
        load_interior_surface_reference,
    )

    view_cache = view_cache_from_viewer_dir(view_dir)
    store = ModelStore(models_root)
    visual = store.load_visual_mesh(view_cache.model_id)
    jar_vertices = np.asarray(visual.vertices, dtype=np.float64)
    jar_faces = np.asarray(visual.faces, dtype=np.int64)
    cameras, center, distance = cameras_from_bounds(
        jar_vertices.min(axis=0),
        jar_vertices.max(axis=0),
    )

    interior_payload: dict[str, Any] | None = None
    interior_error: str | None = None
    try:
        interior = load_interior_surface_reference(
            models_root=models_root,
            model_id=view_cache.model_id,
            step_path=models_root / view_cache.model_id / ModelStore.REFERENCE_STEP,
        )
        interior_payload = {
            **_compact_mesh_payload(
                interior.vertices,
                faces=interior.faces,
                edges=_unique_edges(interior.faces),
            ),
            "source": interior.source,
            "matching_face_count": interior.matching_face_count,
        }
    except (FileNotFoundError, ValueError, OSError) as exc:
        interior_error = str(exc)

    return {
        "model_id": view_cache.model_id,
        "convention": JAR_FRAME_CONVENTION,
        "cameras": cameras,
        "center_mm": center.tolist(),
        "distance_mm": distance,
        "jar": {
            **_compact_mesh_payload(
                jar_vertices,
                faces=jar_faces,
                edges=_unique_edges(jar_faces),
            ),
            "source": "visual.stl",
        },
        "interior": interior_payload,
        "interior_error": interior_error,
    }


def build_interior_contour_response(
    *,
    view_dir: Path,
    models_root: Path,
    clearance_mm: float | None = None,
) -> dict[str, Any]:
    """
    Overlay for the Contour intérieur layer: STEP faces RGB(85,255,255).

    Selection comes solely from the XCAF colour diagnostic already used by the
    debug endpoint — no geometric envelope rebuild.
    """
    del clearance_mm  # reserved; colour selection ignores clearance
    from nutella_scraper.cad_import.step_face_color_diagnostics import (
        extract_target_face_mesh,
    )
    from nutella_scraper.engines.visualization.cad_reference_projector import (
        LAYER_INTERIOR_PROFILE,
    )

    view_cache = view_cache_from_viewer_dir(view_dir)
    store = ModelStore(models_root)
    model_id = view_cache.model_id
    step_path = models_root / model_id / ModelStore.REFERENCE_STEP
    if not step_path.exists():
        raise FileNotFoundError(
            f"reference.step introuvable pour le modèle {model_id} "
            f"(attendu sous {step_path})"
        )

    target_mesh = extract_target_face_mesh(step_path)
    diagnostic = target_mesh.diagnostic
    jar_vertices = np.asarray(store.get(model_id).mesh.vertices, dtype=np.float64)
    projection = TargetFaceColorProjector().project(
        face_vertices=target_mesh.vertices,
        face_triangles=target_mesh.faces,
        jar_vertices=jar_vertices,
        target_face_count=diagnostic.matching_face_count,
        target_area_mm2=diagnostic.total_target_area_mm2,
        fill_rgb_255=diagnostic.target_rgb_255,
        layer_type=LAYER_INTERIOR_PROFILE,
        include_labels=False,
    )
    overlay = ViewOverlayPayload(
        model_id=model_id,
        profile_layers=projection.profile_layers,
        top_layers=projection.top_layers,
        left_layers=projection.left_layers,
        right_layers=projection.right_layers,
        bottom_layers=projection.bottom_layers,
        coverage_score_display=0.0,
    )
    fragments = _map_overlay_fragments(OverlayRenderer().layer_fragments(overlay))
    matching_face_ids = [sample.face_id for sample in diagnostic.matching_faces]
    return {
        "model_id": model_id,
        "color_information_available": diagnostic.color_information_available,
        "total_brep_faces": diagnostic.total_faces,
        "faces_with_readable_color": diagnostic.faces_with_readable_color,
        "interior_colored_faces": diagnostic.matching_face_count,
        "target_faces": diagnostic.matching_face_count,
        "target_area_mm2": diagnostic.total_target_area_mm2,
        "target_rgb_255": list(diagnostic.target_rgb_255),
        "matching_face_ids": matching_face_ids,
        "layer": LAYER_INTERIOR_PROFILE,
        "source": "step_face_color_rgb_85_255_255",
        "overlays": {
            "side": fragments.get("side", {}),
            "top": fragments.get("top", {}),
            "left": fragments.get("left", {}),
            "right": fragments.get("right", {}),
            "bottom": fragments.get("bottom", {}),
        },
    }


def build_debug_step_face_colors_response(
    *,
    view_dir: Path,
    models_root: Path,
) -> dict[str, Any]:
    """
    Debug-only overlay of STEP faces matching RGB(85,255,255).

    Uses the same XCAF colour selection as Contour intérieur; emits the legacy
    ``target-face-colors`` layer name for isolated diagnostics.
    """
    from nutella_scraper.cad_import.step_face_color_diagnostics import (
        extract_target_face_mesh,
    )

    view_cache = view_cache_from_viewer_dir(view_dir)
    store = ModelStore(models_root)
    model_id = view_cache.model_id
    step_path = models_root / model_id / ModelStore.REFERENCE_STEP
    if not step_path.exists():
        raise FileNotFoundError(
            f"reference.step introuvable pour le modèle {model_id} "
            f"(attendu sous {step_path})"
        )

    target_mesh = extract_target_face_mesh(step_path)
    diagnostic = target_mesh.diagnostic
    jar_vertices = np.asarray(store.get(model_id).mesh.vertices, dtype=np.float64)
    projection = TargetFaceColorProjector().project(
        face_vertices=target_mesh.vertices,
        face_triangles=target_mesh.faces,
        jar_vertices=jar_vertices,
        target_face_count=diagnostic.matching_face_count,
        target_area_mm2=diagnostic.total_target_area_mm2,
        fill_rgb_255=diagnostic.target_rgb_255,
        layer_type=LAYER_TARGET_FACE_COLORS,
        include_labels=True,
    )
    overlay = ViewOverlayPayload(
        model_id=model_id,
        profile_layers=projection.profile_layers,
        top_layers=projection.top_layers,
        left_layers=projection.left_layers,
        right_layers=projection.right_layers,
        bottom_layers=projection.bottom_layers,
        coverage_score_display=0.0,
    )
    fragments = _map_overlay_fragments(OverlayRenderer().layer_fragments(overlay))
    return {
        "model_id": model_id,
        "color_information_available": diagnostic.color_information_available,
        "total_brep_faces": diagnostic.total_faces,
        "faces_with_readable_color": diagnostic.faces_with_readable_color,
        "target_faces": diagnostic.matching_face_count,
        "target_area_mm2": diagnostic.total_target_area_mm2,
        "target_rgb_255": list(diagnostic.target_rgb_255),
        "matching_face_ids": [sample.face_id for sample in diagnostic.matching_faces],
        "layer": LAYER_TARGET_FACE_COLORS,
        "overlays": {
            "side": fragments.get("side", {}),
            "top": fragments.get("top", {}),
            "left": fragments.get("left", {}),
            "right": fragments.get("right", {}),
            "bottom": fragments.get("bottom", {}),
        },
    }


def _trajectory_overlay_payload(
    internal: InternalJarSurface,
    pose_store: PoseSnapshotStore | None,
    pose_count: int,
) -> dict[str, dict[str, str]]:
    if pose_store is None or pose_count < 2:
        return {"side": {}, "top": {}, "left": {}, "right": {}, "bottom": {}}
    positions: list[tuple[float, float, float]] = []
    for index in range(pose_count):
        snapshot = pose_store.load(index)
        transform = snapshot.scraper_transform
        positions.append(
            (
                float(transform[0, 3]),
                float(transform[1, 3]),
                float(transform[2, 3]),
            )
        )
    projection = TrajectoryProjector().project(tuple(positions), internal)
    overlay = ViewOverlayPayload(
        model_id=internal.jar_id,
        profile_layers=projection.profile_layers,
        top_layers=projection.top_layers,
        left_layers=projection.left_layers,
        right_layers=projection.right_layers,
        bottom_layers=projection.bottom_layers,
        coverage_score_display=0.0,
    )
    return _map_overlay_fragments(OverlayRenderer().layer_fragments(overlay))


def _envelope_overlay_payload(
    cad_reference: CadReferenceGeometry,
    envelope_data: dict[str, Any] | None,
    *,
    jar_vertices: np.ndarray | None = None,
) -> dict[str, dict[str, str]]:
    if not envelope_data:
        return {"side": {}, "top": {}, "left": {}, "right": {}, "bottom": {}}
    slices = tuple(
        EnvelopeSlice(
            y_mm=float(entry["y_mm"]),
            max_radial_mm=float(entry["max_radial_mm"]),
        )
        for entry in envelope_data.get("slices", [])
    )
    envelope = InteriorEnvelope(
        jar_id=str(envelope_data.get("jar_id", cad_reference.model_id)),
        y_min_mm=float(envelope_data.get("y_min_mm", 0.0)),
        y_max_mm=float(envelope_data.get("y_max_mm", 0.0)),
        neck_radius_mm=float(envelope_data.get("neck_radius_mm", 0.0)),
        clearance_mm=float(envelope_data.get("clearance_mm", 0.0)),
        slices=slices,
    )
    projection = EnvelopeProjector().project(
        envelope,
        cad_reference,
        jar_vertices=jar_vertices,
    )
    overlay = ViewOverlayPayload(
        model_id=cad_reference.model_id,
        profile_layers=projection.profile_layers,
        top_layers=projection.top_layers,
        coverage_score_display=0.0,
    )
    return _map_overlay_fragments(OverlayRenderer().layer_fragments(overlay))


def _map_overlay_fragments(fragments: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map renderer view keys (profile/top/left/right) to viewer keys."""
    return {
        "side": fragments.get("profile", {}),
        "top": fragments.get("top", {}),
        "left": fragments.get("left", {}),
        "right": fragments.get("right", {}),
        "bottom": fragments.get("bottom", {}),
    }
