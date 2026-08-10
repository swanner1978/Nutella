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
from nutella_scraper.domain.models.internal_jar_surface import InternalJarSurface
from nutella_scraper.domain.models.contact import (
    CollisionResult,
    ContactSimulationConfig,
    TrajectoryConfig,
)
from nutella_scraper.domain.models.envelope import EnvelopeSlice, InteriorEnvelope
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
from nutella_scraper.engines.visualization.trajectory_projector import TrajectoryProjector
from nutella_scraper.engines.compute.envelope_builder import EnvelopeBuilder
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
) -> dict[str, Any]:
    """Build Scraper3D geometry only — no contact simulation or overlay side-effects."""
    view_cache = view_cache_from_viewer_dir(view_dir)
    store = ModelStore(models_root)
    jar = store.get(view_cache.model_id)
    internal = store.get_internal(view_cache.model_id)
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
    try:
        posed_mesh = scraper_builder.build_posed(
            DEFAULT_SCRAPER_GEOMETRY,
            DEFAULT_SCRAPER_POSE,
        )
        scraper_pipeline["stages"]["mesh_built"] = {
            "ok": bool(posed_mesh.volume > 0.0),
            "vertex_count": len(posed_mesh.vertices),
            "face_count": len(posed_mesh.faces),
            "volume_mm3": float(posed_mesh.volume),
        }
        if posed_mesh.volume <= 0.0:
            scraper_pipeline["loss_stage"] = "mesh_built"
    except Exception as exc:
        scraper_pipeline["stages"]["mesh_built"] = {"ok": False, "error": str(exc)}
        scraper_pipeline["loss_stage"] = "mesh_built"
        raise

    transform = pose_matrix(DEFAULT_SCRAPER_POSE)
    scraper_projection = ScraperResultProjector().project(
        scraper_vertices=posed_mesh.vertices,
        scraper_faces=posed_mesh.faces,
        internal=internal,
    )
    overlay = ViewOverlayPayload(
        model_id=view_cache.model_id,
        profile_layers=scraper_projection.profile_layers,
        top_layers=scraper_projection.top_layers,
        left_layers=scraper_projection.left_layers,
        right_layers=scraper_projection.right_layers,
        coverage_score_display=0.0,
    )
    fragments = OverlayRenderer().layer_fragments(overlay)
    return {
        "model_id": view_cache.model_id,
        "pose": pose_to_dict(DEFAULT_SCRAPER_POSE),
        "scraper_transform": transform.tolist(),
        "scraper_pipeline": scraper_pipeline,
        "scraper": {
            "vertex_count": scraper_projection.vertex_count,
            "face_count": scraper_projection.face_count,
            "provenance": "procedural_build",
        },
        "overlays": {
            "side": fragments["profile"],
            "top": fragments["top"],
            "left": fragments["left"],
            "right": fragments["right"],
        },
    }


def build_interior_contour_response(
    *,
    view_dir: Path,
    models_root: Path,
    clearance_mm: float | None = None,
) -> dict[str, Any]:
    """Compute the CAD interior contour without contact simulation."""
    del clearance_mm  # reserved; contour uses zero clearance (no artificial offset)
    view_cache = view_cache_from_viewer_dir(view_dir)
    store = ModelStore(models_root)
    cad_reference = store.get_cad_reference(view_cache.model_id)
    jar_vertices = np.asarray(store.get(view_cache.model_id).mesh.vertices, dtype=np.float64)
    projection = EnvelopeProjector().project_geometry(
        cad_reference,
        jar_vertices=jar_vertices,
    )
    overlay = ViewOverlayPayload(
        model_id=view_cache.model_id,
        profile_layers=projection.profile_layers,
        top_layers=projection.top_layers,
        coverage_score_display=0.0,
    )
    envelope_payload = _map_overlay_fragments(OverlayRenderer().layer_fragments(overlay))
    return {
        "model_id": view_cache.model_id,
        "overlays": {
            "side": envelope_payload.get("side", {}),
            "top": envelope_payload.get("top", {}),
            "left": envelope_payload.get("left", {}),
            "right": envelope_payload.get("right", {}),
        },
        "cad_reference": {
            "source": cad_reference.metadata.get("source", "opencascade_brep"),
            "step_sha256": cad_reference.step_sha256,
            "inner_face_count": cad_reference.inner_face_count,
            "profile_edge_count": (
                cad_reference.profile_contour.edge_count
                if cad_reference.profile_contour
                else 0
            ),
            "top_edge_count": (
                cad_reference.top_contour.edge_count if cad_reference.top_contour else 0
            ),
            "profile_plane": (
                cad_reference.profile_contour.plane if cad_reference.profile_contour else None
            ),
            "top_plane": (
                cad_reference.top_contour.plane if cad_reference.top_contour else None
            ),
        },
    }


def _trajectory_overlay_payload(
    internal: InternalJarSurface,
    pose_store: PoseSnapshotStore | None,
    pose_count: int,
) -> dict[str, dict[str, str]]:
    if pose_store is None or pose_count < 2:
        return {"side": {}, "top": {}, "left": {}, "right": {}}
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
        return {"side": {}, "top": {}, "left": {}, "right": {}}
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
    }
