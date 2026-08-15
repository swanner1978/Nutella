"""Build viewer overlays for one exact captured trajectory pose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nutella_scraper.cad_import.model_store import ModelStore
from nutella_scraper.domain.models.contact import ContactOverlayData, ContactResult
from nutella_scraper.domain.models.views import ViewOverlayPayload
from nutella_scraper.engines.visualization.contact_result_projector import (
    ContactResultProjector,
)
from nutella_scraper.engines.visualization.overlay_renderer import OverlayRenderer
from nutella_scraper.engines.visualization.pose_snapshot_store import PoseSnapshotStore
from nutella_scraper.engines.visualization.scraper_result_projector import (
    ScraperResultProjector,
)
from nutella_scraper.engines.compute.scraper_transform import pose_from_matrix, pose_to_dict


def build_pose_visualization_response(
    *,
    pose_snapshot_dir: Path,
    view_dir: Path,
    models_root: Path,
    pose_index: int,
) -> dict[str, Any]:
    """Project contact and scraper geometry captured for exactly one pose."""
    metadata = json.loads((view_dir / "metadata.json").read_text(encoding="utf-8"))
    model_id = str(metadata["model_id"])
    jar = ModelStore(models_root).get(model_id)
    internal = ModelStore(models_root).get_internal(model_id)
    snapshot = PoseSnapshotStore(pose_snapshot_dir).load(pose_index)

    coverage = np.asarray(snapshot.face_coverage, dtype=np.bool_)
    touched = frozenset(int(index) for index in np.flatnonzero(coverage))
    untouched = frozenset(int(index) for index in np.flatnonzero(~coverage))
    overlay = ContactOverlayData(
        contact_points=snapshot.contact_points,
        face_coverage=tuple(bool(value) for value in coverage),
        min_distance_per_face_mm=tuple(float(value) for value in snapshot.face_distances),
        scraper_pose_count=1,
    )
    contact = ContactResult(
        model_id=f"pose-{pose_index}",
        jar_id=jar.id,
        coverage_score=0.0,
        touched_face_ids=touched,
        untouched_face_ids=untouched,
        contact_distance_map=snapshot.face_distances,
        trajectory_pose_count=1,
        overlay=overlay,
        collision=snapshot.collision,
        diagnostics={"pose_index": pose_index, "source": "captured_3d_pose"},
    )
    contact_projection = ContactResultProjector().project(
        contact,
        None,
        jar,
        internal=internal,
    )
    scraper_projection = ScraperResultProjector().project(
        scraper_vertices=snapshot.scraper_vertices,
        scraper_faces=snapshot.scraper_faces,
        internal=internal,
    )
    profile_layers = tuple(
        sorted(
            (*contact_projection.profile_layers, *scraper_projection.profile_layers),
            key=lambda layer: layer.z_index,
        )
    )
    top_layers = tuple(
        sorted(
            (*contact_projection.top_layers, *scraper_projection.top_layers),
            key=lambda layer: layer.z_index,
        )
    )
    left_layers = tuple(
        sorted(
            (*contact_projection.left_layers, *scraper_projection.left_layers),
            key=lambda layer: layer.z_index,
        )
    )
    right_layers = tuple(
        sorted(
            (*contact_projection.right_layers, *scraper_projection.right_layers),
            key=lambda layer: layer.z_index,
        )
    )
    bottom_layers = tuple(
        sorted(
            (*contact_projection.bottom_layers, *scraper_projection.bottom_layers),
            key=lambda layer: layer.z_index,
        )
    )
    combined = ViewOverlayPayload(
        model_id=model_id,
        profile_layers=profile_layers,
        top_layers=top_layers,
        left_layers=left_layers,
        right_layers=right_layers,
        bottom_layers=bottom_layers,
        coverage_score_display=0.0,
    )
    fragments = OverlayRenderer().layer_fragments(combined)
    pose = pose_from_matrix(snapshot.scraper_transform)
    return {
        "model_id": model_id,
        "pose_index": snapshot.index,
        "pose_count": snapshot.total,
        "pose": pose_to_dict(pose),
        "scraper_transform": snapshot.scraper_transform.tolist(),
        "scraper": {
            "vertex_count": scraper_projection.vertex_count,
            "face_count": scraper_projection.face_count,
            "provenance": "captured_exact_simulation_mesh",
        },
        "overlays": {
            "side": fragments["profile"],
            "top": fragments["top"],
            "left": fragments["left"],
            "right": fragments["right"],
            "bottom": fragments["bottom"],
        },
    }
