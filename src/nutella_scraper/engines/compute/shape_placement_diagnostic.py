"""Compare A0 and a short 2 mm blade in the same physical pose.

Does not change collision, the 608-point cloud, or the jar. Placement only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

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
    envelope_contact_frame,
    rigid_transform_between_frames,
    transform_points,
)
from nutella_scraper.engines.compute.shape_families import (
    DEFAULT_SCRAPER_LENGTH_MM,
    FAMILY_BY_ID,
    SCRAPER_THICKNESS_MM,
    SCRAPER_WIDTH_MM,
    build_sagittal_frame,
    sample_profile,
)
from nutella_scraper.engines.compute.shape_materialize import (
    manufacturing_parameters,
    materialize_a0,
    materialize_profile,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import (
    reference_scraper_parameters,
)

DIAGNOSTIC_LABEL = "SHAPE_PLACEMENT_DIAGNOSTIC"
OPENING_AZIMUTH_DEG = 0.0


def _bbox(vertices: NDArray[np.float64]) -> dict[str, list[float]]:
    pts = np.asarray(vertices, dtype=np.float64)
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    return {
        "min_xyz_mm": [float(lo[0]), float(lo[1]), float(lo[2])],
        "max_xyz_mm": [float(hi[0]), float(hi[1]), float(hi[2])],
        "extent_xyz_mm": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])],
    }


def local_section_extents_mm(artifact: RigidScraperArtifact) -> dict[str, float]:
    """Extents in the design frame: axis0=thickness, axis1=length, axis2=width."""
    rotation = np.asarray(artifact.design_frame.rotation, dtype=np.float64)
    origin = np.asarray(artifact.design_frame.origin_mm, dtype=np.float64)
    verts = np.asarray(artifact.mesh.vertices, dtype=np.float64)
    local = (verts - origin) @ rotation
    ptp = np.ptp(local, axis=0)
    return {
        "thickness_mm": float(ptp[0]),
        "length_mm": float(ptp[1]),
        "width_mm": float(ptp[2]),
    }


def _pose_snapshot(
    *,
    name: str,
    artifact: RigidScraperArtifact,
    surface: InteriorSurfaceReference,
    y_mm: float,
    azimuth_deg: float,
) -> dict[str, Any]:
    params = (
        reference_scraper_parameters(surface)
        if name == "A0"
        else manufacturing_parameters(surface, length_mm=DEFAULT_SCRAPER_LENGTH_MM)
    )
    params = params.with_updates(
        position_z_mm=float(y_mm),
        surface_progress_deg=float(azimuth_deg),
    )
    mesh = surface.to_trimesh()
    bind_envelope_proximity(mesh)
    target = envelope_contact_frame(
        surface,
        params,
        surface_progress_deg=float(azimuth_deg),
        surface_mesh=mesh,
    )
    src = np.asarray(artifact.mesh.vertices, dtype=np.float64)
    faces = np.asarray(artifact.mesh.faces, dtype=np.int64)
    edges = np.asarray(artifact.mesh.edges_unique, dtype=np.int64)
    chosen = None
    report = None
    neighborhood_used = False
    posed = src
    for index, frame in enumerate(rigid_pose_neighborhood(target)):
        try:
            transform = rigid_transform_between_frames(artifact.design_frame, frame)
        except np.linalg.LinAlgError:
            continue
        posed = transform_points(src, transform)
        report = evaluate_envelope_collision(
            artifact.mesh,
            surface,
            params,
            surface_mesh=mesh,
            vertices=posed,
            faces=faces,
            edges_unique=edges,
        )
        if report.admissible:
            chosen = frame
            neighborhood_used = index > 0
            break
    if chosen is None:
        chosen = target
        transform = rigid_transform_between_frames(artifact.design_frame, target)
        posed = transform_points(src, transform)
        if report is None:
            report = evaluate_envelope_collision(
                artifact.mesh,
                surface,
                params,
                surface_mesh=mesh,
                vertices=posed,
                faces=faces,
                edges_unique=edges,
            )
    _closest, distances, _ids = closest_on_envelope_surface(mesh, posed)
    extents = local_section_extents_mm(artifact)
    rest = np.asarray(artifact.mesh.vertices, dtype=np.float64)
    origin = tuple(float(v) for v in chosen.origin_mm)
    length_axis = tuple(float(v) for v in chosen.rotation[:, 1])
    return {
        "name": name,
        "y_mm": float(y_mm),
        "azimuth_deg": float(azimuth_deg),
        "origin_mm": list(origin),
        "length_axis": list(length_axis),
        "design_bbox": _bbox(rest),
        "posed_bbox": _bbox(posed),
        "local_extents_mm": extents,
        "admissible": bool(report.admissible),
        "neighborhood_used": bool(neighborhood_used),
        "min_signed_interior_mm": float(report.min_signed_interior_mm),
        "min_unsigned_distance_mm": float(report.min_unsigned_distance_mm),
        "min_envelope_distance_mm": float(np.min(distances)) if len(distances) else None,
        "vertex_count": int(len(posed)),
    }


def compare_a0_and_straight40(
    surface: InteriorSurfaceReference,
    *,
    azimuth_deg: float = OPENING_AZIMUTH_DEG,
) -> dict[str, Any]:
    """Same (Y, azimuth) for A0 40x2.5 and straight 40x2.0. HEURISTIC diagnostic."""
    opening_y, _lower, _span = scraper_length_span(surface)
    a0 = materialize_a0(surface)
    frame = build_sagittal_frame(surface)
    family = FAMILY_BY_ID["straight"]
    window = frame.window_for_length(DEFAULT_SCRAPER_LENGTH_MM)
    profile = sample_profile(
        family,
        family.default_params(window),
        frame,
        length_mm=DEFAULT_SCRAPER_LENGTH_MM,
    )
    blade = materialize_profile(
        profile, surface, length_mm=DEFAULT_SCRAPER_LENGTH_MM
    )
    a0_row = _pose_snapshot(
        name="A0",
        artifact=a0,
        surface=surface,
        y_mm=float(opening_y),
        azimuth_deg=float(azimuth_deg),
    )
    blade_row = _pose_snapshot(
        name="straight_40_2mm",
        artifact=blade,
        surface=surface,
        y_mm=float(opening_y),
        azimuth_deg=float(azimuth_deg),
    )
    blade_ext = blade_row["local_extents_mm"]
    anomalies: list[str] = []
    if abs(float(blade_ext["thickness_mm"]) - SCRAPER_THICKNESS_MM) > 0.6:
        anomalies.append(
            f"straight 40 mm thickness {blade_ext['thickness_mm']:.2f} mm "
            f"≠ {SCRAPER_THICKNESS_MM:.1f} mm"
        )
    if abs(float(blade_ext["width_mm"]) - SCRAPER_WIDTH_MM) > 0.6:
        anomalies.append(
            f"straight 40 mm width {blade_ext['width_mm']:.2f} mm "
            f"≠ {SCRAPER_WIDTH_MM:.1f} mm"
        )
    if abs(float(blade_ext["length_mm"]) - DEFAULT_SCRAPER_LENGTH_MM) > 8.0:
        anomalies.append(
            f"straight 40 mm length extent {blade_ext['length_mm']:.2f} mm "
            f"far from {DEFAULT_SCRAPER_LENGTH_MM:.0f} mm"
        )
    origin_delta = float(
        np.linalg.norm(
            np.asarray(a0_row["origin_mm"]) - np.asarray(blade_row["origin_mm"])
        )
    )
    return {
        "label": DIAGNOSTIC_LABEL,
        "optimization_label": "HEURISTIC",
        "opening_y_mm": float(opening_y),
        "azimuth_deg": float(azimuth_deg),
        "origin_delta_mm": origin_delta,
        "a0": a0_row,
        "straight_40_2mm": blade_row,
        "profile_length_mm": float(profile.length_mm),
        "anomalies": anomalies,
        "same_pose": True,
        "disclaimer": (
            "Diagnostic de placement. Ne modifie pas le moteur de collision."
        ),
    }
