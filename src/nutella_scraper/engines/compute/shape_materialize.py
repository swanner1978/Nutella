"""Build a rigid 3D scraper from a sagittal profile. Reuses the existing loft.

The centreline is NOT snapped onto the envelope: a straight chord stays a
chord. Collision / contact remain the existing engines' job.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import (
    NUMERIC_GAP_MM,
    EnvelopeStation,
    ScraperEnvelopePath,
    assert_no_inverted_station_pairs,
    interior_centroid_mm,
    normalize_row_orders,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    RigidScraperArtifact,
    build_rigid_scraper_artifact,
    build_rigid_scraper_artifact_from_path,
)
from nutella_scraper.engines.compute.shape_families import (
    BLADE_THICKNESS_MM,
    BLADE_WIDTH_MM,
    SampledProfile,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import (
    reference_scraper_parameters,
)

WIDTH_SAMPLES = 33


def manufacturing_parameters(
    surface: InteriorSurfaceReference,
    *,
    length_mm: float,
) -> ScraperParameters:
    base = reference_scraper_parameters(surface)
    return base.with_updates(
        width_mm=BLADE_WIDTH_MM,
        thickness_mm=BLADE_THICKNESS_MM,
        length_mm=float(length_mm),
        bevel_angle_deg=0.0,
        relief_angle_deg=0.0,
        helix_rate_deg_per_mm=0.0,
        clearance_mm=0.0,
    )


def profile_fingerprint(profile: SampledProfile) -> str:
    payload = (
        f"{profile.family_id}|L={float(profile.length_mm):.6g}|"
        + "|".join(f"{v:.10g}" for v in profile.parameters)
        + "|"
        + hashlib.sha256(np.round(profile.points_mm, 5).tobytes()).hexdigest()[:16]
    )
    return payload


def _unit(vector: NDArray[np.float64], fallback: NDArray[np.float64]) -> NDArray[np.float64]:
    nrm = float(np.linalg.norm(vector))
    if nrm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / nrm


def envelope_path_from_profile(
    profile: SampledProfile,
    surface: InteriorSurfaceReference,
    parameters: ScraperParameters,
) -> ScraperEnvelopePath:
    curve = np.asarray(profile.points_mm, dtype=np.float64)
    if len(curve) < 2:
        raise ValueError("Profile needs at least two samples to loft")
    half_w = 0.5 * float(parameters.width_mm)
    clearance = float(parameters.clearance_mm) + NUMERIC_GAP_MM
    interior = interior_centroid_mm(surface)
    wall_rows: list[NDArray[np.float64]] = []
    normal_rows: list[NDArray[np.float64]] = []
    tangents: list[NDArray[np.float64]] = []
    for index, point in enumerate(curve):
        if index + 1 < len(curve):
            raw_t = curve[index + 1] - point
        else:
            raw_t = point - curve[index - 1]
        tangent = _unit(raw_t, np.array([0.0, -1.0, 0.0], dtype=np.float64))
        axis_dir = np.array(
            [interior[0] - point[0], 0.0, interior[2] - point[2]],
            dtype=np.float64,
        )
        normal = _unit(axis_dir, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        normal = normal - tangent * float(np.dot(normal, tangent))
        normal = _unit(normal, np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        width = np.cross(normal, tangent)
        width = _unit(width, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        tangent = np.cross(width, normal)
        tangent = _unit(tangent, np.array([0.0, -1.0, 0.0], dtype=np.float64))
        alphas = np.linspace(-1.0, 1.0, WIDTH_SAMPLES)
        wall = point[None, :] + (alphas * half_w)[:, None] * width[None, :]
        wall_rows.append(np.asarray(wall, dtype=np.float64))
        normal_rows.append(np.tile(normal, (WIDTH_SAMPLES, 1)))
        tangents.append(tangent)
    normalize_row_orders(wall_rows, normal_rows)
    stations: list[EnvelopeStation] = []
    mid = WIDTH_SAMPLES // 2
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    )
    s_mid = 0.5 * float(arc[-1])
    for index in range(len(curve)):
        wall_row = wall_rows[index]
        normal_row = normal_rows[index]
        stations.append(
            EnvelopeStation(
                s_mm=float(arc[index] - s_mid),
                y_mm=float(wall_row[mid, 1]),
                tip_points_mm=wall_row + normal_row * clearance,
                inward_normals=normal_row,
                tangent_length=tangents[index],
                wall_points_mm=wall_row,
            )
        )
    path = ScraperEnvelopePath(
        stations=tuple(stations),
        source=SOURCE_INTERIOR_PRODUCT_SURFACE,
    )
    assert_no_inverted_station_pairs(path.stations)
    return path


def materialize_profile(
    profile: SampledProfile,
    surface: InteriorSurfaceReference,
    *,
    length_mm: float,
) -> RigidScraperArtifact:
    parameters = manufacturing_parameters(surface, length_mm=length_mm)
    path = envelope_path_from_profile(profile, surface, parameters)
    return build_rigid_scraper_artifact_from_path(
        surface,
        parameters,
        path,
        shape_fingerprint=profile_fingerprint(profile),
    )


def materialize_a0(surface: InteriorSurfaceReference) -> RigidScraperArtifact:
    """Historical A0 manufacturing solid. Not an A0 point-grid simulation."""
    return build_rigid_scraper_artifact(surface, reference_scraper_parameters(surface))
