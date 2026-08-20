"""Visualization-only control cage overlay for scraper A.

The cage is a light lattice of contact samples on InteriorSurfaceReference.
It is not a physical mesh, not an editor, and it must not be imported by
collision or rigid-motion code.

Search layout (future optimiser, not blade width):
    5 rows left  +  centreline  +  5 rows right
    offsets −50 … 0 … +50 mm (1 cm between rows)
The yellow scraper is a separate thin loft along the centreline only.

Side rows share one Bishop-transported W per station and sit on the jar
parallel (rotation about +Y). The last-station jumps came from two local
bugs, not from the cage layout itself:

* rotating the *raw* centreline azimuth near the axis (mesh wobble becomes
  a large swing once ``angle = arc / r``);
* ``mesh.nearest`` on each side point, which snaps onto a neighbouring
  triangle and breaks the column.

Placement now uses the *opening* meridian (stable azimuth) and never
projects a side point with independent nearest.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    EnvelopeStation,
    ScraperEnvelopePath,
)

# Overlay identity — not a manufacturing solid.
CAGE_KIND = "control_polyline_overlay"
CAGE_PROFILE = "A"
# Search-space rows (cm): −5 −4 −3 −2 −1  0  +1  +2  +3  +4  +5
CAGE_ROW_OFFSETS_MM: tuple[float, ...] = (
    -50.0,
    -40.0,
    -30.0,
    -20.0,
    -10.0,
    0.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
)
CAGE_CENTER_ROW_INDEX = 5
# Keep wall-span (~50 mm on a ~50 mm radius) and only compress in the fillet.
_MAX_HALF_ANGLE_RAD = 1.0
_AXIS_COLLAPSE_MM = 1.0
_MAX_SCALE_STEP = 0.12
_JAR_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def build_control_cage_overlay(
    path: ScraperEnvelopePath,
    surface: InteriorSurfaceReference | None = None,
) -> dict[str, Any]:
    """Derive a non-editable 11-row cage from scraper A's contact centreline.

    The centreline is ``path.wall_curve_mm`` (validated trajectory A) and is
    copied unchanged. Side rows of a station share one transported W and
    one locked meridian azimuth. After construction, each lateral trajectory
    is radially projected onto InteriorSurfaceReference (constant Y).
    Independent nearest-point is not used to place a side sample.
    """
    if not path.stations:
        raise ValueError("Envelope path has no stations for a control cage")

    spine = np.asarray(path.wall_curve_mm, dtype=np.float64)
    if len(spine) < 2:
        raise ValueError("Control cage needs at least two contact samples")

    axis_xz = _jar_axis_xz(surface, spine)
    widths = _transported_width_dirs(spine, path.stations, axis_xz)
    radii = _tube_radii_mm(spine, axis_xz)
    meridian_az = _locked_meridian_azimuth(spine, axis_xz)

    n_stations = len(spine)
    n_rows = len(CAGE_ROW_OFFSETS_MM)
    grid = np.zeros((n_rows, n_stations, 3), dtype=np.float64)
    for i in range(n_stations):
        origin = spine[i]
        width = widths[i]
        radius = max(float(radii[i]), 0.5)
        for r, offset in enumerate(CAGE_ROW_OFFSETS_MM):
            if abs(offset) < 1e-9:
                grid[r, i] = origin
            else:
                grid[r, i] = _point_on_locked_parallel(
                    origin,
                    width_dir=width,
                    arc_mm=float(offset),
                    radius_mm=radius,
                    axis_xz=axis_xz,
                    meridian_az=meridian_az,
                )

    grid[CAGE_CENTER_ROW_INDEX] = spine

    grid = np.round(grid, 4)
    rows = [grid[r].tolist() for r in range(n_rows)]
    all_points = grid.reshape(-1, 3)
    spine_list = np.round(spine, 4).tolist()
    payload = {
        "kind": CAGE_KIND,
        "profile": CAGE_PROFILE,
        "editable": False,
        "lofted": False,
        "source": path.source or SOURCE_INTERIOR_PRODUCT_SURFACE,
        "row_count": n_rows,
        "center_row_index": CAGE_CENTER_ROW_INDEX,
        "row_offsets_mm": list(CAGE_ROW_OFFSETS_MM),
        "centerline_mm": spine_list,
        "nominal_point_count": int(len(all_points)),
        "nominal_candidates": rows,
        "point_count": int(len(all_points)),
        "points_mm": all_points.tolist(),
        "polylines_mm": rows,
        "candidates": rows,
    }
    if surface is None:
        return payload
    from nutella_scraper.engines.visualization.scraper_shape_space import (
        filter_control_cage,
    )

    return filter_control_cage(payload, surface)


def _jar_axis_xz(
    surface: InteriorSurfaceReference | None,
    spine: NDArray[np.float64],
) -> NDArray[np.float64]:
    if surface is not None and surface.vertex_count > 0:
        verts = np.asarray(surface.vertices, dtype=np.float64)
        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
        return 0.5 * (mins[[0, 2]] + maxs[[0, 2]])
    del spine
    return np.array([0.0, 0.0], dtype=np.float64)


def _smooth_radii(radii: NDArray[np.float64]) -> NDArray[np.float64]:
    """3-point average — kills single-station tube-radius spikes at the floor."""
    raw = np.asarray(radii, dtype=np.float64)
    if len(raw) < 3:
        return raw.copy()
    smooth = raw.copy()
    smooth[1:-1] = (raw[:-2] + raw[1:-1] + raw[2:]) / 3.0
    return smooth


def _locked_meridian_azimuth(
    spine: NDArray[np.float64],
    axis_xz: NDArray[np.float64],
) -> float:
    """Azimuth of the opening (largest tube radius) — not the noisy floor point."""
    radii = _tube_radii_mm(spine, axis_xz)
    index = int(np.argmax(radii))
    dx = float(spine[index, 0]) - float(axis_xz[0])
    dz = float(spine[index, 2]) - float(axis_xz[1])
    if math.hypot(dx, dz) < 1e-9:
        return 0.0
    return math.atan2(dz, dx)


def _tube_radii_mm(
    spine: NDArray[np.float64],
    axis_xz: NDArray[np.float64],
) -> NDArray[np.float64]:
    dx = spine[:, 0] - float(axis_xz[0])
    dz = spine[:, 2] - float(axis_xz[1])
    return np.hypot(dx, dz)


def _offset_scales(radii: NDArray[np.float64]) -> NDArray[np.float64]:
    """Uniform row scale per station: 1 on the wall, tapering in the fillet."""
    max_offset = max(abs(float(CAGE_ROW_OFFSETS_MM[0])), 1e-6)
    raw = np.where(
        radii < _AXIS_COLLAPSE_MM,
        0.0,
        np.minimum(1.0, radii * _MAX_HALF_ANGLE_RAD / max_offset),
    )
    # Stations run floor → opening. Seed at the opening (stable radius)
    # and walk toward the floor so the taper is monotone, not jittery.
    scales = np.asarray(raw, dtype=np.float64).copy()
    for i in range(len(scales) - 2, -1, -1):
        scales[i] = min(float(raw[i]), float(scales[i + 1]) + _MAX_SCALE_STEP)
    return scales


def _transported_width_dirs(
    spine: NDArray[np.float64],
    stations: tuple[EnvelopeStation, ...],
    axis_xz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bishop-transported W along the centreline.

    Seed W at the opening (large tube radius) and carry it toward the
    floor. Independent ``N×T`` at the floor is unstable as C approaches
    the axis and is what produced 180° flips.
    """
    tangents = _spine_tangents(spine)
    widths = np.zeros_like(spine)
    seed = len(spine) - 1
    widths[seed] = _seed_width_dir(stations[seed], tangents[seed], spine[seed], axis_xz)
    for i in range(seed - 1, -1, -1):
        carried = _parallel_transport(widths[i + 1], tangents[i + 1], tangents[i])
        if float(np.dot(carried, widths[i + 1])) < 0.0:
            carried = -carried
        widths[i] = carried
    for i in range(seed + 1, len(spine)):
        carried = _parallel_transport(widths[i - 1], tangents[i - 1], tangents[i])
        if float(np.dot(carried, widths[i - 1])) < 0.0:
            carried = -carried
        widths[i] = carried
    return widths


def _spine_tangents(spine: NDArray[np.float64]) -> NDArray[np.float64]:
    tangents = np.zeros_like(spine)
    tangents[0] = spine[1] - spine[0]
    tangents[-1] = spine[-1] - spine[-2]
    if len(spine) > 2:
        tangents[1:-1] = spine[2:] - spine[:-2]
    for i, raw in enumerate(tangents):
        tangents[i] = _safe_unit(raw, fallback=_JAR_UP)
    return tangents


def _seed_width_dir(
    station: EnvelopeStation,
    tangent: NDArray[np.float64],
    origin: NDArray[np.float64],
    axis_xz: NDArray[np.float64],
) -> NDArray[np.float64]:
    width = _station_width_dir(station)
    width = width - tangent * float(np.dot(width, tangent))
    width = _safe_unit(width, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    radial = np.array(
        [
            float(origin[0]) - float(axis_xz[0]),
            0.0,
            float(origin[2]) - float(axis_xz[1]),
        ],
        dtype=np.float64,
    )
    azimuthal = np.cross(_JAR_UP, radial)
    if float(np.linalg.norm(azimuthal)) > 1e-9 and float(np.dot(width, azimuthal)) < 0.0:
        width = -width
    return width


def _station_width_dir(station: EnvelopeStation) -> NDArray[np.float64]:
    mid = station.inward_normals.shape[0] // 2
    normal = np.asarray(station.inward_normals[mid], dtype=np.float64)
    tangent = np.asarray(station.tangent_length, dtype=np.float64)
    width = np.cross(normal, tangent)
    if float(np.linalg.norm(width)) <= 1e-9:
        width = np.cross(normal, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    if float(np.linalg.norm(width)) <= 1e-9:
        width = np.cross(normal, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    width = width / max(float(np.linalg.norm(width)), 1e-9)
    chord = np.asarray(
        station.wall_points_mm[-1] - station.wall_points_mm[0], dtype=np.float64
    )
    if float(np.linalg.norm(chord)) > 1e-9 and float(np.dot(width, chord)) < 0.0:
        width = -width
    return width


def _parallel_transport(
    vector: NDArray[np.float64],
    tangent_from: NDArray[np.float64],
    tangent_to: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate ``vector`` with T so it stays ⟂ T without independent rebuilds."""
    t0 = _safe_unit(tangent_from, fallback=_JAR_UP)
    t1 = _safe_unit(tangent_to, fallback=_JAR_UP)
    vec = np.asarray(vector, dtype=np.float64)
    axis = np.cross(t0, t1)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
    if sine < 1e-12:
        if cosine < 0.0:
            vec = -vec
    else:
        vec = _rodrigues(vec, axis / sine, math.atan2(sine, cosine))
    vec = vec - t1 * float(np.dot(vec, t1))
    return _safe_unit(vec, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))


def _rodrigues(
    vector: NDArray[np.float64],
    axis: NDArray[np.float64],
    angle: float,
) -> NDArray[np.float64]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        vector * cos_a
        + np.cross(axis, vector) * sin_a
        + axis * float(np.dot(axis, vector)) * (1.0 - cos_a)
    )


def _point_on_locked_parallel(
    origin: NDArray[np.float64],
    *,
    width_dir: NDArray[np.float64],
    arc_mm: float,
    radius_mm: float,
    axis_xz: NDArray[np.float64],
    meridian_az: float,
) -> NDArray[np.float64]:
    """Offset on the jar parallel using the opening meridian, not raw C azimuth.

    Independent nearest-point is never used. Y stays the station's Y so the
    11 samples share one transverse frame. Azimuth is ``meridian_az + θ``
    with the same θ family at every station, so a column stays a meridian.
    """
    radius = max(float(radius_mm), 0.5)
    # Constant azimuth family: θ = offset / 50 mm. Local radius only seeds
    # the parallel; it must not collapse a side row onto the centreline.
    theta = float(arc_mm) / max(abs(float(CAGE_ROW_OFFSETS_MM[0])), 1e-6)
    theta = float(np.clip(theta, -_MAX_HALF_ANGLE_RAD, _MAX_HALF_ANGLE_RAD))
    d_daz = np.array(
        [
            -radius * math.sin(meridian_az),
            0.0,
            radius * math.cos(meridian_az),
        ],
        dtype=np.float64,
    )
    azimuthal = np.asarray(width_dir, dtype=np.float64).copy()
    azimuthal[1] = 0.0
    if float(np.linalg.norm(azimuthal)) > 1e-9 and float(np.dot(d_daz, azimuthal)) < 0.0:
        theta = -theta
    azimuth = meridian_az + theta
    point = np.asarray(origin, dtype=np.float64).copy()
    point[0] = float(axis_xz[0]) + radius * math.cos(azimuth)
    point[2] = float(axis_xz[1]) + radius * math.sin(azimuth)
    return point


def _safe_unit(
    vector: NDArray[np.float64],
    *,
    fallback: NDArray[np.float64],
) -> NDArray[np.float64]:
    vec = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=np.float64).copy()
    return vec / norm
