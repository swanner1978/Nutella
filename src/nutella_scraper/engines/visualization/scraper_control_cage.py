"""Visualization-only control cage overlay for scraper A.

The cage is a light lattice of contact samples on InteriorSurfaceReference.
It is not a physical mesh, not an editor, and it must not be imported by
collision or rigid-motion code.

Search layout (future optimiser, not blade width):
    5 rows left  +  centreline  +  5 rows right
    offsets −50 … 0 … +50 mm (1 cm between rows)
The yellow scraper is a separate thin loft along the centreline only.
"""

from __future__ import annotations

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
_MAX_SEED_DIST_MM = 8.0
_MAX_STEP_JUMP_MM = 8.0
_WALK_STEP_MM = 2.5


def build_control_cage_overlay(
    path: ScraperEnvelopePath,
    surface: InteriorSurfaceReference | None = None,
) -> dict[str, Any]:
    """Derive a non-editable 11-row cage from scraper A's contact centreline.

    The centreline is ``path.wall_curve_mm`` (validated trajectory A). Side
    rows are walks on InteriorSurfaceReference along the local width
    direction. Offsets that leave the envelope are dropped, never placed
    in the jar volume.
    """
    if not path.stations:
        raise ValueError("Envelope path has no stations for a control cage")

    spine = np.asarray(path.wall_curve_mm, dtype=np.float64)
    if len(spine) < 2:
        raise ValueError("Control cage needs at least two contact samples")

    mesh = surface.to_trimesh() if surface is not None else None
    rows: list[list[list[float]]] = []
    candidates: list[list[list[float] | None]] = []
    all_points: list[NDArray[np.float64]] = []

    for offset in CAGE_ROW_OFFSETS_MM:
        row_pts: list[list[float]] = []
        row_cand: list[list[float] | None] = []
        for station, origin in zip(path.stations, spine, strict=True):
            if abs(offset) < 1e-9:
                point: NDArray[np.float64] | None = np.asarray(
                    origin, dtype=np.float64
                )
            elif mesh is None:
                point = None
            else:
                point = _offset_on_envelope(mesh, origin, station, float(offset))
            if point is None:
                row_cand.append(None)
                continue
            xyz = np.round(point, 4)
            row_pts.append(xyz.tolist())
            row_cand.append(xyz.tolist())
            all_points.append(np.asarray(xyz, dtype=np.float64))
        rows.append(row_pts)
        candidates.append(row_cand)

    spine_list = np.round(spine, 4).tolist()
    return {
        "kind": CAGE_KIND,
        "profile": CAGE_PROFILE,
        "editable": False,
        "lofted": False,
        "source": path.source or SOURCE_INTERIOR_PRODUCT_SURFACE,
        "row_count": len(CAGE_ROW_OFFSETS_MM),
        "center_row_index": CAGE_CENTER_ROW_INDEX,
        "row_offsets_mm": list(CAGE_ROW_OFFSETS_MM),
        "centerline_mm": spine_list,
        "point_count": int(len(all_points)),
        "points_mm": [p.tolist() for p in all_points],
        "polylines_mm": rows,
        "candidates": candidates,
    }


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


def _offset_on_envelope(
    mesh: object,
    origin: NDArray[np.float64],
    station: EnvelopeStation,
    offset_mm: float,
) -> NDArray[np.float64] | None:
    """Walk ``offset_mm`` along the local width direction, staying on the envelope."""
    origin = np.asarray(origin, dtype=np.float64)
    sampled = _offset_via_transverse_section(mesh, origin, station, offset_mm)
    if sampled is not None:
        return sampled
    return _offset_via_surface_walk(mesh, origin, station, offset_mm)


def _offset_via_transverse_section(
    mesh: object,
    origin: NDArray[np.float64],
    station: EnvelopeStation,
    offset_mm: float,
) -> NDArray[np.float64] | None:
    tangent = np.asarray(station.tangent_length, dtype=np.float64)
    nrm = float(np.linalg.norm(tangent))
    if nrm <= 1e-9:
        return None
    tangent = tangent / nrm
    section = mesh.section(  # type: ignore[attr-defined]
        plane_origin=origin.tolist(),
        plane_normal=tangent.tolist(),
    )
    if section is None:
        return None
    polylines = _path3d_polylines(section)
    if not polylines:
        return None
    poly = min(
        polylines,
        key=lambda arr: float(np.min(np.linalg.norm(arr - origin, axis=1))),
    )
    if len(poly) < 2:
        return None
    point = _interpolate_along_polyline(poly, origin, offset_mm)
    if point is None:
        return None
    snapped, dist, _tri = mesh.nearest.on_surface(point.reshape(1, 3))  # type: ignore[attr-defined]
    if float(dist[0]) > _MAX_SEED_DIST_MM:
        return None
    return np.asarray(snapped[0], dtype=np.float64)


def _offset_via_surface_walk(
    mesh: object,
    origin: NDArray[np.float64],
    station: EnvelopeStation,
    offset_mm: float,
) -> NDArray[np.float64] | None:
    remaining = abs(float(offset_mm))
    point = np.asarray(origin, dtype=np.float64).copy()
    walk = _station_width_dir(station)
    if offset_mm < 0.0:
        walk = -walk
    tangent = np.asarray(station.tangent_length, dtype=np.float64)
    normal = np.asarray(
        station.inward_normals[station.inward_normals.shape[0] // 2],
        dtype=np.float64,
    )
    while remaining > 1e-6:
        step = min(_WALK_STEP_MM, remaining)
        seed = point + walk * step
        snapped, dist, tri_ids = mesh.nearest.on_surface(seed.reshape(1, 3))  # type: ignore[attr-defined]
        if float(dist[0]) > _MAX_SEED_DIST_MM:
            return None
        nxt = np.asarray(snapped[0], dtype=np.float64)
        delta = nxt - point
        travelled = float(np.linalg.norm(delta))
        if travelled < 1e-4 or travelled > _MAX_STEP_JUMP_MM:
            return None
        point = nxt
        face_n = np.asarray(
            mesh.face_normals[int(tri_ids[0])],  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        if float(np.dot(face_n, normal)) < 0.0:
            face_n = -face_n
        normal = face_n
        walk = np.cross(normal, tangent)
        wn = float(np.linalg.norm(walk))
        if wn <= 1e-9:
            return None
        walk = walk / wn
        if float(np.dot(walk, delta)) < 0.0:
            walk = -walk
        remaining -= min(travelled, step)
    return point


def _path3d_polylines(section: object) -> list[NDArray[np.float64]]:
    discrete = getattr(section, "discrete", None)
    if discrete:
        return [
            np.asarray(part, dtype=np.float64) for part in discrete if len(part) >= 2
        ]
    polylines: list[NDArray[np.float64]] = []
    verts = np.asarray(section.vertices, dtype=np.float64)
    for entity in getattr(section, "entities", []):
        idx = np.asarray(getattr(entity, "points", []), dtype=np.int64)
        if len(idx) >= 2:
            polylines.append(verts[idx])
    return polylines


def _interpolate_along_polyline(
    pts: NDArray[np.float64],
    origin: NDArray[np.float64],
    offset_mm: float,
) -> NDArray[np.float64] | None:
    pts = np.asarray(pts, dtype=np.float64)
    closed = float(np.linalg.norm(pts[0] - pts[-1])) <= 1e-6
    if closed and len(pts) > 2:
        pts = pts[:-1].copy()
    if len(pts) < 2:
        return None
    if closed:
        diffs = np.diff(pts, axis=0, append=pts[:1])
    else:
        diffs = np.diff(pts, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        return None
    nearest = int(np.argmin(np.linalg.norm(pts - origin, axis=1)))
    s0 = float(cum[min(nearest, len(cum) - 1)])
    target = s0 + float(offset_mm)
    if closed:
        target = target % total
    elif target < -1e-6 or target > total + 1e-6:
        return None
    target = float(np.clip(target, 0.0, total))
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    idx = max(0, min(idx, len(pts) - 1))
    s_a = float(cum[idx])
    s_b = float(cum[min(idx + 1, len(cum) - 1)])
    span = max(s_b - s_a, 1e-12)
    t = (target - s_a) / span
    p0 = pts[idx]
    if idx + 1 < len(pts):
        p1 = pts[idx + 1]
    elif closed:
        p1 = pts[0]
    else:
        p1 = pts[idx]
    return p0 * (1.0 - t) + p1 * t
