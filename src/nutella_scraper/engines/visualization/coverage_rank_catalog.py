"""Load already-ranked coverage-100 results without running CoverageSimulator.

The JSON/CSV files are the source of truth for metrics. Control points are
looked up from an already-generated CandidateShape catalog (curves only).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nutella_scraper.domain.models.scraper_parameters import ScraperParameters
from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.scraper_envelope_path import scraper_length_span
from nutella_scraper.engines.compute.scraper_rigid_motion import (
    build_rigid_scraper_artifact,
)
from nutella_scraper.engines.visualization.scraper_control_cage import (
    build_control_cage_overlay,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (
    generate_candidate_shapes,
    lattice_from_cage,
)

DEFAULT_COVERAGE_JSON = Path("output/coverage/candidate_coverage_100.json")
DEFAULT_COVERAGE_CSV = Path("output/coverage/candidate_coverage_100.csv")

# Visualization-only cross-section scale. Physical blade remains 2.5 × 2.5 mm.
# Never feed this into CoverageSimulator, collision, fingerprints, or coverage.
DISPLAY_BLADE_SCALE = 4.0
DISPLAY_BLADE_WIDTH_MM = 2.5
DISPLAY_BLADE_THICKNESS_MM = 2.5

# Fields required to animate rigid SE(3) poses over 0–45° without re-simulating.
COVERAGE_PLAY_REQUIRED_FIELDS: tuple[str, ...] = (
    "best_pose_by_angle",
    "touched_face_ids_by_angle",
)

_VIEWER_METRIC_KEYS: tuple[str, ...] = (
    "rank",
    "candidate_id",
    "family",
    "coverage_percent",
    "covered_area_mm2",
    "target_area_mm2",
    "valid_pose_count",
    "total_pose_count",
    "touched_face_count",
    "curve_length_mm",
    "shape_fingerprint",
)


def coverage_results_path(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path.cwd()
    return (base / DEFAULT_COVERAGE_JSON).resolve()


def load_coverage_rank_json(path: Path | None = None) -> dict[str, Any]:
    """Read the saved 100-candidate ranking. Does not call CoverageSimulator."""
    target = Path(path) if path is not None else DEFAULT_COVERAGE_JSON
    if not target.is_file():
        raise FileNotFoundError(
            f"Classement couverture introuvable: {target}. "
            "Aucun recalcul n'est lancé."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate_coverage_100.json must be an object")
    ranked = payload.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("candidate_coverage_100.json has no ranked list")
    return payload


def load_coverage_rank_csv(path: Path | None = None) -> tuple[dict[str, str], ...]:
    """Read the saved CSV ranking. Does not call CoverageSimulator."""
    target = Path(path) if path is not None else DEFAULT_COVERAGE_CSV
    if not target.is_file():
        raise FileNotFoundError(f"CSV couverture introuvable: {target}")
    with target.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = tuple(dict(row) for row in reader)
    if not rows:
        raise ValueError("candidate_coverage_100.csv is empty")
    return rows


def coverage_play_status(ranked_row: Mapping[str, Any]) -> dict[str, Any]:
    """Play is only possible if per-angle SE(3) poses were saved."""
    missing = [name for name in COVERAGE_PLAY_REQUIRED_FIELDS if name not in ranked_row]
    return {
        "available": not missing,
        "sector_deg": [0.0, 45.0],
        "missing_fields": missing,
        "reason": (
            None
            if not missing
            else (
                "Le JSON de classement n'expose pas les poses SE(3) par angle "
                f"({', '.join(missing)}). Play 0–45° n'invente aucune pose et "
                "ne relance pas CoverageSimulator."
            )
        ),
    }


def ranked_viewer_rows(
    payload: Mapping[str, Any],
    *,
    shapes_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Join saved metrics with catalog curves. Geometry is not mutated."""
    catalog = dict(shapes_by_id or {})
    rows: list[dict[str, Any]] = []
    play = coverage_play_status(payload["ranked"][0] if payload.get("ranked") else {})
    for index, raw in enumerate(payload["ranked"]):
        if not isinstance(raw, Mapping):
            continue
        candidate_id = str(raw["candidate_id"])
        shape = catalog.get(candidate_id, {})
        metrics = {key: raw.get(key) for key in _VIEWER_METRIC_KEYS}
        item = {
            **shape,
            **metrics,
            "index": index,
            "is_reference_a": candidate_id == "A0",
            "is_best": int(metrics.get("rank") or 0) == 1,
            "metrics_source": "saved_json",
            "coverage_play": play,
            "geometry_source": "saved_rank_plus_catalog_curve",
        }
        if "control_points_mm" not in item or item["control_points_mm"] is None:
            item["control_points_mm"] = None
            item["curve_available"] = False
        else:
            item["control_points_mm"] = [list(p) for p in item["control_points_mm"]]
            item["curve_available"] = True
            item["row_indices"] = list(item.get("row_indices") or [])
            width = float(item.get("width_mm") or DISPLAY_BLADE_WIDTH_MM)
            thickness = float(item.get("thickness_mm") or DISPLAY_BLADE_THICKNESS_MM)
            item["scraper_geometry"] = blade_display_mesh_from_curve(
                item["control_points_mm"],
                width_mm=width,
                thickness_mm=thickness,
            )
            item["geometry_source"] = "catalog_curve_blade_display"
            item["display_blade_scale"] = DISPLAY_BLADE_SCALE
        rows.append(item)
    return rows


def shapes_by_id_from_candidates(candidates: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Index CandidateShape (or dict) by id without regenerating coverage."""
    out: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if hasattr(item, "to_dict"):
            payload = item.to_dict()
        elif isinstance(item, Mapping):
            payload = dict(item)
        else:
            continue
        key = str(payload.get("candidate_id", ""))
        if key:
            out[key] = payload
    return out


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vector / norm


def blade_display_mesh_from_curve(
    control_points_mm: Sequence[Sequence[float]],
    *,
    width_mm: float = DISPLAY_BLADE_WIDTH_MM,
    thickness_mm: float = DISPLAY_BLADE_THICKNESS_MM,
    visual_scale: float = DISPLAY_BLADE_SCALE,
) -> dict[str, Any] | None:
    """Sweep a rectangular cross-section along a saved catalog curve.

    Visualization only. ``visual_scale`` thickens the ribbon for the viewer.
    Physical ``width_mm`` / ``thickness_mm`` stay 2.5 mm and must never enter
    CoverageSimulator, collision, fingerprints, or closest-point.
    """
    pts = np.asarray(control_points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
        return None
    scale = float(visual_scale)
    half_w = 0.5 * float(width_mm) * scale
    half_t = 0.5 * float(thickness_mm) * scale
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for index, point in enumerate(pts):
        if index < len(pts) - 1:
            tangent = _unit(pts[index + 1] - point)
        else:
            tangent = _unit(point - pts[index - 1])
        radial = np.array([float(point[0]), 0.0, float(point[2])], dtype=np.float64)
        outward = _unit(radial)
        if float(np.linalg.norm(outward)) <= 1e-9:
            outward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        width_dir = _unit(np.cross(tangent, outward))
        if float(np.linalg.norm(width_dir)) <= 1e-9:
            width_dir = _unit(np.cross(tangent, np.array([0.0, 1.0, 0.0])))
        if float(np.linalg.norm(width_dir)) <= 1e-9:
            width_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        outward = _unit(np.cross(width_dir, tangent))
        if float(np.linalg.norm(outward)) <= 1e-9:
            outward = _unit(radial)
        corners = (
            point - width_dir * half_w - outward * half_t,
            point + width_dir * half_w - outward * half_t,
            point + width_dir * half_w + outward * half_t,
            point - width_dir * half_w + outward * half_t,
        )
        for corner in corners:
            vertices.append([float(corner[0]), float(corner[1]), float(corner[2])])
        if index == 0:
            continue
        a = (index - 1) * 4
        b = index * 4
        quads = (
            (a, a + 1, b + 1, b),
            (a + 1, a + 2, b + 2, b + 1),
            (a + 2, a + 3, b + 3, b + 2),
            (a + 3, a, b, b + 3),
        )
        for i0, i1, i2, i3 in quads:
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])
    return {
        "vertices": vertices,
        "faces": faces,
        "visual_only": True,
        "display_blade_scale": scale,
        "width_mm": float(width_mm),
        "thickness_mm": float(thickness_mm),
    }


def _progress_azimuth_deg(x: float, z: float) -> float:
    return float(np.mod(np.degrees(np.arctan2(-float(z), float(x))), 360.0))


def evaluation_interior_envelope_payload() -> dict[str, Any]:
    """Interior mesh of the saved coverage campaign. Not CoverageSimulator.

    Face ids match angle_audit dumps. This cylinder is NOT visual.stl; the
    viewer must not composite it as if it were the imported Nutella jar.
    """
    surface = _synthetic_evaluation_surface()
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    centroids = vertices[faces].mean(axis=1)
    azimuths = np.array(
        [_progress_azimuth_deg(float(p[0]), float(p[2])) for p in centroids],
        dtype=np.float64,
    )
    radii = np.hypot(vertices[:, 0], vertices[:, 2])
    sector = (azimuths >= -1e-9) & (azimuths <= 45.0 + 1e-9)
    sector_ids = [int(i) for i in np.flatnonzero(sector)]
    return {
        "vertices": np.round(vertices, 6).tolist(),
        "faces": faces.astype(np.int32).tolist(),
        "sector_face_ids": sector_ids,
        "source": "synthetic_evaluation_interior",
        "on_interior_surface": True,
        "radius_mm": 50.0,
        "max_vertex_radius_mm": float(np.max(radii)),
        "min_vertex_radius_mm": float(np.min(radii)),
        "simulator_invoked": False,
        "coverage_recomputed": False,
    }


def filter_ranked_prefix(
    rows: Sequence[Mapping[str, Any]],
    top_n: int | None,
) -> tuple[Mapping[str, Any], ...]:
    """Keep the JSON rank order. ``top_n is None`` means all saved rows."""
    items = tuple(rows)
    if top_n is None:
        return items
    limit = max(1, int(top_n))
    return items[: min(limit, len(items))]


def clamp_rank_index(index: int, count: int) -> int:
    if int(count) <= 0:
        return 0
    return max(0, min(int(count) - 1, int(index)))


def step_rank_index(index: int, delta: int, count: int) -> int:
    """Move ±1 in the visible prefix. Never wraps, never errors."""
    return clamp_rank_index(int(index) + int(delta), count)


def walk_rank_indices(count: int) -> tuple[int, ...]:
    """Index path 0→N-1 then N-1→0, matching viewer clamp (no wrap)."""
    n = int(count)
    if n <= 0:
        return ()
    forward = tuple(range(n))
    back = tuple(range(n - 1, -1, -1))
    return forward + back[1:]


def neighbor_rank_rows(
    rows: Sequence[Mapping[str, Any]],
    index: int,
) -> dict[str, Mapping[str, Any] | None]:
    """Previous / next saved rows in the current visible prefix."""
    items = tuple(rows)
    idx = clamp_rank_index(index, len(items))
    return {
        "previous": items[idx - 1] if idx > 0 else None,
        "next": items[idx + 1] if idx + 1 < len(items) else None,
    }


def _synthetic_evaluation_surface() -> InteriorSurfaceReference:
    """Same cylinder as the saved coverage-100 campaign (r=50, y 0–80, 21×48)."""
    radius = 50.0
    y_min, y_max = 0.0, 80.0
    y_count, angular_count = 21, 48
    thetas = np.linspace(-math.pi, math.pi, angular_count, endpoint=False)
    ys = np.linspace(y_min, y_max, y_count)
    vertices: list[tuple[float, float, float]] = []
    for y in ys:
        for theta in thetas:
            vertices.append(
                (float(radius * math.cos(theta)), float(y), float(radius * math.sin(theta)))
            )
    faces: list[tuple[int, int, int]] = []
    for yi in range(y_count - 1):
        for ti in range(angular_count):
            t2 = (ti + 1) % angular_count
            i00 = yi * angular_count + ti
            i01 = yi * angular_count + t2
            i10 = (yi + 1) * angular_count + ti
            i11 = (yi + 1) * angular_count + t2
            faces.append((i00, i10, i11))
            faces.append((i00, i11, i01))
    return InteriorSurfaceReference.from_arrays(
        model_id="synthetic-interior",
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        matching_face_count=1,
        source=SOURCE_INTERIOR_PRODUCT_SURFACE,
    )


_EVAL_BUNDLE: dict[str, Any] | None = None


def _evaluation_bundle(*, count: int = 1000) -> dict[str, Any]:
    """A0 solid + lattice + curves of the saved campaign. No CoverageSimulator."""
    global _EVAL_BUNDLE
    if _EVAL_BUNDLE is not None and len(_EVAL_BUNDLE["catalog"]) >= int(count):
        return _EVAL_BUNDLE
    surface = _synthetic_evaluation_surface()
    _opening_y, _lower_y, max_length = scraper_length_span(surface)
    parameters = ScraperParameters.default().with_updates(
        bevel_angle_deg=0.0,
        relief_angle_deg=0.0,
        helix_rate_deg_per_mm=0.0,
        width_mm=2.5,
        thickness_mm=2.5,
        length_mm=min(40.0, max_length),
        clearance_mm=0.0,
        position_z_mm=float(0.5 * (surface.y_min_mm + surface.y_max_mm)),
    )
    reference = build_rigid_scraper_artifact(surface, parameters)
    cage = build_control_cage_overlay(reference.design_path, surface)
    lattice = lattice_from_cage(cage, surface)
    catalog = tuple(generate_candidate_shapes(lattice, count=int(count)))
    _EVAL_BUNDLE = {
        "surface": surface,
        "parameters": parameters,
        "reference": reference,
        "lattice": lattice,
        "catalog": catalog,
    }
    return _EVAL_BUNDLE


def evaluation_curve_catalog(*, count: int = 1000) -> tuple[Any, ...]:
    """Regenerate the evaluation *curves* only. Never runs CoverageSimulator."""
    return _evaluation_bundle(count=count)["catalog"][: int(count)]


def evaluation_lattice():
    """Contact lattice of the saved coverage campaign (for cardinality reports)."""
    return _evaluation_bundle()["lattice"]


def build_ranked_coverage_viewer_payload(
    *,
    json_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Viewer payload: ranked 100 + curves. No coverage recompute."""
    payload = load_coverage_rank_json(json_path)
    csv_rows = None
    try:
        csv_rows = load_coverage_rank_csv(csv_path)
    except FileNotFoundError:
        csv_rows = None
    catalog = evaluation_curve_catalog(count=1000)
    shapes = shapes_by_id_from_candidates(catalog)
    ranked = ranked_viewer_rows(payload, shapes_by_id=shapes)
    a0_index = next(
        (i for i, row in enumerate(ranked) if str(row.get("candidate_id")) == "A0"),
        None,
    )
    play = ranked[0]["coverage_play"] if ranked else coverage_play_status({})
    csv_ids = tuple(str(row.get("candidate_id", "")) for row in csv_rows) if csv_rows else ()
    json_ids = tuple(str(row["candidate_id"]) for row in ranked)
    return {
        "source_json": str(Path(json_path or DEFAULT_COVERAGE_JSON)),
        "count": len(ranked),
        "simulator_invoked": False,
        "coverage_recomputed": False,
        "candidates": ranked,
        "a0_index": a0_index,
        "coverage_play": {**play, "available": False},
        "csv_matches_json": csv_ids == json_ids if csv_ids else None,
        "blade_thickness_mm": 2.5,
        "blade_width_mm": 2.5,
        "metrics_source": "saved_json",
        "top_filters": [10, 20, "all"],
        "default_top_filter": 10,
        "best_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "coverage_interior_envelope": evaluation_interior_envelope_payload(),
        "display_blade_scale": DISPLAY_BLADE_SCALE,
    }
