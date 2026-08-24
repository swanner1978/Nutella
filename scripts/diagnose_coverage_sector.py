"""Read-only diagnostic of saved coverage definition. Does not evaluate candidates.

Does not instantiate CoverageSimulator. Does not run collision or closest-point.
Replays S0008 JSON + the same synthetic interior mesh used by the 100-campaign.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from nutella_scraper.engines.compute.coverage_simulator import (
    ANGLE_END_DEG,
    ANGLE_START_DEG,
    ANGLE_STEP_DEG,
    coverage_angle_samples_deg,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    load_interior_surface_reference,
)
from nutella_scraper.engines.compute.mesh_utils import face_areas
from nutella_scraper.engines.visualization.coverage_rank_catalog import (
    evaluation_interior_envelope_payload,
    _synthetic_evaluation_surface,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output" / "coverage" / "angle_audit" / "S0008.json"
RANK_JSON = ROOT / "output" / "coverage" / "candidate_coverage_100.json"
OUT_JSON = ROOT / "output" / "coverage" / "angle_audit" / "s0008_sector_diagnostic.json"
OUT_HTML = ROOT / "output" / "coverage" / "angle_audit" / "S0008_interior_faces_debug.html"
MODELS_ROOT = ROOT / "output" / "models"


def _azimuths_deg(centroids: np.ndarray, axis_xz: np.ndarray) -> np.ndarray:
    dx = centroids[:, 0] - float(axis_xz[0])
    dz = centroids[:, 2] - float(axis_xz[1])
    return np.mod(np.degrees(np.arctan2(-dz, dx)), 360.0)


def _surface_axis_xz(vertices: np.ndarray) -> np.ndarray:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return 0.5 * (mins[[0, 2]] + maxs[[0, 2]])


def _svg_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    covered: set[int],
    sector: set[int],
    *,
    axes: tuple[int, int],
    title: str,
) -> str:
    pts = vertices[:, list(axes)]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 8.0
    size = 420.0
    scale = (size - 2 * pad) / float(np.max(span))

    def xy(index: int) -> tuple[float, float]:
        p = pts[index]
        x = pad + (float(p[0]) - float(lo[0])) * scale
        y = size - pad - (float(p[1]) - float(lo[1])) * scale
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size:.0f}" height="{size:.0f}" '
        f'viewBox="0 0 {size:.0f} {size:.0f}">',
        f'<rect width="100%" height="100%" fill="#111"/>',
        f'<text x="12" y="18" fill="#ddd" font-size="12">{title}</text>',
    ]
    for face_id, tri in enumerate(faces):
        a, b, c = (xy(int(i)) for i in tri)
        in_sector = face_id in sector
        is_covered = face_id in covered
        if not in_sector:
            fill, opacity = "#333333", "0.15"
        elif is_covered:
            fill, opacity = "#3ddc84", "0.85"
        else:
            fill, opacity = "#6b4ea8", "0.35"
        parts.append(
            f'<polygon points="{a[0]:.2f},{a[1]:.2f} {b[0]:.2f},{b[1]:.2f} '
            f'{c[0]:.2f},{c[1]:.2f}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="none"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def real_jar_symmetry(model_id: str) -> dict:
    ref = load_interior_surface_reference(models_root=MODELS_ROOT, model_id=model_id)
    mesh = ref.to_trimesh()
    vertices = np.asarray(ref.vertices, dtype=np.float64)
    areas = face_areas(mesh)
    centroids = vertices[ref.faces].mean(axis=1)
    axis = _surface_axis_xz(vertices)
    az = _azimuths_deg(centroids, axis)
    radii = np.hypot(centroids[:, 0] - axis[0], centroids[:, 2] - axis[1])
    y = centroids[:, 1]
    # Wall band: exclude floor/axis collapse (r < 5 mm) and keep mid height.
    y_lo, y_hi = np.quantile(y, [0.25, 0.75])
    wall = (radii >= 5.0) & (y >= y_lo) & (y <= y_hi)
    bands = ((0.0, 90.0), (90.0, 180.0), (180.0, 270.0), (270.0, 360.0))
    per_band = []
    for i, (lo, hi) in enumerate(bands):
        if i == 0:
            mask = wall & (az >= lo) & (az < hi)
        elif i < 3:
            mask = wall & (az >= lo) & (az < hi)
        else:
            mask = wall & (az >= lo) & (az < 360.0)
        per_band.append(
            {
                "band_deg": f"{lo:.0f}-{hi:.0f}",
                "face_count": int(np.count_nonzero(mask)),
                "area_mm2": float(np.sum(areas[mask])),
                "mean_radius_mm": float(np.mean(radii[mask])) if np.any(mask) else None,
                "std_radius_mm": float(np.std(radii[mask])) if np.any(mask) else None,
            }
        )
    areas_b = np.array([b["area_mm2"] for b in per_band], dtype=np.float64)
    radii_b = np.array([b["mean_radius_mm"] or 0.0 for b in per_band], dtype=np.float64)
    area_rel = float(np.max(np.abs(areas_b - np.mean(areas_b))) / max(np.mean(areas_b), 1e-9))
    radius_rel = float(np.max(np.abs(radii_b - np.mean(radii_b))) / max(np.mean(radii_b), 1e-9))
    # Vertex radii at a mid-height ring.
    y_mid = 0.5 * (float(ref.y_min_mm) + float(ref.y_max_mm))
    ring = np.abs(vertices[:, 1] - y_mid) <= 2.0
    ring_r = np.hypot(vertices[ring, 0] - axis[0], vertices[ring, 2] - axis[1])
    ring_r = ring_r[ring_r >= 5.0]
    return {
        "model_id": model_id,
        "face_count": int(ref.face_count),
        "vertex_count": int(ref.vertex_count),
        "y_min_mm": float(ref.y_min_mm),
        "y_max_mm": float(ref.y_max_mm),
        "source": ref.source,
        "wall_band_y_mm": [float(y_lo), float(y_hi)],
        "quadrants_90deg": per_band,
        "max_relative_area_deviation": area_rel,
        "max_relative_mean_radius_deviation": radius_rel,
        "midheight_ring_radius_mean_mm": float(np.mean(ring_r)) if len(ring_r) else None,
        "midheight_ring_radius_std_mm": float(np.std(ring_r)) if len(ring_r) else None,
        "midheight_ring_radius_ptp_mm": float(np.ptp(ring_r)) if len(ring_r) else None,
        "axisymmetric_wall_approx": bool(radius_rel < 0.02 and area_rel < 0.05),
        "four_fold_rotation_approx": bool(radius_rel < 0.02 and area_rel < 0.05),
        "note": (
            "Measured on STEP interior product surface (Contour intérieur). "
            "This is NOT the mesh scored by the coverage-100 campaign."
        ),
    }


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rank = json.loads(RANK_JSON.read_text(encoding="utf-8"))
    rows = rank["ranked"] if isinstance(rank, dict) else rank
    saved = next(r for r in rows if r["candidate_id"] == "S0008")

    covered_ids = [int(i) for i in audit["covered_face_ids"]]
    covered_set = set(covered_ids)

    surface = _synthetic_evaluation_surface()
    mesh = surface.to_trimesh()
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    areas = face_areas(mesh)
    centroids = vertices[faces].mean(axis=1)
    axis = _surface_axis_xz(vertices)
    az = _azimuths_deg(centroids, axis)
    y = centroids[:, 1]
    opening_y = float(surface.y_max_mm)
    lower_y = float(surface.y_min_mm)
    useful = (y >= lower_y - 1e-3) & (y <= opening_y)
    radii_v = np.hypot(vertices[:, 0], vertices[:, 2])
    radii_c = np.hypot(centroids[:, 0], centroids[:, 2])

    in_45 = useful & (az >= ANGLE_START_DEG - 1e-9) & (az <= ANGLE_END_DEG + 1e-9)
    in_90 = useful & (az >= 0.0 - 1e-9) & (az <= 90.0 + 1e-9)
    ids_45 = [int(i) for i in np.flatnonzero(in_45)]
    ids_90 = [int(i) for i in np.flatnonzero(in_90)]
    area_45 = float(np.sum(areas[in_45]))
    area_90 = float(np.sum(areas[in_90]))
    covered_in_mesh = all(0 <= i < len(faces) for i in covered_ids)
    covered_in_45 = covered_set <= set(ids_45)
    covered_area = float(sum(areas[i] for i in covered_ids))
    percent_area = 100.0 * covered_area / area_45 if area_45 else 0.0
    percent_count = 100.0 * len(covered_ids) / len(ids_45) if ids_45 else 0.0

    payload = evaluation_interior_envelope_payload()
    sector_payload = set(int(i) for i in payload["sector_face_ids"])
    covered_in_payload_sector = covered_set <= sector_payload
    # Viewer onInterior: all three vertices within 0.05 mm of r=50.
    radius = 50.0
    on_interior_faces = []
    for face_id in range(len(faces)):
        tri = faces[face_id]
        ok = all(abs(float(radii_v[int(v)]) - radius) <= 0.05 for v in tri)
        on_interior_faces.append(ok)
    viewer_green = {
        i for i in covered_ids if i in sector_payload and on_interior_faces[i]
    }
    viewer_missing = sorted(covered_set - viewer_green)
    extra_payload_vs_sim = sorted(sector_payload - set(ids_45))
    missing_payload_vs_sim = sorted(set(ids_45) - sector_payload)

    bands_45 = ((0.0, 45.0), (45.0, 90.0), (90.0, 135.0), (135.0, 180.0))
    quadrant_areas = []
    for i, (lo, hi) in enumerate(bands_45):
        if i == 0:
            mask = useful & (az >= lo) & (az <= hi)
        else:
            mask = useful & (az > lo) & (az <= hi)
        quadrant_areas.append(float(np.sum(areas[mask])))

    full_lateral = float(np.sum(areas[useful]))
    angles = coverage_angle_samples_deg()
    angles_90 = coverage_angle_samples_deg(start_deg=0.0, end_deg=90.0, step_deg=2.0)

    # Face-area equality inside the 0-45 sector.
    sector_areas = areas[in_45]
    # Unique face-id modulo 2 (two triangles per quad).
    unique_ids = np.unique(covered_ids)

    real_models = sorted(
        p.name for p in MODELS_ROOT.iterdir() if (p / "interior_product_surface.npz").is_file()
    )
    real = real_jar_symmetry(real_models[0]) if real_models else None

    report = {
        "simulator_invoked": False,
        "coverage_recomputed": False,
        "collision_executed": False,
        "closest_point_executed": False,
        "definition": {
            "angle_start_deg": ANGLE_START_DEG,
            "angle_end_deg": ANGLE_END_DEG,
            "angle_step_deg": ANGLE_STEP_DEG,
            "evaluated_angles": list(angles),
            "evaluated_angle_count": len(angles),
            "azimuth_convention": "0=+X, 90=-Z, atan2(-z,x) mod 360",
            "useful_y_mm": [lower_y, opening_y],
            "useful_filter": "face centroid Y in [y_min, y_max] of interior mesh",
            "target_faces": "useful AND azimuth in [0, 45]",
            "coverage_percent": (
                "100 * covered_area / target_area, CoverageScorer area-weighted; "
                "symmetry_multiplier_applied is always False"
            ),
            "union_not_sum": True,
            "times_four": False,
            "surfaces_included": (
                "All InteriorSurfaceReference triangles in the useful Y band. "
                "Campaign mesh is a lateral cylinder only (no floor, no lid)."
            ),
        },
        "evaluation_mesh": {
            "source": "synthetic cylinder r=50 mm, y 0-80, 21x48",
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "min_vertex_radius_mm": float(np.min(radii_v)),
            "max_vertex_radius_mm": float(np.max(radii_v)),
            "centroid_radius_mean_mm": float(np.mean(radii_c)),
            "is_visual_stl": False,
            "is_step_interior": False,
        },
        "s0008": {
            "saved_json_coverage_percent": float(saved["coverage_percent"]),
            "saved_json_covered_area_mm2": float(saved["covered_area_mm2"]),
            "saved_json_target_area_mm2": float(saved.get("target_area_mm2") or audit["target_area_mm2"]),
            "saved_json_touched_face_count": int(saved.get("touched_face_count") or len(covered_ids)),
            "audit_coverage_percent": float(audit["coverage_percent"]),
            "audit_covered_area_mm2": float(audit["covered_area_mm2"]),
            "audit_target_area_mm2": float(audit["target_area_mm2"]),
            "covered_face_count": len(covered_ids),
            "covered_face_ids_unique": len(unique_ids),
            "covered_faces_in_evaluation_mesh": covered_in_mesh,
            "covered_faces_subset_of_sector_0_45": covered_in_45,
            "recomputed_covered_area_mm2": covered_area,
            "recomputed_target_area_mm2": area_45,
            "recomputed_percent_area": percent_area,
            "recomputed_percent_face_count": percent_count,
            "matches_159_over_240": bool(
                len(covered_ids) == 159 and len(ids_45) == 240
            ),
            "area_equals_count_ratio": abs(percent_area - percent_count) < 1e-9,
        },
        "sector_0_45": {
            "reference_face_count": len(ids_45),
            "reference_area_mm2": area_45,
            "fraction_of_full_useful_faces": len(ids_45) / len(faces),
            "fraction_of_360_perimeter": 45.0 / 360.0,
            "is_quarter_perimeter": False,
            "is_eighth_perimeter": True,
        },
        "sector_0_90_if_same_mesh": {
            "reference_face_count": len(ids_90),
            "reference_area_mm2": area_90,
            "fraction_of_360_perimeter": 90.0 / 360.0,
            "is_quarter_perimeter": True,
            "angle_sample_count": len(angles_90),
            "angle_samples_preview": list(angles_90[:5]) + ["..."] + [angles_90[-1]],
            "s0008_covered_in_0_90": covered_set <= set(ids_90),
            "s0008_coverage_of_90_if_same_union": (
                100.0 * covered_area / area_90 if area_90 else None
            ),
            "note": (
                "This is a mesh-count projection only. S0008 was never evaluated "
                "on 0-90; poses beyond 45° were not searched."
            ),
        },
        "quadrant_useful_areas_mm2": {
            "0_45": quadrant_areas[0],
            "45_90": quadrant_areas[1],
            "90_135": quadrant_areas[2],
            "135_180": quadrant_areas[3],
            "used_as_times_four": False,
            "covers_180_360": False,
            "full_useful_lateral_area_mm2": full_lateral,
        },
        "cylinder_symmetry": {
            "regular_48_fold": True,
            "left_right_reflection": True,
            "four_quarters_equal_area": all(
                abs(a - quadrant_areas[0]) < 1e-6 for a in quadrant_areas
            ),
            "vertical_axis": "Y",
            "applies_to": "synthetic evaluation cylinder only",
        },
        "real_jar_interior": real,
        "viewer": {
            "paints_evaluation_cylinder": True,
            "paints_visual_stl_faces": False,
            "paints_step_interior_faces": False,
            "payload_sector_face_count": len(sector_payload),
            "payload_sector_equals_simulator_0_45": sector_payload == set(ids_45),
            "payload_extra_vs_simulator": extra_payload_vs_sim[:20],
            "payload_missing_vs_simulator": missing_payload_vs_sim[:20],
            "covered_in_payload_sector": covered_in_payload_sector,
            "on_interior_all_mesh_faces": all(on_interior_faces),
            "viewer_green_face_count_if_filters_applied": len(viewer_green),
            "viewer_would_drop_covered_ids": viewer_missing,
            "verdict": (
                "A_same_159_on_evaluation_mesh"
                if viewer_green == covered_set
                else ("B_partial" if viewer_green < covered_set else "C_different_surface")
            ),
            "note": (
                "Green triangles use evaluation_interior_envelope_payload vertices "
                "(same IDs as the engine). They are not the imported Nutella jar. "
                "The surface looks cylindrical because the scored interior is a cylinder."
            ),
        },
        "changes_for_0_to_90": {
            "ANGLE_END_DEG": "45 -> 90",
            "target_face_mask": "azimuth <= 90 instead of <= 45",
            "reference_area": "about 2x current target_area on this mesh",
            "angle_samples": f"{len(angles)} -> {len(angles_90)} (step 2° kept)",
            "poses_searched": "one SE(3) neighbourhood per sample; ~2x envelope frames",
            "ranking": "still coverage_percent of the chosen sector; scores not comparable to 0-45 ranking",
            "percent_definition": "keep sector ratio separate from optional symmetry extrapolation",
            "not_only_a_window_change": True,
            "other_consequences": [
                "phi=0 still uses design_frame neighbourhood; all other phi use envelope frames",
                "saved 100-candidate ranking is 0-45 only and must not be reused",
                "angle_audit dumps are 0-45 only",
                "A0 golden baseline numbers are 0-45 only",
            ],
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    top = _svg_view(
        vertices, faces, covered_set, set(ids_45),
        axes=(0, 2),
        title="S0008 top (X,Z) — evaluation interior faces",
    )
    front = _svg_view(
        vertices, faces, covered_set, set(ids_45),
        axes=(0, 1),
        title="S0008 front (X,Y) — evaluation interior faces",
    )
    OUT_HTML.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        "<title>S0008 interior face debug</title>"
        "<style>body{font-family:sans-serif;background:#000;color:#eee;padding:16px}"
        "code{color:#9fd4ff}</style></head><body>"
        "<h1>S0008 — faces du mesh intérieur d'évaluation</h1>"
        "<p>Pas une reconstruction. Vertices/faces = cylindre r=50 mm de la campagne. "
        f"Vert = {len(covered_ids)} covered_face_ids. Violet = secteur 0–45° non couvert. "
        "Gris = hors secteur.</p>"
        "<p><code>covered_face_ids</code> = faces vertes : "
        f"{'oui' if viewer_green == covered_set else 'non'}</p>"
        f"{top}{front}</body></html>",
        encoding="utf-8",
    )
    print(json.dumps({
        "out_json": str(OUT_JSON),
        "out_html": str(OUT_HTML),
        "covered": len(covered_ids),
        "ref_45": len(ids_45),
        "percent_area": percent_area,
        "percent_count": percent_count,
        "viewer_verdict": report["viewer"]["verdict"],
        "real_axisym": None if real is None else real["axisymmetric_wall_approx"],
        "simulator_invoked": False,
    }, indent=2))


if __name__ == "__main__":
    main()
