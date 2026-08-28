"""HEURISTIC / PRELIMINARY — six 2 mm blade families on pose-graph V2.

Not a 14-family campaign. Not MAX_SHAPE_EVALUATIONS=100.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from scripts.validate_trajectory_model_v2_a0 import (  # noqa: E402
    _svg_replay,
)

from nutella_scraper.engines.compute.coverage_reference_matrix import (  # noqa: E402
    build_coverage_reference_matrix,
)
from nutella_scraper.engines.compute.interior_surface_reference import (  # noqa: E402
    load_interior_surface_reference,
)
from nutella_scraper.engines.compute.pose_contact_cache import (  # noqa: E402
    build_pose_contact_cache,
    cache_entries_payload,
    cache_from_payload,
)
from nutella_scraper.engines.compute.pose_space import TRAJECTORY_MODEL  # noqa: E402
from nutella_scraper.engines.compute.shape_export import (  # noqa: E402
    candidate_payload,
    export_best_candidates,
)
from nutella_scraper.engines.compute.shape_families import (  # noqa: E402
    BLADE_THICKNESS_MM,
    BLADE_WIDTH_MM,
)
from nutella_scraper.engines.compute.shape_search import (  # noqa: E402
    preliminary_config,
    report_to_payload,
    search_scraper_shapes,
)
from nutella_scraper.engines.compute.shape_validation import (  # noqa: E402
    check_covered_indices_in_matrix,
    check_trajectory_rules,
    check_union_not_sum,
)

OUT_DIR = _ROOT / "output" / "coverage" / "shape_search_v2_preliminary"
JSON_PATH = OUT_DIR / "shape_search_v2_preliminary.json"
HTML_PATH = OUT_DIR / "shape_search_v2_preliminary.html"
CACHE_DIR = OUT_DIR / "caches"
A0_V2_CACHE = _ROOT / "output" / "coverage" / "trajectory_model_v2_a0_pose_cache.json"
MODEL_ID = "9a1eee56-d73e-46fb-93d1-25d4ea856874"
LABEL = "HEURISTIC / PRELIMINARY"
A0_V2_REFERENCE_PERCENT = 34.86842105263158
A0_V2_REFERENCE_POINTS = 212


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:80]


def _disk_physics(surface, matrix, specs, *, artifact=None, parameters=None):
    fingerprint = str(artifact.shape_fingerprint) if artifact is not None else "none"
    path = CACHE_DIR / f"cache_{_safe_name(fingerprint)}.json"
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        if int(blob.get("n_specs", -1)) == len(specs):
            print(f"    cache disque {path.name}", flush=True)
            return cache_from_payload(
                blob["entries"],
                n_points=int(blob["n_points"]),
                fingerprint=str(blob["scraper_fingerprint"]),
                angle_window_deg=tuple(blob["angle_window_deg"]),
            )
    if (
        "t=2.5" in fingerprint
        and A0_V2_CACHE.exists()
    ):
        blob = json.loads(A0_V2_CACHE.read_text(encoding="utf-8"))
        if (
            str(blob.get("scraper_fingerprint")) == fingerprint
            and int(blob.get("n_specs", -1)) == len(specs)
        ):
            print("    réutilisation cache A0 V2", flush=True)
            cache = cache_from_payload(
                blob["entries"],
                n_points=int(blob["n_points"]),
                fingerprint=str(blob["scraper_fingerprint"]),
                angle_window_deg=tuple(blob["angle_window_deg"]),
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(blob), encoding="utf-8")
            return cache
    print(f"    physique {fingerprint[:48]}… n={len(specs)}", flush=True)
    cache = build_pose_contact_cache(
        surface, matrix, specs, artifact=artifact, parameters=parameters
    )
    blob = {
        "n_points": int(cache.n_points),
        "n_specs": len(specs),
        "scraper_fingerprint": str(cache.scraper_fingerprint),
        "physics_queries": int(cache.physics_queries),
        "angle_window_deg": list(cache.angle_window_deg),
        "entries": cache_entries_payload(cache),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob), encoding="utf-8")
    return cache


def _profile_svg(profile_xy, title: str, size: float = 420.0) -> str:
    pts = np.asarray(profile_xy, dtype=np.float64)
    lo = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - lo, 1e-6)
    pad = 18.0
    scale = (size - 2 * pad) / float(np.max(span))
    d = []
    for x, y in pts:
        px = pad + (float(x) - float(lo[0])) * scale
        py = size - pad - (float(y) - float(lo[1])) * scale
        d.append(f"{px:.1f},{py:.1f}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size:.0f}" height="{size:.0f}" '
        f'viewBox="0 0 {size:.0f} {size:.0f}">'
        '<rect width="100%" height="100%" fill="#111"/>'
        f'<text x="12" y="18" fill="#ddd" font-size="12">{title}</text>'
        f'<polyline points="{" ".join(d)}" fill="none" stroke="#4da3ff" '
        'stroke-width="3.2"/>'
        "</svg>"
    )


def _html_report(payload, matrix_points, best, a0) -> str:
    rows = []
    for row in payload["ranking_table"]:
        rows.append(
            "<tr>"
            f"<td>{row['forme']}</td><td>{row['parametres']}</td>"
            f"<td>{row['couverture_percent']:.2f}%</td>"
            f"<td>{row['points']}</td>"
            f"<td>{row['trajectory_length_mm']:.1f}</td>"
            f"<td>{row['direction_changes']}</td>"
            f"<td>{row['min_curvature_radius_mm']:.2f}</td>"
            f"<td>{row['thickness_mm']:.1f}</td>"
            "</tr>"
        )
    touched = set(best.covered_point_indices)
    origins = [tuple(item) for item in best.trajectory_origins]
    scraper_lines = []
    if payload["best"]["poses"]:
        last = payload["best"]["poses"][-1]
        origin = tuple(last["origin_mm"])
        axis = (0.0, -1.0, 0.0)
        tip = (
            origin[0] + axis[0] * float(best.scraper_length_mm),
            origin[1] + axis[1] * float(best.scraper_length_mm),
            origin[2] + axis[2] * float(best.scraper_length_mm),
        )
        scraper_lines = [(origin, tip)]
    top = _svg_replay(
        matrix_points,
        axes=(0, 2),
        title="dessus XZ",
        touched=touched,
        origins=origins,
        scraper_lines=scraper_lines,
    )
    side = _svg_replay(
        matrix_points,
        axes=(0, 1),
        title="profil XY (haut→bas)",
        touched=touched,
        origins=origins,
        scraper_lines=scraper_lines,
    )
    iso_pts = np.column_stack(
        (
            matrix_points[:, 0] + 0.6 * matrix_points[:, 2],
            matrix_points[:, 1],
            np.zeros(len(matrix_points)),
        )
    )
    iso_origins = [
        (o[0] + 0.6 * o[2], o[1], 0.0) for o in origins
    ]
    iso = _svg_replay(
        iso_pts,
        axes=(0, 1),
        title="vue 3D simplifiée (X+0.6Z, Y)",
        touched=touched,
        origins=iso_origins,
        scraper_lines=[],
    )
    profile = np.asarray(best.profile_points_mm, dtype=np.float64)
    blade = _profile_svg(profile[:, [0, 1]], "lame 2 mm — profil sagittal r(y)")
    a0_txt = (
        f"A0 V2 référence : {a0.coverage_percent:.3f}% "
        f"({a0.covered_points}/608)"
        if a0 is not None
        else ""
    )
    return (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'/>"
        f"<title>{LABEL}</title>"
        "<style>body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:24px}"
        ".row{display:flex;gap:12px;flex-wrap:wrap}"
        "td,th{border:1px solid #444;padding:6px 8px} table{border-collapse:collapse}"
        ".note{color:#bbb;max-width:980px}</style></head><body>"
        f"<h1>{LABEL}</h1>"
        "<p class='note'>Lame 2,0 mm. Trajectoire = graphe de poses V2. "
        "Pas un optimum global. Nuage = interior_matrix_a0_0_90. "
        f"{a0_txt}</p>"
        "<table><tr><th>Forme</th><th>Paramètres</th><th>Couverture</th>"
        "<th>Points / 608</th><th>Longueur traj.</th><th>Chang. dir.</th>"
        "<th>Rayon courbure min</th><th>Épaisseur</th></tr>"
        + "".join(rows)
        + "</table>"
        f"<h2>Meilleure forme trouvée : {best.family_id} "
        f"({best.coverage_percent:.2f}%)</h2>"
        f"<div class='row'>{blade}</div>"
        "<h2>Trajectoire</h2>"
        f"<div class='row'>{iso}{top}{side}</div>"
        "</body></html>"
    )


def _ranking_table(a0, best_per_family):
    rows = []
    if a0 is not None:
        rows.append(
            {
                "forme": "A0 V2",
                "parametres": "historique",
                "couverture_percent": float(a0.coverage_percent),
                "points": f"{a0.covered_points} / {a0.total_points}",
                "trajectory_length_mm": float(a0.trajectory_length_mm),
                "direction_changes": int(a0.direction_changes),
                "min_curvature_radius_mm": float(a0.min_curvature_radius_mm),
                "thickness_mm": float(a0.thickness_mm),
            }
        )
    for item in best_per_family:
        rows.append(
            {
                "forme": item.family_id,
                "parametres": ", ".join(f"{v:.3f}" for v in item.parameters[:6]),
                "couverture_percent": float(item.coverage_percent),
                "points": f"{item.covered_points} / {item.total_points}",
                "trajectory_length_mm": float(item.trajectory_length_mm),
                "direction_changes": int(item.direction_changes),
                "min_curvature_radius_mm": float(item.min_curvature_radius_mm),
                "thickness_mm": float(item.thickness_mm),
            }
        )
    return rows


def main() -> int:
    print(f"{LABEL} — 6 familles, 1 graine, poses V2, lame {BLADE_THICKNESS_MM} mm", flush=True)
    print("STOP: pas de 14 familles, pas de 100 évaluations", flush=True)
    if abs(BLADE_THICKNESS_MM - 2.0) > 1e-9 or abs(BLADE_WIDTH_MM - 2.0) > 1e-9:
        raise SystemExit("lame candidate doit faire 2.0 × 2.0 mm")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    surface = load_interior_surface_reference(
        models_root=_ROOT / "output" / "models", model_id=MODEL_ID
    )
    matrix = build_coverage_reference_matrix(surface)
    if int(matrix.point_count) != 608:
        raise SystemExit(f"nuage {matrix.point_count} ≠ 608")
    started = time.perf_counter()
    report = search_scraper_shapes(
        surface,
        matrix=matrix,
        config=preliminary_config(),
        physics_builder=_disk_physics,
    )
    elapsed = time.perf_counter() - started
    anomalies: list[str] = []
    if abs(BLADE_THICKNESS_MM - 2.0) > 1e-9:
        anomalies.append("épaisseur candidate ≠ 2 mm")
    for item in report.candidates:
        anomalies.extend(check_union_not_sum(item))
        anomalies.extend(check_covered_indices_in_matrix(item, matrix))
        anomalies.extend(check_trajectory_rules(item, report.grid))
        if item.physical_valid and item.family_id != "A0":
            if abs(item.thickness_mm - 2.0) > 1e-9:
                anomalies.append(f"{item.candidate_id}: thickness={item.thickness_mm}")
        if item.physical_valid:
            ys = [y for y, _az in item.trajectory_poses]
            if ys and ys[0] < report.frame.y_top_mm - 8.0:
                anomalies.append(f"{item.candidate_id}: départ loin de l'ouverture")
    if report.a0_reference is not None:
        anomalies.extend(check_union_not_sum(report.a0_reference))
        anomalies.extend(check_trajectory_rules(report.a0_reference, report.grid))
    best = report.candidates[0] if report.candidates else None
    ranking = _ranking_table(report.a0_reference, report.best_per_family)
    payload = report_to_payload(report)
    payload.update(
        {
            "label": LABEL,
            "stage": "PRELIMINARY",
            "trajectory_model": TRAJECTORY_MODEL,
            "campaign_launched": False,
            "blade_thickness_mm": BLADE_THICKNESS_MM,
            "blade_width_mm": BLADE_WIDTH_MM,
            "a0_v2_reference_expected_percent": A0_V2_REFERENCE_PERCENT,
            "a0_v2_reference_expected_points": A0_V2_REFERENCE_POINTS,
            "elapsed_seconds": elapsed,
            "anomalies": anomalies,
            "ranking_table": ranking,
            "disclaimer": (
                "HEURISTIC / PRELIMINARY. Meilleure forme trouvée dans ce budget, "
                "pas un optimum global."
            ),
        }
    )
    if best is not None:
        origins = list(best.trajectory_origins)
        payload["best"] = {
            **candidate_payload(best),
            "trajectory_poses": list(best.trajectory_poses),
            "poses": [
                {
                    "y_mm": y,
                    "azimuth_deg": az,
                    "origin_mm": list(origin),
                }
                for (y, az), origin in zip(best.trajectory_poses, origins, strict=True)
            ],
            "max_depth_reached_mm": best.max_depth_reached_mm,
        }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best is not None:
        export_best_candidates(
            report.candidates, OUT_DIR / "export", top_k=6
        )
        points = np.asarray(matrix.points_mm, dtype=np.float64)
        HTML_PATH.write_text(
            _html_report(payload, points, best, report.a0_reference),
            encoding="utf-8",
        )
    print(
        f"elapsed={elapsed:.1f}s  physics={report.stats.physics_simulations}  "
        f"queries_note=voir caches  anomalies={len(anomalies)}",
        flush=True,
    )
    if best is not None:
        print(
            f"best={best.family_id}  {best.coverage_percent:.3f}%  "
            f"{best.covered_points}/{best.total_points}  "
            f"poses={best.trajectory_steps}  Y->{best.max_depth_reached_mm:.1f}",
            flush=True,
        )
    if report.a0_reference is not None:
        print(
            f"A0 V2={report.a0_reference.coverage_percent:.3f}%  "
            f"(réf attendue {A0_V2_REFERENCE_PERCENT:.3f}%)",
            flush=True,
        )
    print(f"json {JSON_PATH}", flush=True)
    print(f"html {HTML_PATH}", flush=True)
    print("STOP after PRELIMINARY", flush=True)
    return 1 if anomalies else 0


if __name__ == "__main__":
    raise SystemExit(main())
