"""HEURISTIC length pretest: A0 + 6 short 2 mm blades. Not a 35-form campaign."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

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
from nutella_scraper.engines.compute.shape_constraints import (  # noqa: E402
    MAX_CURVATURE_MM_INV,
    MILD_SAG_FRACTION,
)
from nutella_scraper.engines.compute.shape_families import (  # noqa: E402
    SCRAPER_THICKNESS_MM,
    SCRAPER_WIDTH_MM,
)
from nutella_scraper.engines.compute.shape_placement_diagnostic import (  # noqa: E402
    compare_a0_and_straight40,
)
from nutella_scraper.engines.compute.shape_search import (  # noqa: E402
    descending_contact_trajectory,
    length_pretest_config,
    report_to_payload,
    search_scraper_shapes,
)
from nutella_scraper.engines.compute.shape_validation import (  # noqa: E402
    check_covered_indices_in_matrix,
    check_trajectory_rules,
    check_union_not_sum,
)

OUT_DIR = _ROOT / "output" / "coverage" / "shape_length_pretest"
JSON_PATH = OUT_DIR / "shape_length_pretest.json"
HTML_PATH = OUT_DIR / "shape_length_pretest.html"
CACHE_DIR = OUT_DIR / "caches"
A0_V2_CACHE = _ROOT / "output" / "coverage" / "trajectory_model_v2_a0_pose_cache.json"
MODEL_ID = "9a1eee56-d73e-46fb-93d1-25d4ea856874"
LABEL = "HEURISTIC"
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
    if "t=2.5" in fingerprint and A0_V2_CACHE.exists():
        blob = json.loads(A0_V2_CACHE.read_text(encoding="utf-8"))
        if (
            str(blob.get("scraper_fingerprint")) == fingerprint
            and int(blob.get("n_specs", -1)) == len(specs)
        ):
            print("    reuse cache A0 V2", flush=True)
            cache = cache_from_payload(
                blob["entries"],
                n_points=int(blob["n_points"]),
                fingerprint=str(blob["scraper_fingerprint"]),
                angle_window_deg=tuple(blob["angle_window_deg"]),
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(blob), encoding="utf-8")
            return cache
    print(f"    physique {fingerprint[:48]} n={len(specs)}", flush=True)
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


def _instrumentation_row(item) -> dict:
    return {
        "forme": item.family_id,
        "longueur_mm": float(item.scraper_length_mm),
        "epaisseur_mm": float(item.thickness_mm),
        "largeur_mm": float(item.width_mm),
        "poses_candidates": int(item.n_pose_candidates),
        "poses_admissibles": int(item.n_admissible_poses),
        "poses_contact": int(item.n_contacting_poses),
        "poses_accessibles": int(item.n_reachable_poses),
        "trajectoire": "oui" if item.trajectory_found else "non",
        "nb_poses_trajet": int(item.trajectory_steps),
        "longueur_trajet_mm": float(item.trajectory_length_mm),
        "changements_direction": int(item.direction_changes),
        "points": int(item.covered_points),
        "couverture_percent": float(item.coverage_percent),
        "profondeur_max_mm": float(item.max_depth_reached_mm),
        "depart_ouverture": bool(item.opening_start_available),
        "fond_atteint": bool(item.floor_reached),
        "fin": item.termination_reason,
    }


def _html(payload: dict) -> str:
    rows = []
    for row in payload["table"]:
        rows.append(
            "<tr>"
            f"<td>{row['forme']}</td>"
            f"<td>{row['longueur_mm']:.1f}</td>"
            f"<td>{row['epaisseur_mm']:.1f}</td>"
            f"<td>{row['poses_admissibles']}</td>"
            f"<td>{row['trajectoire']}</td>"
            f"<td>{row['points']}</td>"
            f"<td>{row['couverture_percent']:.2f}%</td>"
            f"<td>{row['fin']}</td>"
            "</tr>"
        )
    place = payload.get("placement_diagnostic", {})
    a0 = place.get("a0", {})
    blade = place.get("straight_40_2mm", {})
    gate = payload.get("pretest_gate_passed")
    rec = payload.get("recommendation", "")
    return (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'/>"
        "<title>HEURISTIC length pretest</title>"
        "<style>body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:24px}"
        "td,th{border:1px solid #444;padding:6px 8px}table{border-collapse:collapse}"
        ".note{color:#bbb;max-width:960px}</style></head><body>"
        "<h1>HEURISTIC — pretest longueurs courtes</h1>"
        "<p class='note'>Lame 2.0 x 2.0 mm. Longueur explicite 20-50 mm. "
        "Pas un optimum. Nuage interior_matrix_a0_0_90. Campagne 5x7 NON lancee.</p>"
        f"<p>Porte pretest: {'OUI' if gate else 'NON — STOP'}</p>"
        f"<p>{rec}</p>"
        "<table><tr><th>Forme</th><th>Longueur</th><th>Epaisseur</th>"
        "<th>Poses adm.</th><th>Trajectoire</th><th>Points</th>"
        "<th>Couverture</th><th>Fin</th></tr>"
        + "".join(rows)
        + "</table>"
        "<h2>Placement A0 vs straight 40 mm / 2 mm</h2>"
        "<pre>"
        + json.dumps(
            {
                "origin_delta_mm": place.get("origin_delta_mm"),
                "a0_extents": a0.get("local_extents_mm"),
                "straight_extents": blade.get("local_extents_mm"),
                "a0_admissible": a0.get("admissible"),
                "straight_admissible": blade.get("admissible"),
                "anomalies": place.get("anomalies"),
            },
            indent=2,
        )
        + "</pre></body></html>"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("HEURISTIC length pretest — 6 lames 2 mm + A0. STOP avant 5x7.", flush=True)
    surface = load_interior_surface_reference(
        models_root=_ROOT / "output" / "models",
        model_id=MODEL_ID,
    )
    print("  diagnostic placement A0 vs straight 40 mm / 2 mm", flush=True)
    placement = compare_a0_and_straight40(surface)
    (OUT_DIR / "placement_a0_vs_straight40.json").write_text(
        json.dumps(placement, indent=2), encoding="utf-8"
    )
    print(
        f"    origin_delta={placement['origin_delta_mm']:.3f} mm  "
        f"straight_extents={placement['straight_40_2mm']['local_extents_mm']}  "
        f"a0_extents={placement['a0']['local_extents_mm']}",
        flush=True,
    )
    for note in placement["anomalies"]:
        print(f"    ANOMALIE placement: {note}", flush=True)

    matrix = build_coverage_reference_matrix(surface)
    started = time.perf_counter()
    report = search_scraper_shapes(
        surface,
        matrix=matrix,
        config=length_pretest_config(),
        physics_builder=_disk_physics,
    )
    elapsed = time.perf_counter() - started
    anomalies: list[str] = list(placement["anomalies"])
    for item in report.candidates:
        anomalies.extend(check_union_not_sum(item))
        anomalies.extend(check_covered_indices_in_matrix(item, matrix))
        anomalies.extend(check_trajectory_rules(item, report.grid))
    if report.a0_reference is not None:
        anomalies.extend(check_union_not_sum(report.a0_reference))
        anomalies.extend(check_trajectory_rules(report.a0_reference, report.grid))

    ranked = []
    if report.a0_reference is not None:
        ranked.append(report.a0_reference)
    ranked.extend(report.candidates)
    ranked.sort(
        key=lambda item: (-int(item.covered_points), float(item.trajectory_length_mm))
    )
    table = [_instrumentation_row(item) for item in ranked]
    gate = any(descending_contact_trajectory(item) for item in report.candidates)
    if gate:
        recommendation = (
            "Poursuivre: au moins une lame 2 mm a une trajectoire descendante "
            "avec contact UNION > 0. La campagne 5 familles x 7 longueurs "
            "peut etre lancee ensuite. Toujours HEURISTIC."
        )
    else:
        recommendation = (
            "STOP. Aucune lame 2 mm n'a produit de trajectoire descendante "
            "avec contact. Ne pas lancer la campagne 5x7. Diagnostiquer "
            "placement / orientation / section avant toute recherche."
        )

    payload = report_to_payload(report)
    payload.update(
        {
            "label": LABEL,
            "stage": "LENGTH_PRETEST",
            "trajectory_model": TRAJECTORY_MODEL,
            "campaign_35_launched": False,
            "blade_thickness_mm": SCRAPER_THICKNESS_MM,
            "blade_width_mm": SCRAPER_WIDTH_MM,
            "max_curvature_mm_inv": MAX_CURVATURE_MM_INV,
            "mild_sag_fraction": MILD_SAG_FRACTION,
            "max_curvature_derivation": (
                "sag <= 8% of L=20 mm -> R = L^2/(8s) = 31.25 mm -> "
                "MAX_CURVATURE_MM_INV = 0.032 mm^-1"
            ),
            "a0_v2_reference_expected_percent": A0_V2_REFERENCE_PERCENT,
            "a0_v2_reference_expected_points": A0_V2_REFERENCE_POINTS,
            "elapsed_seconds": elapsed,
            "anomalies": anomalies,
            "table": table,
            "placement_diagnostic": placement,
            "pretest_gate_passed": gate,
            "recommendation": recommendation,
            "disclaimer": (
                "HEURISTIC. Meilleure trajectoire trouvee dans ce pretest, "
                "pas un optimum global. Campagne 5x7 non lancee."
            ),
        }
    )
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    HTML_PATH.write_text(_html(payload), encoding="utf-8")
    print(
        f"elapsed={elapsed:.1f}s  physics={report.stats.physics_simulations}  "
        f"gate={gate}  anomalies={len(anomalies)}",
        flush=True,
    )
    print(recommendation, flush=True)
    print(f"json {JSON_PATH}", flush=True)
    print(f"html {HTML_PATH}", flush=True)
    print("STOP after LENGTH_PRETEST", flush=True)
    return 0 if gate and not anomalies else 1


if __name__ == "__main__":
    raise SystemExit(main())
