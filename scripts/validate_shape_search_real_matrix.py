"""VALIDATION_REAL_MATRIX — 4 shapes on interior_matrix_a0_0_90.

Does not launch MAX_SHAPE_EVALUATIONS=100. Exits non-zero on STOP.
"""

from __future__ import annotations

import json
import sys
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
from nutella_scraper.engines.compute.shape_search import (  # noqa: E402
    real_matrix_validation_config,
    search_scraper_shapes,
)
from nutella_scraper.engines.compute.shape_validation import (  # noqa: E402
    VALIDATION_LABEL,
    case_row,
    check_labels,
    ordered_validation_cases,
    run_mandatory_checks,
)


def _fail(message: str) -> None:
    print(f"STOP: {message}", file=sys.stderr)
    raise SystemExit(1)


def _load_real_surface():
    models = _ROOT / "output" / "models"
    caches = sorted(
        models.glob("*/interior_product_surface.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not caches:
        _fail("aucun interior_product_surface.npz — géométrie cible introuvable")
    cache = caches[0]
    model_id = cache.parent.name
    print(f"modèle = {model_id}  (npz le plus récent)")
    return load_interior_surface_reference(models_root=models, model_id=model_id), model_id


def main() -> None:
    surface, model_id = _load_real_surface()
    print("construction de interior_matrix_a0_0_90 …")
    matrix = build_coverage_reference_matrix(surface)
    print(
        f"nuage = {matrix.point_count} points  "
        f"région={matrix.coverage_target_region}  "
        f"azimut={matrix.coverage_target_azimuth_range}"
    )
    print("évaluation A0 + straight + arc + bezier_4 (pas de campagne) …")
    report = search_scraper_shapes(
        surface,
        matrix=matrix,
        config=real_matrix_validation_config(),
    )
    ok, errors = run_mandatory_checks(matrix=matrix, surface=surface, report=report)
    cases = ordered_validation_cases(report)
    payload = {
        "label": VALIDATION_LABEL,
        "campaign_launched": False,
        "max_shape_evaluations_campaign": 100,
        "model_id": model_id,
        "target_definition": matrix.coverage_target_region,
        "point_count": int(matrix.point_count),
        "uses_legacy_a0_point_matrix": bool(matrix.uses_legacy_a0_point_matrix),
        "on_interior_envelope": bool(matrix.on_interior_envelope),
        "any_point_outside_envelope": bool(matrix.any_point_outside_envelope),
        "symmetry_multiplier_applied": bool(matrix.symmetry_multiplier_applied),
        "pipeline_ok": bool(ok),
        "anomalies": list(errors),
        "cas": [case_row(item) for item in cases],
        "classement_couverture": [
            {
                "forme": item.family_id,
                "couverture_percent": item.coverage_percent,
                "points": item.covered_points,
            }
            for item in sorted(cases, key=lambda row: -row.covered_points)
        ],
        "stats": report.stats.to_payload(),
        "peut_lancer_la_campagne_complete": bool(ok),
    }
    label_errors = check_labels(json.dumps(payload))
    if label_errors:
        ok = False
        payload["pipeline_ok"] = False
        payload["peut_lancer_la_campagne_complete"] = False
        payload["anomalies"] = list(payload["anomalies"]) + list(label_errors)
    out = _ROOT / "output" / "coverage" / "shape_search_validation_real_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"label": VALIDATION_LABEL, "cas": payload["cas"]}, indent=2))
    print(f"anomalies = {payload['anomalies']}")
    print(f"pipeline_ok = {ok}")
    print(f"wrote {out}")
    if not ok:
        _fail("; ".join(payload["anomalies"]) or "incohérence physique ou géométrique")
    print("VALIDATION_REAL_MATRIX: les 4 cas sont cohérents. Campagne 100×14 NON lancée.")


if __name__ == "__main__":
    main()
