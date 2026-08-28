"""Small validation of the shape-search pipeline. Not a coverage campaign.

Checks:
  1. straight line, historical A0, simple Bézier
  2. target is interior_matrix_a0_0_90 (tiny stand-in in tests)
  3. coverage is a UNION of touched points
  4. trajectories do not climb / teleport
  5. results are labelled HEURISTIC

If any check fails the process exits non-zero with STOP.
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

from nutella_scraper.engines.compute.shape_export import (  # noqa: E402
    export_best_candidates,
)
from nutella_scraper.engines.compute.shape_search import (  # noqa: E402
    report_to_payload,
    search_scraper_shapes,
    validation_config,
)


def _fail(message: str) -> None:
    print(f"STOP: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    from tests.unit.engines.compute.test_coverage_simulator import _fast_surface
    from tests.unit.engines.compute.test_trajectory_search import _tiny_reference_matrix

    surface = _fast_surface()
    matrix = _tiny_reference_matrix()
    report = search_scraper_shapes(
        surface,
        matrix=matrix,
        config=validation_config(),
    )
    if report.target_definition != "interior_matrix_a0_0_90":
        _fail("la cible n'est pas interior_matrix_a0_0_90")
    if report.grid.uses_legacy_a0_point_matrix:
        _fail("la grille A0 historique a été utilisée")
    if report.optimization_label != "HEURISTIC":
        _fail("le résultat n'est pas étiqueté HEURISTIC")
    if report.a0_reference is None:
        _fail("référence A0 absente")
    families = {item.family_id for item in report.candidates}
    if "straight" not in families or "bezier_4" not in families:
        _fail("le trio de validation n'a pas produit droite + Bézier")
    for item in report.candidates:
        if item.physical_valid and item.covered_points != len(item.covered_point_indices):
            _fail("la couverture n'est pas une UNION de points distincts")
        if item.covered_points > item.total_points:
            _fail("plus de points touchés que de points dans le nuage")
        if item.physical_valid:
            ys = [y for y, _az in item.trajectory_poses]
            if any(nxt > prev + 1e-6 for prev, nxt in zip(ys[:-1], ys[1:], strict=True)):
                _fail("Y remonte")
            if item.family_id != "A0" and abs(item.thickness_mm - 2.0) > 1e-9:
                _fail("lame candidate n'a pas 2 mm d'épaisseur")
    payload = report_to_payload(report)
    out = Path("output/coverage/shape_search_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if report.candidates:
        export_best_candidates(report.candidates, out.parent / "shape_search_validation")
    a0 = report.a0_reference
    best = report.candidates[0]
    print("Validation trio OK (HEURISTIC, pas un optimum).")
    print(
        f"A0 historique : couverture = {a0.coverage_percent:.4f}% "
        f"({a0.covered_points}/{a0.total_points})"
    )
    print(
        f"Meilleure forme trouvée : {best.family_id} {best.candidate_id} "
        f"= {best.coverage_percent:.4f}%  ({best.covered_points}/{best.total_points})"
    )
    print(f"Gain : {best.covered_points - a0.covered_points:+d} points de couverture")
    print(f"Stats : {report.stats.to_payload()}")


if __name__ == "__main__":
    main()
