#!/usr/bin/env python3
"""Dump the interior 5 mm reference matrix. Does not run CoverageSimulator."""

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
    diagnose_coverage_reference_matrix,
)
from nutella_scraper.engines.compute.interior_surface_reference import (  # noqa: E402
    load_interior_surface_reference,
)


def main() -> int:
    models = _ROOT / "output" / "models"
    cache = next(iter(sorted(models.glob("*/interior_product_surface.npz"))), None)
    if cache is None:
        raise SystemExit("Aucun interior_product_surface.npz")
    model_id = cache.parent.name
    interior = load_interior_surface_reference(models_root=models, model_id=model_id)
    matrix = build_coverage_reference_matrix(interior)
    payload = diagnose_coverage_reference_matrix(matrix)
    payload["model_id"] = model_id
    out = _ROOT / "output" / "coverage" / "reference_matrix_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "model_id",
        "point_count",
        "y_min_mm",
        "y_max_mm",
        "azimuth_min_deg",
        "azimuth_max_deg",
        "mean_vertical_spacing_mm",
        "mean_tangential_spacing_mm",
        "neighbor_min_mm",
        "neighbor_max_mm",
        "fingerprint",
        "on_interior_envelope",
        "any_point_outside_envelope",
        "max_distance_to_interior_mm",
        "coverage_target_region",
        "coverage_target_azimuth_range",
    )}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
