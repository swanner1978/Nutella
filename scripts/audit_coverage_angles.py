#!/usr/bin/env python3
"""Replay CoverageSimulator for A0 / S0008 / S0010 only. Dump per-angle poses.

Does not evaluate any other candidate. Does not change coverage physics.
Aborts if union(faces) != covered_face_ids or union(area) != covered_area_mm2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.unit.engines.compute.coverage_catalog_fixtures import (  # noqa: E402
    load_generated_catalog,
)

from nutella_scraper.engines.compute.candidate_coverage import (  # noqa: E402
    materialize_catalog_candidate,
)
from nutella_scraper.engines.compute.coverage_angle_audit import (  # noqa: E402
    AUDIT_CANDIDATE_IDS,
    CoverageUnionMismatchError,
    build_angle_audit,
    format_comparative_report,
)
from nutella_scraper.engines.compute.coverage_simulator import (  # noqa: E402
    CoverageSimulator,
)
from nutella_scraper.engines.visualization.coverage_rank_catalog import (  # noqa: E402
    load_coverage_rank_json,
)

OUT_DIR = ROOT / "output" / "coverage" / "angle_audit"
SAVED_JSON = ROOT / "output" / "coverage" / "candidate_coverage_100.json"
REPORT_PATH = ROOT / "output" / "coverage" / "angle_audit_compare.txt"


def main() -> int:
    saved = load_coverage_rank_json(SAVED_JSON)
    saved_by_id = {str(row["candidate_id"]): row for row in saved["ranked"]}
    missing = [cid for cid in AUDIT_CANDIDATE_IDS if cid not in saved_by_id]
    if missing:
        raise SystemExit(f"IDs absents du JSON sauvé: {missing}")

    surface, parameters, reference, catalog = load_generated_catalog(count=1000)
    by_id = {str(item.candidate_id): item for item in catalog}
    simulator = CoverageSimulator(surface, parameters=parameters)
    mesh = simulator._surface_mesh
    audits: list[dict] = []
    for cid in AUDIT_CANDIDATE_IDS:
        shape = by_id[cid]
        artifact = materialize_catalog_candidate(
            shape,
            surface=surface,
            parameters=parameters,
            reference=reference,
        )
        simulator.register(cid, artifact)
        result = simulator.evaluate_candidate(cid)
        try:
            payload = build_angle_audit(
                result,
                areas=simulator._areas,
                centroids=mesh.triangles_center,
                control_points_mm=shape.control_points_mm,
                rest_vertices=artifact.mesh.vertices,
                family=str(shape.family),
                saved_row=saved_by_id[cid],
            )
        except CoverageUnionMismatchError as exc:
            print(f"INCOHERENCE: {exc}", file=sys.stderr)
            return 2
        audits.append(payload)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{cid}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {path}")

    report = format_comparative_report(audits)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    combined = OUT_DIR / "compare.json"
    combined.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    print(report)
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
