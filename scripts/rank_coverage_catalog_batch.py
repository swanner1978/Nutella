"""Sequential coverage ranking of A0 + 9 catalog shapes. Compute only.

Does not change CoverageSimulator, collision, KD-tree proximity, or the viewer.
Does not evaluate the full 100-candidate catalog.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.unit.engines.compute.coverage_catalog_fixtures import (  # noqa: E402
    load_generated_catalog,
)
from tests.unit.engines.compute.test_coverage_simulator import (  # noqa: E402
    A0_BASELINE_COVERAGE_PERCENT,
    A0_BASELINE_COVERED_AREA_MM2,
    A0_BASELINE_TARGET_AREA_MM2,
)

from nutella_scraper.engines.compute.candidate_coverage import (  # noqa: E402
    A0_BASELINE_FINGERPRINT,
    evaluate_rigid_candidate_batch,
    format_coverage_rank_report,
    select_and_materialize_catalog,
)
from nutella_scraper.engines.compute.coverage_simulator import (  # noqa: E402
    CoverageSimulator,
)


def main() -> int:
    surface, parameters, reference, catalog = load_generated_catalog(count=1000)
    selected, entries = select_and_materialize_catalog(
        catalog,
        surface=surface,
        parameters=parameters,
        reference=reference,
        count=10,
    )
    print("Selected candidates (evaluation order):")
    for index, item in enumerate(selected, start=1):
        print(
            f"  {index:2d}. {item.candidate_id:6s}  family={item.family:12s}  "
            f"valid={item.valid}  fp={item.shape_fingerprint[:20]}"
        )
    print()
    print("Evaluating sequentially (0–45°, step 2°, 17 SE(3) poses). A0 first.")
    simulator = CoverageSimulator(surface, parameters=parameters)
    batch = evaluate_rigid_candidate_batch(simulator, entries)
    print()
    print(format_coverage_rank_report(batch))
    print()
    a0 = batch.evaluated[0]
    print(
        "A0 vs baseline: "
        f"{a0.coverage_percent:.4f} % "
        f"({a0.covered_area_mm2:.2f} / {a0.useful_area_mm2:.2f} mm²) "
        f"faces={a0.touched_face_count} "
        f"poses={a0.valid_pose_count}/{a0.total_pose_count} "
        f"fp={a0.shape_fingerprint}"
    )
    print(
        "Expected: "
        f"{A0_BASELINE_COVERAGE_PERCENT:.4f} % "
        f"({A0_BASELINE_COVERED_AREA_MM2:.2f} / {A0_BASELINE_TARGET_AREA_MM2:.2f} mm²) "
        f"fp={A0_BASELINE_FINGERPRINT}"
    )
    print(f"Fingerprints unchanged: {batch.fingerprints_unchanged}")
    print(f"Total elapsed: {batch.total_elapsed_seconds:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
