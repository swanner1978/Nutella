"""A0 closest-point fast vs legacy. Does not change coverage physics."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.unit.engines.compute.test_coverage_simulator import (  # noqa: E402
    A0_BASELINE_COVERAGE_PERCENT,
    A0_BASELINE_COVERED_AREA_MM2,
    A0_BASELINE_FACE_IDS,
    A0_BASELINE_FINGERPRINT,
    A0_BASELINE_TARGET_AREA_MM2,
    _a0_parameters,
    _fast_surface,
)

from nutella_scraper.engines.compute.coverage_simulator import (  # noqa: E402
    CoverageSimulator,
)
from nutella_scraper.engines.compute.envelope_surface_proximity import (  # noqa: E402
    PROXIMITY_STATS,
    EnvelopeSurfaceProximity,
    reset_proximity_stats,
)
from nutella_scraper.engines.compute.scraper_rigid_motion import (  # noqa: E402
    build_rigid_scraper_artifact,
)


def _run(label: str, method_name: str) -> dict[str, object]:
    surface = _fast_surface()
    params = _a0_parameters(surface)
    artifact = build_rigid_scraper_artifact(surface, params)
    original = EnvelopeSurfaceProximity.closest_on_surface
    method = getattr(EnvelopeSurfaceProximity, method_name)
    closest_s = 0.0

    def timed(self, points):  # type: ignore[no-untyped-def]
        nonlocal closest_s
        started = time.perf_counter()
        out = method(self, points)
        closest_s += time.perf_counter() - started
        return out

    EnvelopeSurfaceProximity.closest_on_surface = timed  # type: ignore[method-assign]
    try:
        simulator = CoverageSimulator(surface, parameters=params)
        simulator.register("A0", artifact)
        reset_proximity_stats()
        t0 = time.perf_counter()
        result = simulator.evaluate_candidate("A0")
        elapsed = time.perf_counter() - t0
    finally:
        EnvelopeSurfaceProximity.closest_on_surface = original  # type: ignore[method-assign]
    n_valid = sum(1 for _a, pose in result.best_pose_by_angle if pose is not None)
    print(
        f"{label}: {elapsed:.2f} s  coverage={result.coverage_percent:.4f}%  "
        f"faces={len(result.covered_face_ids)}  poses={n_valid}/24  "
        f"points={PROXIMITY_STATS['points']}  "
        f"triangles={PROXIMITY_STATS['triangles_examined']}  "
        f"fallback={PROXIMITY_STATS['tied_fallback_points']}  "
        f"empty={PROXIMITY_STATS['empty_ball_points']}  "
        f"fast_pts={PROXIMITY_STATS['fast_path_points']}  "
        f"expand={PROXIMITY_STATS.get('bound_expand_points', 0)}  "
        f"closest={closest_s:.2f}s"
    )
    return {
        "elapsed": elapsed,
        "closest_s": closest_s,
        "result": result,
        "n_valid": n_valid,
        "stats": dict(PROXIMITY_STATS),
    }


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    print("A0 closest-point benchmark (legacy then fast). Same 24 angles x 17 poses.")
    before = _run("LEGACY", "closest_on_surface_legacy")
    after = _run("FAST  ", "closest_on_surface_fast")
    r0 = before["result"]
    r1 = after["result"]
    ok = (
        abs(r0.coverage_percent - A0_BASELINE_COVERAGE_PERCENT) <= 1e-6
        and abs(r1.coverage_percent - A0_BASELINE_COVERAGE_PERCENT) <= 1e-6
        and r0.covered_face_ids == A0_BASELINE_FACE_IDS == r1.covered_face_ids
        and abs(r0.covered_area_mm2 - A0_BASELINE_COVERED_AREA_MM2) <= 1e-2
        and abs(r1.covered_area_mm2 - A0_BASELINE_COVERED_AREA_MM2) <= 1e-2
        and abs(r0.target_area_mm2 - A0_BASELINE_TARGET_AREA_MM2) <= 1e-2
        and r0.shape_fingerprint == r1.shape_fingerprint == A0_BASELINE_FINGERPRINT
        and before["n_valid"] == after["n_valid"] == 24
    )
    saved = float(before["elapsed"]) - float(after["elapsed"])
    gain = 100.0 * saved / float(before["elapsed"]) if before["elapsed"] else 0.0
    print(f"A0 identity: {'PASS' if ok else 'FAIL'}")
    print(f"time before={before['elapsed']:.2f} s  after={after['elapsed']:.2f} s  "
          f"saved={saved:.2f} s ({gain:.1f} %)")
    print(f"points before={before['stats']['points']} after={after['stats']['points']}")
    print(
        "triangles before="
        f"{before['stats']['triangles_examined']} after={after['stats']['triangles_examined']}"
    )
    print(
        "fallback before="
        f"{before['stats']['tied_fallback_points']} after={after['stats']['tied_fallback_points']}"
    )
    print(
        "closest-point before="
        f"{before['closest_s']:.2f} s after={after['closest_s']:.2f} s"
    )
    print(f"bound-expand after={after['stats'].get('bound_expand_points', 0)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
