"""Sequential coverage ranking of 100 catalog shapes. Compute only.

Uses the same generate_candidate_shapes catalog as the validated 10-shape
ranking (count=1000 → admissible curves). The first 10 selected are exactly
the known-ten reference set. Does not change CoverageSimulator, closest-point,
collision, SE(3), or the viewer. Sequential only — no parallelization.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.unit.engines.compute.coverage_catalog_fixtures import (  # noqa: E402
    load_generated_catalog,
)

from nutella_scraper.engines.compute.candidate_coverage import (  # noqa: E402
    KNOWN_TEN_RANKING,
    A0BaselineRegressionError,
    assert_a0_matches_baseline,
    geometry_unchanged,
    rank_candidate_coverage,
    select_and_materialize_catalog,
    snapshot_artifact_geometry,
)
from nutella_scraper.engines.compute.coverage_simulator import (  # noqa: E402
    CoverageSimulator,
    coverage_angle_samples_deg,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (  # noqa: E402
    BLADE_THICKNESS_MM,
)

OUT_DIR = ROOT / "output" / "coverage"
JSON_PATH = OUT_DIR / "candidate_coverage_100.json"
CSV_PATH = OUT_DIR / "candidate_coverage_100.csv"
N_CANDIDATES = 100
CATALOG_GENERATE_COUNT = 1000


def _pct4(value: float) -> float:
    return float(f"{float(value):.4f}")


def _known_ten_tuples(items: list) -> list[tuple[str, float]]:
    ranked = rank_candidate_coverage(items)
    return [(str(item.candidate_id), _pct4(item.coverage_percent)) for item in ranked]


def _preflight(catalog, selected, entries) -> dict:
    ids = [str(item.candidate_id) for item in selected]
    fingerprints = [str(item.shape_fingerprint) for item in selected]
    thicknesses = {float(item.shape.thickness_mm) for item in selected}
    has_mesh = [item.shape.vertices is not None for item in selected]
    invalid = [str(item.candidate_id) for item in selected if not item.valid]
    fp_counts = Counter(str(item.shape_fingerprint) for item in catalog)
    duplicate_fps = sorted(fp for fp, n in fp_counts.items() if n > 1)
    known_ids = [cid for cid, _pct in KNOWN_TEN_RANKING]
    missing_known = [cid for cid in known_ids if cid not in ids]
    families = dict(Counter(str(item.family) for item in selected))
    report = {
        "generated_count_arg": CATALOG_GENERATE_COUNT,
        "catalog_returned": len(catalog),
        "catalog_valid": sum(1 for item in catalog if item.valid),
        "catalog_unique_ids": len({str(item.candidate_id) for item in catalog}),
        "selected": len(selected),
        "unique_ids": len(set(ids)) == len(ids) == N_CANDIDATES,
        "unique_fingerprints": len(set(fingerprints)) == len(fingerprints) == N_CANDIDATES,
        "all_valid": not invalid,
        "invalid_ids": invalid,
        "thickness_mm": sorted(thicknesses),
        "thickness_ok": thicknesses == {float(BLADE_THICKNESS_MM)},
        "lightweight_curves": not any(has_mesh),
        "a0_first": ids[0] == "A0" if ids else False,
        "known_ten_present": not missing_known,
        "missing_known_ten": missing_known,
        "known_ten_are_prefix": ids[:10] == known_ids
        or set(ids[:10]) == set(known_ids),
        "selected_ids_prefix10": ids[:10],
        "families": families,
        "catalog_duplicate_fingerprints": len(duplicate_fps),
        "entries": len(entries),
    }
    print("PREFLIGHT")
    for key, value in report.items():
        print(f"  {key}: {value}")
    problems: list[str] = []
    if not report["unique_ids"]:
        problems.append("candidate ids are not unique")
    if not report["unique_fingerprints"]:
        problems.append("selected fingerprints are not unique")
    if not report["all_valid"]:
        problems.append(f"invalid candidates: {invalid}")
    if not report["thickness_ok"]:
        problems.append(f"thickness {thicknesses} != {BLADE_THICKNESS_MM}")
    if not report["lightweight_curves"]:
        problems.append("catalog rows already carry meshes (not 1D curves)")
    if not report["a0_first"]:
        problems.append("A0 is not first")
    if missing_known:
        problems.append(f"known-ten missing from selected 100: {missing_known}")
    if problems:
        raise SystemExit("PREFLIGHT FAILED: " + "; ".join(problems))
    print("PREFLIGHT OK")
    print()
    return report


def _row(
    *,
    rank: int | None,
    item,
    family: str,
    curve_length_mm: float,
    covered_face_ids: list[int],
    target_area_mm2: float,
) -> dict:
    return {
        "rank": rank,
        "candidate_id": str(item.candidate_id),
        "family": str(family),
        "coverage_percent": float(item.coverage_percent),
        "covered_area_mm2": float(item.covered_area_mm2),
        "target_area_mm2": float(target_area_mm2),
        "covered_face_ids": covered_face_ids,
        "valid_pose_count": int(item.valid_pose_count),
        "total_pose_count": int(item.total_pose_count),
        "curve_length_mm": float(curve_length_mm),
        "shape_fingerprint": str(item.shape_fingerprint),
        "evaluation_time_seconds": float(item.elapsed_seconds),
        "touched_face_count": int(item.touched_face_count),
    }


def _write_outputs(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = [
        "rank",
        "candidate_id",
        "family",
        "coverage_percent",
        "covered_area_mm2",
        "target_area_mm2",
        "touched_face_count",
        "valid_pose_count",
        "total_pose_count",
        "curve_length_mm",
        "evaluation_time_seconds",
        "shape_fingerprint",
        "covered_face_ids",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["ranked"]:
            out = dict(row)
            out["covered_face_ids"] = ";".join(str(i) for i in row["covered_face_ids"])
            writer.writerow(out)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    angles = coverage_angle_samples_deg()
    print(
        "Coverage 100 — sequential, 0–45° "
        f"({len(angles)} angles: {angles[0]:g} … {angles[-2]:g}, {angles[-1]:g})."
    )
    print("No parallelization. No viewer. Same generator as the known-ten ranking.")
    print()

    surface, parameters, reference, catalog = load_generated_catalog(
        count=CATALOG_GENERATE_COUNT
    )
    selected, entries = select_and_materialize_catalog(
        catalog,
        surface=surface,
        parameters=parameters,
        reference=reference,
        count=N_CANDIDATES,
    )
    preflight = _preflight(catalog, selected, entries)
    by_id = {str(item.candidate_id): item for item in selected}

    simulator = CoverageSimulator(surface, parameters=parameters)
    snapshots = {}
    for candidate_id, _family, artifact in entries:
        key = str(candidate_id)
        simulator.register(key, artifact)
        snapshots[key] = snapshot_artifact_geometry(artifact)

    from nutella_scraper.engines.compute.candidate_coverage import (  # noqa: E402
        candidate_result_from_coverage,
    )

    evaluated = []
    extra: dict[str, dict] = {}
    t_all = time.perf_counter()
    n = len(entries)

    for index, (candidate_id, family, artifact) in enumerate(entries):
        key = str(candidate_id)
        snapshot = snapshots[key]
        remaining = n - index
        done_times = [item.elapsed_seconds for item in evaluated]
        if done_times:
            mean_s = statistics.fmean(done_times)
            eta_s = mean_s * remaining
            eta = f"ETA {eta_s / 60.0:.1f} min"
        else:
            eta = "ETA n/a"
        print(f"[{index + 1:3d}/{n}] start {key:6s}  family={family:12s}  {eta}")
        t0 = time.perf_counter()
        coverage = simulator.evaluate_candidate(key)
        elapsed = time.perf_counter() - t0
        if not geometry_unchanged(artifact, snapshot):
            raise ValueError(f"candidate {key!r} deformed during coverage evaluation")
        if str(coverage.shape_fingerprint) != snapshot.fingerprint:
            raise ValueError(f"candidate {key!r} fingerprint changed")
        item = candidate_result_from_coverage(
            coverage, family=family, elapsed_seconds=elapsed
        )
        if index == 0:
            assert_a0_matches_baseline(item)
        evaluated.append(item)
        extra[key] = {
            "family": family,
            "curve_length_mm": float(by_id[key].curve_length_mm),
            "covered_face_ids": sorted(int(i) for i in coverage.covered_face_ids),
            "target_area_mm2": float(coverage.target_area_mm2),
        }
        print(
            f"[{index + 1:3d}/{n}] done  {key:6s}  "
            f"{item.coverage_percent:.4f} %  "
            f"faces={item.touched_face_count}  "
            f"poses={item.valid_pose_count}/{item.total_pose_count}  "
            f"{elapsed:.2f} s"
        )

        if index + 1 == len(KNOWN_TEN_RANKING):
            got = _known_ten_tuples(evaluated)
            expected = [(cid, _pct4(pct)) for cid, pct in KNOWN_TEN_RANKING]
            print()
            print("KNOWN-TEN GATE")
            print("  expected:", expected)
            print("  got     :", got)
            if got != expected:
                print("STOP: known-ten ranking changed. Not evaluating the remaining candidates.")
                return 1
            print("  MATCH — continuing to 100.")
            print()

    total_s = time.perf_counter() - t_all
    ranked = rank_candidate_coverage(evaluated)
    times = [float(item.elapsed_seconds) for item in evaluated]
    a0 = next(item for item in evaluated if item.candidate_id == "A0")
    best = ranked[0]
    a0_rank = next(i for i, item in enumerate(ranked, start=1) if item.candidate_id == "A0")
    n_above = sum(1 for item in ranked if item.coverage_percent > a0.coverage_percent + 1e-9)
    n_tie = sum(
        1
        for item in ranked
        if abs(item.coverage_percent - a0.coverage_percent) <= 1e-4
        and item.candidate_id != "A0"
    )
    payload = {
        "meta": {
            "sector_deg": [0.0, 45.0],
            "evaluated_angles_deg": list(angles),
            "n_angles": len(angles),
            "multiply_by_four": False,
            "catalog_generate_count": CATALOG_GENERATE_COUNT,
            "n_selected": n,
            "preflight": preflight,
            "known_ten_gate": "MATCH",
            "fingerprints_unchanged": True,
            "a0_rank": a0_rank,
            "best_candidate_id": best.candidate_id,
            "best_coverage_percent": float(best.coverage_percent),
            "a0_coverage_percent": float(a0.coverage_percent),
            "best_minus_a0_percent": float(best.coverage_percent - a0.coverage_percent),
            "n_above_a0": n_above,
            "n_tie_a0": n_tie,
            "total_seconds": total_s,
            "mean_seconds": float(statistics.fmean(times)),
            "median_seconds": float(statistics.median(times)),
            "min_seconds": float(min(times)),
            "max_seconds": float(max(times)),
            "a0_seconds": float(a0.elapsed_seconds),
        },
        "evaluated_order": [str(item.candidate_id) for item in evaluated],
        "ranked": [
            _row(
                rank=rank,
                item=item,
                family=extra[item.candidate_id]["family"],
                curve_length_mm=extra[item.candidate_id]["curve_length_mm"],
                covered_face_ids=extra[item.candidate_id]["covered_face_ids"],
                target_area_mm2=extra[item.candidate_id]["target_area_mm2"],
            )
            for rank, item in enumerate(ranked, start=1)
        ],
    }
    _write_outputs(payload)
    print()
    print("Rank | Candidate | Family | Coverage | Area | Valid poses | Time")
    print()
    for row in payload["ranked"]:
        print(
            f"{row['rank']} | {row['candidate_id']} | {row['family']} | "
            f"{row['coverage_percent']:.4f} % | {row['covered_area_mm2']:.2f} | "
            f"{row['valid_pose_count']}/{row['total_pose_count']} | "
            f"{row['evaluation_time_seconds']:.2f} s"
        )
    print()
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(
        f"total={total_s:.2f}s mean={payload['meta']['mean_seconds']:.2f}s "
        f"median={payload['meta']['median_seconds']:.2f}s "
        f"min={payload['meta']['min_seconds']:.2f}s max={payload['meta']['max_seconds']:.2f}s "
        f"A0={a0.elapsed_seconds:.2f}s"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A0BaselineRegressionError as exc:
        print(f"STOP: {exc}")
        raise SystemExit(1) from exc
