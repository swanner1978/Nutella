"""Sequential vs batch coverage on the validated 10-shape set. Compute only.

Does not change collision, KD-tree proximity, A0, or the viewer.
Does not evaluate 100 candidates.
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

from nutella_scraper.engines.compute.candidate_coverage import (  # noqa: E402
    KNOWN_TEN_RANKING,
    evaluate_rigid_candidate_batch,
    format_coverage_rank_report,
    select_and_materialize_catalog,
)
from nutella_scraper.engines.compute.coverage_simulator import (  # noqa: E402
    CoverageSimulator,
)


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _ids_and_coverage(batch) -> list[tuple[str, float]]:
    return [
        (item.candidate_id, float(item.coverage_percent))
        for item in batch.ranked
    ]


def _ranking_matches(
    actual: list[tuple[str, float]],
    expected: list[tuple[str, float]],
) -> bool:
    if [item[0] for item in actual] != [item[0] for item in expected]:
        return False
    return all(
        abs(left[1] - right[1]) <= 1e-3
        for left, right in zip(actual, expected, strict=True)
    )


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    surface, parameters, reference, catalog = load_generated_catalog(count=1000)
    selected, entries = select_and_materialize_catalog(
        catalog,
        surface=surface,
        parameters=parameters,
        reference=reference,
        count=10,
    )
    print("Candidates:")
    for item in selected:
        print(f"  {item.candidate_id:6s}  {item.family}")
    print()

    rss0 = _rss_mb()
    sequential_sim = CoverageSimulator(surface, parameters=parameters)
    print("Running sequential evaluate_candidate loop...")
    sequential = evaluate_rigid_candidate_batch(
        sequential_sim, entries, use_batch_invariants=False
    )
    rss_seq = _rss_mb()
    print("Sequential (evaluate_candidate loop)")
    print(format_coverage_rank_report(sequential))
    print(f"elapsed: {sequential.total_elapsed_seconds:.2f} s")
    print()

    batch_sim = CoverageSimulator(surface, parameters=parameters)
    print("Running evaluate_candidates_batch...")
    batched = evaluate_rigid_candidate_batch(
        batch_sim, entries, use_batch_invariants=True
    )
    rss_batch = _rss_mb()
    print("Batch (evaluate_candidates_batch, jar invariants once)")
    print(format_coverage_rank_report(batched))
    print(f"elapsed: {batched.total_elapsed_seconds:.2f} s")
    print()

    seq_ids = _ids_and_coverage(sequential)
    batch_ids = _ids_and_coverage(batched)
    known = [(cid, pct) for cid, pct in KNOWN_TEN_RANKING]
    vs_seq = _ranking_matches(seq_ids, batch_ids)
    vs_known = _ranking_matches(batch_ids, known)
    print("Ranking vs sequential:", "MATCH" if vs_seq else "DIFF")
    print("Ranking vs known 10:", "MATCH" if vs_known else "DIFF")
    if not vs_seq:
        print(" sequential", [(i, round(p, 4)) for i, p in seq_ids])
        print(" batch     ", [(i, round(p, 4)) for i, p in batch_ids])
    if not vs_known:
        print(" known     ", known)
        print(" batch     ", [(i, round(p, 4)) for i, p in batch_ids])

    saved = sequential.total_elapsed_seconds - batched.total_elapsed_seconds
    gain = (
        100.0 * saved / sequential.total_elapsed_seconds
        if sequential.total_elapsed_seconds > 0
        else 0.0
    )
    print()
    print(
        "Fingerprints unchanged sequential="
        f"{sequential.fingerprints_unchanged} batch={batched.fingerprints_unchanged}"
    )
    print(f"Time sequential: {sequential.total_elapsed_seconds:.2f} s")
    print(f"Time batch:      {batched.total_elapsed_seconds:.2f} s")
    print(f"Saved:           {saved:.2f} s  ({gain:.1f} %)")
    if rss0 is not None and rss_seq is not None and rss_batch is not None:
        print(
            f"RSS start={rss0:.1f} MiB  after sequential={rss_seq:.1f} MiB  "
            f"after batch={rss_batch:.1f} MiB"
        )
    per = batched.total_elapsed_seconds / 10.0
    print(
        f"Estimate 100 shapes (no parallel): {per * 100.0:.0f} s "
        f"({per * 100.0 / 60.0:.1f} min)"
    )
    return 0 if vs_seq and vs_known else 1


if __name__ == "__main__":
    raise SystemExit(main())
