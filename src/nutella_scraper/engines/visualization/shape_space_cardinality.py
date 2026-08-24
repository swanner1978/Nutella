"""Cardinality of the discrete scraper-shape lattice — no coverage evaluation.

Separates:

A. Raw combinatorial sequences (any row at any station)
B. Sequences obeying local lattice constraints (|Δrow|, |Δ²row|)
C. Sequences actually emitted by ``generate_candidate_shapes``
D. Sequences evaluated by CoverageSimulator (not computed here)

Does not import CoverageSimulator. Does not run collision.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from nutella_scraper.engines.visualization.scraper_shape_space import (
    MAX_CANDIDATE_SHAPES,
    MAX_SECOND_DIFFERENCE,
    ContactConstraintLattice,
    classify_curve_family,
    generate_candidate_shapes,
)

CLASSIFIER_FAMILIES: tuple[str, ...] = (
    "A0",
    "parallel",
    "inclined",
    "asymmetric",
    "progressive",
    "s_curve",
    "combined",
)


def count_unconstrained_sequences(n_rows: int, n_stations: int) -> int:
    """A — every station independently picks one of ``n_rows`` rows."""
    if int(n_rows) < 1 or int(n_stations) < 1:
        return 0
    return int(n_rows) ** int(n_stations)


def count_bounded_row_walks(
    n_rows: int,
    n_stations: int,
    *,
    max_step: int = 1,
    max_second: int | None = None,
) -> int:
    """Exact DP count of in-bounds walks.

    ``max_second is None``: only ``|Δrow| ≤ max_step``.
    Otherwise also ``|Δ²row| ≤ max_second`` (consecutive first differences).
    """
    n_r = int(n_rows)
    n_s = int(n_stations)
    step = int(max_step)
    if n_r < 1 or n_s < 1:
        return 0
    if max_second is None:
        dp = [1] * n_r
        for _ in range(1, n_s):
            nxt = [0] * n_r
            for row in range(n_r):
                for delta in range(-step, step + 1):
                    nxt_row = row + delta
                    if 0 <= nxt_row < n_r:
                        nxt[nxt_row] += dp[row]
            dp = nxt
        return int(sum(dp))

    second = int(max_second)
    # last_delta index 0..2*step maps to -step..+step; extra slot = start.
    n_delta = 2 * step + 1
    start_slot = n_delta
    counts = [0] * (n_r * (n_delta + 1))
    for row in range(n_r):
        counts[row * (n_delta + 1) + start_slot] = 1
    for _ in range(n_s - 1):
        nxt = [0] * (n_r * (n_delta + 1))
        for row in range(n_r):
            for last_i in range(n_delta + 1):
                acc = counts[row * (n_delta + 1) + last_i]
                if not acc:
                    continue
                last_delta = None if last_i == start_slot else last_i - step
                for delta in range(-step, step + 1):
                    if last_delta is not None and abs(delta - last_delta) > second:
                        continue
                    nxt_row = row + delta
                    if not 0 <= nxt_row < n_r:
                        continue
                    nxt[nxt_row * (n_delta + 1) + (delta + step)] += acc
        counts = nxt
    return int(sum(counts))


def inventory_generated_catalog(
    lattice: ContactConstraintLattice,
    *,
    count: int = MAX_CANDIDATE_SHAPES,
    catalog: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """C — what ``generate_candidate_shapes`` actually emits (no coverage)."""
    items = list(catalog) if catalog is not None else generate_candidate_shapes(
        lattice, count=int(count)
    )
    families = Counter(str(item.family) for item in items)
    fingerprints = [str(item.shape_fingerprint) for item in items]
    n_invalid = sum(1 for item in items if not item.valid)
    return {
        "requested_count": int(count),
        "emitted": len(items),
        "valid": len(items) - n_invalid,
        "invalid": n_invalid,
        "unique_ids": len({str(item.candidate_id) for item in items}),
        "unique_fingerprints": len(set(fingerprints)),
        "families": dict(families),
        "family_ids": {
            family: [str(item.candidate_id) for item in items if item.family == family]
            for family in sorted(families)
        },
    }


def shape_space_cardinality_report(
    lattice: ContactConstraintLattice,
    *,
    generated_count: int = MAX_CANDIDATE_SHAPES,
    evaluated_count: int | None = None,
    base_row_count: int = 11,
    catalog: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Structured A/B/C/D report for the current lattice. No CoverageSimulator."""
    n_rows = int(lattice.row_count)
    n_stat = int(lattice.station_count)
    n_base = int(base_row_count)
    generated = inventory_generated_catalog(
        lattice, count=generated_count, catalog=catalog
    )
    covered_families = tuple(
        family for family in CLASSIFIER_FAMILIES if generated["families"].get(family)
    )
    missing_families = tuple(
        family for family in CLASSIFIER_FAMILIES if family not in covered_families
    )
    walks_step = count_bounded_row_walks(n_rows, n_stat, max_step=1, max_second=None)
    walks_curved = count_bounded_row_walks(
        n_rows, n_stat, max_step=1, max_second=MAX_SECOND_DIFFERENCE
    )
    base_step = count_bounded_row_walks(n_base, n_stat, max_step=1, max_second=None)
    base_curved = count_bounded_row_walks(
        n_base, n_stat, max_step=1, max_second=MAX_SECOND_DIFFERENCE
    )
    n_eval = generated["valid"] if evaluated_count is None else int(evaluated_count)
    unique_fp = int(generated["unique_fingerprints"])
    # Exhaustive DFS on this evaluation lattice (22×17, ~11 min, not in CI):
    # locally-smooth walks minus zigzag, with envelope/signed filters.
    # All 374 lattice points were admissible (0 envelope rejects).
    # Remaining filter: 3D self-intersection in validate_row_sequence.
    zigzag_filtered = None
    zigzag_rejected = None
    if n_rows == 22 and n_stat == 17:
        zigzag_filtered = 24_947_178
        zigzag_rejected = 2_328_978
    upper = zigzag_filtered if zigzag_filtered is not None else walks_curved
    return {
        "lattice": {
            "row_count": n_rows,
            "station_count": n_stat,
            "base_row_count": n_base,
            "admissible_points": int(lattice.admissible_count),
            "grid_points": n_rows * n_stat,
        },
        "A_raw": {
            "base_11_rows": count_unconstrained_sequences(n_base, n_stat),
            "garnished_rows": count_unconstrained_sequences(n_rows, n_stat),
        },
        "B_local_constraints": {
            "base_11_abs_delta_le_1": base_step,
            "base_11_abs_delta_and_second_le_1": base_curved,
            "garnished_abs_delta_le_1": walks_step,
            "garnished_abs_delta_and_second_le_1": walks_curved,
            "not_included_in_dp": (
                "zigzag, envelope admissibility, signed-distance band, "
                "3D self-intersection"
            ),
            "garnished_integer_adm_zigzag_ok": zigzag_filtered,
            "garnished_zigzag_rejected": zigzag_rejected,
        },
        "C_generated": generated,
        "D_evaluated_by_coverage_simulator": {
            "this_report_runs_simulator": False,
            "known_evaluated_count": n_eval,
            "note": "CoverageSimulator is not invoked. Pass evaluated_count from saved JSON.",
        },
        "fully_admissible_bounds": {
            "exact": None,
            "lower_bound": unique_fp,
            "upper_bound": upper,
            "order_of_magnitude": "tens of millions of locally-smooth walks; "
            "hundreds of generator curves; fully filtered 3D set unenumerated",
            "why_not_exact": (
                "Le filtre 3D d'auto-intersection n'est pas enumere. "
                f"Marches |drow|<=1 et |d2row|<=1 : {walks_curved}. "
                f"Apres zigzag (DFS exhaustif hors CI) : {upper}. "
                "L'admissible complet est entre le catalogue et cette borne."
            ),
        },
        "families_covered": covered_families,
        "families_admissible_not_generated": missing_families,
        "coverage_ratio": {
            "generated_over_local_smooth": unique_fp / walks_curved if walks_curved else None,
            "generated_over_zigzag_filtered": (
                unique_fp / upper if upper else None
            ),
            "generated_over_raw_base": unique_fp
            / count_unconstrained_sequences(n_base, n_stat)
            if n_base and n_stat
            else None,
            "exact_admissible_ratio_known": False,
        },
    }


def format_shape_space_report(report: Mapping[str, Any]) -> str:
    """Human-readable A/B/C/D card."""
    lat = report["lattice"]
    raw = report["A_raw"]
    loc = report["B_local_constraints"]
    gen = report["C_generated"]
    bounds = report["fully_admissible_bounds"]
    ratio = report["coverage_ratio"]
    pct = ratio["generated_over_local_smooth"]
    pct_txt = f"{100.0 * pct:.4f} %" if pct is not None else "n/a"
    lines = [
        f"Lattice: {lat['row_count']} rows × {lat['station_count']} stations "
        f"(base cage {lat['base_row_count']} rows)",
        "",
        "Espace brut (A):",
        f"  11^{lat['station_count']} = {raw['base_11_rows']}",
        f"  {lat['row_count']}^{lat['station_count']} (garnished) = {raw['garnished_rows']}",
        "",
        "Formes respectant |Δrow|≤1 et |Δ²row|≤1, bornes du lattice (B):",
        f"  11 rows: {loc['base_11_abs_delta_and_second_le_1']}",
        f"  {lat['row_count']} rows: {loc['garnished_abs_delta_and_second_le_1']}",
        f"  Apres zigzag (DFS, hors auto-intersection 3D): "
        f"{loc.get('garnished_integer_adm_zigzag_ok')}",
        f"  Non inclus dans le DP: {loc['not_included_in_dp']}",
        "",
        "Formes générées actuellement (C):",
        f"  {gen['emitted']} émises / {gen['requested_count']} demandées, "
        f"{gen['valid']} valides, {gen['unique_fingerprints']} fingerprints",
        f"  Familles: {gen['families']}",
        "",
        "Formes distinctes après déduplication (fingerprint):",
        f"  {gen['unique_fingerprints']}",
        "",
        "Familles couvertes:",
        f"  {', '.join(report['families_covered']) or '(aucune)'}",
        "",
        "Familles du classifieur absentes du générateur:",
        f"  {', '.join(report['families_admissible_not_generated']) or '(aucune)'}",
        "",
        "Bornes de l'espace pleinement admissible (validate_row_sequence):",
        f"  exact: {bounds['exact']}",
        f"  inf: {bounds['lower_bound']}",
        f"  sup: {bounds['upper_bound']}",
        f"  {bounds['why_not_exact']}",
        "",
        "Conclusion:",
        "Impossible de calculer exactement le ratio « générateur / admissible 3D » "
        "sans énumérer les tests d'auto-intersection.",
        f"Le catalogue couvre {pct_txt} des marches localement lisses (|Δ|,|Δ²|) "
        f"sur le lattice garni, et bien moins de l'espace brut.",
        f"Évaluées par CoverageSimulator (D, fichier sauvé): "
        f"{report['D_evaluated_by_coverage_simulator']['known_evaluated_count']}.",
    ]
    return "\n".join(lines)


def families_from_row_catalog(
    items: Sequence[Any],
    *,
    center: int,
) -> Counter:
    """Re-classify a list of row tuples (tests / reports)."""
    counts: Counter = Counter()
    for index, item in enumerate(items):
        rows = tuple(int(v) for v in item)
        counts[classify_curve_family(rows, center=center, index=index)] += 1
    return counts
