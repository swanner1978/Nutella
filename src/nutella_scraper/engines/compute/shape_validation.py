"""Mandatory checks for the real-matrix shape-search validation.

Label: VALIDATION_REAL_MATRIX. Not OPTIMAL / EXHAUSTIVE / BEST.
Does not launch MAX_SHAPE_EVALUATIONS=100.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
    LEGACY_A0_QUADRANT_REGION,
    CoverageReferenceMatrix,
    azimuths_deg,
    surface_axis_xz,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    InteriorSurfaceReference,
)
from nutella_scraper.engines.compute.shape_result import ShapeCandidate
from nutella_scraper.engines.compute.shape_search import (
    ShapeSearchReport,
    rank_shape_candidates,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import contact_cache_key
from nutella_scraper.engines.compute.trajectory_search import (
    MAX_DOWNWARD_STEP,
    MAX_LATERAL_STEP,
    TrajectoryGrid,
    trajectory_is_valid,
)

VALIDATION_LABEL = "VALIDATION_REAL_MATRIX"
FORBIDDEN_RESULT_WORDS = ("OPTIMAL", "EXHAUSTIVE", "BEST")
EXPECTED_POINT_COUNT_MIN = 500
EXPECTED_POINT_COUNT_MAX = 750


def check_matrix_target(
    matrix: CoverageReferenceMatrix,
    surface: InteriorSurfaceReference,
) -> list[str]:
    errors: list[str] = []
    if str(matrix.coverage_target_region) != COVERAGE_TARGET_REGION:
        errors.append(
            f"cible {matrix.coverage_target_region!r} ≠ {COVERAGE_TARGET_REGION}"
        )
    if str(matrix.coverage_target_region) == LEGACY_A0_QUADRANT_REGION:
        errors.append("la région A0 historique est utilisée comme grille")
    if matrix.uses_legacy_a0_point_matrix:
        errors.append("uses_legacy_a0_point_matrix=True")
    if matrix.symmetry_multiplier_applied:
        errors.append("un facteur de symétrie ×4/×8 a été appliqué")
    if not matrix.on_interior_envelope:
        errors.append("des points du nuage ne sont pas sur l'enveloppe intérieure")
    if matrix.any_point_outside_envelope:
        errors.append("des points hors enveloppe sont présents dans le nuage")
    n_points = int(matrix.point_count)
    if n_points < EXPECTED_POINT_COUNT_MIN or n_points > EXPECTED_POINT_COUNT_MAX:
        errors.append(
            f"point_count={n_points} hors plage attendue "
            f"{EXPECTED_POINT_COUNT_MIN}–{EXPECTED_POINT_COUNT_MAX} (~608)"
        )
    axis = surface_axis_xz(surface.vertices)
    points = np.asarray(matrix.points_mm, dtype=np.float64)
    az = azimuths_deg(points, axis)
    if float(az.min()) < -1e-3 or float(az.max()) > 90.0 + 1e-2:
        errors.append(
            f"azimut hors 0–90°: [{float(az.min()):.3f}, {float(az.max()):.3f}]"
        )
    if float(az.max()) < 89.0:
        errors.append("le nuage n'atteint pas le méridien +90°")
    return errors


def check_union_not_sum(item: ShapeCandidate) -> list[str]:
    errors: list[str] = []
    unique = tuple(sorted(set(item.covered_point_indices)))
    if unique != tuple(sorted(item.covered_point_indices)):
        errors.append(f"{item.candidate_id}: indices de contact dupliqués")
    if int(item.covered_points) != len(unique):
        errors.append(
            f"{item.candidate_id}: covered_points={item.covered_points} "
            f"≠ UNION {len(unique)}"
        )
    if int(item.covered_points) + len(item.untouched_point_indices) != int(
        item.total_points
    ):
        errors.append(f"{item.candidate_id}: union + non touchés ≠ N")
    if int(item.covered_points) > int(item.total_points):
        errors.append(f"{item.candidate_id}: plus de points touchés que N")
    return errors


def check_covered_indices_in_matrix(
    item: ShapeCandidate,
    matrix: CoverageReferenceMatrix,
) -> list[str]:
    errors: list[str] = []
    n_points = int(matrix.point_count)
    if matrix.any_point_outside_envelope:
        errors.append("refus: des points hors enveloppe ne doivent pas être comptés")
    for index in item.covered_point_indices:
        if int(index) < 0 or int(index) >= n_points:
            errors.append(
                f"{item.candidate_id}: indice {index} hors du nuage 0..{n_points - 1}"
            )
    return errors


def check_trajectory_rules(
    item: ShapeCandidate,
    grid: TrajectoryGrid,
) -> list[str]:
    errors: list[str] = []
    if not item.physical_valid:
        return errors
    if item.trajectory_model == "POSE_GRAPH" or item.trajectory_poses:
        ys = [float(y) for y, _az in item.trajectory_poses]
        if len(ys) < 1:
            errors.append(f"{item.candidate_id}: trajectoire pose vide")
            return errors
        if any(nxt > prev + 1e-6 for prev, nxt in zip(ys[:-1], ys[1:], strict=True)):
            errors.append(f"{item.candidate_id}: Y remonte")
        return errors
    if MAX_LATERAL_STEP != 2 or MAX_DOWNWARD_STEP != 1:
        errors.append("MAX_LATERAL_STEP / MAX_DOWNWARD_STEP ont été altérés")
        return errors
    if not item.physical_valid or not item.trajectory_rows_cols:
        if item.physical_valid:
            errors.append(f"{item.candidate_id}: trajectoire vide alors que physical_valid")
        return errors
    cells = []
    for row, col in item.trajectory_rows_cols:
        cell = grid.cell_at(int(row), int(col))
        if cell is None:
            errors.append(f"{item.candidate_id}: cellule ({row},{col}) absente de la grille")
            return errors
        cells.append(cell)
    rows = [cell.row for cell in cells]
    if any(nxt < prev for prev, nxt in zip(rows[:-1], rows[1:], strict=True)):
        errors.append(f"{item.candidate_id}: la trajectoire remonte")
    jumps = [
        (abs(b.col - a.col), b.row - a.row)
        for a, b in zip(cells[:-1], cells[1:], strict=True)
    ]
    for dcol, drow in jumps:
        if drow == 0 and dcol == 0:
            errors.append(f"{item.candidate_id}: pose identique consécutive (téléport/no-op)")
        if dcol > MAX_LATERAL_STEP:
            errors.append(f"{item.candidate_id}: |Δcol|={dcol} > {MAX_LATERAL_STEP}")
        if drow > MAX_DOWNWARD_STEP:
            errors.append(f"{item.candidate_id}: Δrow={drow} > {MAX_DOWNWARD_STEP} (téléport)")
    if not trajectory_is_valid(
        cells,
        grid,
        max_lateral_step=MAX_LATERAL_STEP,
        max_downward_step=MAX_DOWNWARD_STEP,
    ):
        errors.append(f"{item.candidate_id}: trajectory_is_valid=False")
    return errors


def check_shape_sent_to_collision(item: ShapeCandidate) -> list[str]:
    errors: list[str] = []
    if not item.physical_valid:
        return errors
    if not item.scraper_fingerprint:
        errors.append(f"{item.candidate_id}: scraper_fingerprint vide")
    if item.shape_fingerprint and item.scraper_fingerprint:
        if item.family_id != "A0" and item.scraper_fingerprint != item.shape_fingerprint:
            errors.append(
                f"{item.candidate_id}: fingerprint collision "
                f"{item.scraper_fingerprint!r} ≠ forme {item.shape_fingerprint!r}"
            )
    return errors


def check_cache_shape_and_cell(
    fingerprints: tuple[str, ...],
    grid: TrajectoryGrid,
) -> list[str]:
    errors: list[str] = []
    if len(fingerprints) != len(set(fingerprints)):
        errors.append("deux formes partagent le même cache")
    keys: list[str] = []
    for fp in fingerprints:
        for cell in grid.cells:
            keys.append(contact_cache_key(fp, cell.row, cell.col))
    if len(keys) != len(set(keys)):
        errors.append("collision de clé cache forme+paramètres+cellule")
    if len(fingerprints) >= 2 and grid.cells:
        cell = grid.cells[0]
        key_a = contact_cache_key(fingerprints[0], cell.row, cell.col)
        key_b = contact_cache_key(fingerprints[1], cell.row, cell.col)
        if key_a == key_b:
            errors.append("la même cellule est partagée entre deux formes")
    return errors


def check_ranking_uses_coverage(items: tuple[ShapeCandidate, ...]) -> list[str]:
    errors: list[str] = []
    if len(items) < 2:
        return errors
    ranked = rank_shape_candidates(items)
    coverages = [int(item.covered_points) for item in ranked]
    if coverages != sorted(coverages, reverse=True):
        errors.append("le classement n'est pas par couverture physique décroissante")
    geom = [float(item.mean_geometric_error_mm) for item in ranked]
    if (
        ranked[0].covered_points < max(coverages)
        and geom[0] <= min(geom) + 1e-9
    ):
        errors.append("une erreur géométrique plus faible a primé sur la couverture")
    return errors


def check_labels(payload_text: str) -> list[str]:
    errors: list[str] = []
    if VALIDATION_LABEL not in payload_text:
        errors.append(f"label {VALIDATION_LABEL} absent")
    upper = payload_text.upper()
    for word in FORBIDDEN_RESULT_WORDS:
        if word in upper:
            errors.append(f"mot interdit {word} dans le rapport")
    return errors


def case_row(item: ShapeCandidate) -> dict[str, Any]:
    return {
        "forme": item.family_id,
        "candidate_id": item.candidate_id,
        "paramètres": list(item.parameters),
        "nombre_de_points_touchés": int(item.covered_points),
        "couverture_percent": float(item.coverage_percent),
        "erreur_géométrique_moyenne_mm": float(item.mean_geometric_error_mm),
        "erreur_géométrique_maximale_mm": float(item.max_geometric_error_mm),
        "nombre_de_poses_évaluées": int(item.poses_evaluated),
        "nombre_de_trajectoires_explorées_beam": int(item.beam_trajectories_explored),
        "longueur_de_trajectoire_mm": float(item.trajectory_length_mm),
        "changements_latéraux": int(item.lateral_changes),
        "temps_de_calcul_s": float(item.elapsed_seconds),
        "cache_hits": int(item.cache_hits),
        "cache_misses": int(item.cache_misses),
        "geometric_valid": bool(item.geometric_valid),
        "physical_valid": bool(item.physical_valid),
        "scraper_fingerprint": item.scraper_fingerprint,
    }


def ordered_validation_cases(report: ShapeSearchReport) -> tuple[ShapeCandidate, ...]:
    by_family = {item.family_id: item for item in report.candidates}
    if report.a0_reference is not None:
        by_family["A0"] = report.a0_reference
    ordered: list[ShapeCandidate] = []
    for family_id in ("A0", "straight", "circular_arc", "bezier_4"):
        item = by_family.get(family_id)
        if item is not None:
            ordered.append(item)
    return tuple(ordered)


def run_mandatory_checks(
    *,
    matrix: CoverageReferenceMatrix,
    surface: InteriorSurfaceReference,
    report: ShapeSearchReport,
) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = list(check_matrix_target(matrix, surface))
    cases = ordered_validation_cases(report)
    if len(cases) != 4:
        errors.append(f"attendu 4 cas, obtenu {len(cases)}")
    for item in cases:
        physics_ran = int(item.poses_evaluated) > 0 or bool(item.scraper_fingerprint)
        if not physics_ran:
            reasons = ",".join(item.geometric_reasons) or "unknown"
            errors.append(
                f"{item.candidate_id}: pipeline physique non exécuté ({reasons})"
            )
        errors.extend(check_union_not_sum(item))
        errors.extend(check_covered_indices_in_matrix(item, matrix))
        errors.extend(check_trajectory_rules(item, report.grid))
        errors.extend(check_shape_sent_to_collision(item))
    errors.extend(
        check_cache_shape_and_cell(report.cache_shape_fingerprints, report.grid)
    )
    errors.extend(check_ranking_uses_coverage(cases))
    if (
        cases
        and all(int(item.covered_points) == 0 for item in cases)
        and any(int(item.poses_evaluated) > 0 for item in cases)
    ):
        errors.append("aucune des 4 formes n'a de couverture UNION > 0 sur le nuage")
    if report.grid.uses_legacy_a0_point_matrix:
        errors.append("la grille de trajectoire est encore A0")
    if report.grid.target_definition != COVERAGE_TARGET_REGION:
        errors.append("la grille n'est pas interior_matrix_a0_0_90")
    return (not errors, tuple(errors))
