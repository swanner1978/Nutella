"""Reproducible cardinality of the discrete scraper-shape lattice."""

from __future__ import annotations

from pathlib import Path

from tests.unit.engines.compute.coverage_catalog_fixtures import load_generated_catalog

from nutella_scraper.engines.visualization.scraper_control_cage import (
    build_control_cage_overlay,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (
    MAX_SECOND_DIFFERENCE,
    lattice_from_cage,
)
from nutella_scraper.engines.visualization.shape_space_cardinality import (
    CLASSIFIER_FAMILIES,
    count_bounded_row_walks,
    count_unconstrained_sequences,
    format_shape_space_report,
    shape_space_cardinality_report,
)

SRC = Path("src/nutella_scraper/engines/visualization/shape_space_cardinality.py")


def test_cardinality_module_does_not_import_simulator() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    assert "from nutella_scraper.engines.compute.coverage_simulator" not in text
    assert not any(line.startswith("import CoverageSimulator") for line in lines)
    assert "evaluate_candidate(" not in text


def test_unconstrained_and_walk_counts_are_exact() -> None:
    assert count_unconstrained_sequences(11, 17) == 11**17
    assert count_bounded_row_walks(2, 2, max_step=1, max_second=None) == 4
    assert count_bounded_row_walks(1, 5, max_step=1, max_second=1) == 1
    tiny = count_bounded_row_walks(3, 3, max_step=1, max_second=MAX_SECOND_DIFFERENCE)
    brute = 0
    for a in range(3):
        for b in range(3):
            if abs(b - a) > 1:
                continue
            for c in range(3):
                if abs(c - b) > 1:
                    continue
                if abs((c - b) - (b - a)) > 1:
                    continue
                brute += 1
    assert tiny == brute


def test_shape_space_report_separates_abcd_and_is_reproducible() -> None:
    surface, _params, reference, catalog = load_generated_catalog(count=1000)
    lattice = lattice_from_cage(
        build_control_cage_overlay(reference.design_path, surface),
        surface,
    )
    report = shape_space_cardinality_report(
        lattice,
        generated_count=1000,
        evaluated_count=100,
        catalog=catalog,
    )
    again = shape_space_cardinality_report(
        lattice,
        generated_count=1000,
        evaluated_count=100,
        catalog=catalog,
    )
    assert report == again
    assert report["lattice"]["station_count"] == 17
    assert report["lattice"]["base_row_count"] == 11
    assert report["A_raw"]["base_11_rows"] == 11**17
    assert report["B_local_constraints"]["base_11_abs_delta_and_second_le_1"] == 9600813
    assert report["B_local_constraints"]["garnished_abs_delta_and_second_le_1"] == 27276156
    assert report["B_local_constraints"]["garnished_integer_adm_zigzag_ok"] == 24947178
    assert report["B_local_constraints"]["garnished_zigzag_rejected"] == 2328978
    assert report["C_generated"]["emitted"] == 586
    assert report["C_generated"]["unique_fingerprints"] == 586
    assert report["C_generated"]["families"]["A0"] == 1
    assert report["D_evaluated_by_coverage_simulator"]["this_report_runs_simulator"] is False
    assert report["D_evaluated_by_coverage_simulator"]["known_evaluated_count"] == 100
    assert report["fully_admissible_bounds"]["exact"] is None
    assert report["fully_admissible_bounds"]["lower_bound"] == 586
    assert report["fully_admissible_bounds"]["upper_bound"] == 24947178
    assert set(report["families_covered"]).issubset(set(CLASSIFIER_FAMILIES))
    assert "s_curve" in report["families_admissible_not_generated"]
    assert "combined" in report["families_admissible_not_generated"]
    text = format_shape_space_report(report)
    assert "Espace brut" in text
    assert "Impossible de calculer exactement" in text
