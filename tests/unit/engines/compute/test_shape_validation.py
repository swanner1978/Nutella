"""Fast checks for VALIDATION_REAL_MATRIX — no 608-point campaign."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tests.unit.engines.compute.test_shape_search import _stub
from tests.unit.engines.compute.test_trajectory_search import _cell

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    COVERAGE_TARGET_REGION,
)
from nutella_scraper.engines.compute.shape_search import real_matrix_validation_config
from nutella_scraper.engines.compute.shape_validation import (
    VALIDATION_LABEL,
    check_cache_shape_and_cell,
    check_labels,
    check_ranking_uses_coverage,
    check_union_not_sum,
)
from nutella_scraper.engines.compute.trajectory_contact_cache import contact_cache_key
from nutella_scraper.engines.compute.trajectory_search import trajectory_grid_from_cells

SRC = Path("src/nutella_scraper/engines/compute/shape_validation.py")
SCRIPT = Path("scripts/validate_shape_search_real_matrix.py")
SEARCH = Path("src/nutella_scraper/engines/compute/shape_search.py")


def test_real_matrix_config_is_preliminary_not_a_campaign() -> None:
    config = real_matrix_validation_config()
    assert config.max_shape_evaluations == 1
    assert config.family_ids == (
        "straight",
        "concave",
        "convex",
        "circular_arc",
        "bezier_4",
    )
    assert config.run_a0_reference is True
    assert "poly_6" not in config.family_ids
    assert "bezier_10" not in config.family_ids
    assert "fourier_5" not in config.family_ids


def test_union_rejects_duplicate_indices() -> None:
    item = _stub(candidate_id="X", family_id="straight", covered=2)
    assert check_union_not_sum(item) == []
    bad = replace(item, covered_point_indices=(0, 0, 1), covered_points=3)
    assert check_union_not_sum(bad)


def test_ranking_prefers_coverage_over_geometric_error() -> None:
    low_cov = _stub(
        candidate_id="fit",
        family_id="bezier_4",
        covered=2,
        n_params=4,
        mean_err=0.01,
    )
    high_cov = _stub(
        candidate_id="phys",
        family_id="straight",
        covered=4,
        n_params=2,
        mean_err=9.0,
    )
    assert check_ranking_uses_coverage((low_cov, high_cov)) == []


def test_cache_key_includes_shape_and_cell() -> None:
    key_a = contact_cache_key("straight|1", 0, 1)
    key_b = contact_cache_key("bezier_4|1", 0, 1)
    assert key_a != key_b
    cells = tuple(
        _cell(row, col, n_rows=2, n_cols=2) for row in range(2) for col in range(2)
    )
    grid = trajectory_grid_from_cells(cells)
    assert check_cache_shape_and_cell(("fp-a", "fp-b"), grid) == []
    assert check_cache_shape_and_cell(("same", "same"), grid)


def test_label_and_sources_forbid_simulator() -> None:
    assert VALIDATION_LABEL == "VALIDATION_REAL_MATRIX"
    assert check_labels(VALIDATION_LABEL) == []
    assert check_labels("OPTIMAL")
    assert check_labels("EXHAUSTIVE")
    assert check_labels("BEST")
    for src in (SRC, SCRIPT, SEARCH):
        text = src.read_text(encoding="utf-8")
        assert "evaluate_candidate(" not in text
        assert "from nutella_scraper.engines.compute.coverage_simulator" not in text
    assert COVERAGE_TARGET_REGION in SEARCH.read_text(encoding="utf-8")
