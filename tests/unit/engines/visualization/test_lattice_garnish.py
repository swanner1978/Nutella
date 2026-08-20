"""Local lattice garnish fills under-dense envelope gaps without moving A0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.unit.engines.compute.test_scraper_parametric_v1 import _ellipse
from tests.unit.engines.compute.test_scraper_trajectory_width import _bowl
from tests.unit.engines.visualization.test_scraper_control_cage import _reference_path

from nutella_scraper.engines.visualization.scraper_control_cage import (
    CAGE_ROW_OFFSETS_MM,
    build_control_cage_overlay,
)
from nutella_scraper.engines.visualization.scraper_shape_space import (
    CONTACT_TOLERANCE_MM,
    GAP_THRESHOLD_MM,
    MAX_GARNISHED_TRAJECTORIES,
    envelope_contact_metrics,
    filter_control_cage,
    garnish_contact_lattice,
    generate_candidate_shapes,
    lattice_from_cage,
)

COLLISION_SRC = Path(
    "src/nutella_scraper/engines/compute/scraper_envelope_collision.py"
)
HTML_SRC = Path("scripts/templates/demo_viewer.html")


def _base_lattice(surface):
    path, _params, _max_length = _reference_path(surface)
    raw = build_control_cage_overlay(path, None)
    base_overlay = filter_control_cage(raw, surface, garnish=False)
    return lattice_from_cage(base_overlay, surface), path, surface, base_overlay


def test_existing_lattice_and_a0_are_strictly_preserved() -> None:
    surface = _bowl()
    base, path, _surface, overlay = _base_lattice(surface)
    garnished, report = garnish_contact_lattice(base, surface)
    assert report.trajectories_before == 11
    assert garnished.row_count >= base.row_count
    for offset in CAGE_ROW_OFFSETS_MM:
        old = base.points_mm[list(base.row_offsets_mm).index(offset)]
        new = garnished.points_mm[list(garnished.row_offsets_mm).index(offset)]
        assert np.allclose(old, new, atol=1e-9)
    assert np.allclose(
        garnished.points_mm[garnished.center_row_index],
        path.wall_curve_mm,
        atol=1e-3,
    )
    assert overlay["centerline_mm"] == filter_control_cage(
        overlay, surface, garnish=True
    )["centerline_mm"]


def test_gap_below_threshold_adds_no_trajectory() -> None:
    surface = _bowl()
    base, _path, _surface, _overlay = _base_lattice(surface)
    _garnished, report = garnish_contact_lattice(
        base, surface, gap_threshold_mm=10_000.0, max_garnished=50
    )
    assert report.underdense_zones == 0
    assert report.added == 0
    assert report.generated == 0
    assert report.trajectories_after == report.trajectories_before


def test_gap_above_threshold_can_add_a_complete_envelope_row() -> None:
    surface = _bowl()
    base, _path, surface, _overlay = _base_lattice(surface)
    garnished, report = garnish_contact_lattice(
        base, surface, gap_threshold_mm=GAP_THRESHOLD_MM, max_garnished=50
    )
    assert report.underdense_zones > 0
    assert report.added >= 1
    assert garnished.row_count == base.row_count + report.added
    new_offsets = [
        off
        for off in garnished.row_offsets_mm
        if off not in set(base.row_offsets_mm)
    ]
    assert new_offsets
    for offset in new_offsets:
        row = garnished.points_mm[list(garnished.row_offsets_mm).index(offset)]
        assert row.shape == (garnished.station_count, 3)
        assert np.all(np.isfinite(row))
        signed, unsigned = envelope_contact_metrics(surface, row)
        assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM
        assert float(np.max(unsigned)) <= CONTACT_TOLERANCE_MM + 1e-6
        ys = row[:, 1]
        diffs = np.diff(ys)
        assert bool(np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6))


def test_duplicates_and_max_garnish_are_respected() -> None:
    surface = _bowl()
    base, _path, surface, _overlay = _base_lattice(surface)
    once, first = garnish_contact_lattice(base, surface, max_garnished=2)
    assert first.added <= 2
    twice, second = garnish_contact_lattice(
        once, surface, gap_threshold_mm=0.01, max_garnished=2
    )
    assert second.added <= 2
    assert twice.row_count <= once.row_count + 2
    fingerprints = [
        tuple(np.round(twice.points_mm[row], 3).ravel())
        for row in range(twice.row_count)
    ]
    assert len(fingerprints) == len(set(fingerprints))


def test_garnish_does_not_fill_the_jar_volume() -> None:
    surface = _ellipse(a=53.0, b=32.0)
    overlay = build_control_cage_overlay(_reference_path(surface)[0], surface)
    lattice = lattice_from_cage(overlay, surface)
    signed, unsigned = envelope_contact_metrics(
        surface, lattice.points_mm.reshape(-1, 3)
    )
    assert float(np.min(signed)) >= -CONTACT_TOLERANCE_MM
    assert float(np.max(unsigned)) <= CONTACT_TOLERANCE_MM + 1e-6
    assert int(overlay["garnish"]["max_garnished"]) == MAX_GARNISHED_TRAJECTORIES
    assert int(overlay["garnish"]["added"]) <= MAX_GARNISHED_TRAJECTORIES


def test_candidate_catalog_gains_options_but_keeps_a0() -> None:
    surface = _bowl()
    base, path, surface, overlay = _base_lattice(surface)
    before = generate_candidate_shapes(base, count=40)
    garnished, report = garnish_contact_lattice(base, surface)
    after = generate_candidate_shapes(garnished, count=40)
    assert before[0].candidate_id == "A0"
    assert after[0].candidate_id == "A0"
    assert np.allclose(
        np.asarray(before[0].control_points_mm, dtype=np.float64),
        path.wall_curve_mm,
        atol=1e-3,
    )
    assert np.allclose(
        np.asarray(after[0].control_points_mm, dtype=np.float64),
        path.wall_curve_mm,
        atol=1e-3,
    )
    if report.added:
        assert len({item.shape_fingerprint for item in after}) >= len(
            {item.shape_fingerprint for item in before}
        )
    collision = COLLISION_SRC.read_text(encoding="utf-8")
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "garnish_contact_lattice" not in collision
    assert "const rotationCache = new Map()" in html


def test_viewer_overlay_includes_garnished_polylines() -> None:
    surface = _bowl()
    path, _params, _max_length = _reference_path(surface)
    cage = build_control_cage_overlay(path, surface)
    assert int(cage["base_row_count"]) == 11
    assert cage["row_count"] == len(cage["polylines_mm"])
    assert all(all(pt is not None for pt in row) for row in cage["polylines_mm"])
    assert cage["garnish"]["trajectories_after"] == cage["row_count"]
