"""5 mm interior reference matrix — geometry only, no CoverageSimulator.evaluate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.unit.engines.compute.test_coverage_simulator import _fast_surface

from nutella_scraper.engines.compute.coverage_reference_matrix import (
    A0_MERIDIAN_AZIMUTH_DEG,
    COVERAGE_TARGET_REGION,
    LEGACY_A0_QUADRANT_REGION,
    MATRIX_ANGLE_END_DEG,
    MATRIX_ANGLE_START_DEG,
    MATRIX_SPACING_MM,
    REFERENCE_ZONE_SPAN_DEG,
    _y_samples,
    azimuths_deg,
    build_coverage_reference_matrix,
    progress_azimuth_deg,
    surface_axis_xz,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    load_interior_surface_reference,
)

SRC = Path("src/nutella_scraper/engines/compute/coverage_reference_matrix.py")
SIM_SRC = Path("src/nutella_scraper/engines/compute/coverage_simulator.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")
MODELS_ROOT = Path("output/models")
SAVED_JSON = Path("output/coverage/candidate_coverage_100.json")


def test_module_does_not_evaluate_or_use_legacy_a0_or_visual() -> None:
    text = SRC.read_text(encoding="utf-8")
    assert "evaluate_candidate(" not in text
    assert "engines.visualization" not in text
    assert "bind_envelope_proximity" not in text
    assert "visual.stl" not in text
    assert "from nutella_scraper.engines.compute.envelope_surface_proximity" not in text
    sim = SIM_SRC.read_text(encoding="utf-8")
    assert "build_coverage_reference_matrix" in sim
    assert "build_coverage_target_region" not in sim
    assert LEGACY_A0_QUADRANT_REGION not in sim


def test_azimuth_convention_is_zero_to_ninety_quadrant() -> None:
    assert abs(progress_azimuth_deg(50.0, 0.0) - 0.0) < 1e-9
    assert abs(progress_azimuth_deg(0.0, -50.0) - 90.0) < 1e-9
    assert A0_MERIDIAN_AZIMUTH_DEG == 0.0
    assert MATRIX_ANGLE_START_DEG == 0.0
    assert MATRIX_ANGLE_END_DEG == 90.0
    assert REFERENCE_ZONE_SPAN_DEG == 90.0
    assert MATRIX_SPACING_MM == 5.0
    assert COVERAGE_TARGET_REGION == "interior_matrix_a0_0_90"
    sim = SIM_SRC.read_text(encoding="utf-8")
    assert "ANGLE_END_DEG = 45.0" in sim


def test_synthetic_matrix_is_deterministic_and_on_cylinder() -> None:
    surface = _fast_surface()
    first = build_coverage_reference_matrix(surface)
    second = build_coverage_reference_matrix(surface)
    assert first.fingerprint == second.fingerprint
    assert first.points_mm == second.points_mm
    assert first.simulator_invoked is False
    assert first.uses_legacy_a0_point_matrix is False
    assert first.azimuth_span_deg == 90.0
    assert first.coverage_target_azimuth_range == pytest.approx((0.0, 90.0))
    assert first.y_min_mm == pytest.approx(0.0, abs=0.6)
    assert first.y_max_mm == pytest.approx(80.0, abs=0.6)
    assert first.mean_vertical_spacing_mm == pytest.approx(5.0, abs=0.6)
    assert first.mean_tangential_spacing_mm == pytest.approx(5.0, abs=0.8)
    assert first.on_interior_envelope is True
    assert first.any_point_outside_envelope is False
    assert first.max_distance_to_interior_mm <= 0.5
    axis = surface_axis_xz(np.asarray(surface.vertices))
    points = np.asarray(first.points_mm)
    az = azimuths_deg(points, axis)
    assert float(np.min(az)) >= -1e-6
    assert float(np.max(az)) <= 90.0 + 1e-6
    assert float(np.max(az)) >= 89.0
    radii = np.hypot(points[:, 0] - axis[0], points[:, 2] - axis[1])
    assert np.all(np.abs(radii - 50.0) < 0.6)
    assert first.point_count >= 100
    low = az < 45.0 - 1e-6
    high = az > 45.0 + 1e-6
    assert int(np.count_nonzero(high)) > 0
    assert int(np.count_nonzero(high)) == pytest.approx(
        int(np.count_nonzero(low)), rel=0.25, abs=4
    )
    ys = np.unique(np.round(points[:, 1], 6))
    assert float(ys[0]) == pytest.approx(first.y_min_mm, abs=1e-6)
    assert float(ys[-1]) == pytest.approx(first.y_max_mm, abs=1e-6)
    _assert_edge_row_covers_quadrant(points, axis, float(ys[0]))
    _assert_edge_row_covers_quadrant(points, axis, float(ys[-1]))
    assert float(np.max(np.diff(ys))) <= MATRIX_SPACING_MM + 0.6


def test_y_samples_keeps_five_mm_steps_and_both_edges() -> None:
    exact = _y_samples(0.0, 80.0, 5.0)
    assert exact[0] == pytest.approx(0.0)
    assert exact[-1] == pytest.approx(80.0)
    assert np.all(np.diff(exact) == pytest.approx(5.0, abs=1e-9))
    leftover = _y_samples(0.0, 12.0, 5.0)
    assert leftover[0] == pytest.approx(0.0)
    assert leftover[-1] == pytest.approx(12.0)
    assert leftover[1] == pytest.approx(5.0)
    assert leftover[-1] - leftover[-2] == pytest.approx(2.0)


@pytest.mark.skipif(
    not any(MODELS_ROOT.glob("*/interior_product_surface.npz")),
    reason="no cached STEP interior",
)
def test_real_interior_matrix_stays_on_envelope() -> None:
    model_id = next(
        p.parent.name
        for p in sorted(MODELS_ROOT.glob("*/interior_product_surface.npz"))
    )
    interior = load_interior_surface_reference(
        models_root=MODELS_ROOT,
        model_id=model_id,
    )
    assert interior.source == SOURCE_INTERIOR_PRODUCT_SURFACE
    matrix = build_coverage_reference_matrix(interior)
    assert matrix.point_count > 0
    assert matrix.on_interior_envelope is True
    assert matrix.any_point_outside_envelope is False
    assert matrix.max_distance_to_interior_mm <= 0.5
    assert matrix.azimuth_span_deg == 90.0
    assert matrix.mean_tangential_spacing_mm == pytest.approx(5.0, abs=1.5)
    assert matrix.uses_visual_stl is False
    assert matrix.uses_legacy_a0_point_matrix is False
    axis = surface_axis_xz(np.asarray(interior.vertices))
    points = np.asarray(matrix.points_mm)
    az = azimuths_deg(points, axis)
    delta = np.mod(az - A0_MERIDIAN_AZIMUTH_DEG, 360.0)
    assert float(np.min(delta)) >= -1e-6
    assert float(np.max(delta)) <= 90.0 + 1e-3
    assert float(np.max(delta)) >= 89.0
    low = delta < 45.0 - 1e-6
    high = delta > 45.0 + 1e-6
    assert int(np.count_nonzero(high)) > 0
    assert int(np.count_nonzero(high)) == pytest.approx(
        int(np.count_nonzero(low)), rel=0.35, abs=16
    )
    ys = np.unique(np.round(points[:, 1], 4))
    assert float(ys[0]) == pytest.approx(matrix.y_min_mm, abs=0.05)
    assert float(ys[-1]) == pytest.approx(matrix.y_max_mm, abs=0.05)
    assert matrix.y_min_mm == pytest.approx(interior.y_min_mm, abs=2.0)
    assert matrix.y_max_mm == pytest.approx(interior.y_max_mm, abs=2.0)
    assert float(np.max(np.diff(ys))) <= MATRIX_SPACING_MM + 1.0
    # Oval rim/floor: 0° and 90° do not share the same Y to 0.35 mm.
    _assert_edge_row_covers_quadrant(points, axis, float(ys[0]), band_mm=10.0)
    _assert_edge_row_covers_quadrant(points, axis, float(ys[-1]), band_mm=2.0)
    radii = np.hypot(points[:, 0] - axis[0], points[:, 2] - axis[1])
    assert float(np.min(radii)) < 8.0
    floor = points[points[:, 1] <= float(ys[0]) + 12.0]
    assert len(floor) >= 40
    floor_az = np.mod(azimuths_deg(floor, axis) - A0_MERIDIAN_AZIMUTH_DEG, 360.0)
    assert float(np.min(floor_az)) <= 1.0
    assert float(np.max(floor_az)) >= 89.0
    a0 = points[delta <= 2.0]
    a0 = a0[np.argsort(a0[:, 1])]
    meridian_step = np.linalg.norm(np.diff(a0, axis=0), axis=1)
    meridian_step = meridian_step[meridian_step > 0.2]
    assert float(np.median(meridian_step)) == pytest.approx(5.0, abs=1.5)


def _assert_edge_row_covers_quadrant(
    points: np.ndarray,
    axis: np.ndarray,
    y_mm: float,
    *,
    band_mm: float = 0.35,
) -> None:
    row = points[np.abs(points[:, 1] - y_mm) < band_mm]
    assert len(row) >= 2
    delta = np.mod(azimuths_deg(row, axis) - A0_MERIDIAN_AZIMUTH_DEG, 360.0)
    assert float(np.min(delta)) <= 1.0
    assert float(np.max(delta)) >= 89.0


@pytest.mark.skipif(not SAVED_JSON.is_file(), reason="saved coverage-100 JSON absent")
def test_saved_ranking_is_not_rewritten() -> None:
    import json

    payload = json.loads(SAVED_JSON.read_text(encoding="utf-8"))
    ranked = payload["ranked"]
    a0 = next(row for row in ranked if row["candidate_id"] == "A0")
    s8 = next(row for row in ranked if row["candidate_id"] == "S0008")
    assert float(a0["coverage_percent"]) == pytest.approx(63.3333, abs=1e-4)
    assert float(s8["coverage_percent"]) == pytest.approx(66.25, abs=1e-4)


def test_viewer_toolbar_and_white_cloud() -> None:
    html = HTML_SRC.read_text(encoding="utf-8")
    toolbar = html[html.index('class="view-toolbar"') : html.index('class="view-column"')]
    pot_i = toolbar.index('id="toggle-toolbar-pot"')
    wire_i = toolbar.index('id="toggle-toolbar-wireframe"')
    frame_i = toolbar.index('id="toggle-coordinate-frame"')
    scraper_i = toolbar.index('id="toggle-scene-scraper"')
    points_i = toolbar.index('id="toggle-scraper-points"')
    assert pot_i < wire_i < frame_i < scraper_i < points_i
    assert "Wireframe" in toolbar
    assert "Pot de Nutella" not in html
    assert "Référence A0" not in html
    assert "Repères" not in html
    assert "Nuage de points" in html
    assert "Nuages de points" not in html
    draw = html[html.index("function drawScene3D") : html.index("async function loadViewerScene")]
    assert '"#ffffff"' in draw
    assert "#7af0ff" not in draw
    assert "ensureVisualA0" in html
    assert "toggle-toolbar-pot" in draw
    enter = html[
        html.index("async function enterScraperSoloView") : html.index(
            "function cacheReferenceCandidate"
        )
    ]
    assert "ensureVisualA0" in enter
    assert "applyCachedCandidate(0)" not in enter
    assert "evaluate_candidate(" not in html
    a0 = html[html.index("const SCRAPER_A_REFERENCE") : html.index("let scraperSoloMode")]
    assert "width_mm: 2.5" in a0
    assert "thickness_mm: 2.5" in a0
    assert "rotation_angle_deg: 0" in a0
