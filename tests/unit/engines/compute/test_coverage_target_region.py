"""90° interior-wall quadrant — geometry only, no CoverageSimulator.evaluate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.unit.engines.compute.test_coverage_simulator import _fast_surface

from nutella_scraper.engines.compute.coverage_target_region import (
    A0_REFERENCE_AZIMUTH_DEG,
    COVERAGE_TARGET_AZIMUTH_SPAN_DEG,
    COVERAGE_TARGET_REGION,
    COVERAGE_TARGET_SURFACE,
    WALL_RADIUS_MIN_MM,
    azimuth_in_quadrant_mask,
    azimuths_deg,
    build_coverage_target_region,
    progress_azimuth_deg,
    region_to_payload,
    surface_axis_xz,
)
from nutella_scraper.engines.compute.interior_surface_reference import (
    SOURCE_INTERIOR_PRODUCT_SURFACE,
    load_interior_surface_reference,
)

SRC = Path("src/nutella_scraper/engines/compute/coverage_target_region.py")
SIM_SRC = Path("src/nutella_scraper/engines/compute/coverage_simulator.py")
HTML_SRC = Path("scripts/templates/demo_viewer.html")
MODELS_ROOT = Path("output/models")
SAVED_JSON = Path("output/coverage/candidate_coverage_100.json")


def test_module_does_not_evaluate_or_import_visualization() -> None:
    text = SRC.read_text(encoding="utf-8")
    assert "evaluate_candidate(" not in text
    assert "engines.visualization" not in text
    assert "bind_envelope_proximity" not in text
    assert "visual.stl" not in text


def test_azimuth_convention_and_90_degree_span() -> None:
    assert abs(progress_azimuth_deg(50.0, 0.0) - 0.0) < 1e-9
    assert abs(progress_azimuth_deg(0.0, -50.0) - 90.0) < 1e-9
    assert COVERAGE_TARGET_AZIMUTH_SPAN_DEG == 90.0
    assert A0_REFERENCE_AZIMUTH_DEG == 0.0
    start, end = A0_REFERENCE_AZIMUTH_DEG, (
        A0_REFERENCE_AZIMUTH_DEG + COVERAGE_TARGET_AZIMUTH_SPAN_DEG
    ) % 360.0
    width = (end - start) % 360.0
    if width == 0.0:
        width = 360.0
    assert width == pytest.approx(90.0)
    az = np.array([0.0, 45.0, 90.0, 90.1, 359.0])
    mask = azimuth_in_quadrant_mask(az, start_deg=0.0, span_deg=90.0)
    assert mask.tolist() == [True, True, True, False, False]


def test_synthetic_cylinder_quadrant_is_double_the_old_45_sector() -> None:
    surface = _fast_surface()
    region = build_coverage_target_region(surface)
    assert region.azimuth_span_deg == 90.0
    assert region.coverage_target_surface == COVERAGE_TARGET_SURFACE
    assert region.coverage_target_region == COVERAGE_TARGET_REGION
    assert region.face_count == region.point_count
    assert region.face_count == 480
    assert region.simulator_invoked is False
    assert region.symmetry_multiplier_applied is False
    assert abs(region.coverage_target_azimuth_range[0] - 0.0) < 1e-9
    assert abs(region.coverage_target_azimuth_range[1] - 90.0) < 1e-9
    vertices = np.asarray(surface.vertices)
    faces = np.asarray(surface.faces)
    centroids = vertices[faces].mean(axis=1)
    axis = surface_axis_xz(vertices)
    az = azimuths_deg(centroids, axis)
    for face_id in region.face_ids:
        delta = (az[face_id] - A0_REFERENCE_AZIMUTH_DEG) % 360.0
        assert 0.0 - 1e-6 <= delta <= 90.0 + 1e-6
        radius = float(np.hypot(centroids[face_id, 0], centroids[face_id, 2]))
        assert radius >= WALL_RADIUS_MIN_MM


def test_simulator_source_uses_reference_matrix_not_legacy_quadrant() -> None:
    text = SIM_SRC.read_text(encoding="utf-8")
    assert "build_coverage_reference_matrix" in text
    assert "build_coverage_target_region" not in text
    assert "engines.visualization" not in text


def test_payload_points_are_face_centroids() -> None:
    surface = _fast_surface()
    region = build_coverage_target_region(surface)
    payload = region_to_payload(region)
    assert payload["simulator_invoked"] is False
    assert payload["coverage_recomputed"] is False
    assert payload["uses_visual_stl"] is False
    assert len(payload["points_mm"]) == payload["face_count"]
    vertices = np.asarray(surface.vertices)
    faces = np.asarray(surface.faces)
    for face_id, point in zip(region.face_ids, payload["points_mm"], strict=True):
        expected = vertices[faces[face_id]].mean(axis=0)
        assert np.allclose(point, expected, atol=1e-6)


@pytest.mark.skipif(
    not any(MODELS_ROOT.glob("*/interior_product_surface.npz")),
    reason="no cached STEP interior",
)
def test_real_interior_quadrant_is_on_step_wall() -> None:
    model_id = next(
        p.parent.name
        for p in sorted(MODELS_ROOT.glob("*/interior_product_surface.npz"))
    )
    interior = load_interior_surface_reference(
        models_root=MODELS_ROOT,
        model_id=model_id,
    )
    assert interior.source == SOURCE_INTERIOR_PRODUCT_SURFACE
    region = build_coverage_target_region(interior)
    assert region.face_count > 0
    assert region.area_mm2 > 0.0
    assert region.azimuth_span_deg == 90.0
    assert region.uses_synthetic_evaluation_cylinder is False
    vertices = np.asarray(interior.vertices)
    faces = np.asarray(interior.faces)
    centroids = vertices[faces].mean(axis=1)
    axis = surface_axis_xz(vertices)
    for face_id in region.face_ids:
        c = centroids[face_id]
        radius = float(np.hypot(c[0] - axis[0], c[2] - axis[1]))
        assert radius >= WALL_RADIUS_MIN_MM - 1e-6
        delta = (azimuths_deg(c.reshape(1, 3), axis)[0] - 0.0) % 360.0
        assert delta <= 90.0 + 1e-6


@pytest.mark.skipif(not SAVED_JSON.is_file(), reason="saved coverage-100 JSON absent")
def test_saved_ranking_is_not_rewritten() -> None:
    import json

    payload = json.loads(SAVED_JSON.read_text(encoding="utf-8"))
    ranked = payload["ranked"]
    a0 = next(row for row in ranked if row["candidate_id"] == "A0")
    s8 = next(row for row in ranked if row["candidate_id"] == "S0008")
    assert float(a0["coverage_percent"]) == pytest.approx(63.3333, abs=1e-4)
    assert float(s8["coverage_percent"]) == pytest.approx(66.25, abs=1e-4)


def test_viewer_html_uses_interior_quadrant_and_a0() -> None:
    html = HTML_SRC.read_text(encoding="utf-8")
    assert "Nuage de points" in html
    assert 'id="toggle-scraper-points"' in html
    assert "loadCoverageTargetRegion" in html
    assert "API.coverageTargetRegion" in html
    enter = html[
        html.index("async function enterScraperSoloView") : html.index(
            "function cacheReferenceCandidate"
        )
    ]
    assert "SCRAPER_A_REFERENCE" in enter or "ensureVisualA0" in enter
    assert "applyCachedCandidate(0)" not in enter
    assert 'id="toggle-toolbar-pot"' in html
    assert "evaluate_candidate(" not in html
    a0_ref = html[html.index("const SCRAPER_A_REFERENCE") : html.index("let scraperSoloMode")]
    assert "width_mm: 2.5" in a0_ref
    assert "thickness_mm: 2.5" in a0_ref
    assert "rotation_angle_deg: 0" in a0_ref
    draw = html[html.index("function drawScene3D") : html.index("async function loadViewerScene")]
    assert "coverageTarget" in draw
    assert "toggle-scraper-points" in draw
    assert "#55ffff" in draw
    assert "showEvaluationEnvelope" not in draw
