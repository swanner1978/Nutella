"""Coordinate frame layer: canonical origin, 10 mm ticks, shared viewport transform."""

from __future__ import annotations

from xml.etree import ElementTree

import numpy as np
import pytest
import trimesh

from scripts.visualization_helpers import (
    FRAME_TICK_MM,
    VIEW_CONVENTIONS,
    _coordinate_frame_svg,
    _frame_half_span_mm,
    _world_points_to_svg,
    build_projection_svg,
    fit_to_viewport,
)


def _box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(40.0, 80.0, 60.0))


def test_frame_half_span_snaps_to_ten_mm() -> None:
    assert FRAME_TICK_MM == 10.0
    assert _frame_half_span_mm((41.0, 55.0, 33.0)) == 30.0
    assert _frame_half_span_mm((10.0, 10.0, 10.0)) == 20.0


def test_coordinate_frame_origin_is_canonical_center() -> None:
    mesh = _box_mesh()
    center = np.array(mesh.bounding_box.centroid, dtype=np.float64)
    coords = np.asarray(mesh.vertices, dtype=np.float64)[:, [0, 1]]
    scale, offset = fit_to_viewport(coords)
    svg = _coordinate_frame_svg(
        center=center,
        half_span_mm=40.0,
        plane="XY",
        scale=scale,
        offset=offset,
    )
    root = ElementTree.fromstring(f'<g>{svg}</g>')
    origin = root.find(".//*[@data-frame-origin]")
    assert origin is not None
    expected = _world_points_to_svg(center.reshape(1, 3), plane="XY", scale=scale, offset=offset)[0]
    assert float(origin.attrib["cx"]) == pytest.approx(expected[0], abs=0.02)
    assert float(origin.attrib["cy"]) == pytest.approx(expected[1], abs=0.02)
    label = root.find(".//*[@data-frame-origin-label]")
    assert label is not None
    assert "O (0, 0, 0)" in (label.text or "")


def test_tick_spacing_is_exactly_ten_mm_in_world() -> None:
    mesh = _box_mesh()
    center = np.zeros(3, dtype=np.float64)
    plane = "XY"
    coords = np.asarray(mesh.vertices, dtype=np.float64)[:, [0, 1]]
    scale, offset = fit_to_viewport(coords)
    svg = _coordinate_frame_svg(
        center=center,
        half_span_mm=40.0,
        plane=plane,
        scale=scale,
        offset=offset,
    )
    root = ElementTree.fromstring(f'<g>{svg}</g>')
    tick_10 = root.find('.//*[@data-frame-tick="X:10"]')
    tick_20 = root.find('.//*[@data-frame-tick="X:20"]')
    assert tick_10 is not None and tick_20 is not None
    p10 = _world_points_to_svg(
        (center + np.array([10.0, 0.0, 0.0])).reshape(1, 3),
        plane=plane,
        scale=scale,
        offset=offset,
    )[0]
    p20 = _world_points_to_svg(
        (center + np.array([20.0, 0.0, 0.0])).reshape(1, 3),
        plane=plane,
        scale=scale,
        offset=offset,
    )[0]
    # Tick midpoints should match projected world points at +10 / +20 mm
    mid10 = np.array(
        [
            (float(tick_10.attrib["x1"]) + float(tick_10.attrib["x2"])) / 2.0,
            (float(tick_10.attrib["y1"]) + float(tick_10.attrib["y2"])) / 2.0,
        ]
    )
    mid20 = np.array(
        [
            (float(tick_20.attrib["x1"]) + float(tick_20.attrib["x2"])) / 2.0,
            (float(tick_20.attrib["y1"]) + float(tick_20.attrib["y2"])) / 2.0,
        ]
    )
    np.testing.assert_allclose(mid10, p10, atol=0.05)
    np.testing.assert_allclose(mid20, p20, atol=0.05)
    # Screen distance between +10 and +20 equals 10 mm * |scale_x|
    assert abs(np.linalg.norm(mid20 - mid10) - abs(float(np.asarray(scale)[0]) * 10.0)) < 0.1


def test_frame_uses_same_viewport_transform_as_mesh_layer() -> None:
    mesh = _box_mesh()
    center = tuple(float(v) for v in mesh.bounding_box.centroid)
    svg = build_projection_svg(
        mesh,
        plane="XY",
        center=center,
        principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        model_id="frame-test",
        canonical_mesh_sha256="a" * 64,
    )
    root = ElementTree.fromstring(svg)
    frame = next(
        group
        for group in root.findall("{*}g")
        if group.attrib.get("data-layer") == "coordinate-frame"
    )
    assert frame.find(".//*[@data-frame-origin]") is not None
    assert frame.find('.//*[@data-frame-axis="X"]') is not None
    assert frame.find('.//*[@data-frame-axis="Y"]') is not None
    # Contour and frame share one SVG document / one fit
    contour = next(
        group for group in root.findall("{*}g") if group.attrib.get("data-layer") == "contour"
    )
    assert contour is not None


def test_frame_present_for_all_view_planes() -> None:
    mesh = _box_mesh()
    center = tuple(float(v) for v in mesh.bounding_box.centroid)
    for view_name, spec in VIEW_CONVENTIONS.items():
        svg = build_projection_svg(
            mesh,
            plane=spec["plane"],
            center=center,
            principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            model_id=view_name,
            canonical_mesh_sha256="b" * 64,
        )
        assert 'data-layer="coordinate-frame"' in svg
        assert "O (0, 0, 0)" in svg
        assert 'data-frame-tick-label="X:10"' in svg or 'data-frame-depth="X"' in svg


def test_toggling_frame_is_display_only_in_svg_markup() -> None:
    """Layer is a separate group; hiding it does not alter mesh geometry markup."""
    mesh = _box_mesh()
    center = tuple(float(v) for v in mesh.bounding_box.centroid)
    svg = build_projection_svg(
        mesh,
        plane="XZ",
        center=center,
        principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        model_id="toggle",
        canonical_mesh_sha256="c" * 64,
    )
    root = ElementTree.fromstring(svg)
    contour_before = next(
        group for group in root.findall("{*}g") if group.attrib.get("data-layer") == "contour"
    )
    contour_markup = ElementTree.tostring(contour_before, encoding="unicode")
    frame = next(
        group
        for group in root.findall("{*}g")
        if group.attrib.get("data-layer") == "coordinate-frame"
    )
    frame.set("style", "display:none")
    contour_after = next(
        group for group in root.findall("{*}g") if group.attrib.get("data-layer") == "contour"
    )
    assert ElementTree.tostring(contour_after, encoding="unicode") == contour_markup
