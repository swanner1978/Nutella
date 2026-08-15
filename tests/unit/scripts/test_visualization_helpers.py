"""Tests for visualization-only SVG layers."""



from __future__ import annotations



from xml.etree import ElementTree



import pytest

import trimesh

from scripts.visualization_helpers import (

    VIEW_CONVENTIONS,

    build_projection_svg,

    displayed_view_entry,

    projected_extent_mm,

)





def test_projection_contains_one_canonical_model_and_toggleable_layers() -> None:

    mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))

    mesh_hash = "a" * 64



    svg = build_projection_svg(

        mesh,

        plane="XZ",

        center=(0.0, 0.0, 0.0),

        principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),

        model_id="model-1",

        canonical_mesh_sha256=mesh_hash,

    )



    root = ElementTree.fromstring(svg)

    layers = {group.attrib["data-layer"]: group for group in root.findall("{*}g")}



    assert root.attrib["data-model-id"] == "model-1"

    assert root.attrib["data-canonical-mesh-sha256"] == mesh_hash

    assert set(layers) == {

        "contour",

        "wireframe",

        "vertices",

        "bounding-box",

        "principal-axes",

        "coordinate-frame",

    }

    assert "display:none" not in layers["contour"].attrib.get("style", "")

    assert "display:none" not in layers["coordinate-frame"].attrib.get("style", "")

    for name in ("wireframe", "vertices", "bounding-box", "principal-axes"):

        assert layers[name].attrib["style"] == "display:none"

    assert "jar-layer" not in svg

    assert "scraper-layer" not in svg





def test_mesh_projections_use_world_planes_and_extents() -> None:

    mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))

    mesh_hash = "c" * 64



    side_svg = build_projection_svg(

        mesh,

        plane="XZ",

        center=(0.0, 0.0, 0.0),

        principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),

        model_id="model-3",

        canonical_mesh_sha256=mesh_hash,

    )

    top_svg = build_projection_svg(

        mesh,

        plane="XY",

        center=(0.0, 0.0, 0.0),

        principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),

        model_id="model-3",

        canonical_mesh_sha256=mesh_hash,

    )



    side_root = ElementTree.fromstring(side_svg)

    top_root = ElementTree.fromstring(top_svg)

    assert side_root.attrib["data-plane"] == "XZ"

    assert side_root.attrib["data-view-axis"] == "-Y"

    assert top_root.attrib["data-plane"] == "XY"

    assert top_root.attrib["data-view-axis"] == "Z"



    side_layer = next(

        group

        for group in side_root.findall("{*}g")

        if group.attrib.get("data-layer") == "bounding-box"

    )

    top_layer = next(

        group

        for group in top_root.findall("{*}g")

        if group.attrib.get("data-layer") == "bounding-box"

    )

    side_rect = side_layer.find("{*}rect")

    top_rect = top_layer.find("{*}rect")

    assert side_rect is not None

    assert top_rect is not None



    side_w = float(side_rect.attrib["width"])

    side_h = float(side_rect.attrib["height"])

    top_w = float(top_rect.attrib["width"])

    top_h = float(top_rect.attrib["height"])



    side_span_x, side_span_z = 10.0, 30.0

    top_span_x, top_span_y = 10.0, 20.0

    assert side_w / side_h == pytest.approx(side_span_x / side_span_z, rel=1e-3)

    assert top_w / top_h == pytest.approx(top_span_x / top_span_y, rel=1e-3)





def test_displayed_view_entry_documents_side_and_top_conventions() -> None:

    side = displayed_view_entry(

        view_name="side",

        filename="side_composite.svg",

        sha256="d" * 64,

        canonical_mesh_sha256="e" * 64,

    )

    top = displayed_view_entry(

        view_name="top",

        filename="top_composite.svg",

        sha256="f" * 64,

        canonical_mesh_sha256="e" * 64,

    )



    assert side["plane"] == VIEW_CONVENTIONS["side"]["plane"] == "XY"

    assert side["view_axis"] == "Z"

    assert side["label_en"] == "Profile View"

    assert top["plane"] == VIEW_CONVENTIONS["top"]["plane"] == "X-Z"

    assert top["view_axis"] == "Y"

    assert top["label_en"] == "Top View"





def test_analytical_view_extents_use_profile_and_top_planes() -> None:

    assert projected_extent_mm((10.0, 20.0, 30.0), "XZ") == (10.0, 30.0)
    assert projected_extent_mm((10.0, 20.0, 30.0), "X-Z") == (10.0, 30.0)

    assert projected_extent_mm((10.0, 20.0, 30.0), "XY") == (10.0, 20.0)





def test_box_side_contour_is_not_the_complete_wireframe() -> None:

    mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))



    svg = build_projection_svg(

        mesh,

        plane="XZ",

        center=(0.0, 0.0, 0.0),

        principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),

        model_id="model-2",

        canonical_mesh_sha256="b" * 64,

    )



    root = ElementTree.fromstring(svg)

    layers = {group.attrib["data-layer"]: group for group in root.findall("{*}g")}

    contour_path = layers["contour"].find("{*}path")

    wireframe_path = layers["wireframe"].find("{*}path")



    assert contour_path is not None

    assert wireframe_path is not None

    assert contour_path.attrib["d"].count("M") == 4

    assert wireframe_path.attrib["d"].count("M") > 4

