"""Exact Scraper3D projection tests."""

from __future__ import annotations

from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.scraper_geometry import ScraperGeometryBuilder
from nutella_scraper.engines.visualization.scraper_result_projector import (
    LAYER_SCRAPER_CONTOUR,
    LAYER_SCRAPER_VOLUME,
    ScraperResultProjector,
)


def test_projects_exact_posed_scraper_mesh_in_both_views(
    internal_jar_surface: object,
) -> None:
    geometry = ScraperGeometry(width_mm=13.0, length_mm=76.0, thickness_mm=3.0)
    pose = ScraperPose(
        position_mm=(31.0, 42.0, -7.0),
        yaw_deg=27.0,
        pitch_deg=11.0,
        roll_deg=-8.0,
    )
    mesh = ScraperGeometryBuilder().build_posed(geometry, pose)

    projection = ScraperResultProjector().project(
        scraper_vertices=mesh.vertices,
        scraper_faces=mesh.faces,
        internal=internal_jar_surface,
    )

    assert projection.vertex_count == len(mesh.vertices)
    assert projection.face_count == len(mesh.faces)
    for layers in (projection.profile_layers, projection.top_layers):
        assert {layer.layer_type for layer in layers} == {
            LAYER_SCRAPER_VOLUME,
            LAYER_SCRAPER_CONTOUR,
        }
        assert all("<path" in layer.svg_fragment for layer in layers)


def test_pose_changes_projected_scraper_silhouette(
    internal_jar_surface: object,
) -> None:
    geometry = ScraperGeometry(width_mm=10.0, length_mm=70.0, thickness_mm=3.0)
    builder = ScraperGeometryBuilder()
    first = builder.build_posed(geometry, ScraperPose(position_mm=(20.0, 30.0, 0.0)))
    second = builder.build_posed(
        geometry,
        ScraperPose(position_mm=(35.0, 45.0, 5.0), yaw_deg=45.0),
    )
    projector = ScraperResultProjector()

    first_projection = projector.project(
        scraper_vertices=first.vertices,
        scraper_faces=first.faces,
        internal=internal_jar_surface,
    )
    second_projection = projector.project(
        scraper_vertices=second.vertices,
        scraper_faces=second.faces,
        internal=internal_jar_surface,
    )

    assert (
        first_projection.profile_layers[0].svg_fragment
        != second_projection.profile_layers[0].svg_fragment
    )
    assert (
        first_projection.top_layers[0].svg_fragment
        != second_projection.top_layers[0].svg_fragment
    )
