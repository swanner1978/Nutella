"""Procedural Scraper3D V1 mesh generation — never a hardcoded mesh."""

from __future__ import annotations

import trimesh

from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.scraper_transform import (
    apply_tip_radius,
    bend_solid_around_y,
    pose_matrix,
)


class ScraperBuilder:
    """
    Builds watertight Scraper3D V1 volumes from parametric ScraperGeometry.

    All geometry is reconstructed procedurally on every build call.
    """

    def build(self, geometry: ScraperGeometry) -> trimesh.Trimesh:
        mesh = trimesh.creation.box(
            extents=(geometry.thickness_mm, geometry.length_mm, geometry.width_mm)
        )
        if geometry.tip_radius_mm > 0.0:
            mesh = apply_tip_radius(mesh, tip_radius_mm=geometry.tip_radius_mm)
        if geometry.curvature_radius_mm is not None and geometry.bend_angle_deg != 0.0:
            mesh = bend_solid_around_y(
                mesh,
                radius_mm=geometry.curvature_radius_mm,
                bend_angle_deg=geometry.bend_angle_deg,
            )
        return mesh

    def build_posed(
        self,
        geometry: ScraperGeometry,
        pose: ScraperPose,
    ) -> trimesh.Trimesh:
        mesh = self.build(geometry)
        mesh.apply_transform(pose_matrix(pose))
        return mesh
