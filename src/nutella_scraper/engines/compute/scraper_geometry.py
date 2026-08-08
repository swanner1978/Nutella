"""Parametric scraper solid mesh generation and pose application."""

from __future__ import annotations

import trimesh

from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.scraper_builder import ScraperBuilder
from nutella_scraper.engines.compute.scraper_transform import pose_matrix

__all__ = ["ScraperGeometryBuilder", "pose_matrix"]


class ScraperGeometryBuilder:
    """Backward-compatible facade over procedural ScraperBuilder."""

    def __init__(self, *, builder: ScraperBuilder | None = None) -> None:
        self._builder = builder or ScraperBuilder()

    def build(self, geometry: ScraperGeometry) -> trimesh.Trimesh:
        return self._builder.build(geometry)

    def build_posed(
        self,
        geometry: ScraperGeometry,
        pose: ScraperPose,
    ) -> trimesh.Trimesh:
        return self._builder.build_posed(geometry, pose)
