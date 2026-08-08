"""Signed distance field queries between scraper and jar."""

from __future__ import annotations

import numpy as np
import trimesh

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder
from nutella_scraper.engines.compute.scraper_geometry import ScraperGeometryBuilder


class DistanceFieldQuery:
    """3D distance queries between scraper mesh and jar inner surface."""

    def __init__(
        self,
        *,
        jar_mesh_builder: JarMeshBuilder | None = None,
        scraper_geometry: ScraperGeometryBuilder | None = None,
    ) -> None:
        self._jar_mesh_builder = jar_mesh_builder or JarMeshBuilder()
        self._scraper_geometry = scraper_geometry or ScraperGeometryBuilder()

    def min_distance(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
    ) -> float:
        jar_mesh = self._jar_mesh_builder.from_canonical(jar)
        scraper_mesh = self._scraper_geometry.build_posed(geometry, pose)
        return self._min_distance_meshes(jar_mesh, scraper_mesh)

    def mean_distance(
        self,
        jar: CanonicalModel3D,
        geometry: ScraperGeometry,
        pose: ScraperPose,
    ) -> float:
        jar_mesh = self._jar_mesh_builder.from_canonical(jar)
        scraper_mesh = self._scraper_geometry.build_posed(geometry, pose)
        sample_points = jar_mesh.triangles_center
        _, distances, _ = trimesh.proximity.closest_point(scraper_mesh, sample_points)
        finite = distances[np.isfinite(distances)]
        if len(finite) == 0:
            return float("inf")
        return float(np.mean(finite))

    @staticmethod
    def _min_distance_meshes(jar_mesh: trimesh.Trimesh, scraper_mesh: trimesh.Trimesh) -> float:
        sample_points = jar_mesh.triangles_center
        _, distances, _ = trimesh.proximity.closest_point(scraper_mesh, sample_points)
        finite = distances[np.isfinite(distances)]
        if len(finite) == 0:
            return float("inf")
        return float(np.min(finite))
