"""Interior surface builder and contact analyzer tests."""

from __future__ import annotations

import numpy as np
import pytest

from nutella_scraper.domain.models.canonical import CanonicalModel3D
from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.interior_contact_analyzer import analyze_interior_contact
from nutella_scraper.engines.compute.internal_jar_surface_builder import InternalJarSurfaceBuilder
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder
from nutella_scraper.engines.compute.scraper_builder import ScraperBuilder


class TestInteriorSurfaceBuilder:
    def test_samples_are_inside_canonical_mesh(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
    ) -> None:
        jar_mesh = JarMeshBuilder().from_internal(
            InternalJarSurfaceBuilder().from_canonical(cylindrical_jar_canonical)
        )
        interior = InternalJarSurfaceBuilder().from_canonical(cylindrical_jar_canonical)

        assert interior.sample_count > 0
        assert interior.sample_count <= len(jar_mesh.faces)
        mesh_radial = np.sqrt(jar_mesh.vertices[:, 0] ** 2 + jar_mesh.vertices[:, 2] ** 2)
        max_wall_radius = float(mesh_radial.max())
        for point in interior.sample_points_mm:
            radial = float(np.sqrt(point[0] ** 2 + point[2] ** 2))
            assert radial <= max_wall_radius + 1e-3


class TestInteriorContactAnalyzer:
    def test_produces_contact_for_near_wall_scraper(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
        wall_scraper_geometry: ScraperGeometry,
        wall_scraper_pose: ScraperPose,
    ) -> None:
        jar_mesh = JarMeshBuilder().from_internal(
            InternalJarSurfaceBuilder().from_canonical(cylindrical_jar_canonical)
        )
        interior = InternalJarSurfaceBuilder().from_canonical(cylindrical_jar_canonical)
        scraper_mesh = ScraperBuilder().build_posed(wall_scraper_geometry, wall_scraper_pose)

        distances, contacts = analyze_interior_contact(
            interior,
            scraper_mesh,
            contact_threshold_mm=1.0,
        )

        assert len(distances) == len(jar_mesh.faces)
        assert np.any(np.isfinite(distances))
        assert len(contacts) > 0

    def test_sample_count_is_lower_than_triangle_count(
        self,
        cylindrical_jar_canonical: CanonicalModel3D,
    ) -> None:
        jar_mesh = JarMeshBuilder().from_internal(
            InternalJarSurfaceBuilder().from_canonical(cylindrical_jar_canonical)
        )
        interior = InternalJarSurfaceBuilder().from_canonical(cylindrical_jar_canonical)

        assert interior.sample_count <= len(jar_mesh.faces)
