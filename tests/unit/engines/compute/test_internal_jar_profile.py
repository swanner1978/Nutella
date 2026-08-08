"""InternalJarProfile extraction and mesh-independence tests."""

from __future__ import annotations

import numpy as np

from nutella_scraper.domain.models.canonical import JarCanonicalModel, JarProfilePoint
from nutella_scraper.engines.compute.internal_jar_profile_builder import InternalJarProfileBuilder
from nutella_scraper.engines.compute.internal_jar_surface_builder import InternalJarSurfaceBuilder
from nutella_scraper.engines.compute.jar_mesh_builder import JarMeshBuilder
from nutella_scraper.engines.visualization.profile_projector import (
    profile_side_path,
    profile_top_path,
)

MAX_RECONSTRUCTION_ERROR_MM = 0.15


def _jar_model(model_id: str) -> JarCanonicalModel:
    return JarCanonicalModel(
        id=model_id,
        version="1",
        meridian_profile=(
            JarProfilePoint(z_mm=0.0, r_mm=40.0),
            JarProfilePoint(z_mm=100.0, r_mm=40.0),
        ),
        neck_inner_diameter_mm=80.0,
        total_height_mm=100.0,
    )


class TestInternalJarProfileBuilder:
    def test_coarse_and_fine_mesh_produce_same_meridian(self) -> None:
        coarse = JarMeshBuilder().to_canonical(_jar_model("coarse"), theta_segments=16)
        fine = JarMeshBuilder().to_canonical(_jar_model("fine"), theta_segments=96)

        profile_coarse = InternalJarProfileBuilder().from_internal(
            InternalJarSurfaceBuilder().from_canonical(coarse)
        )
        profile_fine = InternalJarProfileBuilder().from_internal(
            InternalJarSurfaceBuilder().from_canonical(fine)
        )

        assert len(profile_coarse.meridian) == len(profile_fine.meridian)
        for left, right in zip(profile_coarse.meridian, profile_fine.meridian, strict=True):
            assert abs(left.y_mm - right.y_mm) <= 0.5
            assert abs(left.radius_mm - right.radius_mm) <= 1.0
        assert abs(profile_coarse.top_inner_radius_mm - 40.0) <= 1.0

    def test_reconstruction_error_within_tolerance(self) -> None:
        jar = JarMeshBuilder().to_canonical(_jar_model("jar"), theta_segments=32)
        profile = InternalJarProfileBuilder().from_internal(
            InternalJarSurfaceBuilder().from_canonical(jar)
        )
        quality = profile.reconstruction
        assert quality.meridian_hausdorff_mm <= MAX_RECONSTRUCTION_ERROR_MM
        assert quality.top_contour_hausdorff_mm <= MAX_RECONSTRUCTION_ERROR_MM
        assert quality.meridian_max_error_mm <= MAX_RECONSTRUCTION_ERROR_MM
        assert quality.top_contour_max_error_mm <= MAX_RECONSTRUCTION_ERROR_MM

    def test_side_profile_is_single_closed_polyline(self) -> None:
        jar = JarMeshBuilder().to_canonical(_jar_model("jar"), theta_segments=32)
        profile = InternalJarProfileBuilder().from_internal(
            InternalJarSurfaceBuilder().from_canonical(jar)
        )
        side = profile_side_path(profile)
        assert len(side) >= 4
        assert profile.meridian_point_count >= 4

    def test_top_profile_is_near_circular(self) -> None:
        jar = JarMeshBuilder().to_canonical(_jar_model("jar"), theta_segments=32)
        profile = InternalJarProfileBuilder().from_internal(
            InternalJarSurfaceBuilder().from_canonical(jar)
        )
        top = profile_top_path(profile)
        radial = np.sqrt(top[:, 0] ** 2 + top[:, 1] ** 2)
        assert float(radial.max()) - float(radial.min()) <= 0.5
        assert profile.reconstruction.top_contour_is_circular is True
