"""ScraperBuilder procedural V1 tests."""

from __future__ import annotations

import numpy as np
import pytest

from nutella_scraper.domain.models.scraper import ScraperGeometry, ScraperPose
from nutella_scraper.engines.compute.scraper_builder import ScraperBuilder
from nutella_scraper.io.scraper_config_loader import load_scraper_geometry


class TestScraperBuilder:
    def test_builds_v1_volume_with_tip_radius(self) -> None:
        geometry = ScraperGeometry(
            width_mm=18.0,
            length_mm=95.0,
            thickness_mm=2.5,
            tip_radius_mm=1.5,
            curvature_radius_mm=35.0,
            bend_angle_deg=12.0,
        )
        mesh = ScraperBuilder().build(geometry)

        assert len(mesh.vertices) >= 8
        assert len(mesh.faces) >= 12
        assert mesh.volume > 0.0
        y_max = float(mesh.vertices[:, 1].max())
        tip_vertices = mesh.vertices[mesh.vertices[:, 1] >= y_max - 1.6]
        tip_radial = np.sqrt(tip_vertices[:, 0] ** 2 + tip_vertices[:, 2] ** 2)
        assert float(tip_radial.max()) <= geometry.tip_radius_mm + geometry.thickness_mm

    def test_rebuilds_after_parameter_change(self) -> None:
        builder = ScraperBuilder()
        first = builder.build(
            ScraperGeometry(width_mm=10.0, length_mm=80.0, thickness_mm=3.0)
        )
        second = builder.build(
            ScraperGeometry(width_mm=12.0, length_mm=80.0, thickness_mm=3.0)
        )

        assert not np.isclose(first.volume, second.volume)

    def test_posed_mesh_matches_transform(self) -> None:
        geometry = ScraperGeometry(width_mm=10.0, length_mm=10.0, thickness_mm=2.0)
        pose = ScraperPose(position_mm=(5.0, 0.0, 0.0))
        builder = ScraperBuilder()
        local = builder.build(geometry)
        posed = builder.build_posed(geometry, pose)

        assert not np.allclose(local.centroid, posed.centroid)


class TestScraperConfigLoader:
    def test_loads_racloir_v1_yaml(self, config_dir) -> None:
        geometry = load_scraper_geometry(config_dir / "scrapers" / "racloir_v1.yaml")

        assert geometry.id == "racloir_v1"
        assert geometry.width_mm == 18.0
        assert geometry.tip_radius_mm == 1.5
        assert geometry.bend_angle_deg == 12.0

    def test_rejects_missing_parameters(self, tmp_path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("id: broken\nparameters: []\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_scraper_geometry(path)
