"""Domain model tests."""

from __future__ import annotations

import numpy as np

from nutella_scraper.domain.models.canonical import (
    BoundingBox,
    CanonicalModel3D,
    GeometricMetadata,
    JarCanonicalModel,
    JarProfilePoint,
    MeshData,
)
from nutella_scraper.domain.models.contact import ContactResult
from nutella_scraper.domain.models.views import ViewProjectionCache


class TestCanonicalModel3D:
    def test_provenance_is_canonical_3d(self) -> None:
        geometry = GeometricMetadata(
            bounding_box=BoundingBox(0, 0, 0, 1, 1, 1),
            dimensions_mm=(1.0, 1.0, 1.0),
            center_mm=(0.5, 0.5, 0.5),
            principal_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            volume_mm3=None,
            is_watertight=False,
            vertex_count=0,
            face_count=0,
        )
        model = CanonicalModel3D(
            id="test",
            source_hash="abc",
            format="stl",
            source_path=__import__("pathlib").Path("test.stl"),
            mesh=MeshData(vertices=(), faces=()),
            bounds=BoundingBox(0, 0, 0, 1, 1, 1),
            geometry=geometry,
            frame=__import__(
                "nutella_scraper.domain.models.canonical", fromlist=["RigidTransform"]
            ).RigidTransform(),
        )
        assert model.provenance == "canonical_3d"


class TestViewProjectionCache:
    def test_provenance_is_visualization_only(self) -> None:
        from nutella_scraper.domain.models.views import ProjectedView, ProjectionMetadata

        meta = ProjectionMetadata(
            plane="XZ", camera={}, scale=1.0, width_px=100, height_px=100
        )
        view = ProjectedView(plane="XZ", asset_path=None, svg_content=None, metadata=meta)
        cache = ViewProjectionCache(
            model_id="test", profile_view=view, top_view=view
        )
        assert cache.provenance == "visualization_projection"


class TestContactResult:
    def test_provenance_is_computed_metric(self) -> None:
        result = ContactResult(
            model_id="m",
            jar_id="j",
            coverage_score=0.0,
            touched_face_ids=frozenset(),
            untouched_face_ids=frozenset(),
            contact_distance_map=np.array([]),
        )
        assert result.provenance == "computed_metric"


class TestJarCanonicalModel:
    def test_load_from_profile(self) -> None:
        jar = JarCanonicalModel(
            id="nutella_400g",
            version="1",
            meridian_profile=(JarProfilePoint(0, 47.5),),
            neck_inner_diameter_mm=58.0,
            total_height_mm=105.0,
        )
        assert jar.id == "nutella_400g"
